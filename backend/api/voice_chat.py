# backend/api/voice_chat.py
import re
import io
import json
import uuid
import asyncio
from contextlib import suppress as contextlib_suppress

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse

from core.stt_whisper import transcribe_bytes
from core.tts_coqui import stream_tts_pcm
from config.settings import TTS_SAMPLE_RATE, AUDIO_SAMPLE_RATE, FRAME_MS, USE_LOCAL_GGUF

# Database integration
from db import save_message, load_history, clear_session

# Conditional LLM import
if USE_LOCAL_GGUF:
    from core.llm_gguf import stream_hf_chat
else:
    from core.llm_hf import stream_hf_chat

# ---- Audio utils ---
try:
    from utils.audio_utils import pcm16_to_wav_bytes
except Exception:
    import wave
    def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

CHUNK_SIZE = int((FRAME_MS / 1000.0) * AUDIO_SAMPLE_RATE * 2)
PARTIAL_EVERY = 2

router = APIRouter(tags=["Voice"])

# --- Core system prompt ---
SYSTEM_CORE = (
    "You are a helpful, concise voice assistant. "
    "Reply in one or two short sentences. "
    "Do not invent a multi-turn dialogue. "
    "Do not propose plans, meetings, times, prices, or locations unless the user explicitly asks. "
    "Do not write 'User:' or 'Assistant:'. "
    "Do not wrap your reply in quotes and do not start with a newline. "
    "Match the user's language."
)
BAD_MARKERS_RE = re.compile(r"(Assistant:|User:|</s>|<\|eot_id\|>)", re.IGNORECASE)


def _clean_leading_strict(s: str) -> str:
    """Remove leading quotes and spaces."""
    s = s.lstrip()
    QUOTES = {'"', "'", "“", "”", "‘", "’"}
    while s and s[0] in QUOTES:
        s = s[1:].lstrip()
    return s


def extract_tags(message: str) -> str:
    """Extract simple tags from a message (basic keyword-based)."""
    words = re.findall(r"\b\w+\b", message.lower())
    stopwords = {"the", "is", "are", "a", "an", "and", "or", "in", "of", "on", "to", "about"}
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]
    unique = list(dict.fromkeys(keywords))[:5]
    return ",".join(unique)


async def _tts_stream_send(websocket: WebSocket, text: str):
    """Send text-to-speech audio chunks to the client."""
    if not (text or "").strip():
        return

    with contextlib_suppress(WebSocketDisconnect):
        await websocket.send_json({
            "type": "tts_start",
            "sample_rate": int(TTS_SAMPLE_RATE or 22050),
            "format": "pcm16"
        })

    try:
        for chunk in stream_tts_pcm(text):
            if not chunk:
                continue
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_bytes(chunk)
            await asyncio.sleep(0)

        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_bytes(b"")

    finally:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "tts_done"})


async def stream_llm_safely(websocket: WebSocket, session_id: str, prompt: str,
                            *, max_new_tokens: int = 256, max_sentences: int = 5) -> str:
    """Stream LLM output safely, save assistant reply to DB, and send via WebSocket."""
    buffer = ""
    last_sent_idx = 0
    max_chars = 600  

    # 🔹 Load past conversation for context
    history = load_history(session_id, limit=20)
    full_prompt = ""
    for h in history:
        full_prompt += f"{h['role'].capitalize()}: {h['content']}\n"
    full_prompt += f"User: {prompt}\nAssistant:"

    for tok in stream_hf_chat(full_prompt, max_new_tokens=max_new_tokens):
        if not tok:
            continue
        if BAD_MARKERS_RE.search(tok):
            break

        buffer += tok

        if last_sent_idx == 0:
            cleaned = _clean_leading_strict(buffer)
            if cleaned != buffer:
                buffer = cleaned

        # 🔹 Stop if too long
        if len(buffer) >= max_chars:
            buffer = buffer[:max_chars].rstrip()
            delta = buffer[last_sent_idx:]
            if delta.strip():
                with contextlib_suppress(WebSocketDisconnect):
                    await websocket.send_json({"type": "delta", "text": delta})
            break

        # 🔹 Stop if enough sentences
        if len(re.findall(r"[\.!?]", buffer)) >= max_sentences:
            delta = buffer[last_sent_idx:]
            if delta.strip():
                with contextlib_suppress(WebSocketDisconnect):
                    await websocket.send_json({"type": "delta", "text": delta})
            break

        # 🔹 Send incremental tokens
        delta = buffer[last_sent_idx:]
        if delta.strip():
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_json({"type": "delta", "text": delta})
            last_sent_idx = len(buffer)

    final_text = buffer.strip()
    if final_text:
        # Save assistant reply to DB with tags
        tags = extract_tags(final_text)
        save_message(session_id, "assistant", final_text, tags)

        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "assistant_final", "text": final_text})
        await _tts_stream_send(websocket, final_text)

    return final_text


# ---- REST Endpoints ----

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    name = (file.filename or "").lower()
    if not name.endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Only wav/mp3/m4a/ogg/webm supported")

    try:
        data = await file.read()
        text = transcribe_bytes(data, mime=file.content_type)
        return {"transcript": text}
    except Exception as e:
        print("Error in /transcribe:", e)
        raise HTTPException(status_code=500, detail="Internal transcription error")


@router.post("/tts")
async def tts_http(payload: dict = Body(...)):
    """Convert text to speech (TTS)."""
    text = payload.get("text", "")
    if not text.strip():
        return {"error": "Text is empty"}

    pcm_chunks = stream_tts_pcm(text)
    pcm_data = b"".join(pcm_chunks)
    wav_bytes = pcm16_to_wav_bytes(pcm_data, sample_rate=TTS_SAMPLE_RATE or 22050)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


# ---- WebSocket Endpoint ----

@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """Main WebSocket handler for voice chat."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    buf = io.BytesIO()
    chunk_count = 0
    last_partial = ""

    await websocket.send_json({"type": "ready", "session": session_id})

    try:
        while True:
            msg = await websocket.receive()

            # --- Handle text events ---
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                low = (txt or "").strip().lower()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = None

                # --- Start event ---
                if isinstance(data, dict) and data.get("type") == "start":
                    await websocket.send_json({
                        "type": "started",
                        "sample_rate": data.get("sample_rate", AUDIO_SAMPLE_RATE)
                    })
                    continue

                # --- End/Done event ---
                if low in ("done", "end") or (isinstance(data, dict) and data.get("event") in ("end", "done")):
                    pcm = buf.getvalue()
                    if pcm:
                        wav = pcm16_to_wav_bytes(pcm, sample_rate=AUDIO_SAMPLE_RATE)
                        final_text = transcribe_bytes(wav, mime="audio/wav")
                    else:
                        final_text = ""

                    await websocket.send_json({"type": "transcript", "final": True, "text": final_text})
                    chunk_count = 0
                    buf = io.BytesIO()

                    if final_text:
                        # Save user message with tags
                        tags = extract_tags(final_text)
                        save_message(session_id, "user", final_text, tags)

                        # Generate + save assistant reply
                        await stream_llm_safely(websocket, session_id, final_text)
                    continue

                continue

            # --- Handle audio chunks ---
            if "bytes" in msg and msg["bytes"] is not None:
                payload: bytes = msg["bytes"]
                if not payload:
                    continue
                buf.write(payload)
                chunk_count += 1
                await websocket.send_json({"type": "ack", "chunk": chunk_count, "bytes": len(payload)})

                # --- Partial STT ---
                if chunk_count % PARTIAL_EVERY == 0:
                    recent = buf.getvalue()[-(PARTIAL_EVERY * CHUNK_SIZE):]
                    if recent:
                        try:
                            wav = pcm16_to_wav_bytes(recent, sample_rate=AUDIO_SAMPLE_RATE)
                            partial = transcribe_bytes(wav, mime="audio/wav")
                            partial = (partial or "").strip()
                        except Exception as e:
                            partial = ""
                            print("partial stt error:", e)

                        if partial and partial != last_partial:
                            last_partial = partial
                            await websocket.send_json({"type": "stt_partial", "text": partial})

                            # (Optional) save partials if needed
                            if len(partial) > 10 and partial[-1] in (" ", ".", "?", "!"):
                                tags = extract_tags(partial)
                                save_message(session_id, "user", partial, tags)
                                await stream_llm_safely(websocket, session_id, partial,
                                                        max_new_tokens=64, max_sentences=1)
                continue

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            await websocket.close()
        return


# ---- New DB Endpoints ----

@router.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    """Fetch conversation history for a session."""
    history = load_history(session_id, limit=limit)
    return {"session_id": session_id, "history": history}


@router.delete("/clear/{session_id}")
async def clear_history(session_id: str):
    """Clear all history for a session."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
