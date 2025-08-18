# backend/api/voice_chat.py
import time
import re
import io
import json
import uuid
import asyncio
from typing import Optional
from contextlib import suppress as contextlib_suppress

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse, Response

from core.stt_whisper import transcribe_bytes, transcribe_audio  # file/bytes helpers
from core.llm_hf import stream_hf_chat  # should yield incremental text tokens

# TTS
from core.tts_coqui import stream_tts_pcm
from config.settings import TTS_SAMPLE_RATE

# ---- Audio utils ---
try:
    from utils.audio_utils import pcm16_to_wav_bytes, frame_generator, collect_segments_vad
except Exception:
    # fallback pcm->wav writer
    import wave
    def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()
    frame_generator = None
    collect_segments_vad = None

# Optional config for partial behaviour
try:
    from config.settings import AUDIO_SAMPLE_RATE, FRAME_MS
except Exception:
    AUDIO_SAMPLE_RATE = 16000
    FRAME_MS = 20

CHUNK_SIZE = int((FRAME_MS / 1000.0) * AUDIO_SAMPLE_RATE * 2)  # bytes per frame window
PARTIAL_EVERY = 2  # emit partial every N chunks

router = APIRouter(tags=["Voice"])

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
    s = s.lstrip()
    QUOTES = {'"', "'", "“", "”", "‘", "’"}
    while s and s[0] in QUOTES:
        s = s[1:].lstrip()
    return s

async def _tts_stream_send(websocket: WebSocket, text: str):
    """
    Send TTS as: tts_start(json) -> binary PCM16 chunks -> tts_done(json)
    """
    if not (text or "").strip():
        return
    with contextlib_suppress(WebSocketDisconnect):
        await websocket.send_json({"type": "tts_start", "sample_rate": int(TTS_SAMPLE_RATE or 22050), "format": "pcm16"})
    try:
        for chunk in stream_tts_pcm(text):
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_bytes(chunk)
    finally:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "tts_done"})

async def stream_llm_safely(websocket: WebSocket, prompt: str, *, max_new_tokens: int = 128, max_sentences: int = 2) -> str:
    """
    Stream tokens (delta) and return the final assistant text for TTS.
    Sends JSON messages:
      {"type":"delta","text": "..."}
      {"type":"assistant_final","text":"..."}
    After assistant_final, TTS streaming is started.
    """
    buffer = ""
    last_sent_idx = 0
    max_chars = 240

    for tok in stream_hf_chat(prompt, max_new_tokens=max_new_tokens):
        if not tok:
            continue
        if BAD_MARKERS_RE.search(tok):
            break

        buffer += tok

        if last_sent_idx == 0:
            cleaned = _clean_leading_strict(buffer)
            if cleaned != buffer:
                buffer = cleaned

        # Cap by chars
        if len(buffer) >= max_chars:
            buffer = buffer[:max_chars].rstrip()
            delta = buffer[last_sent_idx:]
            if delta.strip():
                with contextlib_suppress(WebSocketDisconnect):
                    await websocket.send_json({"type": "delta", "text": delta})
            break

        # Cap by sentences
        if len(re.findall(r"[\.!?]", buffer)) >= max_sentences:
            delta = buffer[last_sent_idx:]
            if delta.strip():
                with contextlib_suppress(WebSocketDisconnect):
                    await websocket.send_json({"type": "delta", "text": delta})
            break

        # Regular flush
        delta = buffer[last_sent_idx:]
        if delta.strip():
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_json({"type": "delta", "text": delta})
            last_sent_idx = len(buffer)

    final_text = buffer.strip()
    if final_text:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "assistant_final", "text": final_text})
        # TTS stream
        await _tts_stream_send(websocket, final_text)

    return final_text

# -----------------------------
# HTTP: file upload -> transcript
# -----------------------------
@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
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

# -----------------------------
# Simple HTTP TTS for quick test (used by test_http_streaming_tts.py)
# -----------------------------
@router.post("/tts")
async def tts_http(payload: dict = Body(...)):
    """
    HTTP endpoint to generate TTS audio (wav) from text.
    Returns a streaming WAV (assembled from PCM chunks).
    """
    text = payload.get("text", "")
    if not text.strip():
        return {"error": "Text is empty"}

    pcm_chunks = stream_tts_pcm(text)
    pcm_data = b"".join(pcm_chunks)
    wav_bytes = pcm16_to_wav_bytes(pcm_data, sample_rate=TTS_SAMPLE_RATE or 22050)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")

# -------------------------------------------------------------
# WebSocket: audio bytes -> STT (partials + finals) -> LLM stream -> TTS stream
# Frontend protocol (recommended):
#  - Client sends JSON: {"type":"start", "sample_rate":16000}
#  - Client sends binary PCM16 ArrayBuffer frames
#  - Client sends text message "done" (or {"event":"end"}) to indicate finalization
# Server sends:
#  - {"type":"ready"}
#  - {"type":"ack","chunk":n}
#  - {"type":"stt_partial","text":"..."} or {"type":"transcript","final":true,"text":"..."}
#  - {"type":"delta","text":"..."} (LLM streaming)
#  - {"type":"assistant_final","text":"..."}
#  - {"type":"tts_start","sample_rate":..., "format":"pcm16"}
#  - binary PCM16 chunks
#  - {"type":"tts_done"}
# -------------------------------------------------------------
@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    buf = io.BytesIO()
    chunk_count = 0
    last_partial = ""
    history = []  # you can keep chat history per session here

    await websocket.send_json({"type": "ready", "session": session_id})

    try:
        while True:
            msg = await websocket.receive()

            # Text/json messages
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                # some clients send simple "done"
                low = (txt or "").strip().lower()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = None

                # Start message from client, e.g. {"type":"start","sample_rate":16000}
                if isinstance(data, dict) and data.get("type") == "start":
                    await websocket.send_json({"type": "started", "sample_rate": data.get("sample_rate", AUDIO_SAMPLE_RATE)})
                    continue

                # End signal -> final STT on buffer
                if low in ("done", "end") or (isinstance(data, dict) and data.get("event") in ("end", "done")):
                    # Finalize STT on full buffer
                    pcm = buf.getvalue()
                    if pcm:
                        # Wrap PCM into WAV bytes and transcribe via transcribe_bytes
                        wav = pcm16_to_wav_bytes(pcm, sample_rate=AUDIO_SAMPLE_RATE)
                        final_text = transcribe_bytes(wav, mime="audio/wav")
                    else:
                        final_text = ""

                    await websocket.send_json({"type": "transcript", "final": True, "text": final_text})
                    chunk_count = 0
                    buf = io.BytesIO()  # reset buffer

                    # If we have final_text, call LLM stream -> TTS
                    if final_text:
                        prompt = final_text  # optionally enrich prompt/history here
                        try:
                            await stream_llm_safely(websocket, prompt)
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": f"llm error: {e}"})
                    # keep connection open for more turns
                    continue

                # Unknown text, ignore
                continue

            # Binary audio frames
            if "bytes" in msg and msg["bytes"] is not None:
                payload: bytes = msg["bytes"]
                if not payload:
                    continue
                buf.write(payload)
                chunk_count += 1
                # ack
                await websocket.send_json({"type": "ack", "chunk": chunk_count, "bytes": len(payload)})

                # produce partial STT occasionally
                if chunk_count % PARTIAL_EVERY == 0:
                    # take recent tail
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
                            # Optionally kick LLM on long-enough partials
                            if len(partial) > 10 and partial[-1] in (" ", ".", "?", "!"):
                                # start a quick LLM stream on the partial (optional)
                                try:
                                    await stream_llm_safely(websocket, partial, max_new_tokens=64, max_sentences=1)
                                except Exception as e:
                                    await websocket.send_json({"type": "error", "message": f"llm error: {e}"})
                continue

    except WebSocketDisconnect:
        # client disconnected
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            await websocket.close()
        return
