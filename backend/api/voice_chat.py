import time
import re
import io
import json
import uuid
import asyncio
from typing import Optional
from contextlib import suppress as contextlib_suppress

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body

from core.stt_whisper import transcribe_bytes, transcribe_audio
from core.llm_hf import stream_hf_chat

from core.tts_coqui import stream_tts_with_preset
from config.settings import TTS_SAMPLE_RATE, AUDIO_SAMPLE_RATE, FRAME_MS, PERSONAS, TTS_PRESETS, DEFAULT_PERSONA_KEY

# ---- Audio utils ---
try:
    from utils.audio_utils import pcm16_to_wav_bytes, frame_generator, collect_segments_vad
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
    frame_generator = None
    collect_segments_vad = None

CHUNK_SIZE = int((FRAME_MS / 1000.0) * AUDIO_SAMPLE_RATE * 2)
PARTIAL_EVERY = 2

router = APIRouter(tags=["Voice"])

BAD_MARKERS_RE = re.compile(r"(Assistant:|User:|</s>|<\|eot_id\|>)", re.IGNORECASE)

def _clean_leading_strict(s: str) -> str:
    s = s.lstrip()
    QUOTES = {'"', "'", "“", "”", "‘", "’"}
    while s and s[0] in QUOTES:
        s = s[1:].lstrip()
    return s

# -------- Personas: session state ----------
from dataclasses import dataclass

@dataclass
class SessionState:
    persona_key: str = DEFAULT_PERSONA_KEY
    tts_preset_key: str = PERSONAS.get(DEFAULT_PERSONA_KEY, {}).get("tts_preset", "")
    system_prompt: str = PERSONAS.get(DEFAULT_PERSONA_KEY, {}).get("system_prompt", "")

    def apply(self, persona: str | None = None, tts_preset: str | None = None):
        if persona and persona in PERSONAS:
            self.persona_key = persona
            self.system_prompt = PERSONAS[persona]["system_prompt"]
            self.tts_preset_key = PERSONAS[persona]["tts_preset"]
        if tts_preset and tts_preset in TTS_PRESETS:
            self.tts_preset_key = tts_preset

# -------- HTTP: personas list ----------
@router.get("/personas")
def list_personas():
    return {
        "default": DEFAULT_PERSONA_KEY,
        "personas": [{"key": k, "label": v.get("label", k), "tts_preset": v.get("tts_preset")} for k, v in PERSONAS.items()],
        "tts_presets": list(TTS_PRESETS.keys()),
    }

# -------- TTS streaming ----------
async def _tts_stream_send(websocket: WebSocket, text: str, preset: dict | None = None):
    if not (text or "").strip():
        return
    with contextlib_suppress(WebSocketDisconnect):
        await websocket.send_json({"type": "tts_start", "sample_rate": int(TTS_SAMPLE_RATE or 22050), "format": "pcm16"})
    try:
        for chunk in stream_tts_with_preset(text, preset):
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_bytes(chunk)
    finally:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "tts_done"})

# -------- LLM streaming ----------
async def stream_llm_safely(websocket: WebSocket, prompt: str, *, system_prompt: str = "", tts_preset: dict | None = None, max_new_tokens: int = 128, max_sentences: int = 2) -> str:
    buffer = ""
    last_sent_idx = 0
    max_chars = 240

    for tok in stream_hf_chat(prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens):
        if not tok:
            continue
        if BAD_MARKERS_RE.search(tok):
            break

        buffer += tok

        if last_sent_idx == 0:
            cleaned = _clean_leading_strict(buffer)
            if cleaned != buffer:
                buffer = cleaned

        if len(buffer) >= max_chars:
            buffer = buffer[:max_chars].rstrip()
            delta = buffer[last_sent_idx:]
            if delta.strip():
                with contextlib_suppress(WebSocketDisconnect):
                    await websocket.send_json({"type": "delta", "text": delta})
            break

        if len(re.findall(r"[\.!?]", buffer)) >= max_sentences:
            delta = buffer[last_sent_idx:]
            if delta.strip():
                with contextlib_suppress(WebSocketDisconnect):
                    await websocket.send_json({"type": "delta", "text": delta})
            break

        delta = buffer[last_sent_idx:]
        if delta.strip():
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_json({"type": "delta", "text": delta})
            last_sent_idx = len(buffer)

    final_text = buffer.strip()
    if final_text:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "assistant_final", "text": final_text})
        await _tts_stream_send(websocket, final_text, preset=tts_preset)

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
# Simple HTTP TTS for quick test
# -----------------------------
@router.post("/tts")
async def tts_http(payload: dict = Body(...)):
    text = payload.get("text", "")
    if not text.strip():
        return {"error": "Text is empty"}

    from core.tts_coqui import stream_tts_pcm
    pcm_chunks = stream_tts_pcm(text)
    pcm_data = b"".join(pcm_chunks)
    wav_bytes = pcm16_to_wav_bytes(pcm_data, sample_rate=TTS_SAMPLE_RATE or 22050)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")

# -------------------------------------------------------------
# WebSocket: audio bytes -> STT -> LLM stream -> TTS stream
# Path: /api/ws/voice  (mounted with /api prefix in main.py)
# -------------------------------------------------------------
@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    buf = io.BytesIO()
    chunk_count = 0
    last_partial = ""

    # per-connection persona config
    session = SessionState()

    await websocket.send_json({"type": "ready", "session": session_id, "persona": session.persona_key})

    try:
        while True:
            msg = await websocket.receive()

            # Text/json messages
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                low = (txt or "").strip().lower()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = None

                # Persona/tts preset config
                if isinstance(data, dict) and data.get("type") == "config":
                    session.apply(
                        persona=(data.get("persona") or "").strip() or None,
                        tts_preset=(data.get("tts_preset") or "").strip() or None,
                    )
                    await websocket.send_json({
                        "type": "config-ack",
                        "persona": session.persona_key,
                        "tts_preset": session.tts_preset_key
                    })
                    continue

                # Start recording
                if isinstance(data, dict) and data.get("type") == "start":
                    await websocket.send_json({"type": "started", "sample_rate": data.get("sample_rate", AUDIO_SAMPLE_RATE)})
                    continue

                # End signal -> finalize STT
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
                        try:
                            await stream_llm_safely(
                                websocket,
                                final_text,
                                system_prompt=session.system_prompt,
                                tts_preset=TTS_PRESETS.get(session.tts_preset_key, {}),
                            )
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": f"llm error: {e}"})
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
                await websocket.send_json({"type": "ack", "chunk": chunk_count, "bytes": len(payload)})

                # partial STT occasionally (optional)
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
                            # optional: quick single-sentence LLM on partials
                            if len(partial) > 10 and partial[-1] in (" ", ".", "?", "!"):
                                try:
                                    await stream_llm_safely(
                                        websocket,
                                        partial,
                                        system_prompt=session.system_prompt,
                                        tts_preset=TTS_PRESETS.get(session.tts_preset_key, {}),
                                        max_new_tokens=64,
                                        max_sentences=1
                                    )
                                except Exception as e:
                                    await websocket.send_json({"type": "error", "message": f"llm error: {e}"})
                continue

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            await websocket.close()
        return
