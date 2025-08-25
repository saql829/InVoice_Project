# backend/api/voice_chat.py
import time, re, io, json, uuid, asyncio, logging
from contextlib import suppress as contextlib_suppress
from typing import Optional
from dataclasses import dataclass

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body

from core.stt_whisper import transcribe_bytes, transcribe_audio
from core.llm_hf import stream_hf_chat, count_tokens
from core.tts_coqui import stream_tts_with_preset

from config.settings import (
    TTS_SAMPLE_RATE, AUDIO_SAMPLE_RATE, FRAME_MS,
    PERSONAS, TTS_PRESETS, DEFAULT_PERSONA_KEY
)

# metrics
from config.settings import METRICS_ENABLED
if METRICS_ENABLED:
    from utils.metrics import (
        ws_open, ws_close, add_audio_bytes, observe_stt_latency,
        add_llm_in_tokens, observe_llm_out, observe_tts, note_error, log_event
    )

log = logging.getLogger("vma.ws")

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

# Block common spurious speaker labels or end markers
BAD_MARKERS_RE = re.compile(
    r"\b(assistant|user|system|ai|bot|responder|machine|teacher)\s*:\s*|</s>|<\|eot_id\|>",
    re.IGNORECASE
)

GREET_RE = re.compile(r"\b(hi|hello|hey|how are you|what'?s up|sup)\b", re.I)

# Guardrail injected before persona prompt
DEFAULT_GUARDRAIL = (
    "You are a real-time voice assistant. Reply in ONE short sentence (<=25 words). "
    "Be warm and conversational. Do not use any speaker labels. "
    "Do not explain grammar or give examples unless explicitly asked."
)

def _join_prompts(*parts: str) -> str:
    return "\n".join([p.strip() for p in parts if (p or "").strip()])

def _clean_leading_strict(s: str) -> str:
    s = s.lstrip()
    QUOTES = {'"', "'", "“", "”", "‘", "’"}
    while s and s[0] in QUOTES:
        s = s[1:].lstrip()
    return s

# -------- Personas: session state ----------
@dataclass
class SessionState:
    persona_key: str = DEFAULT_PERSONA_KEY
    tts_preset_key: str = PERSONAS.get(DEFAULT_PERSONA_KEY, {}).get("tts_preset", "")
    system_prompt: str = PERSONAS.get(DEFAULT_PERSONA_KEY, {}).get("system_prompt", "")

    def apply(self, persona: str | None = None, tts_preset: str | None = None):
        if persona and persona in PERSONAS:
            self.persona_key = persona
            self.system_prompt = PERSONAS[persona].get("system_prompt", "")
            self.tts_preset_key = PERSONAS[persona].get("tts_preset", self.tts_preset_key)
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

# -------- TTS streaming (instrumented) ----------
async def _tts_stream_send(websocket: WebSocket, text: str, preset: dict | None = None):
    if not (text or "").strip():
        return
    with contextlib_suppress(WebSocketDisconnect):
        await websocket.send_json({"type": "tts_start", "sample_rate": int(TTS_SAMPLE_RATE or 22050), "format": "pcm16"})

    t0 = time.perf_counter()
    bytes_out = 0
    try:
        for chunk in stream_tts_with_preset(text, preset):
            bytes_out += len(chunk)
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_bytes(chunk)
    finally:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "tts_done"})
        dt = time.perf_counter() - t0
        if METRICS_ENABLED:
            observe_tts(bytes_out, dt)
        log.info(f"TTS streamed {bytes_out/1024:.1f} KiB in {dt*1000:.1f} ms")

# -------- LLM streaming (instrumented) ----------
async def stream_llm_safely(
    websocket: WebSocket,
    prompt: str,
    *,
    system_prompt: str = "",
    tts_preset: dict | None = None,
    max_new_tokens: int = 128,
    max_sentences: int = 2
) -> str:
    buffer = ""
    last_sent_idx = 0
    max_chars = 220
    llm_t0 = time.perf_counter()

    for tok in stream_hf_chat(
        prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy by default; clean & short
    ):
        if not tok:
            continue

        # Drop speaker labels or end markers mid-stream
        if BAD_MARKERS_RE.search(tok):
            continue

        buffer += tok

        # Clean leading quotes etc only once
        if last_sent_idx == 0:
            cleaned = _clean_leading_strict(buffer)
            if cleaned != buffer:
                buffer = cleaned

        # Emit deltas as we go
        delta = buffer[last_sent_idx:]
        if delta.strip():
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_json({"type": "delta", "text": delta})
            last_sent_idx = len(buffer)

        # Cut after sentence/char limits
        if len(re.findall(r"[\.!?]", buffer)) >= max_sentences or len(buffer) >= max_chars:
            break

    final_text = buffer.strip()
    llm_dt = time.perf_counter() - llm_t0
    if final_text:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "assistant_final", "text": final_text})
        await _tts_stream_send(websocket, final_text, preset=tts_preset)

    # record LLM metrics once we know the final emitted text
    if METRICS_ENABLED:
        out_tok = count_tokens(final_text) if final_text else 0
        observe_llm_out(out_tok, llm_dt)

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
        t0 = time.perf_counter()
        text = transcribe_bytes(data, mime=file.content_type)
        if METRICS_ENABLED:
            observe_stt_latency(time.perf_counter() - t0)
        return {"transcript": text}
    except Exception as e:
        if METRICS_ENABLED: note_error("stt_http", e)
        log.exception("Error in /transcribe")
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
    t0 = time.perf_counter()
    pcm_chunks = list(stream_tts_pcm(text))
    pcm_data = b"".join(pcm_chunks)
    dt = time.perf_counter() - t0
    if METRICS_ENABLED:
        observe_tts(len(pcm_data), dt)
    wav_bytes = pcm16_to_wav_bytes(pcm_data, sample_rate=TTS_SAMPLE_RATE or 22050)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")

# -------------------------------------------------------------
# WebSocket: audio bytes -> STT -> LLM stream -> TTS stream
# -------------------------------------------------------------
@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    buf = io.BytesIO()
    chunk_count = 0
    last_partial = ""
    session = SessionState()

    if METRICS_ENABLED: ws_open()
    if METRICS_ENABLED: log_event("ws", "ready", session=session_id, persona=session.persona_key)
    try:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "ready", "session": session_id, "persona": session.persona_key})

        while True:
            try:
                msg = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                # RuntimeError may appear after disconnect; treat same as WS close.
                log.info(f"[{session_id}] client disconnected")
                break

            # Text/json messages
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                low = (txt or "").strip().lower()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = None

                if isinstance(data, dict) and data.get("type") == "config":
                    session.apply(
                        persona=(data.get("persona") or "").strip() or None,
                        tts_preset=(data.get("tts_preset") or "").strip() or None,
                    )
                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({
                            "type": "config-ack",
                            "persona": session.persona_key,
                            "tts_preset": session.tts_preset_key
                        })
                    log.info(f"[{session_id}] config persona={session.persona_key} tts={session.tts_preset_key}")
                    if METRICS_ENABLED: log_event("ws", "config", persona=session.persona_key, tts=session.tts_preset_key)
                    continue

                if isinstance(data, dict) and data.get("type") == "start":
                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({"type": "started", "sample_rate": data.get("sample_rate", AUDIO_SAMPLE_RATE)})
                    log.info(f"[{session_id}] start (sr={data.get('sample_rate', AUDIO_SAMPLE_RATE)})")
                    continue

                # End signal -> finalize STT
                if low in ("done", "end") or (isinstance(data, dict) and data.get("event") in ("end", "done")):
                    pcm = buf.getvalue()
                    log.info(f"[{session_id}] finalize: buf={len(pcm)} bytes")
                    if pcm:
                        wav = pcm16_to_wav_bytes(pcm, sample_rate=AUDIO_SAMPLE_RATE)
                        stt_t0 = time.perf_counter()
                        try:
                            final_text = transcribe_bytes(wav, mime="audio/wav")
                        except Exception as e:
                            if METRICS_ENABLED: note_error("stt_final", e)
                            final_text = ""
                            log.exception(f"[{session_id}] STT final error")
                        finally:
                            if METRICS_ENABLED:
                                observe_stt_latency(time.perf_counter() - stt_t0)
                    else:
                        final_text = ""

                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({"type": "transcript", "final": True, "text": final_text})

                    # reset buffer for next turn
                    chunk_count = 0
                    buf = io.BytesIO()

                    # Quick greeting router to avoid off-topic replies
                    if final_text and GREET_RE.search(final_text):
                        quick = "hey! I'm good — how's your day going?"
                        with contextlib_suppress(WebSocketDisconnect):
                            await websocket.send_json({"type": "assistant_final", "text": quick})
                        await _tts_stream_send(websocket, quick, preset=TTS_PRESETS.get(session.tts_preset_key, {}))
                        continue

                    # LLM phase
                    if final_text:
                        if METRICS_ENABLED:
                            in_tok = count_tokens(final_text)
                            add_llm_in_tokens(in_tok)
                        try:
                            sys_prompt = _join_prompts(DEFAULT_GUARDRAIL, session.system_prompt)
                            await stream_llm_safely(
                                websocket,
                                final_text,
                                system_prompt=sys_prompt,
                                tts_preset=TTS_PRESETS.get(session.tts_preset_key, {}),
                            )
                        except Exception as e:
                            if METRICS_ENABLED: note_error("llm_stream", e)
                            log.exception(f"[{session_id}] LLM error")
                            with contextlib_suppress(WebSocketDisconnect):
                                await websocket.send_json({"type": "error", "message": f"llm error: {e}"})
                    continue

                # Unknown text -> ignore
                continue

            # Binary audio frames
            if "bytes" in msg and msg["bytes"] is not None:
                payload: bytes = msg["bytes"]
                if not payload:
                    continue
                buf.write(payload)
                chunk_count += 1
                if METRICS_ENABLED: add_audio_bytes(len(payload))
                with contextlib_suppress(WebSocketDisconnect):
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
                            if METRICS_ENABLED: note_error("stt_partial", e)

                        if partial and partial != last_partial:
                            last_partial = partial
                            with contextlib_suppress(WebSocketDisconnect):
                                await websocket.send_json({"type": "stt_partial", "text": partial})
                            # Optional: quick one-sentence LLM on partials
                            if len(partial) > 10 and partial[-1] in (" ", ".", "?", "!"):
                                try:
                                    sys_prompt = _join_prompts(DEFAULT_GUARDRAIL, session.system_prompt)
                                    await stream_llm_safely(
                                        websocket,
                                        partial,
                                        system_prompt=sys_prompt,
                                        tts_preset=TTS_PRESETS.get(session.tts_preset_key, {}),
                                        max_new_tokens=64,
                                        max_sentences=1
                                    )
                                except Exception as e:
                                    if METRICS_ENABLED: note_error("llm_partial", e)
                                    with contextlib_suppress(WebSocketDisconnect):
                                        await websocket.send_json({"type": "error", "message": f"llm error: {e}"})
                continue

    except WebSocketDisconnect:
        log.info(f"[{session_id}] client disconnected")
    except Exception as e:
        if METRICS_ENABLED: note_error("ws_voice", e)
        try:
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.close()
    finally:
        if METRICS_ENABLED: ws_close()
