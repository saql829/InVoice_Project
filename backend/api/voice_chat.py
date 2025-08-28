# backend/api/voice_chat.py
import time, re, io, json, uuid, asyncio, logging
from contextlib import suppress as contextlib_suppress
from typing import Optional
from dataclasses import dataclass

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Body

from core.stt_whisper import transcribe_bytes, transcribe_audio
from core.llm_hf import stream_hf_chat, count_tokens
from core.tts_coqui import stream_tts_with_preset
from utils.db import get_db
from utils.topics import guess_topic

from config.settings import (
    TTS_SAMPLE_RATE, AUDIO_SAMPLE_RATE, FRAME_MS,
    PERSONAS, TTS_PRESETS, DEFAULT_PERSONA_KEY,
    METRICS_ENABLED
)

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

# ---------------- Regex Guardrails ----------------
BAD_MARKERS_RE = re.compile(
    r"\b(assistant|user|system|ai|bot|responder|machine|teacher)\s*:\s*|</s>|<\|eot_id\|>",
    re.IGNORECASE
)
GREET_RE = re.compile(r"\b(hi|hello|hey|how are you|what'?s up|sup)\b", re.I)

# ---------------- Persona Guardrail ----------------
DEFAULT_GUARDRAIL = (
    "You are a real-time voice assistant. "
    "Keep replies short (<=50 words). "
    "Always follow the persona style strictly — "
    "if persona is Friendly use emojis, "
    "if persona is Teacher explain step by step, etc."
)

def _join_prompts(*parts: str) -> str:
    return "\n".join([p.strip() for p in parts if (p or "").strip()])

def _clean_leading_strict(s: str) -> str:
    s = s.lstrip()
    QUOTES = {'"', "'", "“", "”", "‘", "’"}
    while s and s[0] in QUOTES:
        s = s[1:].lstrip()
    return s

# ---------------- Personas: session state ----------------
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

# ---------------- HTTP: personas list ----------------
@router.get("/personas")
def list_personas():
    return {
        "default": DEFAULT_PERSONA_KEY,
        "personas": [{"key": k, "label": v.get("label", k), "tts_preset": v.get("tts_preset")} for k, v in PERSONAS.items()],
        "tts_presets": list(TTS_PRESETS.keys()),
    }

# ---------------- TTS streaming ----------------
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
        if METRICS_ENABLED:
            observe_tts(bytes_out, time.perf_counter() - t0)

# ---------------- LLM streaming ----------------
async def stream_llm_safely(
    websocket: WebSocket,
    prompt: str,
    *,
    system_prompt: str = "",
    tts_preset: dict | None = None,
    max_new_tokens: int = 256,
    max_sentences: int = 5
) -> str:
    buffer = ""
    last_sent_idx = 0
    max_chars = 600
    llm_t0 = time.perf_counter()

    for tok in stream_hf_chat(
        prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_sentences=max_sentences,
    ):
        if not tok:
            continue
        if BAD_MARKERS_RE.search(tok):
            continue
        buffer += tok
        if last_sent_idx == 0:
            cleaned = _clean_leading_strict(buffer)
            if cleaned != buffer:
                buffer = cleaned
        delta = buffer[last_sent_idx:]
        if delta.strip():
            with contextlib_suppress(WebSocketDisconnect):
                await websocket.send_json({"type": "delta", "text": delta})
            last_sent_idx = len(buffer)
        if len(re.findall(r"[\.!?]", buffer)) >= max_sentences or len(buffer) >= max_chars:
            break

    final_text = buffer.strip()
    if final_text:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "assistant_final", "text": final_text})
        await _tts_stream_send(websocket, final_text, preset=tts_preset)

        # 🔹 Save assistant reply to DB
        conn = get_db()
        conn.execute("INSERT INTO conversations (session_id, role, text, topic) VALUES (?, ?, ?, ?)",
                     (websocket.headers.get("x-session-id", "default"), "assistant", final_text, guess_topic(final_text)))
        conn.commit()
        conn.close()

    if METRICS_ENABLED:
        out_tok = count_tokens(final_text) if final_text else 0
        observe_llm_out(out_tok, time.perf_counter() - llm_t0)

    return final_text

# ---------------- File upload -> transcript ----------------
@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Only wav/mp3/m4a/ogg/webm supported")
    try:
        data = await file.read()
        text = transcribe_bytes(data, mime=file.content_type)
        if METRICS_ENABLED:
            observe_stt_latency(time.perf_counter())
        return {"transcript": text}
    except Exception as e:
        if METRICS_ENABLED: note_error("stt_http", e)
        log.exception("Error in /transcribe")
        raise HTTPException(status_code=500, detail="Internal transcription error")

# ---------------- WebSocket: STT -> LLM -> TTS ----------------
@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    websocket.headers["x-session-id"] = session_id  # 🔹 custom attach for DB saving
    buf = io.BytesIO()
    chunk_count = 0
    last_partial = ""
    session = SessionState()

    if METRICS_ENABLED: ws_open()
    try:
        with contextlib_suppress(WebSocketDisconnect):
            await websocket.send_json({"type": "ready", "session": session_id, "persona": session.persona_key})

        while True:
            try:
                msg = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            # ---------- Text / JSON control ----------
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                low = (txt or "").strip().lower()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = None

                # --- Config ---
                if isinstance(data, dict) and data.get("type") == "config":
                    session.apply(persona=data.get("persona"), tts_preset=data.get("tts_preset"))
                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({"type": "config-ack", "persona": session.persona_key, "tts_preset": session.tts_preset_key})
                    continue

                # --- Start ---
                if isinstance(data, dict) and data.get("type") == "start":
                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({"type": "started", "sample_rate": data.get("sample_rate", AUDIO_SAMPLE_RATE)})
                    continue

                # --- End / Finalize ---
                if low in ("done", "end") or (isinstance(data, dict) and data.get("event") in ("end", "done")):
                    pcm = buf.getvalue()
                    if pcm:
                        wav = pcm16_to_wav_bytes(pcm, sample_rate=AUDIO_SAMPLE_RATE)
                        try:
                            final_text = transcribe_bytes(wav, mime="audio/wav")
                        except Exception as e:
                            final_text = ""
                    else:
                        final_text = ""

                    # 🔹 Save user message to DB
                    if final_text:
                        conn = get_db()
                        conn.execute("INSERT INTO conversations (session_id, role, text, topic) VALUES (?, ?, ?, ?)",
                                     (session_id, "user", final_text, guess_topic(final_text)))
                        conn.commit()
                        conn.close()

                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({"type": "transcript", "final": True, "text": final_text})

                    buf = io.BytesIO()
                    chunk_count = 0

                    if final_text and GREET_RE.search(final_text):
                        quick = "Hey! I'm good — how’s your day?"
                        with contextlib_suppress(WebSocketDisconnect):
                            await websocket.send_json({"type": "assistant_final", "text": quick})
                        await _tts_stream_send(websocket, quick, preset=TTS_PRESETS.get(session.tts_preset_key, {}))
                        continue

                    if final_text:
                        await stream_llm_safely(websocket, final_text, system_prompt=_join_prompts(DEFAULT_GUARDRAIL, session.system_prompt), tts_preset=TTS_PRESETS.get(session.tts_preset_key, {}))
                    continue

            # ---------- Binary audio ----------
            if "bytes" in msg and msg["bytes"] is not None:
                payload: bytes = msg["bytes"]
                if payload:
                    buf.write(payload)
                    chunk_count += 1
                    if METRICS_ENABLED: add_audio_bytes(len(payload))
                    with contextlib_suppress(WebSocketDisconnect):
                        await websocket.send_json({"type": "ack", "chunk": chunk_count, "bytes": len(payload)})

    finally:
        if METRICS_ENABLED: ws_close()

# ---------------- Resume API ----------------
@router.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 10):
    conn = get_db()
    cur = conn.execute("SELECT role, text, topic, ts FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]
