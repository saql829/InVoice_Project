# backend/api/ws_voice.py
import json
import io
import tempfile
import uuid
from contextlib import suppress as contextlib_suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

import numpy as np
import soundfile as sf

from core.llm_hf import generate_full
from core.stt_whisper import transcribe_audio
from config.settings import FRONTEND_ORIGINS

# 🔹 new imports for DB + topic tagging
from utils.db import get_db
from utils.topics import guess_topic

router = APIRouter(tags=["WebSocket"])

# -----------------------------
# WebSocket main flow
# -----------------------------
@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    # ---- Origin guard ----
    origin = ws.headers.get("origin")
    if FRONTEND_ORIGINS and origin not in FRONTEND_ORIGINS:
        with contextlib_suppress(Exception):
            await ws.close(code=1008)
        return

    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    print(f"[WS {session_id}] accepted (origin={origin})")

    audio_buf = io.BytesIO()
    sample_rate: int = 16000
    receiving_audio = False

    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "ready", "sample_rate": sample_rate, "session": session_id})

    try:
        while True:
            msg = await ws.receive()

            # ---------------- Text frames ----------------
            if "text" in msg:
                data = (msg["text"] or "").strip()
                lo = data.lower()
                print(f"[WS {session_id}] <text> {lo[:80]}")

                # Plain controls
                if lo == "start":
                    receiving_audio = True
                    audio_buf = io.BytesIO()
                    with contextlib_suppress(WebSocketDisconnect):
                        await ws.send_json({"type": "ack", "event": "start"})
                    print(f"[WS {session_id}] start ack")
                    continue

                if lo in {"done", "end", "finish", "stop"} and receiving_audio:
                    print(f"[WS {session_id}] finalize (done)")
                    await _finalize_turn(ws, audio_buf, sample_rate, session_id)
                    receiving_audio = False
                    audio_buf = io.BytesIO()
                    continue

                # JSON controls (optional)
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = None

                if payload is None:
                    # Freeform chat (no audio)
                    await _send_llm_reply(ws, f"User: {data}\nAssistant:", session_id)
                    continue

                t = (payload.get("type") or "").lower()
                print(f"[WS {session_id}] <json type={t}>")

                if t == "start":
                    receiving_audio = True
                    audio_buf = io.BytesIO()
                    sample_rate = int(payload.get("sample_rate", sample_rate or 16000))
                    with contextlib_suppress(WebSocketDisconnect):
                        await ws.send_json({"type": "ack", "event": "start"})
                    print(f"[WS {session_id}] start ack (json) sr={sample_rate}")

                elif t == "text":
                    user_text = (payload.get("text") or "").strip()
                    if not user_text:
                        with contextlib_suppress(WebSocketDisconnect):
                            await ws.send_json({"type": "error", "message": "empty text"})
                    else:
                        await _send_llm_reply(ws, f"User: {user_text}\nAssistant:", session_id)

                elif t == "stop" and receiving_audio:
                    print(f"[WS {session_id}] finalize (stop)")
                    await _finalize_turn(ws, audio_buf, sample_rate, session_id)
                    receiving_audio = False
                    audio_buf = io.BytesIO()

                else:
                    with contextlib_suppress(WebSocketDisconnect):
                        await ws.send_json({"type": "error", "message": "unknown message type"})

            # --------------- Binary frames (PCM) ---------------
            elif "bytes" in msg:
                b = msg["bytes"]
                if receiving_audio:
                    audio_buf.write(b)
                    print(f"[WS {session_id}] <bytes> +{len(b)}B total={audio_buf.tell()}B")
                else:
                    with contextlib_suppress(WebSocketDisconnect):
                        await ws.send_json({"type": "warn", "message": "binary ignored; send 'start' first"})

    except WebSocketDisconnect:
        print(f"[WS {session_id}] client disconnected")
        return
    except Exception as e:
        print(f"[WS {session_id}] ERROR: {e}")
        with contextlib_suppress(Exception):
            await ws.send_json({"type": "error", "message": str(e)})

# ---------------- helpers ----------------

async def _finalize_turn(ws: WebSocket, audio_buf: io.BytesIO, sample_rate: int, session_id: str):
    """
    PCM16 buffer -> WAV -> STT -> transcript -> LLM reply.
    """
    raw = audio_buf.getvalue()
    print(f"[WS {session_id}] finalize: buf={len(raw)}B")
    if len(raw) == 0:
        with contextlib_suppress(WebSocketDisconnect):
            await ws.send_json({"type": "error", "message": "no audio received"})
        return

    # Write wav temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    pcm = np.frombuffer(raw, dtype=np.int16)
    sf.write(wav_path, pcm, samplerate=sample_rate, subtype="PCM_16")

    # STT
    try:
        text = transcribe_audio(wav_path) or ""
    except Exception as e:
        text = ""
        print(f"[WS {session_id}] STT error: {e}")
    print(f"[WS {session_id}] STT -> {repr(text)}")

    # Save user message
    if text:
        topic = guess_topic(text)
        conn = get_db()
        conn.execute("INSERT INTO conversations (session_id, role, text, topic) VALUES (?, ?, ?, ?)",
                     (session_id, "user", text, topic))
        conn.commit()
        conn.close()

    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "transcript", "text": text, "final": True})

    # LLM
    await _send_llm_reply(ws, f"User: {text}\nAssistant:", session_id)


async def _send_llm_reply(ws: WebSocket, prompt: str, session_id: str):
    """
    Reliable path: non-streaming generate -> send once.
    """
    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "started"})

    try:
        reply = (generate_full(prompt, max_new_tokens=128) or "").strip()
    except Exception as e:
        reply = f"(model error: {e})"

    # Save assistant reply
    if reply:
        topic = guess_topic(reply)
        conn = get_db()
        conn.execute("INSERT INTO conversations (session_id, role, text, topic) VALUES (?, ?, ?, ?)",
                     (session_id, "assistant", reply, topic))
        conn.commit()
        conn.close()

    safe_prompt = (prompt or "")[:120].replace("\n", " ")
    safe_reply  = (reply or "")[:120].replace("\n", " ")
    print(f"[WS {session_id}] LLM prompt: {safe_prompt}")
    print(f"[WS {session_id}] LLM reply : {safe_reply}")

    if not reply:
        reply = "Okay."

    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "delta", "text": reply})
    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "assistant_final", "text": reply})
    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "done"})


# -----------------------------
# Resume/retry: get history
# -----------------------------
@router.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 10):
    conn = get_db()
    cur = conn.execute(
        "SELECT role, text, topic, ts FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]
