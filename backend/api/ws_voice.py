import json
import io
import tempfile
import uuid
from contextlib import suppress as contextlib_suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import numpy as np
import soundfile as sf

from core.llm_gguf import stream_hf_chat
from core.stt_whisper import transcribe_audio
from config.settings import FRONTEND_ORIGINS

# DB + helpers
from db import save_message, load_history
from api.voice_chat import extract_tags
from core.llm_hf import _postprocess
from config.personalities import PERSONALITIES, DEFAULT_PERSONALITY

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    origin = ws.headers.get("origin")
    if FRONTEND_ORIGINS and origin not in FRONTEND_ORIGINS:
        with contextlib_suppress(Exception):
            await ws.close(code=1008)
        return

    await ws.accept()
    cid = str(uuid.uuid4())[:8]
    print(f"[WS {cid}] accepted (origin={origin})")

    session_id = None
    session_personality = DEFAULT_PERSONALITY
    session_voice = PERSONALITIES[DEFAULT_PERSONALITY]["voice"]

    audio_buf = io.BytesIO()
    sample_rate: int = 16000
    receiving_audio = False

    try:
        while True:
            msg = await ws.receive()

            # ---------------- Text frames ----------------
            if "text" in msg:
                data = (msg["text"] or "").strip()
                lo = data.lower()
                print(f"[WS {cid}] <text> {lo[:80]}")

                # ---------------- JSON controls ----------------
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = None

                if payload:
                    t = (payload.get("type") or "").lower()
                    print(f"[WS {cid}] <json type={t}>")

                    if t == "start":
                        receiving_audio = True
                        audio_buf = io.BytesIO()
                        sample_rate = int(payload.get("sample_rate", sample_rate or 16000))
                        session_id = payload.get("session") or str(uuid.uuid4())

                        # 🔹 personality + voice
                        session_personality = payload.get("personality", DEFAULT_PERSONALITY)
                        if session_personality not in PERSONALITIES:
                            session_personality = DEFAULT_PERSONALITY
                        session_voice = payload.get("voice") or PERSONALITIES[session_personality]["voice"]

                        with contextlib_suppress(WebSocketDisconnect):
                            await ws.send_json({
                                "type": "ack",
                                "event": "start",
                                "session": session_id,
                                "personality": session_personality,
                                "voice": session_voice
                            })
                        print(f"[WS {cid}] start ack session={session_id} personality={session_personality} voice={session_voice}")

                    elif t == "text":
                        user_text = (payload.get("text") or "").strip()
                        if not user_text:
                            with contextlib_suppress(WebSocketDisconnect):
                                await ws.send_json({"type": "error", "message": "empty text"})
                        else:
                            tags = extract_tags(user_text)
                            save_message(session_id, "user", user_text, tags)
                            await _send_llm_reply(ws, session_id, user_text, cid, session_personality, session_voice, lang="en")

                    elif t in {"stop", "done", "end"} and receiving_audio:
                        print(f"[WS {cid}] finalize (stop)")
                        await _finalize_turn(ws, audio_buf, sample_rate, session_id, cid, session_personality, session_voice)
                        receiving_audio = False
                        audio_buf = io.BytesIO()

                    else:
                        with contextlib_suppress(WebSocketDisconnect):
                            await ws.send_json({"type": "error", "message": "unknown message type"})
                    continue

                # ---------------- Plain controls ----------------
                if lo == "start":
                    receiving_audio = True
                    audio_buf = io.BytesIO()
                    session_id = str(uuid.uuid4())
                    with contextlib_suppress(WebSocketDisconnect):
                        await ws.send_json({
                            "type": "ack",
                            "event": "start",
                            "session": session_id,
                            "personality": session_personality,
                            "voice": session_voice
                        })
                    print(f"[WS {cid}] start ack session={session_id}")
                    continue

                if lo in {"done", "end", "finish", "stop"} and receiving_audio:
                    print(f"[WS {cid}] finalize (done)")
                    await _finalize_turn(ws, audio_buf, sample_rate, session_id, cid, session_personality, session_voice)
                    receiving_audio = False
                    audio_buf = io.BytesIO()
                    continue

                # Freeform text
                if data:
                    session_id = session_id or str(uuid.uuid4())
                    tags = extract_tags(data)
                    save_message(session_id, "user", data, tags)
                    await _send_llm_reply(ws, session_id, data, cid, session_personality, session_voice, lang="en")

            # --------------- Binary frames (PCM) ---------------
            elif "bytes" in msg:
                b = msg["bytes"]
                if receiving_audio:
                    audio_buf.write(b)
                    print(f"[WS {cid}] <bytes> +{len(b)}B total={audio_buf.tell()}B")
                else:
                    with contextlib_suppress(WebSocketDisconnect):
                        await ws.send_json({"type": "warn", "message": "binary ignored; send 'start' first"})

    except WebSocketDisconnect:
        print(f"[WS {cid}] client disconnected (session={session_id})")
        return
    except Exception as e:
        print(f"[WS {cid}] ERROR: {e}")
        with contextlib_suppress(Exception):
            await ws.send_json({"type": "error", "message": str(e)})


# ---------------- helpers ----------------

async def _finalize_turn(ws: WebSocket, audio_buf: io.BytesIO, sample_rate: int, session_id: str, cid: str, personality_key: str, voice: str):
    raw = audio_buf.getvalue()
    print(f"[WS {cid}] finalize: buf={len(raw)}B")
    if len(raw) == 0:
        with contextlib_suppress(WebSocketDisconnect):
            await ws.send_json({"type": "error", "message": "no audio received"})
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    pcm = np.frombuffer(raw, dtype=np.int16)
    sf.write(wav_path, pcm, samplerate=sample_rate, subtype="PCM_16")

    try:
        result = transcribe_audio(wav_path) or {"text": "", "language": "unknown"}
        text = result["text"]
        detected_lang = result["language"]
    except Exception as e:
        text = ""
        detected_lang = "error"
        print(f"[WS {cid}] STT error: {e}")
    print(f"[WS {cid}] STT -> {repr(text)} (lang={detected_lang})")

    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "transcript", "text": text, "language": detected_lang, "final": True})

    if text:
        tags = extract_tags(text)
        save_message(session_id, "user", text, tags)
        await _send_llm_reply(ws, session_id, text, cid, personality_key, voice, lang=detected_lang)


async def _send_llm_reply(ws: WebSocket, session_id: str, user_text: str, cid: str, personality_key: str, voice: str, lang: str = "en"):
    history = load_history(session_id, limit=20)
    personality = PERSONALITIES.get(personality_key or DEFAULT_PERSONALITY, PERSONALITIES[DEFAULT_PERSONALITY])
    style_prefix = personality["prompt_prefix"]

    # ✅ FIX: Put prefix as system role instead of conversation content
    full_prompt = f"System: {style_prefix}\n\n"
    for h in history:
        full_prompt += f"{h['role'].capitalize()}: {h['content']}\n"
    full_prompt += f"User: {user_text}\nAssistant:"

    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "started"})

    reply_accum = ""
    last_sent = ""

    try:
        for piece in stream_hf_chat(full_prompt, max_new_tokens=128):
            if not piece:
                continue
            reply_accum += piece
            safe_piece = piece.replace("▁", " ").replace("\u2581", " ")
            safe_piece = safe_piece.replace("\n", " ").strip()
            if safe_piece and safe_piece != last_sent:
                with contextlib_suppress(WebSocketDisconnect):
                    await ws.send_json({"type": "delta", "text": safe_piece})
                last_sent = safe_piece

    except Exception as e:
        print(f"[WS {cid}] LLM stream error: {e}")
        reply_accum = f"(model error: {e})"

    final_reply = _postprocess(reply_accum)
    if not final_reply:
        final_reply = "Okay."

    tags = extract_tags(final_reply)
    save_message(session_id, "assistant", final_reply, tags)

    safe_prompt = (user_text or "")[:120].replace("\n", " ")
    safe_reply = (final_reply or "")[:120].replace("\n", " ")
    print(f"[WS {cid}] LLM prompt: {safe_prompt}")
    print(f"[WS {cid}] LLM reply : {safe_reply}")

    # 🔹 If detected language not supported by selected voice → fallback to default
    final_voice = voice
    if lang not in {"en", "english"}:
        final_voice = "alloy"  # fallback voice

    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({
            "type": "assistant_final",
            "text": final_reply,
            "voice": final_voice,
            "language": lang
        })
    with contextlib_suppress(WebSocketDisconnect):
        await ws.send_json({"type": "done"})
