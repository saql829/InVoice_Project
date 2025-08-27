import asyncio
import json
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.stt_whisper import transcribe_pcm_bytes
from core.llm_model import stream_llm as generate_llm_stream
from core.tts_coqui import synthesize_tts_stream

router = APIRouter()

# Temporary storage for audio chunks
TMP_ROOT = Path("tmp")
TMP_ROOT.mkdir(exist_ok=True)

SR = 16000          # sample rate
CHUNK_SIZE = 1024   # audio chunk size
PARTIAL_EVERY = 5   # every N chunks run partial STT


async def safe_send(ws: WebSocket, payload: dict):
    """Wrapper to safely send JSON messages over WebSocket."""
    try:
        await ws.send_json(payload)
    except Exception as e:
        print(f"[WS send error] {e}")
        return False
    return True


@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    print("WS: accepted")

    session_id = str(uuid.uuid4())[:8]
    buf = bytearray()
    chunk_count = 0
    last_partial = ""

    history = []
    llm_task = None
    tts_worker_task = None
    tts_queue = asyncio.Queue()

    # Background worker for TTS
    async def tts_worker():
        while True:
            try:
                payload = await tts_queue.get()
                if payload is None:
                    break
                ok = await safe_send(websocket, {
                    "type": "tts_chunk",
                    "sr": payload["sr"],
                    "b64": payload["b64"],
                    "text": payload["text"]
                })
                if not ok:
                    break
            except Exception as e:
                print(f"[TTS worker error] {e}")
                break

    # Background task for LLM streaming
    async def start_llm_stream(prompt: str, tag="partial"):
        nonlocal llm_task

        async def _run():
            try:
                async for delta in generate_llm_stream(prompt, history=history):
                    ok = await safe_send(websocket, {
                        "type": "llm",
                        "delta": delta,
                        "src": tag,
                        "t": time.time()
                    })
                    if not ok:
                        break
                    await tts_queue.put({
                        "sr": SR,
                        "b64": synthesize_tts_stream(delta),
                        "text": delta
                    })
            except Exception as e:
                print(f"[LLM stream error] {e}")

        llm_task = asyncio.create_task(_run())

    tts_worker_task = asyncio.create_task(tts_worker())

    try:
        await safe_send(websocket, {"type": "ready", "session": session_id})

        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                print("WS: client disconnected (receive break)")
                break

            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("event") == "end":
                    t0 = time.monotonic()
                    final_text = transcribe_pcm_bytes(bytes(buf), sr=SR)
                    t1 = time.monotonic()
                    (TMP_ROOT / f"{session_id}.pcm").write_bytes(buf)
                    await safe_send(websocket, {
                        "type": "final",
                        "text": final_text,
                        "chunks": chunk_count,
                        "t": time.time()
                    })
                    print(f"[STT final] {t1 - t0:.2f}s -> '{final_text}'")
                    if final_text:
                        history.append({"role": "user", "content": final_text})
                        await start_llm_stream(final_text, tag="final")
                    continue
                continue

            payload: bytes = msg.get("bytes", b"")
            if not payload:
                continue

            buf.extend(payload)
            chunk_count += 1
            await safe_send(websocket, {
                "type": "ack",
                "chunk": chunk_count,
                "bytes": len(payload)
            })

            if chunk_count % PARTIAL_EVERY == 0:
                tail = bytes(buf[-(PARTIAL_EVERY * CHUNK_SIZE):])
                t0 = time.monotonic()
                text = transcribe_pcm_bytes(tail, sr=SR)
                t1 = time.monotonic()
                text_clean = (text or "").strip()
                if text_clean and text_clean != last_partial:
                    last_partial = text_clean
                    await safe_send(websocket, {
                        "type": "stt", "partial": True, "text": text_clean,
                        "latency_s": round(t1 - t0, 3), "t": time.time()
                    })
                    print(f"[STT partial] {t1 - t0:.2f}s -> '{text_clean}'")
                    if len(text_clean) > 8 and (text_clean.endswith((" ", ".", "?", "!", ","))):
                        await start_llm_stream(text_clean, tag="partial")

    except Exception as e:
        print(f"[WS error] {e}")

    finally:
        if llm_task and not llm_task.done():
            llm_task.cancel()
        if tts_worker_task and not tts_worker_task.done():
            await tts_queue.put(None)
            tts_worker_task.cancel()

        try:
            if not websocket.client_state.name.lower().startswith("closed"):
                await websocket.close()
        except Exception:
            pass

        print("WS: closed gracefully")
