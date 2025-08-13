import asyncio, base64
from core.tts_coqui import split_into_tts_units, tts_chunk_to_b64
import asyncio
from core.llm_model import stream_llm
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pathlib import Path
from io import BytesIO
import json, uuid, time

from core.stt_whisper import transcribe_pcm_bytes

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1].parent
TMP_ROOT = ROOT / "shared" / "temp_audio"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 16000     
PARTIAL_EVERY = 2     
SR = 16000

@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    print("WS: incoming connection")
    await websocket.accept()
    print("WS: accepted")

    session_id = str(uuid.uuid4())
    buf = BytesIO()
    chunk_count = 0
    last_partial = ""
    history = [{"role": "system", "content": "You are a concise assistant."}]
    llm_task: asyncio.Task | None = None
    tts_queue: asyncio.Queue[str] = asyncio.Queue()
    tts_worker_task: asyncio.Task | None = None

    async def tts_worker():
        """
        Pull small text units from queue, synthesize, and send as base64 PCM16.
        Runs parallel to LLM stream for low latency 'start speaking on first tokens'.
        """
        while True:
            try:
                piece = await tts_queue.get()
                if piece is None:
                    break
                payload = tts_chunk_to_b64(piece)
                await websocket.send_json({"type": "tts_chunk", "sr": payload["sr"], "b64": payload["b64"], "text": payload["text"]})
            except asyncio.CancelledError:
                break
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"tts error: {e}"})

        await websocket.send_json({"type": "tts_done"})

    async def start_llm_stream(prompt_text: str, tag: str):
        """Start streaming tokens for latest text; cancels any prior stream."""
        nonlocal llm_task, history, tts_worker_task

        # Cancel previous LLM task if running
        if llm_task and not llm_task.done():
            llm_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                pass

        # Start TTS worker if not running
        if tts_worker_task is None or tts_worker_task.done():
            tts_worker_task = asyncio.create_task(tts_worker())

        async def _run():
            # Prepare temporary chat context
            tmp = history[:-1] if (history and history[-1]["role"] == "user") else history[:]
            tmp = tmp + [{"role": "user", "content": prompt_text}]

            # Buffer LLM deltas into human chunks for TTS
            acc = ""
            try:
                for delta in stream_llm(tmp, max_new_tokens=128, temperature=0.7, top_p=0.9):
                    await websocket.send_json({"type": "llm", "delta": delta, "src": tag, "t": time.time()})
                    acc += delta

                    # Greedy sentence/phrase boundary
                    units = split_into_tts_units(acc)
                    # If we have >=1 full unit, send all except last tail back to queue
                    if len(units) > 1:
                        for u in units[:-1]:
                            await tts_queue.put(u)
                        acc = units[-1]  # keep tail

                # Flush remaining tail (if any)
                tail = acc.strip()
                if tail:
                    await tts_queue.put(tail)

                await websocket.send_json({"type": "done_llm", "src": tag})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"llm stream error: {e}"})
            finally:
                # Signal TTS worker end for this turn
                await tts_queue.put(None)

        llm_task = asyncio.create_task(_run())

    try:
        await websocket.send_json({"type": "ready", "session": session_id})

        while True:
            msg = await websocket.receive()

            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("event") == "end":
                    # Final STT on full buffer
                    t0 = time.monotonic()
                    final_text = transcribe_pcm_bytes(buf.getvalue(), sr=SR)
                    t1 = time.monotonic()
                    (TMP_ROOT / f"{session_id}.pcm").write_bytes(buf.getvalue())
                    await websocket.send_json({"type": "final", "text": final_text, "chunks": chunk_count, "t": time.time()})
                    print(f"[STT final] {t1 - t0:.2f}s -> '{final_text}'")

                    if final_text:
                        history.append({"role": "user", "content": final_text})
                        await start_llm_stream(final_text, tag="final")

                    # Don't close immediately; allow llm + tts_worker to finish
                    # Auto-close will be handled by client or by idle-timeout (optional)
                    continue
                continue

            # Binary audio frames
            payload: bytes = msg.get("bytes", b"")
            if not payload:
                continue
            buf.write(payload)
            chunk_count += 1
            await websocket.send_json({"type": "ack", "chunk": chunk_count, "bytes": len(payload)})

            if chunk_count % PARTIAL_EVERY == 0:
                tail = buf.getvalue()[-(PARTIAL_EVERY * CHUNK_SIZE):]
                t0 = time.monotonic()
                text = transcribe_pcm_bytes(tail, sr=SR)
                t1 = time.monotonic()
                text_clean = (text or "").strip()

                if text_clean and text_clean != last_partial:
                    last_partial = text_clean
                    await websocket.send_json({
                        "type": "stt", "partial": True, "text": text_clean,
                        "latency_s": round(t1 - t0, 3), "t": time.time()
                    })
                    print(f"[STT partial] {t1 - t0:.2f}s -> '{text_clean}'")

                    # Kick LLM (and thus TTS) for sentence-ish partials
                    if len(text_clean) > 8 and (text_clean.endswith((" ", ".", "?", "!", ","))):
                        await start_llm_stream(text_clean, tag="partial")

    except WebSocketDisconnect:
        print("WS: client disconnected")
        # Cancel pending LLM/TTS tasks
        try:
            if llm_task and not llm_task.done():
                llm_task.cancel()
            if tts_worker_task and not tts_worker_task.done():
                tts_worker_task.cancel()
        except Exception:
            pass
        return


