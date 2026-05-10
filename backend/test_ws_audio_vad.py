# test_ws_audio_vad.py
import sys
import json
import wave
import asyncio
import contextlib
import websockets

# NOTE: agar tumhara route /api ke under ho to:
# URI = "ws://127.0.0.1:8001/api/ws/voice?vad=1"
URI = "ws://127.0.0.1:8001/api/ws/voice?vad=1"

FRAME_MS = 20
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SLEEP_SEC = FRAME_MS / 1000.0

async def sender(ws, path, started_evt: asyncio.Event):
    # server se 'started' aane ka wait karo (optional but safe)
    try:
        await asyncio.wait_for(started_evt.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        pass

    wf = wave.open(path, "rb")
    assert wf.getnchannels() == 1, "WAV must be mono"
    assert wf.getframerate() == SAMPLE_RATE, "WAV must be 16kHz"
    assert wf.getsampwidth() == BYTES_PER_SAMPLE, "WAV must be PCM16 (16-bit)"

    # 1) handshake (IMPORTANT)
    await ws.send("start")

    # 2) stream frames with realtime pacing
    frame_samples = int((FRAME_MS / 1000.0) * SAMPLE_RATE)  # 320 @ 20ms
    data = wf.readframes(frame_samples)
    while data:
        await ws.send(data)
        await asyncio.sleep(SLEEP_SEC)
        data = wf.readframes(frame_samples)

    # 3) end signal
    await ws.send("DONE")

async def run(path):
    done_evt = asyncio.Event()
    started_evt = asyncio.Event()

    # Disable keepalive pings so long STT work doesn't trigger ping timeout
    async with websockets.connect(
        URI,
        max_size=None,
        ping_interval=None  # <-- Option A: no keepalive pings
    ) as ws:

        async def reader():
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    print(f"[binary {len(msg)} bytes]")
                    continue

                try:
                    data = json.loads(msg)
                except Exception:
                    print(msg)
                    continue

                t = data.get("type")
                if t == "ready":
                    print({"type": "ready", "sample_rate": data.get("sample_rate")})
                elif t in ("started", "ack", "info", "warn"):
                    print(data)
                    if t == "started":
                        started_evt.set()
                elif t == "stt_partial":
                    print({"type": "stt_partial", "text": data.get("text")})
                elif t == "transcript":
                    # VAD mode me multiple finals aa sakte hain (per segment)
                    print({"type": "transcript", "final": data.get("final", True), "text": data.get("text")})
                elif t == "delta":
                    print({"type": "delta", "text": data.get("text")})
                elif t == "done":
                    print({"type": "done"})
                    done_evt.set()
                    break
                elif t == "error":
                    print({"type": "error", "message": data.get("message")})
                    done_evt.set()
                    break
                else:
                    print(data)

        reader_task = asyncio.create_task(reader())
        sender_task = asyncio.create_task(sender(ws, path, started_evt))

        try:
            await asyncio.wait_for(done_evt.wait(), timeout=180)
        except asyncio.TimeoutError:
            print("[client] timeout waiting for done")
        finally:
            sender_task.cancel()
            reader_task.cancel()
            with contextlib.suppress(Exception):
                await ws.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ws_audio_vad.py sample_16k.wav")
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))
