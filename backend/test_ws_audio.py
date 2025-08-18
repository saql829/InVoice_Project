# test_ws_audio.py
import sys
import json
import wave
import asyncio
import contextlib
import websockets

# NOTE: agar tumhara route /api ke under hai to niche wali line ko:
# URI = "ws://127.0.0.1:8000/api/ws/voice"
# warna default (no prefix):
URI = "ws://127.0.0.1:8000/api/ws/voice"


FRAME_MS = 20         # 20ms frames
SAMPLE_RATE = 16000   # PCM16 mono 16k
BYTES_PER_SAMPLE = 2  # 16-bit
SLEEP_SEC = FRAME_MS / 1000.0


async def sender(ws, path, started_evt: asyncio.Event):
    # Wait for server to be ready for audio (optional but safer)
    try:
        await asyncio.wait_for(started_evt.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        # agar 'started' na aaye to bhi aage barh jate hain
        pass

    wf = wave.open(path, "rb")
    # WAV must be PCM16 mono 16k
    assert wf.getnchannels() == 1, "WAV must be mono"
    assert wf.getframerate() == SAMPLE_RATE, "WAV must be 16kHz"
    assert wf.getsampwidth() == BYTES_PER_SAMPLE, "WAV must be PCM16 (16-bit)"

    # 1) handshake (IMPORTANT) – kuch servers bina 'start' ke binary ignore karte hain
    await ws.send("start")

    # 2) stream audio frames with realtime pacing
    frame_samples = int((FRAME_MS / 1000.0) * SAMPLE_RATE)  # e.g., 320 samples @ 20ms
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

    async with websockets.connect(URI, max_size=None) as ws:
        # background reader
        async def reader():
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    # server se agar kabhi binary aaye
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
                    # 'started' milte hi audio bhejna safe ho jata hai
                    if t == "started":
                        started_evt.set()
                elif t == "stt_partial":
                    print({"type": "stt_partial", "text": data.get("text")})
                elif t == "transcript":
                    print({"type": "transcript", "final": data.get("final", True), "text": data.get("text")})
                elif t == "delta":
                    # LLM token-by-token
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
        # parallel me sender chalao (handshake + audio)
        sender_task = asyncio.create_task(sender(ws, path, started_evt))

        # wait for server 'done' (or timeout fallback)
        try:
            await asyncio.wait_for(done_evt.wait(), timeout=60)
        except asyncio.TimeoutError:
            print("[client] timeout waiting for done")
        finally:
            sender_task.cancel()
            reader_task.cancel()
            with contextlib.suppress(Exception):
                await ws.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ws_audio.py sample_16k.wav")
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))
