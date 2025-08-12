import asyncio, json, os
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/audio"

async def send_audio_chunks(path: str, chunk_size: int = 16000):
    async with websockets.connect(WS_URL, max_size=None, open_timeout=20) as ws:
        await ws.send(json.dumps({"event": "start", "sr": 16000, "ch": 1}))
        print(">> sent: start")

        with open(path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                await ws.send(data)
                msg = await ws.recv()
                print("<<", msg)

        await ws.send(json.dumps({"event": "end"}))
        print(">> sent: end")

        try:
            while True:
                msg = await ws.recv()
                print("<<", msg)
        except websockets.ConnectionClosedOK:
            print("<< connection closed OK")
        except websockets.ConnectionClosedError:
            print("<< connection closed with error")

if __name__ == "__main__":
    pcm_path = "test.pcm"
    if not os.path.exists(pcm_path):
        raise SystemExit("Please create test.pcm (ffmpeg -i test.wav -f s16le -ac 1 -ar 16000 test.pcm)")
    asyncio.run(send_audio_chunks(pcm_path))
