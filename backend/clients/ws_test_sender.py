# backend/clients/ws_test_stream_all.py
import asyncio, json, os, websockets

WS_URL = "ws://127.0.0.1:8000/ws/audio"

async def run(path: str):
    async with websockets.connect(WS_URL, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"event":"start"}))
        print(">> sent: start")

        with open(path, "rb") as f:
            while True:
                data = f.read(16000)  # 0.5s
                if not data:
                    break
                await ws.send(data)
                msg = await ws.recv()
                print("<<", msg)

        await ws.send(json.dumps({"event":"end"}))
        print(">> sent: end")
        try:
            while True:
                msg = await ws.recv()
                if '"type":"tts_chunk"' in msg:
                    d = json.loads(msg)
                    size = len(d.get("b64","")) * 3 // 4
                    print(f"<< TTS chunk: {size} bytes @ {d.get('sr')} Hz | text='{d.get('text')[:40]}'")
                else:
                    print("<<", msg)
        except websockets.ConnectionClosed:
            print("<< connection closed")

if __name__ == "__main__":
    pcm_path = "test.pcm"
    if not os.path.exists(pcm_path):
        raise SystemExit("Please create test.pcm (ffmpeg -i test.wav -f s16le -ac 1 -ar 16000 test.pcm)")
    asyncio.run(run(pcm_path))
