# backend/test_ws_text.py
import asyncio, json, websockets

async def main():
    uri = "ws://127.0.0.1:8001/ws/voice"
    async with websockets.connect(uri, max_size=None) as ws:
        # sirf text bhej rahe hain (audio nahi)
        await ws.send(json.dumps({"type": "text", "text": "Hello from WS. How are you?"}))
        while True:
            try:
                msg = await ws.recv()
            except websockets.ConnectionClosed:
                break
            print(msg)
            # "done" aate hi loop band
            try:
                data = json.loads(msg)
                if data.get("type") == "done":
                    break
            except Exception:
                pass

asyncio.run(main())
