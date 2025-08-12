# backend/clients/ws_test_llm.py
import asyncio, json, websockets

URL = "ws://127.0.0.1:8000/ws/llm"
IDLE_TIMEOUT = 15  # optional safety

async def run():
    async with websockets.connect(URL, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({
            "event": "start",
            "system": "You are a concise assistant. Keep answers short."
        }))
        print(">> start sent")

        await ws.send(json.dumps({
            "event": "user",
            "text": "Hello, my name is Zeeshan. How are you?"
        }))
        print(">> user partial sent")

        # Read stream until 'done'
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                print(f"<< timeout ({IDLE_TIMEOUT}s) — exiting")
                break

            print("<<", raw)
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if data.get("type") in ("done", "done_llm"):
                # politely end the turn and break
                await ws.send(json.dumps({"event": "end"}))
                print(">> end sent")
                break

        # ensure socket closes cleanly
        try:
            await ws.close(code=1000)
        except websockets.ConnectionClosed:
            pass

if __name__ == "__main__":
    asyncio.run(run())
