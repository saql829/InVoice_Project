# backend/api/ws_llm.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import asyncio
import time
from core.llm_model import stream_llm

router = APIRouter()


async def safe_send(ws, payload):
    try:
        await ws.send_json(payload)
    except Exception as e:
        print(f"[LLM WS send error] {e}")
        return False
    return True


@router.websocket("/ws/llm")
async def ws_llm(websocket: WebSocket):
    await websocket.accept()
    history: List[Dict[str, str]] = []
    system_msg = "You are a helpful assistant."

    try:
        while True:
            msg = await websocket.receive_json()
            event = msg.get("event")

            if event == "start":
                sys = msg.get("system")
                if sys:
                    system_msg = sys
                history = [{"role": "system", "content": system_msg}]
                await safe_send(websocket, {"type": "ack", "event": "start"})
                continue

            if event == "user":
                text = (msg.get("text") or "").strip()
                if not text:
                    await safe_send(websocket, {"type": "error", "message": "empty user text"})
                    continue
                history.append({"role": "user", "content": text})
                await safe_send(websocket, {"type": "ack", "event": "user", "len": len(text)})

                t0 = time.monotonic()
                partial = ""

                async def produce():
                    nonlocal partial
                    loop = asyncio.get_running_loop()

                    def run_llm():
                        return [chunk for chunk in stream_llm(history, max_new_tokens=128, temperature=0.7, top_p=0.9)]

                    chunks = await loop.run_in_executor(None, run_llm)

                    for chunk in chunks:
                        if not isinstance(chunk, str):
                            chunk = str(chunk)  # 🔥 force string
                        partial += chunk
                        ok = await safe_send(websocket, {"type": "llm", "delta": chunk, "t": time.time()})
                        if not ok:
                            return

                    history.append({"role": "assistant", "content": partial})
                    await safe_send(websocket, {"type": "done", "t_total": round(time.monotonic() - t0, 3)})

                await produce()
                continue

            if event == "end":
                await safe_send(websocket, {"type": "ack", "event": "end"})
                await websocket.close()
                break

            await safe_send(websocket, {"type": "error", "message": f"unknown event: {event}"})

    except WebSocketDisconnect:
        print("LLM WS: client disconnected")
