# backend/api/ws_llm.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json
import time

from core.llm_model import stream_llm

router = APIRouter()

@router.websocket("/ws/llm")
async def ws_llm(websocket: WebSocket):
    """
    Protocol:
      -> {"event":"start", "system":"You are helpful.", "persona":"friendly"}  (optional)
      -> {"event":"user", "text":"partial or full text"}  (can be called multiple times)
      -> {"event":"end"}  (signals the turn is over; server streams completion and then closes)

    Server messages:
      <- {"type":"ack", "event":"start/user/end"}
      <- {"type":"llm", "delta":"He", "t": 123.45}  (token stream)
      <- {"type":"done"} (when complete)
    """
    await websocket.accept()
    history: List[Dict[str, str]] = []
    system_msg = "You are a helpful assistant."

    try:
        while True:
            msg = await websocket.receive_json()
            event = msg.get("event")

            if event == "start":
                # Optional persona/system prompt
                sys = msg.get("system")
                if sys:
                    system_msg = sys
                history = [{"role": "system", "content": system_msg}]
                await websocket.send_json({"type": "ack", "event": "start"})
                continue

            if event == "user":
                text = (msg.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "empty user text"})
                    continue

                # Append/merge user content into the working history
                history.append({"role": "user", "content": text})
                await websocket.send_json({"type": "ack", "event": "user", "len": len(text)})

                # Kick off LLM stream for the current turn
                t0 = time.monotonic()
                partial = ""
                async def produce():
                    nonlocal partial
                    for chunk in stream_llm(history, max_new_tokens=128, temperature=0.7, top_p=0.9):
                        partial += chunk
                        await websocket.send_json({"type": "llm", "delta": chunk, "t": time.time()})
                    # Add assistant message to history for future context
                    history.append({"role": "assistant", "content": partial})
                    await websocket.send_json({"type": "done", "t_total": round(time.monotonic() - t0, 3)})

                # Run in an asyncio task (non-blocking)
                await produce()
                continue

            if event == "end":
                await websocket.send_json({"type": "ack", "event": "end"})
                await websocket.close()
                break

            await websocket.send_json({"type":"error","message":f"unknown event: {event}"})

    except WebSocketDisconnect:
        return
