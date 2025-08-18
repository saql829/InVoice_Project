# backend/api/chat.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request_response import ChatRequest
from core.llm_hf import stream_hf_chat, generate_full

# ⚠️ Prefix yahan se hata diya
router = APIRouter(tags=["Chat"])

def _stream_gen(prompt: str, params: dict):
    # None values hata do
    clean = {k: v for k, v in params.items() if v is not None}
    for tok in stream_hf_chat(prompt, **clean):
        yield tok

@router.post("/chat")
async def chat(req: ChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")

    prompt = f"User: {msg}\nAssistant:"

    params = dict(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop,
    )

    if req.stream:
        # Swagger live chunks nahi dikhata; curl -N se dekho
        return StreamingResponse(
            _stream_gen(prompt, params),
            media_type="text/plain; charset=utf-8"
        )
    else:
        clean = {k: v for k, v in params.items() if v is not None}
        text = generate_full(prompt, **clean)
        return {"response": text}
