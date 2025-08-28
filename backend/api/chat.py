from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request_response import ChatRequest
from core.llm_gguf import stream_hf_chat, generate_full
from db import save_message, load_history
import re

router = APIRouter(tags=["Chat"])


def clean_text(text: str) -> str:
    """Fix LLM glued text problems."""
    if not text:
        return text
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)  # space after punctuation
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)      # lowercase → Uppercase merge
    return text.strip()


def _stream_gen(prompt: str, params: dict, session_id: str):
    """Stream LLM tokens and clean them on the fly."""
    clean = {k: v for k, v in params.items() if v is not None}
    buffer = ""
    for tok in stream_hf_chat(prompt, **clean):
        tok = clean_text(tok)
        buffer += tok
        yield tok
    # Save assistant reply once full
    save_message(session_id, "assistant", buffer.strip())


@router.post("/chat")
async def chat(req: ChatRequest):
    msg = (req.text or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")

    # Save user message
    save_message(req.session_id, "user", msg)

    # Load history for prompt
    history = load_history(req.session_id)
    prompt_parts = [f"{h['role'].capitalize()}: {h['content']}" for h in history]
    prompt = "\n".join(prompt_parts) + "\nAssistant:"

    params = dict(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop,
    )

    if req.stream:
        return StreamingResponse(
            _stream_gen(prompt, params, req.session_id),
            media_type="text/plain; charset=utf-8"
        )
    else:
        clean = {k: v for k, v in params.items() if v is not None}
        text = generate_full(prompt, **clean)
        text = clean_text(text)

        # Save assistant reply
        save_message(req.session_id, "assistant", text)

        return {"response": text, "history": history + [{"role": "assistant", "content": text}]}
