# backend/core/llm_hf.py

import os
import re
import threading
import torch

# --- Torch dtype aliases (older torch builds) ---
for t in ("uint8", "uint16", "uint32", "uint64"):
    if not hasattr(torch, t):
        setattr(torch, t, torch.int64)

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from dotenv import load_dotenv
from config.settings import LLM_MODEL, HF_API_KEY
from huggingface_hub import login

# Ensure .env loaded
load_dotenv(override=True)

# Optional: login for gated/private models
if HF_API_KEY:
    try:
        login(token=HF_API_KEY)
        os.environ.setdefault("HF_TOKEN", HF_API_KEY)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_API_KEY)
    except Exception:
        pass  # non-fatal

# ------------ Model & Tokenizer init ------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_auth_token=bool(HF_API_KEY))

# set pad -> eos if missing
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_auth_token=bool(HF_API_KEY),
)
model.to(device)
model.eval()

_model_lock = threading.Lock()

# ------------ Helpers ------------
_BAD_MARKERS_RE = re.compile(r"(Assistant:|User:|</s>|<\|eot_id\|>|<\|endoftext\|>)", re.IGNORECASE)

def _postprocess(text: str) -> str:
    text = (text or "").strip()
    # strip leading quotes/space
    while text and text[0] in "\"'“”‘’ \n\t":
        text = text[1:].lstrip()
    text = _BAD_MARKERS_RE.sub("", text)
    return text[:800].strip()

def _build_kwargs(
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    streamer=None,
):
    kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if streamer is not None:
        kw["streamer"] = streamer
    return kw

def _encode(prompt: str):
    return tokenizer(prompt, return_tensors="pt").to(device)

# ------------ Streaming generation ------------
def stream_hf_chat(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
):
    """
    Token-by-token generator (optional). Your WS currently uses non-streaming.
    """
    inputs = _encode(prompt)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = _build_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    def _run():
        with _model_lock, torch.no_grad():
            model.generate(**inputs, **gen_kwargs)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    for piece in streamer:
        if not piece:
            continue
        yield _postprocess(piece)

# ------------ Non-stream full generation ------------
def generate_full(
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    stop=None,
) -> str:
    """
    Reliable one-shot completion. Returns a clean plain string.
    """
    inputs = _encode(prompt)
    gen_kwargs = _build_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    with _model_lock, torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # decode only the new tokens
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # optional stop markers
    if stop:
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break

    return _postprocess(text)
