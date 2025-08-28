# backend/core/llm_hf.py

import os

import re

import threading

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from huggingface_hub import login

from config.settings import LLM_MODEL, HF_API_KEY
 
_tokenizer = None

_model = None

_device = None

_dtype = None

_init_lock = threading.Lock()

_generate_lock = threading.Lock()
 
_BAD_MARKERS_RE = re.compile(r"(Assistant:|User:|</s>|<\|eot_id\|>|<\|endoftext\|>)", re.IGNORECASE)
 
def _postprocess(text: str) -> str:

    text = (text or "").strip()

    while text and text[0] in "\"'“”‘’ \n\t":

        text = text[1:].lstrip()

    text = _BAD_MARKERS_RE.sub("", text)

    return text[:800].strip()
 
def _build_kwargs(*, tokenizer, max_new_tokens=128, temperature=0.7, top_p=0.9, repetition_penalty=1.15, streamer=None):

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
 
def _ensure_hf_logged_in():

    if HF_API_KEY:

        try:

            login(token=HF_API_KEY)

            os.environ.setdefault("HF_TOKEN", HF_API_KEY)

            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_API_KEY)

        except Exception:

            pass
 
def _lazy_init():

    global _tokenizer, _model, _device, _dtype

    if _model is not None and _tokenizer is not None:

        return

    with _init_lock:

        if _model is not None and _tokenizer is not None:

            return

        if not LLM_MODEL:

            raise RuntimeError(

                "HF LLM requested but LLM_MODEL is not set. "

                "Set USE_LOCAL_GGUF=true or provide LLM_MODEL."

            )

        _ensure_hf_logged_in()

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _dtype = torch.float16 if torch.cuda.is_available() else torch.float32
 
        _tokenizer = AutoTokenizer.from_pretrained(

            LLM_MODEL,

            token=HF_API_KEY if HF_API_KEY else None,

        )

        if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:

            _tokenizer.pad_token = _tokenizer.eos_token
 
        _model = AutoModelForCausalLM.from_pretrained(

            LLM_MODEL,

            torch_dtype=_dtype,

            low_cpu_mem_usage=True,

            token=HF_API_KEY if HF_API_KEY else None,

        )

        _model.to(_device)

        _model.eval()
 
def _encode(prompt: str):

    _lazy_init()

    return _tokenizer(prompt, return_tensors="pt").to(_device)
 
def stream_hf_chat(prompt: str, *, max_new_tokens=128, temperature=0.7, top_p=0.9, repetition_penalty=1.15):

    _lazy_init()

    inputs = _encode(prompt)

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = _build_kwargs(

        tokenizer=_tokenizer,

        max_new_tokens=max_new_tokens,

        temperature=temperature,

        top_p=top_p,

        repetition_penalty=repetition_penalty,

        streamer=streamer,

    )
 
    def _run():

        with _generate_lock, torch.no_grad():

            _model.generate(**inputs, **gen_kwargs)
 
    t = threading.Thread(target=_run, daemon=True)

    t.start()
 
    for piece in streamer:

        if not piece:

            continue

        yield _postprocess(piece)
 
def generate_full(prompt: str, *, max_new_tokens=128, temperature=0.7, top_p=0.9, repetition_penalty=1.15, stop=None) -> str:

    _lazy_init()

    inputs = _encode(prompt)

    gen_kwargs = _build_kwargs(

        tokenizer=_tokenizer,

        max_new_tokens=max_new_tokens,

        temperature=temperature,

        top_p=top_p,

        repetition_penalty=repetition_penalty,

    )
 
    with _generate_lock, torch.no_grad():

        out = _model.generate(**inputs, **gen_kwargs)
 
    new_tokens = out[0, inputs["input_ids"].shape[1]:]

    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
 
    if stop:

        for s in stop:

            if not s:

                continue

            idx = text.find(s)

            if idx != -1:

                text = text[:idx]

                break
 
    return _postprocess(text)

 