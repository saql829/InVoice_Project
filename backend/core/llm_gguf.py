# backend/core/llm_gguf.py

import os

import re

import threading

from llama_cpp import Llama

import config.settings as S
 
_model = None

_model_lock = threading.Lock()
 
# Remove chat markers/specials

_BAD_MARKERS_RE = re.compile(r"(Assistant:|User:|</s>|<\|eot_id\|>|<\|endoftext\|>)", re.IGNORECASE)
 
def _postprocess(text: str, *, for_stream: bool = False) -> str:

    """

    Cleanup helper.

    - Streaming chunks: preserve leading spaces (warna words jud jaate hain).

    - Full outputs: normal trim ok.

    """

    t = text or ""

    # remove junk markers but DON'T kill normal spaces

    t = _BAD_MARKERS_RE.sub("", t)
 
    if for_stream:

        # streaming: keep left side intact, sirf trailing whitespace hatao

        return t.rstrip()
 
    # non-stream: full cleanup

    t = t.strip()

    # trim starting quotes if any

    while t and t[0] in "\"'“”‘’":

        t = t[1:].lstrip()

    return t[:800].strip()
 
def _get_model() -> Llama:

    global _model

    if _model is not None:

        return _model

    with _model_lock:

        if _model is not None:

            return _model

        if not S.GGUF_MODEL_PATH or not os.path.exists(S.GGUF_MODEL_PATH):

            raise FileNotFoundError(f"GGUF model not found at {S.GGUF_MODEL_PATH!r}")

        # n_threads=0 -> auto; adjust n_ctx as needed

        _model = Llama(

            model_path=S.GGUF_MODEL_PATH,

            n_ctx=2048,

            n_threads=0,

            verbose=False,

        )

        return _model
 
def generate_full(

    prompt: str,

    *,

    max_new_tokens: int = 128,

    temperature: float = 0.7,

    top_p: float = 0.95,

    stop=None

) -> str:

    llm = _get_model()

    out = llm(

        prompt=prompt,

        max_tokens=max_new_tokens,

        temperature=temperature,

        top_p=top_p,

        stop=stop or ["User:", "Assistant:"],

        echo=False,

    )

    text = (out.get("choices") or [{}])[0].get("text", "")

    return _postprocess(text, for_stream=False)
 
def stream_chat(

    prompt: str,

    *,

    max_new_tokens: int = 128,

    temperature: float = 0.7,

    top_p: float = 0.95,

    stop=None

):

    llm = _get_model()

    stream = llm.create_completion(

        prompt=prompt,

        max_tokens=max_new_tokens,

        temperature=temperature,

        top_p=top_p,

        stop=stop or ["User:", "Assistant:"],

        stream=True,

        echo=False,

    )
 
    first = True

    for chunk in stream:

        piece = (chunk.get("choices") or [{}])[0].get("text", "")

        if not piece:

            continue

        if first:

            # first chunk: left-trim ok so response doesn't start with a space

            out_piece = _postprocess(piece, for_stream=False)

            first = False

        else:

            # subsequent chunks: preserve leading spaces to keep word boundaries

            out_piece = _postprocess(piece, for_stream=True)

        if out_piece:

            yield out_piece
 
# --- Backward-compat alias so old imports don't break ---

def stream_hf_chat(*args, **kwargs):

    # Old code may call this name; keep it working.

    yield from stream_chat(*args, **kwargs)
 
__all__ = ["generate_full", "stream_chat", "stream_hf_chat"]

 