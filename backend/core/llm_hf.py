# -*- coding: utf-8 -*-
"""
HF LLM wrapper: streaming + full generation with robust, short, single-sentence replies.

- Exports: stream_hf_chat, generate_full, count_tokens, count_chat_tokens
- Hard-stops after N sentences (default=1)
- Stops on role markers (User:/Assistant:/System:) + blank line ("\n\n")
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, Generator, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# -------------------
# Global state
# -------------------
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None

# Helpful defaults (override via env if you like)
_HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
_DEVICE_MAP = os.getenv("HF_DEVICE_MAP", "auto")
_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Role/format stop strings (plus blank line)
_STOP_STRINGS: List[str] = [
    "\nUser:", "\n\nUser:", "User:",
    "\nAssistant:", "\n\nAssistant:", "Assistant:",
    "\nSystem:", "System:",
    "\n\n",  # blank line — end turn
]

# Tokenized stop sequences (populated after tokenizer loads)
_STOP_SEQ_IDS: List[List[int]] = []


# -------------------
# Utils
# -------------------
def _ensure_loaded() -> None:
    global _model, _tokenizer, _STOP_SEQ_IDS
    if _model is not None and _tokenizer is not None and _STOP_SEQ_IDS:
        return

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL_ID, use_fast=True)
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            _HF_MODEL_ID,
            torch_dtype=_DTYPE,
            device_map=_DEVICE_MAP,
        )
        _model.eval()

    # build stop sequences once
    if not _STOP_SEQ_IDS:
        for s in _STOP_STRINGS:
            ids = _tokenizer.encode(s, add_special_tokens=False)
            if ids:
                _STOP_SEQ_IDS.append(ids)


def _apply_chat_template(system_prompt: str, user_prompt: str) -> str:
    """
    Use tokenizer's chat_template if present; otherwise fall back to a simple format.
    """
    assert _tokenizer is not None
    try:
        if getattr(_tokenizer, "chat_template", None):
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": user_prompt})
            return _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass

    # Fallback prompt text
    sys = f"System: {system_prompt.strip()}\n" if system_prompt.strip() else ""
    return f"{sys}User: {user_prompt}\nAssistant:"


def _sanitize_output(text: str) -> str:
    """
    Remove accidental role echoes and trim.
    """
    # Drop anything after another role marker that the model may emit
    cut_markers = ["\nUser:", "\n\nUser:", "User:", "\nAssistant:", "\n\nAssistant:", "Assistant:", "\nSystem:", "System:"]
    cut_pos = None
    for m in cut_markers:
        i = text.find(m)
        if i != -1:
            cut_pos = i if cut_pos is None else min(cut_pos, i)
    if cut_pos is not None:
        text = text[:cut_pos]

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cut_to_n_sentences(text: str, n: int) -> str:
    """
    Hard truncate text to the first n sentence terminators.
    Sentence enders: ., !, ? followed by space or end.
    """
    n = max(1, int(n))
    end = None
    count = 0
    for m in re.finditer(r"[\.!\?](?:\s|$)", text, flags=re.S):
        count += 1
        end = m.end()
        if count >= n:
            break
    if end is None:
        return text.strip()
    return text[:end].strip()


# -------------------
# Stopping criteria
# -------------------
class StopOnSequences(StoppingCriteria):
    """
    Stop when the last tokens match any of the provided stop sequences.
    """
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.shape[0] == 0:
            return False
        seq = input_ids[0].tolist()
        for stop in self.stop_sequences:
            if len(seq) >= len(stop) and seq[-len(stop):] == stop:
                return True
        return False


class StopAfterSentences(StoppingCriteria):
    """
    Stop after N sentence enders across the recent token window.
    """
    def __init__(self, tokenizer: AutoTokenizer, max_sentences: int = 1, window_tokens: int = 160):
        super().__init__()
        self.tok = tokenizer
        self.max_sentences = max(1, int(max_sentences))
        self.window_tokens = int(window_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.shape[0] == 0:
            return False
        tail = input_ids[0].tolist()[-self.window_tokens:]
        try:
            txt = self.tok.decode(tail, skip_special_tokens=True)
        except Exception:
            return False
        ends = re.findall(r"[\.!\?](?:\s|$)", txt, flags=re.S)
        return len(ends) >= self.max_sentences


# -------------------
# Generation kwargs
# -------------------
def _build_kwargs(
    *,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.10,
    streamer=None,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_sentences: int = 1,
) -> Dict[str, Any]:
    _ensure_loaded()
    assert _tokenizer is not None

    # Use HF eos if available; stop strings handled by StoppingCriteria
    eos_ids: Optional[int] | List[int] = None
    if _tokenizer.eos_token_id is not None:
        eos_ids = _tokenizer.eos_token_id

    stop_criteria = StoppingCriteriaList([
        StopOnSequences(_STOP_SEQ_IDS),
        StopAfterSentences(_tokenizer, max_sentences=max_sentences, window_tokens=160),
    ])

    kw: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        repetition_penalty=float(repetition_penalty),
        pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        eos_token_id=eos_ids,
        do_sample=bool(do_sample),
        stopping_criteria=stop_criteria,
    )
    if streamer is not None:
        kw["streamer"] = streamer
    if do_sample:
        kw.update(temperature=float(temperature), top_p=float(top_p))
    return kw


# -------------------
# Public APIs
# -------------------
def stream_hf_chat(
    user_prompt: str,
    *,
    system_prompt: str = "",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
    max_sentences: int = 1,
) -> Generator[str, None, None]:
    """
    Yield short response chunks; hard-stops at `max_sentences`.
    """
    _ensure_loaded()
    assert _tokenizer is not None and _model is not None

    prompt_text = _apply_chat_template(system_prompt, user_prompt)
    inputs = _tokenizer([prompt_text], return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(_model.device)

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = _build_kwargs(
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_sentences=max_sentences,
    )

    def _gen():
        with torch.no_grad():
            _ = _model.generate(**inputs, **gen_kwargs)

    thread = threading.Thread(target=_gen, daemon=True)
    thread.start()

    buf = ""
    for piece in streamer:
        if not piece:
            continue
        buf += piece
        # cut as soon as we hit the N-th sentence end
        truncated = _cut_to_n_sentences(buf, max_sentences)
        if truncated != buf:
            yield _sanitize_output(truncated)
            return

        # stream progressive sanitized chunks
        yield _sanitize_output(piece)

    # model stopped before N sentences -> flush remainder
    if buf.strip():
        yield _sanitize_output(_cut_to_n_sentences(buf, max_sentences))


def generate_full(
    user_prompt: str,
    *,
    system_prompt: str = "",
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
    max_sentences: int = 1,
) -> str:
    """
    Return a single short response string; hard-stops at `max_sentences`.
    """
    _ensure_loaded()
    assert _tokenizer is not None and _model is not None

    prompt_text = _apply_chat_template(system_prompt, user_prompt)
    inputs = _tokenizer([prompt_text], return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(_model.device)

    gen_kwargs = _build_kwargs(
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        streamer=None,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_sentences=max_sentences,
    )

    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = _sanitize_output(text)
    return _sanitize_output(_cut_to_n_sentences(text, max_sentences))


# -------------------
# Token counting helpers
# -------------------
def count_tokens(text: str) -> int:
    """
    Count tokenizer tokens for raw text (no chat template).
    """
    _ensure_loaded()
    assert _tokenizer is not None
    return len(_tokenizer.encode(text or "", add_special_tokens=False))


def count_chat_tokens(system_prompt: str, user_prompt: str) -> int:
    """
    Count tokens after applying chat template (useful for budgeting).
    """
    _ensure_loaded()
    assert _tokenizer is not None
    prompt_text = _apply_chat_template(system_prompt or "", user_prompt or "")
    return len(_tokenizer.encode(prompt_text, add_special_tokens=False))


__all__ = ["stream_hf_chat", "generate_full", "count_tokens", "count_chat_tokens"]
