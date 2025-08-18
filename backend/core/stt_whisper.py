# backend/core/stt_whisper.py
"""
Whisper STT helpers:
- Path + bytes transcription (thread-safe)
- English forced by default via STT_LANGUAGE (.env) else "en"
- No circular imports
"""

import os
import threading
from typing import Optional, Iterable, Iterator

import torch
import whisper
import numpy as np

from utils.audio_utils import (
    decode_to_pcm16,
    pcm16_to_float32,
    frame_generator,
    collect_segments_vad,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
try:
    from config.settings import (
        WHISPER_MODEL,
        AUDIO_SAMPLE_RATE,
        FRAME_MS,
        SILENCE_MS as VAD_SILENCE_MS,  # your settings used SILENCE_MS earlier
        STT_LANGUAGE,
    )
except Exception:
    WHISPER_MODEL     = os.getenv("WHISPER_MODEL", "tiny.en")
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    FRAME_MS          = int(os.getenv("FRAME_MS", "20"))
    VAD_SILENCE_MS    = int(os.getenv("SILENCE_MS", "600"))
    STT_LANGUAGE      = os.getenv("STT_LANGUAGE", "en")

_DEFAULT_LANG = (STT_LANGUAGE or "en").strip().lower() or "en"

# ---------------------------------------------------------------------
# Load Whisper once + global lock
# ---------------------------------------------------------------------
_device   = "cuda" if torch.cuda.is_available() else "cpu"
_model    = whisper.load_model(WHISPER_MODEL, device=_device)
_STT_LOCK = threading.Lock()

def _transcribe_f32(audio_f32: np.ndarray, language: Optional[str] = None) -> str:
    """
    audio_f32: mono float32 @16k in [-1, 1]
    Returns a trimmed transcription string (no extra logs).
    """
    lang = (language or _DEFAULT_LANG)
    with _STT_LOCK:
        result = _model.transcribe(
            audio_f32,
            fp16=(_device == "cuda"),
            language=lang,
            condition_on_previous_text=False,
            verbose=False,
            task="transcribe",
        )
    return (result.get("text") or "").strip()

# ---------------------------------------------------------------------
# 1) File path based
# ---------------------------------------------------------------------
def transcribe_audio(path: str, language: Optional[str] = None) -> str:
    """
    File path -> Whisper transcribe.
    """
    lang = (language or _DEFAULT_LANG)
    with _STT_LOCK:
        result = _model.transcribe(
            path,
            fp16=(_device == "cuda"),
            language=lang,
            condition_on_previous_text=False,
            verbose=False,
            task="transcribe",
        )
    return (result.get("text") or "").strip()

# ---------------------------------------------------------------------
# 2) Bytes based
# ---------------------------------------------------------------------
def transcribe_bytes(data: bytes, mime: Optional[str] = None, language: Optional[str] = None) -> str:
    """
    Any audio (wav/mp3/webm/ogg/etc.) bytes -> text.
    Decodes to PCM16 mono @16k, converts to float32, then transcribes.
    """
    pcm16     = decode_to_pcm16(data, mime=mime, target_rate=AUDIO_SAMPLE_RATE, target_channels=1)
    audio_f32 = pcm16_to_float32(pcm16)
    return _transcribe_f32(audio_f32, language=language or _DEFAULT_LANG)

# ---------------------------------------------------------------------
# 3) (optional) VAD streaming helper
# ---------------------------------------------------------------------
def transcribe_vad_segments(pcm16_iter: Iterable[bytes], language: Optional[str] = None) -> Iterator[str]:
    """
    Feed PCM16 mono @16k chunks, segment via VAD, and yield per-segment transcripts.
    """
    lang = (language or _DEFAULT_LANG)
    buf = b""
    for chunk in pcm16_iter:
        buf += chunk
        segments = list(collect_segments_vad(
            frame_generator(buf, sample_rate=AUDIO_SAMPLE_RATE, frame_ms=FRAME_MS),
            silence_ms=VAD_SILENCE_MS,
            sample_rate=AUDIO_SAMPLE_RATE,
            frame_ms=FRAME_MS,
        ))
        if segments:
            buf = b""
        for seg in segments:
            wav_f32 = pcm16_to_float32(seg)
            yield _transcribe_f32(wav_f32, language=lang)
