# backend/core/tts_coqui.py
"""
Coqui TTS wrapper with simple streaming.
- Streams PCM16 chunks (headerless) so frontend can play incrementally.
- Produces the full audio then yields small chunks (works reliably).
"""
from typing import Iterable, Optional
import numpy as np

from config.settings import (
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    TTS_LANGUAGE,
    TTS_CHUNK_MS,
)

# Local float32->pcm16
def _float32_to_pcm16(f32: np.ndarray) -> bytes:
    x = np.clip(f32, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    return x.tobytes()

_TTS_INSTANCE = None
_TTS_SR = None
_IS_XTTS = False

def _ensure_init():
    global _TTS_INSTANCE, _TTS_SR, _IS_XTTS
    if _TTS_INSTANCE is not None:
        return
    try:
        from TTS.api import TTS
    except Exception as e:
        raise RuntimeError(
            "Coqui TTS not installed. Add `TTS` to requirements and pip install TTS."
        ) from e

    model_name = TTS_MODEL or "tts_models/multilingual/multi-dataset/xtts_v2"
    _TTS_INSTANCE = TTS(model_name)
    sr = getattr(_TTS_INSTANCE, "output_sample_rate", None)
    _TTS_SR = int(sr or TTS_SAMPLE_RATE or 22050)
    _IS_XTTS = "xtts" in model_name.lower()

def tts_full_pcm16(
    text: str,
    *,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: Optional[int] = None,
) -> bytes:
    """
    Synthesize full utterance -> PCM16 raw bytes.
    """
    if not (text or "").strip():
        return b""
    _ensure_init()
    lang = (language or TTS_LANGUAGE or "en").lower()
    sr = int(sample_rate or _TTS_SR)
    if _IS_XTTS:
        wav = _TTS_INSTANCE.tts(text=text, speaker_wav=speaker_wav, language=lang)
    else:
        wav = _TTS_INSTANCE.tts(text=text)
    wav = np.asarray(wav, dtype=np.float32)
    pcm = _float32_to_pcm16(wav)
    return pcm

def stream_tts_pcm(
    text: str,
    *,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None,
    chunk_ms: Optional[int] = None,
) -> Iterable[bytes]:
    """
    Generate full audio then yield PCM16 in small chunks (fake streaming).
    Works reliably and starts delivering chunks quickly.
    """
    pcm = tts_full_pcm16(text, speaker_wav=speaker_wav, language=language)
    if not pcm:
        return
    _ensure_init()
    sr = _TTS_SR
    ms = int(chunk_ms or TTS_CHUNK_MS or 80)  # default chunk size (ms)
    # bytes per ms: (sr * 2 bytes * 1ch) / 1000
    b_per_ms = (sr * 2) / 1000.0
    step = max(1, int(ms * b_per_ms))
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]

def tts_info():
    _ensure_init()
    return {"model": TTS_MODEL, "sample_rate": _TTS_SR, "is_xtts": _IS_XTTS}
