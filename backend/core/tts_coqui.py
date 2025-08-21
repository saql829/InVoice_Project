"""
Coqui TTS wrapper with simple streaming.
- Streams PCM16 chunks (headerless) so frontend can play incrementally.
- Preset-aware helpers (speaker_wav/lang/model) for personas.
"""
from typing import Iterable, Optional, Dict, Tuple
import numpy as np

from config.settings import (
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    TTS_LANGUAGE,
    TTS_CHUNK_MS,
)

# ---------------- Utils ----------------

def _float32_to_pcm16(f32: np.ndarray) -> bytes:
    x = np.clip(f32, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    return x.tobytes()

# Aliases so short names won't crash
_XTTS_ALIASES = {
    "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "coqui-xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
}
def _normalize_model(name: Optional[str]) -> str:
    name = (name or "").strip()
    if not name:
        return "tts_models/multilingual/multi-dataset/xtts_v2"
    return _XTTS_ALIASES.get(name, name)

# ---------------- Global cache ----------------
# model_id -> (TTS instance, sample_rate, is_xtts)
_TTS_CACHE: Dict[str, Tuple[object, int, bool]] = {}

def _ensure_model(model_name: Optional[str] = None) -> Tuple[object, int, bool]:
    """
    Ensure a TTS model is loaded and cached. Returns (instance, sample_rate, is_xtts)
    """
    try:
        from TTS.api import TTS  # lazy import with friendly error if missing
    except Exception as e:
        raise RuntimeError(
            "Coqui TTS not installed. Add `TTS` to requirements and pip install TTS."
        ) from e

    model_id = _normalize_model(model_name or TTS_MODEL)

    if model_id in _TTS_CACHE:
        return _TTS_CACHE[model_id]

    inst = TTS(model_id)
    sr = getattr(inst, "output_sample_rate", None)
    sr = int(sr or TTS_SAMPLE_RATE or 22050)
    is_xtts = "xtts" in model_id.lower()
    _TTS_CACHE[model_id] = (inst, sr, is_xtts)
    return _TTS_CACHE[model_id]

# ---------------- Core synth ----------------

def tts_full_pcm16(
    text: str,
    *,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: Optional[int] = None,
    model: Optional[str] = None,
) -> bytes:
    """
    Return full utterance as PCM16 bytes (mono).
    """
    if not (text or "").strip():
        return b""

    inst, sr_model, is_xtts = _ensure_model(model)
    lang = (language or TTS_LANGUAGE or "en").lower()
    sr = int(sample_rate or sr_model)

    if is_xtts:
        wav = inst.tts(text=text, speaker_wav=speaker_wav, language=lang)
    else:
        # non-XTTS models ignore speaker/lang
        wav = inst.tts(text=text)

    wav = np.asarray(wav, dtype=np.float32)
    pcm = _float32_to_pcm16(wav)
    return pcm

def tts_with_preset(text: str, preset: dict | None = None) -> bytes:
    """
    preset keys (all optional):
      - model:       coqui model id or alias ('xtts_v2')
      - speaker_wav: path to reference wav (XTTS)
      - language:    'en', 'hi', 'ur', etc. (XTTS)
    """
    p = preset or {}
    return tts_full_pcm16(
        text,
        speaker_wav=p.get("speaker_wav"),
        language=p.get("language"),
        model=p.get("model") or TTS_MODEL,
    )

def stream_tts_pcm(
    text: str,
    *,
    speaker_wav: Optional[str] = None,
    language: Optional[str] = None,
    chunk_ms: Optional[int] = None,
    model: Optional[str] = None,
) -> Iterable[bytes]:
    pcm = tts_full_pcm16(text, speaker_wav=speaker_wav, language=language, model=model)
    if not pcm:
        return
    _, sr, _ = _ensure_model(model)
    ms = int(chunk_ms or TTS_CHUNK_MS or 80)
    b_per_ms = (sr * 2) / 1000.0  # 16-bit mono
    step = max(1, int(ms * b_per_ms))
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]

def stream_tts_with_preset(text: str, preset: dict | None = None) -> Iterable[bytes]:
    p = preset or {}
    pcm = tts_with_preset(text, p)
    if not pcm:
        return
    _, sr, _ = _ensure_model(p.get("model"))
    ms = int(TTS_CHUNK_MS or 80)
    b_per_ms = (sr * 2) / 1000.0
    step = max(1, int(ms * b_per_ms))
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]

def tts_info():
    inst, sr, is_xtts = _ensure_model()
    return {"model": type(inst).__name__, "sample_rate": sr, "is_xtts": is_xtts}
