# backend/core/tts_coqui.py
from TTS.api import TTS
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import base64
import os
import re

load_dotenv()
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

# One-time model load (warm)
tts = TTS(model_name=TTS_MODEL, progress_bar=False)
DEFAULT_SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)

_SENT_BOUNDARY = re.compile(r"([\.!\?]+[\)\]\"']?\s+)|(\n+)")
_MIN_CHARS = 30  # minimum chars before we try to synth

def _text_to_pcm16(text: str) -> tuple[bytes, int]:
    """
    TTS -> float32 [-1,1] -> PCM16 bytes, return (bytes, sample_rate).
    """
    if not text.strip():
        return b"", DEFAULT_SR
    wav = tts.tts(text)
    if not isinstance(wav, np.ndarray):
        wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16).tobytes()
    return pcm, DEFAULT_SR

def split_into_tts_units(incoming: str) -> list[str]:
    """
    Greedy sentence-ish splitter: break on . ! ? or newline.
    Falls back to length threshold.
    """
    acc = incoming.strip()
    if not acc:
        return []
    # Hard splits on boundaries
    parts = []
    last = 0
    for m in _SENT_BOUNDARY.finditer(acc + " "):  # sentinel space
        end = m.end()
        if end - last > 0:
            parts.append(acc[last:end].strip())
            last = end
    tail = acc[last:].strip()
    if tail:
        parts.append(tail)

    # If no punctuation, chunk by length
    out = []
    buf = ""
    for p in parts if parts else [acc]:
        buf += (p + " ")
        if len(buf) >= _MIN_CHARS or _SENT_BOUNDARY.search(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [s for s in out if s]

def tts_chunk_to_b64(text: str) -> dict:
    """
    Synthesize a small piece of text and return JSON-safe payload with base64 audio.
    """
    pcm, sr = _text_to_pcm16(text)
    if not pcm:
        return {"sr": sr, "b64": "", "samples": 0, "text": text}
    b64 = base64.b64encode(pcm).decode("ascii")
    return {"sr": sr, "b64": b64, "samples": len(pcm) // 2, "text": text}

def text_to_speech(text: str, file_id: str, output_dir: Path) -> str:
    """
    Existing file-output TTS (kept for /chat/voice route).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{file_id}_reply.wav"
    tts.tts_to_file(text=text, file_path=str(out_path))
    return str(out_path)
