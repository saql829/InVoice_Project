from transformers import pipeline
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
HF_API_KEY = os.getenv("HF_API_KEY")

asr = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL,
    token=HF_API_KEY,
    return_timestamps=False,
)

def transcribe_audio(audio_path: str) -> str:
    """WAV/MP3 file path -> text (used by /chat/voice)."""
    if not audio_path:
        return ""
    try:
        result = asr(audio_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[STT error:file] {e}")
        return ""

def transcribe_pcm_bytes(pcm_bytes: bytes, sr: int = 16000) -> str:
    """Raw S16_LE mono PCM bytes -> text (used by WebSocket partials)."""
    if not pcm_bytes:
        return ""
    try:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = asr({"array": audio, "sampling_rate": sr})
        txt = result.get("text", "").strip()
        return txt
    except Exception as e:
        print(f"[STT error:pcm] {e}")
        return ""
