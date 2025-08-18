# backend/config/settings.py
from dotenv import load_dotenv
import os, pathlib

BASE = pathlib.Path(__file__).parent.parent
load_dotenv(BASE / ".env")

# ======================
# Model & API Settings
# ======================
LLM_MODEL     = os.getenv("LLM_MODEL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
TTS_MODEL     = os.getenv("TTS_MODEL")  # e.g., "tts_models/multilingual/multi-dataset/xtts_v2"

HF_API_KEY    = os.getenv("HF_API_KEY")

# ======================
# Server Settings
# ======================
API_HOST      = os.getenv("API_HOST", "0.0.0.0")
API_PORT      = int(os.getenv("API_PORT", 8000))
LOG_LEVEL     = os.getenv("LOG_LEVEL", "info")

# ======================
# Audio / STT / VAD
# ======================
VAD_ENABLED       = os.getenv("VAD_ENABLED", "true").lower() == "true"
FRAME_MS          = int(os.getenv("FRAME_MS", 20))
SILENCE_MS        = int(os.getenv("SILENCE_MS", 800))
PARTIAL_STT_MS    = int(os.getenv("PARTIAL_STT_MS", 1200))
MAX_BUFFER_MS     = int(os.getenv("MAX_BUFFER_MS", 10000))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
AUDIO_CHANNELS    = int(os.getenv("AUDIO_CHANNELS", 1))
STT_LANGUAGE      = os.getenv("STT_LANGUAGE", "")

# ======================
# WebSocket tuning
# ======================
WS_MAX_MESSAGE_BYTES = int(os.getenv("WS_MAX_MESSAGE_BYTES", 2 * 1024 * 1024))

# ======================
# TTS Settings
# ======================
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", 22050))
TTS_LANGUAGE    = os.getenv("TTS_LANGUAGE", "en")
TTS_CHUNK_MS    = int(os.getenv("TTS_CHUNK_MS", 80))  # per-chunk duration for streaming

# ======================
# Frontend origins
# ======================
raw_origins = os.getenv("FRONTEND_ORIGINS", "")
FRONTEND_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
