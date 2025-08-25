from dotenv import load_dotenv
import os, pathlib, yaml

BASE = pathlib.Path(__file__).parent.parent
load_dotenv(BASE / ".env")

# ======================
# Model & API Settings
# ======================
LLM_MODEL     = os.getenv("LLM_MODEL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
TTS_MODEL     = os.getenv("TTS_MODEL")
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
STT_LANGUAGE      = os.getenv("STT_LANGUAGE", "en")

# ======================
# WebSocket tuning
# ======================
WS_MAX_MESSAGE_BYTES = int(os.getenv("WS_MAX_MESSAGE_BYTES", 2 * 1024 * 1024))

# ======================
# TTS Settings
# ======================
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", 22050))
TTS_LANGUAGE    = os.getenv("TTS_LANGUAGE", "en")
TTS_CHUNK_MS    = int(os.getenv("TTS_CHUNK_MS", 80))

# ======================
# Frontend origins
# ======================
raw_origins = os.getenv("FRONTEND_ORIGINS", "")
FRONTEND_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]


# ======================
# Metrics / Logging
# ======================
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"


# ======================
# Personas (system prompts + TTS presets)
# ======================
DEFAULT_PERSONA_KEY = os.getenv("DEFAULT_PERSONA", "friendly")

# Try load YAML
PERSONAS = {}
TTS_PRESETS = {}
try:
    personas_path = BASE / "config" / "personas.yml"
    if personas_path.exists():
        with open(personas_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        PERSONAS = cfg.get("personas", {}) or {}
        TTS_PRESETS = cfg.get("tts_presets", {}) or {}
except Exception as e:
    print("personas.yml load error:", e)

# Fallback defaults if YAML missing/empty
if not PERSONAS:
    PERSONAS = {
        "friendly": {
            "label": "Friendly",
            "system_prompt": "You are warm and upbeat. Use simple words and short sentences.",
            "tts_preset": "friendly_voice",
        },
        "neutral": {
            "label": "Neutral",
            "system_prompt": "You are concise and helpful. Keep answers short.",
            "tts_preset": "neutral_voice",
        },
        "teacher": {
            "label": "Teacher",
            "system_prompt": "Explain step by step. Define terms briefly before using them.",
            "tts_preset": "teacher_voice",
        },
    }
if not TTS_PRESETS:
    # If files don't exist, TTS will fall back to default voice, but UI will still populate.
    TTS_PRESETS = {
        "friendly_voice": {"speaker_wav": "assets/voices/friendly.wav", "language": "en"},
        "neutral_voice":  {"speaker_wav": "assets/voices/neutral.wav",  "language": "en"},
        "teacher_voice":  {"speaker_wav": "assets/voices/teacher.wav",  "language": "en"},
    }

if DEFAULT_PERSONA_KEY not in PERSONAS:
    DEFAULT_PERSONA_KEY = "friendly"
