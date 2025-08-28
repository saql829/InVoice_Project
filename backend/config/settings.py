# backend/config/settings.py

from __future__ import annotations

import os

import pathlib

from dotenv import load_dotenv, find_dotenv
 
# Load .env reliably

load_dotenv(find_dotenv(), override=True)
 
BASE = pathlib.Path(__file__).resolve().parent.parent
 
def to_bool(x: str | None, default: bool = False) -> bool:

    if x is None:

        return default

    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}
 
def getenv_int(key: str, default: int) -> int:

    try:

        return int(os.getenv(key, str(default)))

    except Exception:

        return default
 
def getenv_list(key: str) -> list[str]:

    raw = os.getenv(key, "") or ""

    return [p.strip() for p in raw.split(",") if p.strip()]
 
# -------- Server --------

API_HOST  = os.getenv("API_HOST", "0.0.0.0")

API_PORT  = getenv_int("API_PORT", 8000)

LOG_LEVEL = (os.getenv("LOG_LEVEL", "info") or "info").lower()
 
# -------- Audio / STT / VAD --------

VAD_ENABLED       = to_bool(os.getenv("VAD_ENABLED"), True)

FRAME_MS          = getenv_int("FRAME_MS", 20)

SILENCE_MS        = getenv_int("SILENCE_MS", 800)

PARTIAL_STT_MS    = getenv_int("PARTIAL_STT_MS", 1200)

MAX_BUFFER_MS     = getenv_int("MAX_BUFFER_MS", 10000)

AUDIO_SAMPLE_RATE = getenv_int("AUDIO_SAMPLE_RATE", 16000)

AUDIO_CHANNELS    = getenv_int("AUDIO_CHANNELS", 1)

STT_LANGUAGE      = os.getenv("STT_LANGUAGE", "en")
 
# -------- WebSocket --------

WS_MAX_MESSAGE_BYTES = getenv_int("WS_MAX_MESSAGE_BYTES", 2 * 1024 * 1024)
 
# -------- TTS --------

TTS_MODEL       = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

TTS_SAMPLE_RATE = getenv_int("TTS_SAMPLE_RATE", 22050)

TTS_LANGUAGE    = os.getenv("TTS_LANGUAGE", "en")

TTS_CHUNK_MS    = getenv_int("TTS_CHUNK_MS", 80)
 
# -------- Frontend CORS --------

FRONTEND_ORIGINS = getenv_list("FRONTEND_ORIGINS")
 
# -------- STT model --------

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
 
# -------- HF / GGUF selection --------

_env_use_local      = os.getenv("USE_LOCAL_GGUF")   # may be None

_env_llm_model      = os.getenv("LLM_MODEL") or None

_env_gguf_modelpath = os.getenv("GGUF_MODEL_PATH") or "./models/phi-2/phi-2.Q4_K_M.gguf"

HF_API_KEY          = os.getenv("HF_API_KEY")  # may be None
 
# Rule:

# 1) If USE_LOCAL_GGUF explicitly set -> follow it.

# 2) Else if LLM_MODEL set -> HF mode.

# 3) Else GGUF mode.

if _env_use_local is not None:

    USE_LOCAL_GGUF = to_bool(_env_use_local, False)

else:

    USE_LOCAL_GGUF = False if _env_llm_model else True
 
if USE_LOCAL_GGUF:

    GGUF_MODEL_PATH = _env_gguf_modelpath

    LLM_MODEL = None

else:

    LLM_MODEL = _env_llm_model

    GGUF_MODEL_PATH = None
 
def validate_and_log():

    mode = "GGUF" if USE_LOCAL_GGUF else "HF"

    print(f"[config] mode={mode}")

    print(f"[config] API_HOST={API_HOST} API_PORT={API_PORT}")

    print(f"[config] WS_MAX_MESSAGE_BYTES={WS_MAX_MESSAGE_BYTES}")

    print(f"[config] FRONTEND_ORIGINS={FRONTEND_ORIGINS}")

    print(f"[config] WHISPER_MODEL={WHISPER_MODEL}")

    print(f"[config] TTS_MODEL={TTS_MODEL}")
 
    if USE_LOCAL_GGUF:

        print(f"[config] GGUF_MODEL_PATH={GGUF_MODEL_PATH}")

        try:

            p = pathlib.Path(GGUF_MODEL_PATH)

            if not p.exists():

                print(f"[warn] GGUF model not found at {GGUF_MODEL_PATH}.")

        except Exception:

            pass

    else:

        print(f"[config] LLM_MODEL={LLM_MODEL}")

        if not LLM_MODEL:

            raise ValueError(

                "LLM_MODEL must be set when USE_LOCAL_GGUF=false "

                "(either set USE_LOCAL_GGUF=true or provide LLM_MODEL)."

            )
