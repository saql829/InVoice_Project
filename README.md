# InVoice — Streaming Voice Assistant (MVP)

**Goal:** Real‑time voice chat with AI — live STT (Whisper), streaming LLM reply, and TTS.

## Stack
- **Backend:** FastAPI, WebSockets, Uvicorn  
- **STT:** Hugging Face Whisper (`openai/whisper-tiny` by default)  
- **LLM:** TinyLlama locally (dev) / Mistral via HF Inference (prod)  
- **TTS:** Coqui TTS (Tacotron2 + HiFiGAN)

## Quick Start
```bash
# backend
cd backend
python -m venv venv310 && source venv310/bin/activate
pip install -r requirements.txt
cp ../.env.example .env   # fill your HF key
uvicorn main:app --reload
