from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from core.stt_whisper import transcribe_audio
from core.llm_model import generate_response
from core.tts_coqui import text_to_speech
from pathlib import Path
import uuid

router = APIRouter()

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
TEMP_DIR = ROOT_DIR / "shared" / "temp_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/voice")
async def chat_via_voice(audio: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}.wav"
        data = await audio.read()
        input_path.write_bytes(data)

        text = transcribe_audio(str(input_path))
        if not text:
            raise HTTPException(status_code=400, detail="STT failed")

        reply = generate_response(text)
        if not reply:
            raise HTTPException(status_code=500, detail="LLM did not return a response")

        out_path = text_to_speech(reply, file_id, output_dir=TEMP_DIR)
        p = Path(out_path)
        if not p.exists() or p.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="TTS output not generated")

        return FileResponse(str(p), media_type="audio/wav", filename="reply.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
