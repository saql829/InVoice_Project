from TTS.api import TTS
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
TTS_MODEL = os.getenv("TTS_MODEL")

tts = TTS(model_name=TTS_MODEL, progress_bar=False)

def text_to_speech(text: str, file_id: str, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{file_id}_reply.wav"
    tts.tts_to_file(text=text, file_path=str(out_path))
    return str(out_path)
