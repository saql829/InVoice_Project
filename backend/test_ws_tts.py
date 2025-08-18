# test_http_tts.py
import sys
import io
import requests
import wave
from config.settings import TTS_SAMPLE_RATE  # agar ye import ho sakta hai

if len(sys.argv) < 2:
    print("Usage: python test_http_tts.py 'Your text here'")
    sys.exit(1)

text_input = " ".join(sys.argv[1:])
OUTPUT_FILE = "output.wav"

resp = requests.post("http://127.0.0.1:8000/api/tts", json={"text": text_input}, stream=True)

if resp.status_code != 200:
    print("Error:", resp.status_code, resp.text)
    sys.exit(1)

wav_bytes = io.BytesIO(resp.content)
with wave.open(wav_bytes, "rb") as w:
    frames = w.readframes(w.getnframes())
    with wave.open(OUTPUT_FILE, "wb") as out_w:
        out_w.setnchannels(w.getnchannels())
        out_w.setsampwidth(w.getsampwidth())
        out_w.setframerate(w.getframerate())
        out_w.writeframes(frames)

print(f"[INFO] Saved TTS audio to {OUTPUT_FILE}")
