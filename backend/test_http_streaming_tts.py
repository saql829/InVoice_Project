# test_http_streaming_tts.py
import sys
import io
import requests
import wave
from config.settings import TTS_SAMPLE_RATE  # agar ye available hai

TTS_URL = "http://127.0.0.1:8000/api/tts"
OUTPUT_FILE = "output_streaming.wav"

if len(sys.argv) < 2:
    print("Usage: python test_http_streaming_tts.py 'Your text here'")
    sys.exit(1)

text_input = " ".join(sys.argv[1:])

# HTTP POST request with streaming
resp = requests.post(TTS_URL, json={"text": text_input}, stream=True)

if resp.status_code != 200:
    print("Error:", resp.status_code, resp.text)
    sys.exit(1)

pcm_data = b""
print("[INFO] Receiving TTS audio in streaming mode...")

for chunk in resp.iter_content(chunk_size=1024):
    if chunk:
        pcm_data += chunk
        print(f"[CHUNK] Received {len(chunk)} bytes")

# Save received PCM/WAV bytes
wav_buffer = io.BytesIO(pcm_data)
try:
    # Agar server ne WAV return kiya
    with wave.open(wav_buffer, "rb") as w:
        frames = w.readframes(w.getnframes())
        with wave.open(OUTPUT_FILE, "wb") as out_w:
            out_w.setnchannels(w.getnchannels())
            out_w.setsampwidth(w.getsampwidth())
            out_w.setframerate(w.getframerate())
            out_w.writeframes(frames)
except wave.Error:
    # Agar server ne raw PCM16 bheja
    with wave.open(OUTPUT_FILE, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # PCM16
        w.setframerate(TTS_SAMPLE_RATE or 22050)
        w.writeframes(pcm_data)

print(f"[INFO] Saved streaming TTS audio to {OUTPUT_FILE}")
