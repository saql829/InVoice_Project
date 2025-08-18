# test_tts.py
import requests

# Updated URL with /api prefix
TTS_URL = "http://127.0.0.1:8000/api/tts"

text_to_speak = "Hello, this is a backend TTS test."

payload = {
    "text": text_to_speak
}

print(f"Sending text to TTS: {text_to_speak}")
response = requests.post(TTS_URL, json=payload)

if response.status_code == 200:
    with open("tts_output.wav", "wb") as f:
        f.write(response.content)
    print("✅ TTS audio saved as tts_output.wav")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
