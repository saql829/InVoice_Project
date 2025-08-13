# backend/clients/ws_test_stream_all.py
import asyncio, json, os, sys, base64, websockets, wave, subprocess, shlex

WS_URL = "ws://127.0.0.1:8000/ws/audio"
CHUNK_BYTES = 16000   # MUST match backend CHUNK_SIZE

def ensure_pcm(path_in: str) -> str:
    """
    Accepts .pcm (s16le mono 16k) OR .wav (auto-convert to .pcm).
    Returns path to a .pcm file.
    """
    path_in = os.path.abspath(path_in)
    root, ext = os.path.splitext(path_in)

    if ext.lower() == ".pcm":
        if os.path.exists(path_in):
            return path_in
        raise SystemExit(f"PCM not found: {path_in}")

    if ext.lower() == ".wav":
        if not os.path.exists(path_in):
            raise SystemExit(f"WAV not found: {path_in}")
        out_pcm = root + ".pcm"
        cmd = f"ffmpeg -y -i {shlex.quote(path_in)} -f s16le -ac 1 -ar 16000 {shlex.quote(out_pcm)}"
        print(">> converting wav->pcm:", cmd)
        subprocess.check_call(cmd, shell=True)
        return out_pcm

    # No extension, try as-is
    if os.path.exists(path_in):
        return path_in

    raise SystemExit("Provide test.pcm (raw s16le mono 16k) or test.wav to auto-convert.")

async def run_stream(inp_path: str):
    pcm_path = ensure_pcm(inp_path)

    tts_pcm = bytearray()
    tts_sr = None
    got_llm_done = False
    got_tts_done = False

    async with websockets.connect(WS_URL, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"event": "start"}))
        print(">> sent: start")

        # Send audio in chunks
        with open(pcm_path, "rb") as f:
            while True:
                data = f.read(CHUNK_BYTES)
                if not data:
                    break
                await ws.send(data)
                # read fast acks if any
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print("<<", msg)
                except asyncio.TimeoutError:
                    pass

        # Signal end of utterance
        await ws.send(json.dumps({"event": "end"}))
        print(">> sent: end")

        # Read stream until both LLM and TTS finish, then exit
        try:
            while True:
                raw = await ws.recv()

                # Binary? (we don't send raw TTS binary here; server sends JSON+b64)
                if isinstance(raw, bytes):
                    print("<< (binary)", len(raw))
                    continue

                # JSON path
                print("<<", raw)

                # TTS chunk (base64) collect
                if '"type":"tts_chunk"' in raw:
                    d = json.loads(raw)
                    b64 = d.get("b64") or ""
                    if b64:
                        tts_pcm.extend(base64.b64decode(b64))
                    if tts_sr is None:
                        tts_sr = d.get("sr", 22050)

                if '"type":"done_llm"' in raw:
                    got_llm_done = True
                if '"type":"tts_done"' in raw:
                    got_tts_done = True

                # Auto-exit condition
                if got_llm_done and got_tts_done:
                    print("<< all done signals received")
                    break

        except websockets.ConnectionClosed:
            print("<< connection closed")

    # Always save TTS if received
    if tts_pcm and tts_sr:
        out_wav = "tts_stream.wav"
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)     # 16‑bit
            wf.setframerate(int(tts_sr))
            wf.writeframes(bytes(tts_pcm))
        print(f">> saved TTS audio -> {out_wav}")

if __name__ == "__main__":
    # Run from inside backend/:
    #   python clients/ws_test_stream_all.py test.wav
    #   or
    #   python clients/ws_test_stream_all.py test.pcm
    arg = sys.argv[1] if len(sys.argv) > 1 else "test.pcm"
    asyncio.run(run_stream(arg))
