from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional

from core.tts_coqui import synthesize_to_wav, stream_tts_frames
from config.settings import (
    TTS_OUTPUT_FORMAT, TTS_SAMPLE_RATE, TTS_STREAM_CHUNK_MS, WS_MAX_MESSAGE_BYTES
)

router = APIRouter()

# ---------- HTTP: generate full audio (WAV or raw PCM16) ----------
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    format: Optional[str] = None  # "wav" | "pcm16"

@router.post("/tts")
def tts_http(req: TTSRequest):
    fmt = (req.format or TTS_OUTPUT_FORMAT).lower()
    if fmt not in ("wav", "pcm16"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'pcm16'")

    if fmt == "wav":
        wav_bytes = synthesize_to_wav(
            text=req.text, voice=req.voice, language=req.language, speed=req.speed, sample_rate=TTS_SAMPLE_RATE
        )
        return Response(content=wav_bytes, media_type="audio/wav")

    # raw pcm16 (mono)
    # client side must know sample rate = TTS_SAMPLE_RATE, 16-bit, mono
    from core.tts_coqui import synthesize_to_pcm16
    pcm16 = synthesize_to_pcm16(
        text=req.text, voice=req.voice, language=req.language, speed=req.speed, sample_rate=TTS_SAMPLE_RATE
    )
    return Response(content=pcm16, media_type="application/octet-stream")


# ---------- WebSocket: streaming chunks ----------
# Protocol (mirrors your STT style):
# Client -> { "type": "start", "text": "...", "voice": "...", "language":"...", "speed": 1.0, "format":"pcm16|wav" }
# Server -> { "type": "start", "sr": 22050, "frame_ms": 40, "format":"pcm16" }   # ack
# Server -> <binary bytes>  (many chunks)
# Server -> { "type": "end" }
@router.websocket("/ws/tts")
async def tts_ws(ws: WebSocket):
    await ws.accept(max_size=WS_MAX_MESSAGE_BYTES)

    try:
        init = await ws.receive_json()
        if not isinstance(init, dict) or init.get("type") != "start":
            await ws.send_json({"type": "error", "error": "first message must be {type:'start', ...}"})
            await ws.close()
            return

        text      = init.get("text", "")
        voice     = init.get("voice")
        language  = init.get("language")
        speed     = init.get("speed")
        fmt       = (init.get("format") or "pcm16").lower()

        if not text.strip():
            await ws.send_json({"type": "error", "error": "empty text"})
            await ws.close()
            return

        # We stream PCM16 frames for both modes; if client wants WAV, they can wrap or we can send a full WAV at the end.
        # Keeping it simple/low-latency: stream headerless PCM16 frames.
        await ws.send_json({
            "type": "start",
            "sr": TTS_SAMPLE_RATE,
            "frame_ms": TTS_STREAM_CHUNK_MS,
            "format": "pcm16"
        })

        # Produce frames and push
        async def _push_frames():
            for chunk in stream_tts_frames(
                text=text, voice=voice, language=language, speed=speed,
                sample_rate=TTS_SAMPLE_RATE, frame_ms=TTS_STREAM_CHUNK_MS
            ):
                await ws.send_bytes(chunk)

        await _push_frames()
        await ws.send_json({"type": "end"})

    except WebSocketDisconnect:
        # client closed early; ignore
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        finally:
            await ws.close()
