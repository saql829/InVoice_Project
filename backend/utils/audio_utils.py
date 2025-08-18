import io
import wave
from dataclasses import dataclass
from typing import Iterator, Iterable, Optional, List

import numpy as np
from pydub import AudioSegment

# ---- Safe defaults if settings not present ----
try:
    from config.settings import (
        AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, FRAME_MS,
        TTS_SAMPLE_RATE
    )
except Exception:
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    FRAME_MS = 20  # VAD-friendly (10/20/30ms)
    TTS_SAMPLE_RATE = 22050

# ---------------------------------------
# Decode / Normalize helpers
# ---------------------------------------
def decode_to_pcm16(
    audio_bytes: bytes,
    mime: Optional[str] = None,
    target_rate: int = AUDIO_SAMPLE_RATE,
    target_channels: int = AUDIO_CHANNELS,
) -> bytes:
    """
    Any input (wav/mp3/webm/ogg/m4a) -> 16 kHz, mono, PCM16 raw bytes.
    Needs ffmpeg installed (pydub backend).
    """
    seg = AudioSegment.from_file(
        io.BytesIO(audio_bytes),
        format=None if not mime else mime.split("/")[-1],
    )
    seg = (
        seg.set_frame_rate(target_rate)
           .set_channels(target_channels)
           .set_sample_width(2)  # 16-bit PCM
    )
    return seg.raw_data  # little-endian PCM16


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    """
    PCM16 raw bytes -> float32 numpy array in [-1, 1].
    """
    arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    return arr


def float32_to_pcm16(f32: np.ndarray) -> bytes:
    """
    float32 [-1, 1] -> PCM16 raw bytes.
    """
    x = np.clip(f32, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    return x.tobytes()


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int = AUDIO_SAMPLE_RATE) -> bytes:
    """
    PCM16 raw -> WAV container (in-memory). Handy for playback/clients.
    """
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return bio.getvalue()


def wav_bytes_to_pcm16(wav_bytes: bytes) -> bytes:
    """
    WAV container -> PCM16 raw bytes (mono).
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getsampwidth() == 2, "WAV must be 16-bit"
        assert w.getnchannels() == 1, "WAV must be mono"
        pcm = w.readframes(w.getnframes())
    return pcm

# ---------------------------------------
# Framing (for VAD & streaming)
# ---------------------------------------
@dataclass
class Frame:
    bytes: bytes
    timestamp: float
    duration: float


def frame_generator(
    pcm16: bytes,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
) -> Iterator[Frame]:
    """
    Split PCM16 stream into fixed-size frames (10/20/30ms typical for VAD).
    """
    frame_len = int(sample_rate * frame_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    ts = 0.0
    dur = frame_ms / 1000.0
    total = len(pcm16)

    while offset + frame_len <= total:
        chunk = pcm16[offset : offset + frame_len]
        yield Frame(chunk, ts, dur)
        offset += frame_len
        ts += dur

    # tail (pad-safe optional): ignore partial (most players don't need padding for streaming)
    # if offset < total: yield Frame(pcm16[offset:], ts, (total-offset)/(2*sample_rate))

# ---------------------------------------
# Simple VAD segment collector (webrtcvad)
# ---------------------------------------
def collect_segments_vad(
    frames: Iterable[Frame],
    silence_ms: int,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
    aggressiveness: int = 2,
) -> Iterator[bytes]:
    """
    Collect voiced segments using WebRTC VAD:
      - Emits concatenated PCM16 when trailing silence > silence_ms.
      - Emits last buffered audio on stream end.

    NOTE: requires `webrtcvad` to be installed.
    """
    try:
        import webrtcvad
    except Exception:
        raise RuntimeError("webrtcvad not installed. Add `webrtcvad` to requirements.txt")

    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be one of (10, 20, 30) for webrtcvad")

    vad = webrtcvad.Vad(int(aggressiveness))
    silence_frames_needed = max(1, int(silence_ms / frame_ms))

    voiced: List[bytes] = []
    silence_count = 0

    for f in frames:
        is_speech = vad.is_speech(f.bytes, sample_rate)
        if is_speech:
            voiced.append(f.bytes)
            silence_count = 0
        else:
            if voiced:
                silence_count += 1
                if silence_count >= silence_frames_needed:
                    yield b"".join(voiced)
                    voiced.clear()
                    silence_count = 0

    if voiced:
        yield b"".join(voiced)


# ---------------------------------------
# TTS helpers (chunking for stream)
# ---------------------------------------
def tts_pcm_to_stream_frames(
    pcm16: bytes,
    sample_rate: int = TTS_SAMPLE_RATE,
    frame_ms: int = 40,
) -> Iterator[bytes]:
    """
    Convert full TTS PCM16 buffer into streaming-sized binary frames (ms).
    Client can concatenate these bytes (headerless PCM) or you may wrap first chunk into WAV externally.
    """
    for f in frame_generator(pcm16, sample_rate=sample_rate, frame_ms=frame_ms):
        yield f.bytes
