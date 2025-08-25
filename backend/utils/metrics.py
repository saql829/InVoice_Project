# backend/utils/metrics.py
from __future__ import annotations
import time
from collections import deque
from typing import Any, Dict, List, Tuple

from prometheus_client import (
    Counter, Histogram, Gauge,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

# single registry so we don't double-register on hot reload
registry = CollectorRegistry()

# ---------- Prometheus metrics ----------
REQUESTS = Counter(
    "vma_http_requests_total", "HTTP requests",
    ["route", "method", "code"], registry=registry
)

WS_SESSIONS = Counter(
    "vma_ws_sessions_total", "WebSocket session events",
    ["event"], registry=registry  # open|close
)

AUDIO_BYTES = Counter(
    "vma_audio_bytes_total", "Audio bytes received from client",
    registry=registry
)

STT_LATENCY = Histogram(
    "vma_stt_latency_seconds", "Final STT (Whisper) latency, seconds",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20),
    registry=registry
)

LLM_LATENCY = Histogram(
    "vma_llm_latency_seconds", "LLM end-to-end latency (until first/full text sent)",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 6, 12),
    registry=registry
)

LLM_TOKENS = Counter(
    "vma_llm_tokens_total", "LLM token counts",
    ["direction"],  # in|out
    registry=registry
)

LLM_TPS = Histogram(
    "vma_llm_tokens_per_second", "Tokens per second for completions",
    buckets=(1, 3, 5, 10, 20, 40, 80, 120, 200),
    registry=registry
)

TTS_LATENCY = Histogram(
    "vma_tts_latency_seconds", "TTS synthesis latency (time to stream all bytes)",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
    registry=registry
)

TTS_BYTES = Counter(
    "vma_tts_bytes_total", "PCM bytes streamed to client (TTS)",
    registry=registry
)

ERRORS = Counter(
    "vma_errors_total", "Errors by component",
    ["where", "type"], registry=registry
)

# ---------- lightweight in-memory log (for a tiny dashboard) ----------
_MAX_EVENTS = 500
_events: deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)

def _now_ms() -> int:
    return int(time.time() * 1000)

def log_event(kind: str, msg: str, **fields: Any) -> None:
    _events.appendleft({
        "ts": _now_ms(),
        "kind": kind,
        "msg": msg,
        "fields": fields,
    })

def recent_events(limit: int = 200) -> List[Dict[str, Any]]:
    return list(list(_events)[:limit])

# ---------- helpers the app can call ----------
def observe_http(route: str, method: str, code: int) -> None:
    REQUESTS.labels(route=route, method=method, code=str(code)).inc()

def ws_open() -> None:
    WS_SESSIONS.labels(event="open").inc()
    log_event("ws", "open")

def ws_close() -> None:
    WS_SESSIONS.labels(event="close").inc()
    log_event("ws", "close")

def add_audio_bytes(n: int) -> None:
    AUDIO_BYTES.inc(n)

def observe_stt_latency(seconds: float) -> None:
    STT_LATENCY.observe(max(0.0, seconds))
    log_event("stt", "finalized", latency_s=round(seconds, 3))

def add_llm_in_tokens(n: int) -> None:
    if n > 0:
        LLM_TOKENS.labels(direction="in").inc(n)

def observe_llm_out(tokens: int, seconds: float) -> None:
    tokens = max(0, tokens)
    seconds = max(0.000001, seconds)
    LLM_TOKENS.labels(direction="out").inc(tokens)
    LLM_LATENCY.observe(seconds)
    LLM_TPS.observe(tokens / seconds)
    log_event("llm", "completed", tokens=tokens, latency_s=round(seconds, 3),
              tps=round(tokens / seconds, 2))

def observe_tts(bytes_out: int, seconds: float) -> None:
    TTS_BYTES.inc(max(0, bytes_out))
    TTS_LATENCY.observe(max(0.0, seconds))
    log_event("tts", "streamed", bytes=bytes_out, latency_s=round(seconds, 3))

def note_error(where: str, exc: BaseException) -> None:
    ERRORS.labels(where=where, type=exc.__class__.__name__).inc()
    log_event("error", f"{where} failed", type=exc.__class__.__name__, detail=str(exc))

# ---------- JSON snapshot for a tiny UI ----------
def json_snapshot() -> Dict[str, Any]:
    # small, human-friendly summary (no heavy math; Prometheus is source of truth)
    return {
        "counters": {
            "http_total": REQUESTS._value.get(),  # type: ignore[attr-defined]
            "ws_open_close": {"open": WS_SESSIONS._metrics.get(("open",), 0).get(),  # type: ignore
                              "close": WS_SESSIONS._metrics.get(("close",), 0).get()},  # type: ignore
        },
        "latest": list(recent_events(50)),
    }

def prometheus_export() -> Tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST
