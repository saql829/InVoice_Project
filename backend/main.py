# backend/main.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import logging, time

from api.voice_chat import router as voice_router
from api.chat import router as chat_router
from api.ws_voice import router as ws_router  # keep as-is (your older socket)
from config.settings import API_HOST, API_PORT, FRONTEND_ORIGINS, LOG_LEVEL, METRICS_ENABLED

# metrics
if METRICS_ENABLED:
    from utils.metrics import (
        prometheus_export, json_snapshot, recent_events,
        observe_http
    )

app = FastAPI(title="Voice Mode Assistant Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS or ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- logging setup --------
logging.basicConfig(
    level=getattr(logging, (LOG_LEVEL or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("vma")

# -------- HTTP metrics middleware --------
if METRICS_ENABLED:
    @app.middleware("http")
    async def _metrics_mw(request: Request, call_next):
        t0 = time.perf_counter()
        code = 500
        try:
            resp: Response = await call_next(request)
            code = resp.status_code
            return resp
        finally:
            dt = time.perf_counter() - t0
            path = request.url.path
            method = request.method
            observe_http(path, method, code)
            log.info(f"{method} {path} -> {code} ({dt*1000:.1f} ms)")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Routers
app.include_router(voice_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(ws_router)  # NOTE: also defines /ws/voice (legacy)

# -------- metrics endpoints --------
if METRICS_ENABLED:
    @app.get("/metrics")
    def metrics_prom():
        blob, ctype = prometheus_export()
        return Response(content=blob, media_type=ctype)

    @app.get("/api/metrics")
    def metrics_json():
        return json_snapshot()

    @app.get("/api/metrics/logs")
    def metrics_logs(limit: int = 200):
        return {"events": recent_events(limit)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
