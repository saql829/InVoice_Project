# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.voice_chat import router as voice_router
from api.chat import router as chat_router
from api.ws_voice import router as ws_router  # keep as-is (your older socket)
from config.settings import API_HOST, API_PORT, FRONTEND_ORIGINS

app = FastAPI(title="Voice Mode Assistant Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS or ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Routers
app.include_router(voice_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(ws_router)  # NOTE: also defines /ws/voice (legacy)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
