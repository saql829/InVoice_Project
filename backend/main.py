# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import Routers
from api.voice_chat import router as voice_router
from api.chat import router as chat_router
from api.ws_voice import router as ws_router

# Config & DB
from config.settings import API_HOST, API_PORT, FRONTEND_ORIGINS
from db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    print("🔹 Initializing database...")
    init_db()
    print(" Database initialized and ready.")
    yield
    # ---- Shutdown ----
    print(" Shutting down application.")


# ---- FastAPI App ----
app = FastAPI(
    title="Voice Mode Assistant Backend",
    version="0.1.0",
    lifespan=lifespan
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS or ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Health Check ----
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---- Routers ----
app.include_router(voice_router, prefix="/api")   # Voice Chat + DB Endpoints
app.include_router(chat_router, prefix="/api")    # Text Chat
app.include_router(ws_router)                     # WebSocket


# ---- Run Server ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
