from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.voice_chat import router as voice_router
from api.ws_audio import router as ws_router
from api.ws_llm import router as llm_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(voice_router, prefix="/chat")
app.include_router(ws_router)
app.include_router(llm_router)

@app.get("/")
def root():
    return {"status": "Voice Assistant Backend Running"}

@app.get("/health")
def health():
    return {"ok": True}
