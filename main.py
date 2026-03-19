"""
Digital Sentience - Servidor de Memoria Neural
Corre en Render free tier (512MB RAM, 0.1 CPU)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.routers import memory, chat, profile, health
from app.routers.upload import upload_router
from app.core.database import init_db
from app.core.model import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup y shutdown del servidor."""
    logger.info("🧠 Inicializando Digital Sentience Server...")
    init_db()
    load_model()
    logger.info("✅ Servidor listo.")
    yield
    logger.info("💤 Servidor apagándose...")


app = FastAPI(
    title="Digital Sentience API",
    description="Servidor de memoria y modelo para IA personal local",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, restringir a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,  prefix="/health",  tags=["health"])
app.include_router(memory.router,  prefix="/memory",  tags=["memory"])
app.include_router(upload_router,  prefix="/memory",  tags=["memory"])
app.include_router(chat.router,    prefix="/chat",    tags=["chat"])
app.include_router(profile.router, prefix="/profile", tags=["profile"])


@app.get("/")
async def root():
    return {
        "name": "Digital Sentience Server",
        "version": "1.0.0",
        "status": "alive",
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
