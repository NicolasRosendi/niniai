"""
Health check endpoints.
GET /health       — estado general del servidor
GET /health/ping  — ping simple para evitar cold start
GET /health/model — estado del modelo LLM
"""

import psutil
import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from app.core.model import get_model
from app.core.database import get_collection

logger = logging.getLogger(__name__)
router = APIRouter()

_start_time = datetime.now(timezone.utc)


@router.get("/ping")
async def ping():
    """
    Endpoint ultrasimple para mantener el servidor despierto.
    La app lo llama al abrirse para evitar el cold start de 30s.
    """
    return {"pong": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("")
async def health():
    """Estado completo del servidor."""
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()

    # RAM
    mem = psutil.virtual_memory()
    ram_used_mb = round((mem.total - mem.available) / 1024 / 1024, 1)
    ram_total_mb = round(mem.total / 1024 / 1024, 1)

    # Modelo
    model_loaded = get_model() is not None

    # Memoria vectorial
    try:
        collection = get_collection()
        memories_count = collection.count()
        db_ok = True
    except Exception:
        memories_count = 0
        db_ok = False

    return {
        "status": "healthy" if model_loaded and db_ok else "degraded",
        "uptime_seconds": round(uptime, 1),
        "model": {
            "loaded": model_loaded,
            "name": "TinyLlama-1.1B-Q4_K_M" if model_loaded else None,
        },
        "memory_db": {
            "ok": db_ok,
            "total_memories": memories_count,
        },
        "system": {
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "ram_percent": round(mem.percent, 1),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/model")
async def model_status():
    """Estado específico del modelo LLM."""
    model = get_model()
    if model is None:
        return {
            "loaded": False,
            "message": "Modelo no disponible. Puede estar descargando o hubo un error.",
        }
    return {
        "loaded": True,
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "quantization": "Q4_K_M",
        "context_window": 2048,
        "threads": 1,
    }
