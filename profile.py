"""
Perfil de la IA — nombre, personalidad, system prompt base.
Se guarda en un archivo JSON simple en ./data/profile.json
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

PROFILE_PATH = Path("./data/profile.json")

DEFAULT_PROFILE = {
    "ai_name": "Sentience",
    "model": "llama",
    "system_prompt": "Sos una IA personal que aprende y crece con el usuario.",
    "personality": "",
    "sessions": 0,
    "iq": 100.0,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}


# ─── Schemas ────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    ai_name: Optional[str] = Field(default=None, max_length=50)
    model: Optional[str] = None
    system_prompt: Optional[str] = Field(default=None, max_length=2000)
    personality: Optional[str] = Field(default=None, max_length=1000)


class SessionUpdate(BaseModel):
    iq_delta: float = Field(default=0.1, description="Cuánto IQ ganó en esta sesión")
    messages_count: int = Field(default=1)


# ─── Helpers ────────────────────────────────────────────────

def get_profile_data() -> dict:
    """Lee el perfil del archivo. Crea uno default si no existe."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PROFILE_PATH.exists():
        save_profile_data(DEFAULT_PROFILE)
        return DEFAULT_PROFILE.copy()
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profile_data(data: dict):
    """Guarda el perfil en disco."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Endpoints ──────────────────────────────────────────────

@router.get("")
async def get_profile():
    """Obtiene el perfil completo de la IA."""
    return get_profile_data()


@router.patch("")
async def update_profile(update: ProfileUpdate):
    """Actualiza campos del perfil."""
    profile = get_profile_data()

    if update.ai_name is not None:
        profile["ai_name"] = update.ai_name
    if update.model is not None:
        profile["model"] = update.model
    if update.system_prompt is not None:
        profile["system_prompt"] = update.system_prompt
    if update.personality is not None:
        profile["personality"] = update.personality

    save_profile_data(profile)
    logger.info(f"✏️  Perfil actualizado: {profile['ai_name']}")
    return profile


@router.post("/session")
async def register_session(update: SessionUpdate):
    """
    Registra una sesión completada.
    Incrementa el contador de sesiones y el IQ simbólico.
    """
    profile = get_profile_data()
    profile["sessions"] = profile.get("sessions", 0) + 1
    profile["iq"] = round(profile.get("iq", 100.0) + update.iq_delta, 2)
    save_profile_data(profile)
    return {
        "sessions": profile["sessions"],
        "iq": profile["iq"],
        "message": f"Sesión registrada. IQ actual: {profile['iq']}",
    }


@router.post("/reset")
async def reset_profile():
    """Resetea el perfil a valores default (no borra la memoria vectorial)."""
    save_profile_data(DEFAULT_PROFILE.copy())
    return {"message": "Perfil reseteado.", "profile": DEFAULT_PROFILE}
