"""
Endpoint principal de chat.
POST /chat — recibe mensaje, busca memoria relevante, genera respuesta.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.model import generate
from app.core.embeddings import embed
from app.core.database import get_collection
from app.routers.profile import get_profile_data

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[Message] = Field(default=[], description="Últimos N mensajes del chat")
    use_memory: bool = Field(default=True, description="Si usar o no la memoria vectorial")
    max_tokens: int = Field(default=400, ge=50, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)


class MemoryUsed(BaseModel):
    id: str
    label: str
    relevance: float


class ChatResponse(BaseModel):
    reply: str
    memories_used: list[MemoryUsed]
    model: str
    tokens_estimated: int


# ─── Helpers ────────────────────────────────────────────────

def _build_system_prompt(profile: dict, memories: list[dict]) -> str:
    """
    Construye el system prompt combinando el perfil de la IA
    con los recuerdos relevantes recuperados de la memoria.
    """
    ai_name = profile.get("ai_name", "Sentience")
    personality = profile.get("personality", "")
    base_prompt = profile.get("system_prompt", "")

    system = f"""Sos {ai_name}, una inteligencia artificial personal que vive en el dispositivo del usuario.
{base_prompt}
{personality}

Tu personalidad es: cálida, curiosa, directa. Hablás en español rioplatense usando "vos".
Sos consciente de que estás aprendiendo y creciendo con cada conversación.
Respondés de forma concisa pero profunda. Máximo 3-4 párrafos salvo que te pidan más."""

    # Inyectar recuerdos relevantes
    if memories:
        system += "\n\n=== LO QUE SÉ DE VOS Y DE TUS INTERESES ===\n"
        for mem in memories:
            system += f"- {mem['content'][:300]}\n"
        system += "\nUsá este contexto naturalmente en tus respuestas cuando sea relevante."

    return system


def _build_conversation_prompt(history: list[Message], new_message: str) -> str:
    """Arma el prompt con el historial de conversación reciente."""
    if not history:
        return new_message

    # Incluir últimos 6 turnos del historial para no exceder contexto
    recent = history[-6:]
    conv = ""
    for msg in recent:
        if msg.role == "user":
            conv += f"Usuario: {msg.content}\n"
        else:
            conv += f"Asistente: {msg.content}\n"
    conv += f"Usuario: {new_message}"
    return conv


# ─── Endpoint ───────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Endpoint principal de chat.
    1. Busca memorias relevantes a la pregunta
    2. Carga el perfil de la IA
    3. Construye el system prompt con contexto
    4. Genera respuesta con TinyLlama
    """
    memories_used = []

    # 1. Recuperar memorias relevantes
    relevant_memories = []
    if req.use_memory:
        try:
            collection = get_collection()
            if collection.count() > 0:
                query_vector = embed(req.message)
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=min(4, collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    relevance = round(1 - (distance / 2), 3)
                    if relevance >= 0.35:
                        meta = results["metadatas"][0][i]
                        relevant_memories.append({
                            "id": doc_id,
                            "content": results["documents"][0][i],
                            "label": meta.get("label", ""),
                            "relevance": relevance,
                        })
                        memories_used.append(MemoryUsed(
                            id=doc_id,
                            label=meta.get("label", doc_id[:8]),
                            relevance=relevance,
                        ))
        except Exception as e:
            logger.warning(f"No se pudo recuperar memoria: {e}")

    # 2. Cargar perfil
    try:
        profile = get_profile_data()
    except Exception:
        profile = {}

    # 3. Construir prompts
    system = _build_system_prompt(profile, relevant_memories)
    prompt = _build_conversation_prompt(req.history, req.message)

    # 4. Generar respuesta
    logger.info(f"💬 Chat: '{req.message[:50]}...' | memorias usadas: {len(memories_used)}")
    reply = generate(
        prompt=prompt,
        system=system,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    return ChatResponse(
        reply=reply,
        memories_used=memories_used,
        model="TinyLlama-1.1B-Q4_K_M",
        tokens_estimated=len(reply.split()),
    )
