"""
Endpoints de memoria vectorial.

POST /memory/add     — guarda un nuevo recuerdo
POST /memory/search  — busca recuerdos similares a una query
GET  /memory/list    — lista todos los recuerdos
DELETE /memory/{id}  — elimina un recuerdo
GET  /memory/stats   — estadísticas de la memoria
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.database import get_collection
from app.core.embeddings import embed, embed_batch

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Schemas ────────────────────────────────────────────────

class MemoryAddRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000, description="Texto a memorizar")
    source: str = Field(default="manual", description="Origen: 'chat', 'training', 'file', 'manual'")
    label: Optional[str] = Field(default=None, description="Etiqueta opcional para mostrar en UI")
    tags: list[str] = Field(default=[], description="Tags para filtrar")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Texto de búsqueda")
    n_results: int = Field(default=4, ge=1, le=10, description="Cuántos recuerdos traer")
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0, description="Score mínimo de relevancia")


class MemoryItem(BaseModel):
    id: str
    content: str
    source: str
    label: Optional[str]
    tags: list[str]
    created_at: str
    relevance: Optional[float] = None


class MemoryAddResponse(BaseModel):
    id: str
    message: str
    iq_gain: float  # IQ simbólico que gana la IA


class MemorySearchResponse(BaseModel):
    results: list[MemoryItem]
    total_searched: int
    query: str


# ─── Endpoints ──────────────────────────────────────────────

@router.post("/add", response_model=MemoryAddResponse)
async def add_memory(req: MemoryAddRequest):
    """Agrega un nuevo recuerdo a la memoria vectorial."""
    try:
        collection = get_collection()
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Crear embedding del contenido
        vector = embed(req.content)

        # Metadata que se guarda junto al vector
        metadata = {
            "source": req.source,
            "label": req.label or req.content[:60],
            "tags": ",".join(req.tags),
            "created_at": now,
        }

        collection.add(
            ids=[memory_id],
            embeddings=[vector],
            documents=[req.content],
            metadatas=[metadata],
        )

        # IQ gain simbólico basado en longitud del contenido
        iq_gain = round(min(len(req.content) / 500 * 0.5 + 0.2, 3.0), 2)

        logger.info(f"💾 Memoria guardada: {memory_id[:8]}... ({len(req.content)} chars)")

        return MemoryAddResponse(
            id=memory_id,
            message=f"Recuerdo almacenado correctamente.",
            iq_gain=iq_gain,
        )

    except Exception as e:
        logger.error(f"Error guardando memoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(req: MemorySearchRequest):
    """Busca recuerdos similares semánticamente a la query."""
    try:
        collection = get_collection()
        total = collection.count()

        if total == 0:
            return MemorySearchResponse(results=[], total_searched=0, query=req.query)

        query_vector = embed(req.query)
        n = min(req.n_results, total)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Distancia coseno → similitud (0=idéntico, 2=opuesto)
            relevance = round(1 - (distance / 2), 3)

            if relevance < req.min_relevance:
                continue

            meta = results["metadatas"][0][i]
            memories.append(MemoryItem(
                id=doc_id,
                content=results["documents"][0][i],
                source=meta.get("source", "unknown"),
                label=meta.get("label"),
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                created_at=meta.get("created_at", ""),
                relevance=relevance,
            ))

        logger.info(f"🔍 Búsqueda: '{req.query[:40]}...' → {len(memories)} resultados")
        return MemorySearchResponse(results=memories, total_searched=total, query=req.query)

    except Exception as e:
        logger.error(f"Error buscando memoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=list[MemoryItem])
async def list_memories(limit: int = 20, offset: int = 0):
    """Lista los recuerdos más recientes."""
    try:
        collection = get_collection()
        total = collection.count()

        if total == 0:
            return []

        # ChromaDB no tiene paginación nativa, traemos todo y paginamos en Python
        results = collection.get(
            include=["documents", "metadatas"],
            limit=min(limit, 100),
            offset=offset,
        )

        memories = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            memories.append(MemoryItem(
                id=doc_id,
                content=results["documents"][i],
                source=meta.get("source", "unknown"),
                label=meta.get("label"),
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                created_at=meta.get("created_at", ""),
            ))

        # Ordenar por fecha descendente
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Elimina un recuerdo específico."""
    try:
        collection = get_collection()
        collection.delete(ids=[memory_id])
        return {"message": f"Recuerdo {memory_id} eliminado."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def memory_stats():
    """Estadísticas generales de la memoria."""
    try:
        collection = get_collection()
        total = collection.count()

        # Contar por fuente
        if total > 0:
            all_items = collection.get(include=["metadatas"])
            sources: dict[str, int] = {}
            for meta in all_items["metadatas"]:
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
        else:
            sources = {}

        return {
            "total_memories": total,
            "by_source": sources,
            "iq_estimate": round(100 + total * 0.8, 1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
