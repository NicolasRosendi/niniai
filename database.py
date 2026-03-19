"""
Base de datos vectorial con ChromaDB.
Guarda los recuerdos como vectores para búsqueda semántica.
ChromaDB usa SQLite por debajo — liviano, perfecto para Render.
"""

import os
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Directorio donde ChromaDB guarda sus datos
DB_DIR = Path(os.getenv("CHROMA_DIR", "./data/chroma"))

# Singleton del cliente
_client = None
_collection = None

COLLECTION_NAME = "sentience_memory"


def init_db():
    """Inicializa ChromaDB y crea la colección si no existe."""
    global _client, _collection

    DB_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _client = chromadb.PersistentClient(
            path=str(DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Similitud coseno
        )
        count = _collection.count()
        logger.info(f"✅ ChromaDB listo. Recuerdos almacenados: {count}")
    except Exception as e:
        logger.error(f"❌ Error iniciando ChromaDB: {e}")
        raise


def get_collection():
    """Retorna la colección activa."""
    if _collection is None:
        raise RuntimeError("Base de datos no inicializada. Llamá init_db() primero.")
    return _collection


def get_client():
    return _client
