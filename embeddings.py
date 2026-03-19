"""
Motor de embeddings liviano.
Usa sentence-transformers/all-MiniLM-L6-v2:
  - Tamaño: ~90MB
  - Velocidad: muy rápido en CPU
  - Calidad: excelente para búsqueda semántica en español/inglés
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_embedder = None


def get_embedder():
    """Carga el modelo de embeddings (singleton)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("📐 Cargando modelo de embeddings...")
            _embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./data/embeddings_cache",
            )
            logger.info("✅ Embedder listo.")
        except Exception as e:
            logger.error(f"❌ Error cargando embedder: {e}")
            raise
    return _embedder


def embed(text: str) -> list[float]:
    """Convierte un texto en un vector de embeddings."""
    embedder = get_embedder()
    vector = embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convierte múltiples textos en vectores (más eficiente que uno por uno)."""
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, batch_size=8)
    return vectors.tolist()
