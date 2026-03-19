"""
Carga y gestión del modelo de lenguaje.
Usa TinyLlama 1.1B Q4 cuantizado (~650MB RAM).
El modelo se descarga automáticamente al primer arranque.
"""

import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Ruta donde se guarda el modelo (persiste en Render con disk)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./data/models"))
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_FILE

# Singleton del modelo
_model = None


def load_model():
    """
    Descarga el modelo si no existe y lo carga en memoria.
    TinyLlama Q4_K_M: ~650MB RAM, buena calidad/velocidad.
    """
    global _model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Descargar si no existe
    if not MODEL_PATH.exists():
        logger.info(f"📥 Descargando modelo {MODEL_FILE}...")
        logger.info("   Esto tarda ~2 min la primera vez, luego queda guardado.")
        try:
            hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                filename=MODEL_FILE,
                local_dir=str(MODEL_DIR),
            )
            logger.info("✅ Modelo descargado.")
        except Exception as e:
            logger.error(f"❌ Error descargando modelo: {e}")
            logger.warning("⚠️  El servidor arrancó sin modelo. Solo memoria disponible.")
            return

    # Cargar en RAM
    try:
        from llama_cpp import Llama
        logger.info("🔄 Cargando modelo en RAM...")
        _model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,        # Ventana de contexto
            n_threads=1,       # Render free tier: 1 CPU
            n_gpu_layers=0,    # Sin GPU en Render
            verbose=False,
        )
        logger.info("✅ Modelo cargado en RAM.")
    except ImportError:
        logger.error("❌ llama-cpp-python no instalado. Revisar requirements.txt")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")


def get_model():
    """Retorna el modelo cargado o None si no está disponible."""
    return _model


def generate(
    prompt: str,
    system: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop: list[str] | None = None,
) -> str:
    """
    Genera una respuesta dado un prompt y un system prompt.
    Formato ChatML compatible con TinyLlama.
    """
    model = get_model()
    if model is None:
        return "⚠️ Modelo no disponible. El servidor está iniciando o hubo un error de carga."

    # Formato ChatML que usa TinyLlama
    full_prompt = f"<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"

    try:
        result = model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "<|user|>", "<|system|>"],
            echo=False,
        )
        text = result["choices"][0]["text"].strip()
        return text
    except Exception as e:
        logger.error(f"Error en generación: {e}")
        return "Error generando respuesta."
