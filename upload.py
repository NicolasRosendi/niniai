"""
Endpoint para subir archivos (PDF, TXT, MD) y procesarlos como memoria.
POST /memory/upload — sube un archivo, lo parsea y lo guarda fragmentado
"""

import logging
import io
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.core.database import get_collection
from app.core.embeddings import embed_batch
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Router separado para uploads (se registra en main.py como /memory/upload)
upload_router = APIRouter()


class UploadResponse(BaseModel):
    filename: str
    chunks_saved: int
    total_chars: int
    iq_gain: float
    message: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Divide el texto en chunks solapados para mejor cobertura semántica.
    chunk_size: caracteres por chunk
    overlap: cuántos caracteres se repiten entre chunks consecutivos
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def extract_text_from_pdf(content: bytes) -> str:
    """Extrae texto de un PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parseando PDF: {e}")


def extract_text_from_docx(content: bytes) -> str:
    """Extrae texto de un .docx"""
    try:
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parseando DOCX: {e}")


@upload_router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    label: Optional[str] = Form(default=None),
    tags: Optional[str] = Form(default=""),
):
    """
    Sube un archivo y lo guarda fragmentado en la memoria vectorial.
    Soporta: .txt, .md, .pdf, .docx
    """
    # Validar tipo de archivo
    allowed = [".txt", ".md", ".pdf", ".docx"]
    filename = file.filename or "archivo"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {ext}. Soportados: {allowed}",
        )

    # Leer contenido
    content_bytes = await file.read()

    # Extraer texto según tipo
    if ext in [".txt", ".md"]:
        text = content_bytes.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        text = extract_text_from_pdf(content_bytes)
    elif ext == ".docx":
        text = extract_text_from_docx(content_bytes)
    else:
        text = content_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="El archivo no contiene texto extraíble.")

    # Fragmentar el texto
    chunks = chunk_text(text, chunk_size=400, overlap=40)
    if not chunks:
        raise HTTPException(status_code=400, detail="No se pudieron crear fragmentos del texto.")

    # Crear embeddings en batch
    vectors = embed_batch(chunks)

    # Guardar en ChromaDB
    collection = get_collection()
    now = datetime.now(timezone.utc).isoformat()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    file_label = label or filename

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "source": "file",
            "label": f"{file_label} (fragmento {i+1}/{len(chunks)})",
            "tags": ",".join(tag_list + ["file", ext.lstrip(".")]),
            "created_at": now,
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )

    iq_gain = round(len(chunks) * 0.3 + 0.5, 2)
    logger.info(f"📄 Archivo procesado: {filename} → {len(chunks)} fragmentos")

    return UploadResponse(
        filename=filename,
        chunks_saved=len(chunks),
        total_chars=len(text),
        iq_gain=iq_gain,
        message=f"'{filename}' procesado y guardado en {len(chunks)} fragmentos de memoria.",
    )
