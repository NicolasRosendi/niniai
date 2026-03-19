# 🧠 Digital Sentience — Servidor de Memoria Neural

Servidor backend para la app Digital Sentience. Corre **TinyLlama 1.1B** cuantizado
con memoria vectorial **ChromaDB**, diseñado para vivir en el free tier de Render (512MB RAM).

---

## Stack

| Componente | Tecnología |
|---|---|
| Framework | FastAPI + Uvicorn |
| Modelo LLM | TinyLlama 1.1B Q4_K_M (via llama-cpp-python) |
| Memoria vectorial | ChromaDB (SQLite backend) |
| Embeddings | all-MiniLM-L6-v2 (~90MB) |
| Deploy | Render free tier |

---

## Endpoints

```
GET  /                    → info del servidor
GET  /health              → estado completo (RAM, modelo, DB)
GET  /health/ping         → ping para evitar cold start

POST /chat                → mensaje → respuesta con contexto de memoria
POST /memory/add          → guardar nuevo conocimiento
POST /memory/search       → buscar recuerdos similares
GET  /memory/list         → listar recuerdos
GET  /memory/stats        → estadísticas (total, IQ, por fuente)
DELETE /memory/{id}       → eliminar recuerdo

GET  /profile             → perfil de la IA
PATCH /profile            → actualizar nombre, personalidad, system prompt
POST /profile/session     → registrar sesión completada (+IQ)
```

Documentación interactiva: `https://tu-servidor.onrender.com/docs`

---

## Deploy en Render

### 1. Subir a GitHub

```bash
git init
git add .
git commit -m "feat: initial Digital Sentience server"
git remote add origin https://github.com/TU_USUARIO/sentience-server.git
git push -u origin main
```

### 2. Crear servicio en Render

1. Ir a [render.com](https://render.com) → New → Web Service
2. Conectar tu repo de GitHub
3. Render detecta `render.yaml` automáticamente
4. Click **Deploy**

> ⚠️ El primer deploy tarda ~10 minutos porque compila `llama-cpp-python` desde source.

### 3. Agregar disco persistente (importante)

Sin disco, el modelo (~650MB) se descarga en CADA restart. Con disco:

1. En Render → tu servicio → **Disks** → Add Disk
2. Name: `sentience-data`
3. Mount path: `/data`
4. Size: 1 GB (~$1/mes)

---

## Correr localmente

```bash
# Clonar y crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Correr servidor
uvicorn app.main:app --reload --port 8000

# El modelo se descarga automáticamente al primer arranque (~650MB)
# Documentación: http://localhost:8000/docs
```

---

## Uso desde la app (React Native)

```javascript
const API_URL = 'https://tu-servidor.onrender.com';

// Chat con memoria
const response = await fetch(`${API_URL}/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: '¿Qué sabés de mí?',
    history: conversationHistory,
    use_memory: true,
  }),
});
const { reply, memories_used } = await response.json();

// Guardar conocimiento
await fetch(`${API_URL}/memory/add`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    content: 'Al usuario le gusta la arquitectura de software.',
    source: 'training',
    label: 'Intereses del usuario',
  }),
});
```

---

## Limitaciones del free tier de Render

| Limitación | Valor |
|---|---|
| RAM | 512MB |
| CPU | 0.1 (compartido) |
| Cold start | ~30s si estuvo inactivo 15min |
| Tiempo de respuesta | 3-8 segundos por mensaje |
| Uptime | 750hs/mes gratis |

**Solución al cold start:** la app hace un `GET /health/ping` al abrirse
para despertar el servidor antes de que el usuario escriba.

---

## Arquitectura de la memoria

```
Tu texto/PDF/audio
       ↓
  Embedding (MiniLM)
       ↓
  Vector 384D
       ↓
  ChromaDB (disco)
       ↓
Al chatear:
  Tu pregunta → vector → buscar top-4 similares → inyectar en prompt → TinyLlama responde
```

---

## Licencia

MIT — Proyecto personal. Hacé lo que quieras con esto.
