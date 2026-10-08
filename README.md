# Chishiki-AI-Agent-RAG-pipeline
Generate a lean and production-ready application named GCK. Use Streamlit for the user interface and Python with FastAPI for the backend logic. Structure the project with a minimal, readable, and well-organized directory, avoiding complex patterns while ensuring it is scalable enough for deployment. Keep the code descriptive and easy to understand.

# Chishiki-AI Agent & RAG Pipeline

Chishiki-AI Agent & RAG Pipeline is a small, single-user RAG application. Upload a
PDF, Markdown, or text file, then ask questions answered from the indexed content.
The stack runs locally with Streamlit, FastAPI, Ollama, and Qdrant; no external AI
account is required.

## Run with Docker

Requirements:

- Docker Engine with Compose
- Enough disk and memory for `llama3.2:3b` and `nomic-embed-text`

Start the application:

```bash
cp .env.example .env
docker compose up --build
```

On the first run, the `ollama-models` service downloads both configured models.
This can take several minutes. Docker volumes preserve the models and indexed data.

- UI: http://localhost:8501
- API documentation: http://localhost:8001/docs
- Readiness: http://localhost:8001/health/ready

Stop services with `docker compose down`. Add `--volumes` only when you intentionally
want to delete downloaded models and indexed documents.

## Configuration

Environment values can be set in `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_CHAT_MODEL` | `llama3.2:3b` | Model used to answer questions |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model used for vector embeddings |
| `RAG_TOP_K` | `4` | Number of chunks supplied as context |
| `MAX_UPLOAD_MB` | `10` | Maximum document size |

Changing the embedding model after indexing may change vector dimensions. If that
happens, remove the `qdrant_data` volume and re-index documents.

## API

- `POST /documents` accepts one multipart file under the `file` field.
- `POST /chat` accepts `{"question": "..."}` and returns an answer with sources.
- `GET /health/live` reports process health.
- `GET /health/ready` checks Ollama and Qdrant.

Supported document types are `.pdf`, `.md`, and `.txt`. Chishiki stores extracted
text and vector metadata in Qdrant. It does not retain the original uploaded file.

## Local development

Use Python 3.12 and provide reachable Ollama and Qdrant instances:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
streamlit run frontend/app.py
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. Override `OLLAMA_URL`, `QDRANT_URL`, or
`BACKEND_URL` when the services are not using their defaults.

Run quality checks with:

```bash
ruff check .
ruff format --check .
pytest
docker compose config --quiet
```

## Deployment note

The first release intentionally has no authentication and is suitable for a trusted,
single-user network. Put the UI and API behind HTTPS and access control before exposing
them publicly. Qdrant and Ollama are only available on the internal Compose network.
