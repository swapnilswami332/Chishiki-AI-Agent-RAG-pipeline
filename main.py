import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from backend.app.clients import OllamaClient, QdrantStore
from backend.app.config import get_settings
from backend.app.documents import DocumentError, validate_upload
from backend.app.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from backend.app.service import RagService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    ollama = OllamaClient(
        settings.ollama_url,
        settings.ollama_chat_model,
        settings.ollama_embed_model,
        settings.request_timeout_seconds,
    )
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    app.state.rag = RagService(
        ollama,
        store,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.rag_top_k,
    )
    app.state.settings = settings
    try:
        yield
    finally:
        await ollama.close()
        await store.close()


app = FastAPI(
    title="Chishiki-AI Agent & RAG Pipeline API",
    version="0.1.0",
    description="Document ingestion and retrieval-augmented question answering.",
    lifespan=lifespan,
)


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def ready(request: Request, response: Response) -> HealthResponse:
    services: dict[str, str] = {}
    checks = {
        "ollama": request.app.state.rag.ollama.is_ready,
        "qdrant": request.app.state.rag.store.is_ready,
    }
    for name, check in checks.items():
        try:
            await check()
            services[name] = "ok"
        except Exception:  # Readiness must report a dependency outage, not crash.
            logger.warning("%s readiness check failed", name, exc_info=True)
            services[name] = "unavailable"

    is_ready = all(value == "ok" for value in services.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if is_ready else "unavailable", services=services)


@app.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def ingest_document(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> IngestResponse:
    settings = request.app.state.settings
    content = await file.read(settings.max_upload_bytes + 1)
    try:
        filename = validate_upload(file.filename or "", content, settings.max_upload_bytes)
        return await request.app.state.rag.ingest(filename, content)
    except DocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (httpx.HTTPError, UnexpectedResponse, ResponseHandlingException, RuntimeError) as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="A model or vector-store service is unavailable.",
        ) from exc
    finally:
        await file.close()


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be blank.",
        )
    try:
        return await request.app.state.rag.chat(question)
    except (httpx.HTTPError, UnexpectedResponse, ResponseHandlingException, RuntimeError) as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="A model or vector-store service is unavailable.",
        ) from exc
