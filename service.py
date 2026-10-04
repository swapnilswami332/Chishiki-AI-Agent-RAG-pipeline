from uuid import uuid4

from anyio import to_thread

from backend.app.clients import OllamaClient, QdrantStore, RetrievedChunk
from backend.app.documents import chunk_text, extract_text
from backend.app.schemas import ChatResponse, IngestResponse, Source


def build_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    sources: list[Source] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.filename, chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        excerpt = chunk.text if len(chunk.text) <= 220 else f"{chunk.text[:217].rstrip()}..."
        sources.append(
            Source(
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                excerpt=excerpt,
                score=round(chunk.score, 4),
            )
        )
    return sources


class RagService:
    def __init__(
        self,
        ollama: OllamaClient,
        store: QdrantStore,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> None:
        self.ollama = ollama
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    async def _embed_batches(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(await self.ollama.embed(texts[start : start + batch_size]))
        return embeddings

    async def ingest(self, filename: str, content: bytes) -> IngestResponse:
        text = await to_thread.run_sync(extract_text, filename, content)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        embeddings = await self._embed_batches(chunks)
        document_id = str(uuid4())
        await self.store.add(document_id, filename, chunks, embeddings)
        return IngestResponse(
            document_id=document_id,
            filename=filename,
            chunks_indexed=len(chunks),
        )

    async def chat(self, question: str) -> ChatResponse:
        question_embedding = (await self.ollama.embed([question]))[0]
        chunks = await self.store.search(question_embedding, self.top_k)
        if not chunks:
            return ChatResponse(
                answer="I do not know yet. Upload a relevant document and try again.",
                sources=[],
            )

        context = "\n\n".join(
            f"[{chunk.filename}, chunk {chunk.chunk_index}]\n{chunk.text}" for chunk in chunks
        )
        answer = await self.ollama.answer(question, context)
        return ChatResponse(answer=answer, sources=build_sources(chunks))
