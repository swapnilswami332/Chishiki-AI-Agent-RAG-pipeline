from dataclasses import dataclass
from uuid import uuid4

import httpx
from qdrant_client import AsyncQdrantClient, models


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    chunk_index: int
    score: float


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embed_model: str,
        timeout: float,
    ) -> None:
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self) -> None:
        await self.http.aclose()

    async def is_ready(self) -> bool:
        response = await self.http.get("/api/tags")
        response.raise_for_status()
        return True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.http.post(
            "/api/embed",
            json={"model": self.embed_model, "input": texts},
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected number of embeddings.")
        return embeddings

    async def answer(self, question: str, context: str) -> str:
        response = await self.http.post(
            "/api/chat",
            json={
                "model": self.chat_model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied context. If the context does not "
                            "contain the answer, say that you do not know. Be concise and factual."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
            },
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer.")
        return answer


class QdrantStore:
    def __init__(self, url: str, collection: str) -> None:
        self.collection = collection
        self.client = AsyncQdrantClient(url=url)

    async def close(self) -> None:
        await self.client.close()

    async def is_ready(self) -> bool:
        await self.client.get_collections()
        return True

    async def _ensure_collection(self, vector_size: int) -> None:
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    async def add(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if not embeddings or not embeddings[0]:
            raise ValueError("At least one embedding is required.")
        await self._ensure_collection(len(embeddings[0]))
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        await self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )

    async def search(self, embedding: list[float], limit: int) -> list[RetrievedChunk]:
        if not await self.client.collection_exists(self.collection):
            return []
        result = await self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                text=str(point.payload.get("text", "")),
                filename=str(point.payload.get("filename", "Unknown")),
                chunk_index=int(point.payload.get("chunk_index", 0)),
                score=float(point.score),
            )
            for point in result.points
            if point.payload
        ]
