from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str] | None = None


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    filename: str
    chunk_index: int
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
