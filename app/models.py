from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Hit(BaseModel):
    chunk: Chunk
    score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    directory: str = "data/docs"
    metadata: dict[str, str] = Field(default_factory=dict)
