from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException

from .config import get_settings
from .models import IngestRequest, QueryRequest
from .service import RagService

_svc: RagService | None = None


def svc() -> RagService:
    global _svc
    if _svc is None:
        _svc = RagService(get_settings())
    return _svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = svc()
        sources = await service.store.sources()
        if not sources:
            default_docs = Path(__file__).resolve().parent.parent / "data" / "docs"
            if default_docs.is_dir():
                await service.ingest_dir(str(default_docs))
    except Exception:
        pass
    yield


app = FastAPI(title="Vishnu RAG", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "vishnu-rag", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/query")
async def query(req: QueryRequest):
    return await svc().answer(req.question, req.top_k, req.filters)


@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    hits = await svc().retrieve(req.question, req.top_k, req.filters)
    return {"results": [h.model_dump() for h in hits]}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        return await svc().ingest_dir(req.directory, req.metadata)
    except FileNotFoundError:
        raise HTTPException(404, "directory not found")


@app.get("/sources")
async def sources():
    return {"sources": await svc().store.sources()}


@app.get("/sources/{doc_id:path}")
async def source(doc_id: str):
    for s in await svc().store.sources():
        if s["doc_id"] == doc_id:
            return s
    raise HTTPException(404, "source not found")
