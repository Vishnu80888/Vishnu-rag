from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Response

from .config import Settings, get_settings
from .models import IngestRequest, QueryRequest
from .service import RagService

_svc: RagService | None = None


def get_rag_service(settings: Settings = Depends(get_settings)) -> RagService:
    """Dependency provider for RagService.
    Supports clean testing via app.dependency_overrides[get_rag_service] = lambda: mock_svc.
    """
    global _svc
    if _svc is not None and _svc.s == settings:
        return _svc
    if _svc is None or _svc.s != settings:
        _svc = RagService(settings)
    return _svc


def svc() -> RagService:
    """Backward-compatible access to default RagService."""
    return get_rag_service(get_settings())


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = get_settings()
        service = get_rag_service(settings)
        sources = await service.store.sources()
        if not sources:
            project_root = Path(__file__).resolve().parent.parent
            docs_dir = Path(settings.docs_dir)
            if not docs_dir.is_absolute():
                docs_dir = project_root / docs_dir
            if docs_dir.is_dir():
                await service.ingest_dir(str(docs_dir))
    except Exception:
        pass
    yield


app = FastAPI(title="Vishnu RAG", lifespan=lifespan)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def root():
    return {"service": "vishnu-rag", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/query")
async def query(req: QueryRequest, service: RagService = Depends(get_rag_service)):
    return await service.answer(req.question, req.top_k, req.filters)


@app.post("/retrieve")
async def retrieve(req: QueryRequest, service: RagService = Depends(get_rag_service)):
    hits = await service.retrieve(req.question, req.top_k, req.filters)
    return {"results": [h.model_dump() for h in hits]}


@app.post("/ingest")
async def ingest(req: IngestRequest, service: RagService = Depends(get_rag_service)):
    try:
        return await service.ingest_dir(req.directory, req.metadata)
    except FileNotFoundError:
        raise HTTPException(404, "directory not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/sources")
async def sources(service: RagService = Depends(get_rag_service)):
    return {"sources": await service.store.sources()}


@app.get("/sources/{doc_id:path}")
async def source(doc_id: str, service: RagService = Depends(get_rag_service)):
    for s in await service.store.sources():
        if s["doc_id"] == doc_id:
            return s
    raise HTTPException(404, "source not found")
