import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.models import Chunk, Hit
from app.retrieval import rrf_fuse
from app.service import NO_ANSWER, RagService

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "docs"


def H(i):
    return Hit(chunk=Chunk(chunk_id=i, doc_id="d", text=i), score=1)


def test_rrf_dedup_and_order():
    out = rrf_fuse([[H("a"), H("b")], [H("b"), H("c")]])
    assert [h.chunk.chunk_id for h in out] == ["b", "a", "c"]


@pytest.fixture
def client():
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    svc = RagService(s)
    # Use FastAPI dependency_overrides instead of mutating globals
    main.app.dependency_overrides[main.get_rag_service] = lambda: svc
    with TestClient(main.app) as c:
        res = c.post("/ingest", json={"directory": str(FIXTURES_DIR)})
        assert res.status_code == 200
        assert res.json()["documents"] == 3
        yield c
    main.app.dependency_overrides.clear()


def top(c, q):
    res = c.post("/query", json={"question": q})
    assert res.status_code == 200
    return {x["doc_id"] for x in res.json()["sources"]}


def test_site_questions(client):
    assert "faq.md" in top(client, "How long does the admission process take?")
    assert "expos.md" in top(client, "Are virtual expos free to attend?")
    assert "advisors.md" in top(client, "Who is Priya?")
    r = client.post("/query", json={"question": "Who is Priya?"}).json()
    assert "Oxford" in r["answer"]


def test_filter_by_page(client):
    res = client.post("/retrieve", json={"question": "free", "filters": {"page": "expos"}}).json()["results"]
    assert res and all(h["chunk"]["metadata"]["page"] == "expos" for h in res)


def test_fail_closed_on_empty_store():
    import asyncio
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    assert asyncio.run(RagService(s).answer("anything"))["answer"] == NO_ANSWER
