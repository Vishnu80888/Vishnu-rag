from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.models import Chunk, Hit
from app.retrieval import rrf_fuse
from app.service import NO_ANSWER, RagService


def H(i):
    return Hit(chunk=Chunk(chunk_id=i, doc_id="d", text=i), score=1)


def test_rrf_dedup_and_order():
    out = rrf_fuse([[H("a"), H("b")], [H("b"), H("c")]])
    assert [h.chunk.chunk_id for h in out] == ["b", "a", "c"]


def client():
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    main._svc = RagService(s)
    c = TestClient(main.app)
    assert c.post("/ingest", json={"directory": "data/docs"}).json()["documents"] >= 7
    return c


def top(c, q):  # doc ids among the retrieved sources
    return {x["doc_id"] for x in c.post("/query", json={"question": q}).json()["sources"]}


def test_site_questions():
    c = client()
    assert "faq.md" in top(c, "How long does the admission process take?")
    assert "expos.md" in top(c, "Are virtual expos free to attend?")
    assert "advisors.md" in top(c, "How many verified advisors are there?")
    r = c.post("/query", json={"question": "Who is Priya?"}).json()
    assert "Oxford" in r["answer"] and r["sources"][0]["url"].startswith("https://www.codejobz.com")


def test_filter_by_page():
    c = client()
    res = c.post("/retrieve", json={"question": "free", "filters": {"page": "expos"}}).json()["results"]
    assert res and all(h["chunk"]["metadata"]["page"] == "expos" for h in res)


def test_fail_closed_on_empty_store():
    import asyncio
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    assert asyncio.run(RagService(s).answer("anything"))["answer"] == NO_ANSWER
