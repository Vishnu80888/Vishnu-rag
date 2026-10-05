"""Comprehensive multi-category test suite for Vishnu RAG.
Covers:
1. Unit Tests (Chunking, Hashing, Tokenizer, RRF Fusion)
2. Store & Retrieval Tests (Dense, Sparse/BM25, Hybrid RRF, Filtering)
3. API Endpoint Tests (GET /, GET /health, POST /ingest, GET /sources, POST /retrieve, POST /query)
4. Validation & Error Handling Tests (422 field mismatch, min_length, top_k bounds, 404 sources)
5. Safety / Fallback Tests (Empty store, ungrounded queries)
"""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
INNER_ROOT = ROOT_DIR / "vishnu-rag"
for p in (ROOT_DIR, INNER_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

DOCS_DIR = (INNER_ROOT / "data" / "docs") if (INNER_ROOT / "data" / "docs").exists() else (ROOT_DIR / "data" / "docs")

import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.models import Chunk, Hit
from app.chunking import chunk_text
from app.providers import tokenize, HashEmbedder, EchoLLM
from app.retrieval import rrf_fuse
from app.service import RagService, NO_ANSWER
from app import main

# ==========================================
# 1. UNIT TESTS: Chunking & Tokenizing
# ==========================================

def test_tokenizer_stopwords():
    tokens = tokenize("What is the process for an international student in 2026?")
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "for" not in tokens
    assert "an" not in tokens
    assert "in" not in tokens
    assert "process" in tokens
    assert "international" in tokens
    assert "student" in tokens
    assert "2026" in tokens

def test_chunking_sliding_window_and_overlap():
    text = ("Paragraph one with some detailed content.\n\n"
            "Paragraph two with even more detailed content.\n\n"
            "Paragraph three with concluding thoughts.")
    chunks = chunk_text("doc1", text, size=50, overlap=10)
    assert len(chunks) >= 3
    # Check deterministic hash IDs
    assert all(len(c.chunk_id) == 24 for c in chunks)
    assert all(c.doc_id == "doc1" for c in chunks)
    # Check metadata indexing
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

def test_chunking_empty_and_whitespace():
    assert chunk_text("empty", "", size=100, overlap=20) == []
    assert chunk_text("blank", "    \n\n   ", size=100, overlap=20) == []

def test_hash_embedder_normalization():
    import math, asyncio
    embedder = HashEmbedder(dim=128)
    vecs = asyncio.run(embedder.embed(["Sample document text for embedding"]))
    assert len(vecs) == 1
    assert len(vecs[0]) == 128
    # Norm should be approximately 1.0
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) < 1e-4

# ==========================================
# 2. UNIT TESTS: RRF Fusion
# ==========================================

def _h(cid, score=1.0):
    return Hit(chunk=Chunk(chunk_id=cid, doc_id="doc", text=cid), score=score)

def test_rrf_scoring_and_deduplication():
    list_dense = [_h("A"), _h("B"), _h("C")]
    list_sparse = [_h("B"), _h("D"), _h("A")]
    fused = rrf_fuse([list_dense, list_sparse], k=60)
    
    # B was rank 2 in dense, rank 1 in sparse -> highest total
    # Score formula: 1/(k+rank)
    score_b = 1.0/(60+2) + 1.0/(60+1)
    score_a = 1.0/(60+1) + 1.0/(60+3)
    
    assert fused[0].chunk.chunk_id == "B"
    assert fused[1].chunk.chunk_id == "A"
    assert abs(fused[0].score - score_b) < 1e-6
    assert abs(fused[1].score - score_a) < 1e-6

def test_rrf_empty_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []

# ==========================================
# 3. INTEGRATION TESTS: RAG Service & Retrieval
# ==========================================

@pytest.fixture(scope="module")
def rag_service():
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    svc = RagService(s)
    import asyncio
    docs_dir = str(DOCS_DIR)
    res = asyncio.run(svc.ingest_dir(docs_dir))
    assert res["documents"] >= 7
    return svc

def test_retrieval_ranking(rag_service):
    import asyncio
    hits = asyncio.run(rag_service.retrieve("How long does the admission process take?", top_k=3))
    assert len(hits) > 0
    top_docs = [h.chunk.doc_id for h in hits]
    assert any("faq.md" in d for d in top_docs)

def test_retrieval_metadata_filter(rag_service):
    import asyncio
    hits = asyncio.run(rag_service.retrieve("universities", top_k=5, filters={"page": "universities"}))
    assert len(hits) > 0
    assert all(h.chunk.metadata.get("page") == "universities" for h in hits)

def test_fail_closed_empty_service():
    import asyncio
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    empty_svc = RagService(s)
    ans = asyncio.run(empty_svc.answer("Can you tell me about scholarships?"))
    assert ans["answer"] == NO_ANSWER
    assert ans["grounded"] is False
    assert ans["sources"] == []

# ==========================================
# 4. API CONTRACT & VALIDATION TESTS
# ==========================================

@pytest.fixture(scope="module")
def api_client():
    s = Settings(llm_provider="local", embedding_provider="local", vector_store_provider="memory")
    main._svc = RagService(s)
    c = TestClient(main.app)
    docs_dir = str(DOCS_DIR)
    res = c.post("/ingest", json={"directory": docs_dir})
    assert res.status_code == 200
    return c

def test_api_root_and_health(api_client):
    r_root = api_client.get("/")
    assert r_root.status_code == 200
    assert r_root.json() == {"service": "vishnu-rag", "status": "ok"}

    r_health = api_client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "healthy"}

def test_api_sources_endpoints(api_client):
    r_sources = api_client.get("/sources")
    assert r_sources.status_code == 200
    sources = r_sources.json()["sources"]
    assert len(sources) >= 7

    # Check valid source detail
    sample_doc = sources[0]["doc_id"]
    r_detail = api_client.get(f"/sources/{sample_doc}")
    assert r_detail.status_code == 200
    assert r_detail.json()["doc_id"] == sample_doc

    # Check non-existent source 404
    r_missing = api_client.get("/sources/non_existent_doc_12345.md")
    assert r_missing.status_code == 404

def test_api_retrieve_endpoint(api_client):
    res = api_client.post("/retrieve", json={"question": "visa interview assistance", "top_k": 3})
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) <= 3
    assert "chunk" in results[0]
    assert "score" in results[0]

def test_api_query_success(api_client):
    res = api_client.post("/query", json={"question": "What is ProfileSity?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert data["grounded"] is True
    assert len(data["sources"]) > 0

def test_api_validation_errors(api_client):
    # Test 1: Wrong field name ("query" instead of "question") -> 422
    r1 = api_client.post("/query", json={"query": "What courses are available?"})
    assert r1.status_code == 422
    assert "Field required" in str(r1.json()) or "missing" in str(r1.json())

    # Test 2: Empty question (min_length=1 constraint) -> 422
    r2 = api_client.post("/query", json={"question": ""})
    assert r2.status_code == 422

    # Test 3: Invalid top_k (> 50 constraint) -> 422
    r3 = api_client.post("/query", json={"question": "hello", "top_k": 999})
    assert r3.status_code == 422
