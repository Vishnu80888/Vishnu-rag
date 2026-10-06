"""Vector store contract with in-memory and Qdrant adapters."""
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Protocol

from .config import Settings
from .models import Chunk, Hit
from .providers import tokenize


def _match(chunk: Chunk, filters: dict) -> bool:
    return all(str(chunk.metadata.get(k)) == str(v) for k, v in filters.items())


def _cosine(a: list[float], b: list[float]) -> float:
    """True cosine similarity — safe against non-unit-norm vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (mag_a * mag_b)


class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def search_dense(self, vector: list[float], k: int, filters: dict) -> list[Hit]: ...
    async def search_sparse(self, query: str, k: int, filters: dict) -> list[Hit]: ...
    async def sources(self) -> list[dict]: ...


class MemoryStore:
    def __init__(self):
        self.chunks: dict[str, Chunk] = {}
        self.vecs: dict[str, list[float]] = {}

    async def upsert(self, chunks, vectors):
        for c, v in zip(chunks, vectors):
            self.chunks[c.chunk_id], self.vecs[c.chunk_id] = c, v

    async def search_dense(self, vector, k, filters):
        hits = [Hit(chunk=c, score=_cosine(vector, self.vecs[cid]))
                for cid, c in self.chunks.items() if _match(c, filters)]
        return sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:k]

    async def search_sparse(self, query, k, filters, k1=1.5, b=0.75):
        docs = [(c, tokenize(c.text)) for c in self.chunks.values() if _match(c, filters)]
        if not docs:
            return []
        n = len(docs)
        avg = sum(len(t) for _, t in docs) / n or 1.0
        df = Counter(tok for _, t in docs for tok in set(t))
        q = set(tokenize(query))
        hits = []
        for c, toks in docs:
            tf = Counter(toks)
            s = 0.0
            for term in q:
                if tf[term]:
                    idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(toks) / avg))
            if s > 0:
                hits.append(Hit(chunk=c, score=s))
        return sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:k]

    async def sources(self):
        agg: dict[str, int] = {}
        for c in self.chunks.values():
            agg[c.doc_id] = agg.get(c.doc_id, 0) + 1
        return [{"doc_id": d, "chunks": n} for d, n in sorted(agg.items())]


class QdrantStore:
    """Hybrid collection: named dense vector + FastEmbed BM25 sparse vector (IDF modifier)."""

    def __init__(self, s: Settings):
        from fastembed import SparseTextEmbedding
        from qdrant_client import AsyncQdrantClient, models

        self.m = models
        self.s = s
        if s.qdrant_path:
            p = Path(s.qdrant_path)
            if not p.is_absolute():
                p = (Path(__file__).resolve().parent.parent / p).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=str(p))
        else:
            self.client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
        self.bm25 = SparseTextEmbedding("Qdrant/bm25")
        self.ready = False

    async def _ensure(self):
        if self.ready:
            return
        m = self.m
        if not await self.client.collection_exists(self.s.collection_name):
            await self.client.create_collection(
                self.s.collection_name,
                vectors_config={"dense": m.VectorParams(size=self.s.embedding_dimension, distance=m.Distance.COSINE)},
                sparse_vectors_config={"sparse": m.SparseVectorParams(modifier=m.Modifier.IDF)})
        self.ready = True

    def _sparse(self, text: str):
        e = next(iter(self.bm25.embed([text])))
        return self.m.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    def _filter(self, filters):
        if not filters:
            return None
        m = self.m
        return m.Filter(must=[m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in filters.items()])

    async def upsert(self, chunks, vectors):
        await self._ensure()
        m = self.m
        pts = [m.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector={"dense": v, "sparse": self._sparse(c.text)},
            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, **c.metadata})
            for c, v in zip(chunks, vectors)]
        await self.client.upsert(self.s.collection_name, points=pts)

    def _hits(self, res):
        out = []
        for p in res.points:
            pl = dict(p.payload)
            cid, did, text = pl.pop("chunk_id"), pl.pop("doc_id"), pl.pop("text")
            out.append(Hit(chunk=Chunk(chunk_id=cid, doc_id=did, text=text, metadata=pl), score=p.score))
        return out

    async def search_dense(self, vector, k, filters):
        await self._ensure()
        return self._hits(await self.client.query_points(
            self.s.collection_name, query=vector, using="dense", limit=k,
            query_filter=self._filter(filters), with_payload=True))

    async def search_sparse(self, query, k, filters):
        await self._ensure()
        return self._hits(await self.client.query_points(
            self.s.collection_name, query=self._sparse(query), using="sparse", limit=k,
            query_filter=self._filter(filters), with_payload=True))

    async def sources(self):
        await self._ensure()
        agg: dict[str, int] = {}
        offset = None
        while True:
            pts, offset = await self.client.scroll(self.s.collection_name, limit=256, offset=offset,
                                                   with_payload=["doc_id"])
            for p in pts:
                if p.payload and "doc_id" in p.payload:
                    doc = str(p.payload["doc_id"])
                    agg[doc] = agg.get(doc, 0) + 1
            if offset is None:
                break
        return [{"doc_id": d, "chunks": n} for d, n in sorted(agg.items())]


def build_store(s: Settings) -> VectorStore:
    return QdrantStore(s) if s.vector_store_provider == "qdrant" else MemoryStore()
