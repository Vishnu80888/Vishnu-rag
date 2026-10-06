import re
from pathlib import Path

from .chunking import chunk_text
from .config import Settings
from .models import Hit
from .providers import build_embedder, build_llm
from .retrieval import rrf_fuse
from .stores import build_store

SYSTEM = ("You are Vishnu RAG, the knowledge assistant for the ProfileSity study-abroad platform. You answer strictly from the provided CONTEXT. Cite sources as [doc_id#chunk_index]. "
          "If the context does not contain the answer, say you don't have enough information. Never invent facts.")
NO_ANSWER = "I don't have enough grounded information in the knowledge base to answer that."


class RagService:
    def __init__(self, s: Settings):
        self.s, self.embedder, self.llm, self.store = s, build_embedder(s), build_llm(s), build_store(s)

    async def ingest_dir(self, directory: str | None = None, metadata: dict | None = None) -> dict:
        project_root = Path(__file__).resolve().parent.parent.resolve()
        configured_docs = Path(getattr(self.s, "docs_dir", "data/docs"))
        if not configured_docs.is_absolute():
            configured_docs = (project_root / configured_docs).resolve()

        if directory is None or (isinstance(directory, str) and directory.strip() == ""):
            root = configured_docs
        else:
            cand = Path(directory)
            if not cand.is_absolute():
                cand = (project_root / cand).resolve()
            else:
                cand = cand.resolve()

            # Security path validation: must be within project root
            try:
                cand.relative_to(project_root)
            except ValueError:
                if not getattr(self.s, "allow_arbitrary_ingest_path", False):
                    raise ValueError(f"Access denied: directory '{directory}' is outside the allowed project directories.")

            if not cand.is_dir():
                raise FileNotFoundError(f"Directory not found: {directory}")
            root = cand
        docs = chunks_total = 0
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in {".txt", ".md"}:
                continue
            text = p.read_text(encoding="utf-8")
            meta = dict(metadata or {})
            m = re.search(r"<!--\s*source:\s*(\S+)\s*\|\s*page:\s*(\S+)\s*-->", text)
            if m:
                meta.update(source_url=m.group(1), page=m.group(2))
                text = (text[:m.start()] + text[m.end():]).strip()
            chunks = chunk_text(p.relative_to(root).as_posix(), text,
                                self.s.chunk_size, self.s.chunk_overlap, meta)
            if chunks:
                vecs = await self.embedder.embed([c.text for c in chunks])
                await self.store.upsert(chunks, vecs)
                docs += 1
                chunks_total += len(chunks)
        return {"documents": docs, "chunks": chunks_total}

    async def retrieve(self, question: str, top_k: int | None = None, filters: dict | None = None) -> list[Hit]:
        filters = filters or {}
        n = self.s.max_candidates
        qv = (await self.embedder.embed([question]))[0]
        dense = await self.store.search_dense(qv, n, filters)
        sparse = await self.store.search_sparse(question, n, filters)
        return rrf_fuse([dense, sparse])[: top_k or self.s.top_k]

    async def answer(self, question: str, top_k=None, filters=None) -> dict:
        hits = await self.retrieve(question, top_k, filters)
        if not hits:  # fail closed
            return {"answer": NO_ANSWER, "grounded": False, "sources": []}
        ctx = "\n\n".join(f"[{h.chunk.doc_id}#{h.chunk.metadata.get('chunk_index')}] {h.chunk.text}" for h in hits)
        text = await self.llm.generate(SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}")
        return {"answer": text, "grounded": True,
                "sources": [{"doc_id": h.chunk.doc_id, "url": h.chunk.metadata.get("source_url"), "chunk_id": h.chunk.chunk_id, "score": h.score} for h in hits]}
