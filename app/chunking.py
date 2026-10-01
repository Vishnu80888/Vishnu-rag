import hashlib
from .models import Chunk


def chunk_text(doc_id: str, text: str, size: int, overlap: int, metadata: dict | None = None) -> list[Chunk]:
    """Sliding-window chunking preferring paragraph/sentence boundaries; deterministic ids."""
    text = text.strip()
    chunks: list[Chunk] = []
    start, i = 0, 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            for sep in ("\n\n", ". ", "\n", " "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            cid = hashlib.sha256(f"{doc_id}:{i}:{piece}".encode()).hexdigest()[:24]
            chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, text=piece,
                                metadata={**(metadata or {}), "chunk_index": i}))
            i += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
