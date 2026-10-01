from .models import Hit


def rrf_fuse(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion with chunk_id dedup and stable tie-breaking."""
    scores: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for lst in lists:
        for rank, h in enumerate(lst, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, h)
    ordered = sorted(scores, key=lambda c: (-scores[c], c))
    return [Hit(chunk=by_id[c].chunk, score=scores[c]) for c in ordered]
