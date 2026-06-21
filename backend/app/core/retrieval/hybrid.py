"""
Reciprocal Rank Fusion: combines two ranked lists (FAISS semantic results,
BM25 keyword results) into one ranking, without needing to normalize or
compare raw scores from two completely different scales.
"""


def rrf_fuse(
    faiss_ids: list[int],
    bm25_ids: list[int],
    top_k: int = 10,
    k: int = 60,
) -> list[int]:
    """
    faiss_ids / bm25_ids: lists of vector_ids already sorted best-first.
    k: RRF constant (60 is the standard default from the original paper).
    Returns the fused top_k vector_ids, best first.
    """
    scores: dict[int, float] = {}

    for rank, vector_id in enumerate(faiss_ids):
        scores[vector_id] = scores.get(vector_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, vector_id in enumerate(bm25_ids):
        scores[vector_id] = scores.get(vector_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [vector_id for vector_id, _ in fused[:top_k]]