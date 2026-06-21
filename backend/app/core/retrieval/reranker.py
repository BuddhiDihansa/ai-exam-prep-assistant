"""
FAISS + BM25 are fast but approximate. The cross-encoder reads the actual
query+chunk pair together (slower, but far more accurate) and re-sorts the
~10 fused candidates down to the best 3 that actually go to the LLM.
"""
from sentence_transformers import CrossEncoder
from app.core.config import settings

_model: CrossEncoder | None = None


def load_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.RERANKER_MODEL)
    return _model


def get_reranker() -> CrossEncoder:
    if _model is None:
        raise RuntimeError("Reranker not loaded - did startup run?")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    candidates: [{"text": ..., ...other fields...}, ...]
    Returns the top_k candidates re-sorted by cross-encoder relevance score,
    each with a "rerank_score" field added.
    """
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]