"""
BM25 (keyword search) complements FAISS (semantic search) - it catches exact
terms FAISS sometimes misses (acronyms, codes, names). The pickle is small
enough to store directly as a Mongo field, unlike FAISS which needs GridFS.
"""
import pickle
import re
from rank_bm25 import BM25Okapi

_cache: dict[str, BM25Okapi] = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_bm25(texts: list[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized_corpus)


def serialize_bm25(bm25: BM25Okapi) -> bytes:
    return pickle.dumps(bm25)


def deserialize_bm25(raw: bytes) -> BM25Okapi:
    return pickle.loads(raw)


def cache_bm25(session_id: str, bm25: BM25Okapi):
    _cache[session_id] = bm25


def get_cached_bm25(session_id: str) -> BM25Okapi | None:
    return _cache.get(session_id)


def invalidate_cache(session_id: str):
    _cache.pop(session_id, None)


def search(bm25: BM25Okapi, query: str, top_k: int = 20):
    """Returns list of (vector_id, score) sorted by score descending."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]