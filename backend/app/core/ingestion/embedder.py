"""
Loads the embedding model ONCE (singleton) and reuses it for every request.
Reloading per-request is the #1 way to blow through Render's free 512MB RAM
and add multi-second latency to every call - don't do it.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model: SentenceTransformer | None = None


def load_embedder() -> SentenceTransformer:
    """Call this once from main.py's startup/lifespan handler."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def get_embedder() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Embedder not loaded - did startup run?")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns a (N, dim) float32 array, L2-normalized so dot product == cosine similarity."""
    model = get_embedder()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Returns a (1, dim) float32 array for a single query string."""
    return embed_texts([text])