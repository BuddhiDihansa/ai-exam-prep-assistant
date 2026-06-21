"""
FAISS has no built-in persistence like ChromaDB, so we serialize the index
to bytes and store those bytes in MongoDB GridFS. An in-memory dict caches
the deserialized index per session so a warm server doesn't hit GridFS on
every single query - only on cold start or after a rebuild.
"""
import faiss
import numpy as np
from bson import ObjectId
from app.db.mongodb import get_db, get_gridfs_bucket

_cache: dict[str, faiss.Index] = {}  # session_id -> in-memory FAISS index


def build_index(vectors: np.ndarray) -> faiss.Index:
    """vectors must be float32, L2-normalized -> inner product == cosine similarity."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


async def save_index(session_id: str, index: faiss.Index) -> str:
    """Serializes the index and stores it in GridFS, replacing any previous one for this session."""
    db = get_db()
    bucket = get_gridfs_bucket()

    # Remove the old file for this session, if any
    existing = await db.session_indices.find_one({"_id": session_id})
    if existing and existing.get("faiss_gridfs_id"):
        try:
            await bucket.delete(ObjectId(existing["faiss_gridfs_id"]))
        except Exception:
            pass  # already gone - fine

    raw_bytes = faiss.serialize_index(index)
    file_id = await bucket.upload_from_stream(
        f"{session_id}.faiss", raw_bytes.tobytes()
    )
    _cache[session_id] = index
    return str(file_id)


async def load_index(session_id: str) -> faiss.Index | None:
    """Returns the FAISS index for a session, from cache or GridFS. None if it doesn't exist yet."""
    if session_id in _cache:
        return _cache[session_id]

    db = get_db()
    bucket = get_gridfs_bucket()
    record = await db.session_indices.find_one({"_id": session_id})
    if not record or not record.get("faiss_gridfs_id"):
        return None

    stream = await bucket.open_download_stream(ObjectId(record["faiss_gridfs_id"]))
    raw_bytes = await stream.read()
    index = faiss.deserialize_index(np.frombuffer(raw_bytes, dtype=np.uint8))
    _cache[session_id] = index
    return index


def invalidate_cache(session_id: str):
    _cache.pop(session_id, None)


def search(index: faiss.Index, query_vector: np.ndarray, top_k: int = 20):
    """Returns (scores, vector_ids) for the top_k nearest chunks."""
    scores, ids = index.search(query_vector, top_k)
    return scores[0], ids[0]