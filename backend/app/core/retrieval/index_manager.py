"""
Whenever a document is added to or removed from a session, we rebuild that
session's combined FAISS + BM25 index from ALL chunks currently in the
session. This is the piece that lets a query search across every document
the student has uploaded, not just the most recent one.

Rebuilding from scratch (rather than incrementally updating) is simple and
plenty fast at the scale of a student's lecture notes (hundreds of chunks,
not millions) - don't over-engineer this for a portfolio project.
"""
import numpy as np
from app.db.mongodb import get_db
from app.core.retrieval import faiss_store, bm25_index


async def rebuild_session_index(session_id: str):
    db = get_db()
    cursor = db.chunks.find({"session_id": session_id}).sort("_id", 1)
    chunks = await cursor.to_list(length=None)

    faiss_store.invalidate_cache(session_id)
    bm25_index.invalidate_cache(session_id)

    if not chunks:
        # No chunks left (e.g. last doc in session was deleted) - clear the index record
        await db.session_indices.delete_one({"_id": session_id})
        return

    # vector_id_map[i] = chunk _id stored at FAISS row i / BM25 corpus index i.
    # Order must match exactly between the embeddings array and this map.
    vector_id_map = [str(c["_id"]) for c in chunks]
    texts = [c["text"] for c in chunks]
    vectors = np.array([c["embedding"] for c in chunks], dtype="float32")

    index = faiss_store.build_index(vectors)
    faiss_gridfs_id = await faiss_store.save_index(session_id, index)

    bm25 = bm25_index.build_bm25(texts)
    bm25_index.cache_bm25(session_id, bm25)
    bm25_bytes = bm25_index.serialize_bm25(bm25)

    await db.session_indices.update_one(
        {"_id": session_id},
        {
            "$set": {
                "faiss_gridfs_id": faiss_gridfs_id,
                "bm25_pickle": bm25_bytes,
                "vector_id_map": vector_id_map,
            }
        },
        upsert=True,
    )