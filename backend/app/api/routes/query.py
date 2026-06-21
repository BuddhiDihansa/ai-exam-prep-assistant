from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from bson import ObjectId
import datetime

from app.db.mongodb import get_db
from app.core.ingestion.embedder import embed_query
from app.core.retrieval import faiss_store, bm25_index, reranker
from app.core.retrieval.hybrid import rrf_fuse
from app.core.generation.rag_chain import generate_answer
from app.core.generation.query_expander import expand_query

router = APIRouter()


class QueryRequest(BaseModel):
    session_id: str
    question: str
    use_hyde: bool = False


@router.post("/")
async def query(req: QueryRequest):
    db = get_db()

    index = await faiss_store.load_index(req.session_id)
    record = await db.session_indices.find_one({"_id": req.session_id})
    if index is None or not record:
        raise HTTPException(
            400, "No documents indexed for this session yet - upload something first."
        )

    bm25 = bm25_index.get_cached_bm25(req.session_id)
    if bm25 is None:
        bm25 = bm25_index.deserialize_bm25(record["bm25_pickle"])
        bm25_index.cache_bm25(req.session_id, bm25)

    vector_id_map = record["vector_id_map"]

    # Optional HyDE expansion: embed a hypothetical answer instead of the raw question
    search_text = expand_query(req.question) if req.use_hyde else req.question
    query_vector = embed_query(search_text)

    faiss_scores, faiss_positions = faiss_store.search(index, query_vector, top_k=20)
    faiss_ids = [int(p) for p in faiss_positions if p != -1]

    bm25_ranked = bm25_index.search(bm25, req.question, top_k=20)
    bm25_ids = [pos for pos, _score in bm25_ranked]

    fused_positions = rrf_fuse(faiss_ids, bm25_ids, top_k=10)
    chunk_object_ids = [ObjectId(vector_id_map[p]) for p in fused_positions]

    cursor = db.chunks.find({"_id": {"$in": chunk_object_ids}})
    candidates = await cursor.to_list(length=None)
    for c in candidates:
        c["chunk_id"] = str(c.pop("_id"))

    top_chunks = reranker.rerank(req.question, candidates, top_k=3)

    if not top_chunks:
        raise HTTPException(404, "No relevant content found for this question.")

    result = generate_answer(req.question, top_chunks)

    await db.conversations.update_one(
        {"session_id": req.session_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {
                            "role": "user",
                            "content": req.question,
                            "timestamp": datetime.datetime.utcnow(),
                        },
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "timestamp": datetime.datetime.utcnow(),
                        },
                    ]
                }
            }
        },
        upsert=True,
    )

    return result