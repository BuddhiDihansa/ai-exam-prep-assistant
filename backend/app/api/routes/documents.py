import os
import tempfile
import datetime
from fastapi import APIRouter, UploadFile, Form, HTTPException
from bson import ObjectId

from app.core.config import settings
from app.core.ingestion.parsers import parse_file
from app.core.ingestion.chunker import chunk_blocks
from app.core.ingestion.embedder import embed_texts
from app.core.retrieval.index_manager import rebuild_session_index
from app.db.mongodb import get_db

router = APIRouter()

ALLOWED_TYPES = {"pdf", "docx", "pptx"}


@router.post("/upload")
async def upload_document(file: UploadFile, session_id: str = Form(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_TYPES}")

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            400, f"File is {size_mb:.1f}MB, max allowed is {settings.MAX_FILE_SIZE_MB}MB"
        )

    # Parsers need a real file path, so write to a temp file then clean up.
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        blocks = parse_file(tmp_path, ext)
    finally:
        os.unlink(tmp_path)

    if not blocks:
        raise HTTPException(400, "Couldn't extract any text from this file - is it scanned/empty?")

    chunks = chunk_blocks(blocks)
    if not chunks:
        raise HTTPException(400, "Text was extracted but produced no usable chunks.")

    vectors = embed_texts([c["text"] for c in chunks])

    db = get_db()
    doc_result = await db.documents.insert_one(
        {
            "filename": file.filename,
            "file_type": ext,
            "session_id": session_id,
            "chunk_count": len(chunks),
            "uploaded_at": datetime.datetime.utcnow(),
        }
    )
    doc_id = str(doc_result.inserted_id)

    chunk_docs = [
        {
            "doc_id": doc_id,
            "session_id": session_id,
            "filename": file.filename,
            "text": c["text"],
            "chunk_index": c["chunk_index"],
            "page_number": c["page_number"],
            "embedding": vectors[i].tolist(),
        }
        for i, c in enumerate(chunks)
    ]
    await db.chunks.insert_many(chunk_docs)

    # Rebuild the session's combined FAISS+BM25 index to include this doc
    await rebuild_session_index(session_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunk_count": len(chunks),
    }


@router.get("/")
async def list_documents(session_id: str):
    db = get_db()
    cursor = db.documents.find({"session_id": session_id}).sort("uploaded_at", -1)
    docs = await cursor.to_list(length=None)
    for d in docs:
        d["doc_id"] = str(d.pop("_id"))
    return {"documents": docs}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, session_id: str):
    db = get_db()
    doc = await db.documents.find_one({"_id": ObjectId(doc_id), "session_id": session_id})
    if not doc:
        raise HTTPException(404, "Document not found in this session")

    await db.chunks.delete_many({"doc_id": doc_id})
    await db.documents.delete_one({"_id": ObjectId(doc_id)})

    # Rebuild so the deleted doc's chunks disappear from the index too
    await rebuild_session_index(session_id)

    return {"deleted": doc_id}