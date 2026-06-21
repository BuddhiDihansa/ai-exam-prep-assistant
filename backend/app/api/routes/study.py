from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.mongodb import get_db
from app.core.generation import study_tools

router = APIRouter()

# Cap how much chunk text we send to the LLM per request - keeps cost/latency
# predictable even for a 100-page document.
MAX_CONTEXT_CHARS = 12000


class StudyRequest(BaseModel):
    doc_id: str


async def _get_doc_text(doc_id: str) -> str:
    db = get_db()
    cursor = db.chunks.find({"doc_id": doc_id}).sort("chunk_index", 1)
    chunks = await cursor.to_list(length=None)
    if not chunks:
        raise HTTPException(404, "Document not found or has no content")

    text = ""
    for c in chunks:
        if len(text) + len(c["text"]) > MAX_CONTEXT_CHARS:
            break
        text += c["text"] + "\n\n"
    return text


@router.post("/quiz")
async def quiz(req: StudyRequest):
    text = await _get_doc_text(req.doc_id)
    try:
        return study_tools.generate_quiz(text)
    except ValueError:
        raise HTTPException(502, "Model returned malformed JSON - try again")


@router.post("/flashcards")
async def flashcards(req: StudyRequest):
    text = await _get_doc_text(req.doc_id)
    try:
        return study_tools.generate_flashcards(text)
    except ValueError:
        raise HTTPException(502, "Model returned malformed JSON - try again")


@router.post("/summary")
async def summary(req: StudyRequest):
    text = await _get_doc_text(req.doc_id)
    try:
        return study_tools.generate_summary(text)
    except ValueError:
        raise HTTPException(502, "Model returned malformed JSON - try again")