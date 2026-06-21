"""
Calls Groq directly via the official `groq` SDK (lighter than pulling in
the full LangChain dependency tree for one chat call). Builds the answer
from the reranked chunks and returns sources for citation in the UI.
"""
from groq import Groq
from app.core.config import settings

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "You are a helpful study assistant for a university student. "
    "Answer the question using ONLY the provided context from the student's "
    "own lecture notes. If the context doesn't contain the answer, say so "
    "clearly instead of guessing. Keep answers concise and well-structured. "
    "When useful, mention which source number you drew from, like [1] or [2]."
)


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = c.get("page_number", "?")
        parts.append(f"[{i}] (page {page}): {c['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    chunks: reranked top-N chunks, each with at least {"text", "page_number", "doc_id"... }
    Returns {"answer": str, "sources": [...]}
    """
    context = build_context_block(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    client = get_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "doc_id": c.get("doc_id"),
            "filename": c.get("filename"),
            "page_number": c.get("page_number"),
            "snippet": c["text"][:200],
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}