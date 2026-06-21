"""
HyDE (Hypothetical Document Embeddings): instead of embedding the student's
raw, often-short question, we ask the LLM to write a hypothetical answer
first, then embed THAT. A fuller hypothetical answer tends to match the
phrasing of real lecture notes better than a 5-word question does.

This is OFF by default (use_hyde=False in the query endpoint) because it
costs one extra Groq call per query - turn it on if recall feels weak.
"""
from app.core.generation.rag_chain import get_client
from app.core.config import settings

HYDE_PROMPT = (
    "Write a short, plausible paragraph (3-4 sentences) that could answer "
    "the following question, as if it came from a university lecture note. "
    "It does not need to be factually correct - it only needs to be "
    "plausible in style and content, since it will be used purely to "
    "improve a search query."
)


def expand_query(question: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.5,
        max_tokens=200,
    )
    return response.choices[0].message.content