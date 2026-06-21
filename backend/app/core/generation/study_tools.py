"""
Generates quiz/flashcard/summary content from a document's chunks.
Groq is asked to return ONLY JSON (no markdown fences) so we can parse it
directly - we still defensively strip fences in case the model adds them.
"""
import json
from app.core.generation.rag_chain import get_client
from app.core.config import settings


def _call_groq_json(system_prompt: str, content: str) -> dict | list:
    client = get_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.4,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content.strip()
    # Defensive cleanup in case the model wraps JSON in ```json fences anyway
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


QUIZ_SYSTEM_PROMPT = (
    "You generate multiple-choice quiz questions from lecture note excerpts. "
    "Return ONLY valid JSON, no preamble, no markdown fences, in this exact "
    "shape: "
    '{"questions": [{"question": str, "options": [str, str, str, str], '
    '"correct_index": int, "explanation": str}]}. '
    "Generate exactly 5 questions. correct_index is 0-based."
)

FLASHCARD_SYSTEM_PROMPT = (
    "You generate flashcards (front/back Q&A pairs) from lecture note "
    "excerpts, covering the key concepts a student should memorize. "
    "Return ONLY valid JSON, no preamble, no markdown fences, in this exact "
    'shape: {"flashcards": [{"front": str, "back": str}]}. '
    "Generate 8-10 flashcards."
)

SUMMARY_SYSTEM_PROMPT = (
    "You summarize lecture note excerpts into a structured study summary. "
    "Return ONLY valid JSON, no preamble, no markdown fences, in this exact "
    'shape: {"title": str, "key_points": [str], "summary": str}. '
    "key_points should be 5-8 short bullet-style strings."
)


def generate_quiz(chunks_text: str) -> dict:
    return _call_groq_json(QUIZ_SYSTEM_PROMPT, chunks_text)


def generate_flashcards(chunks_text: str) -> dict:
    return _call_groq_json(FLASHCARD_SYSTEM_PROMPT, chunks_text)


def generate_summary(chunks_text: str) -> dict:
    return _call_groq_json(SUMMARY_SYSTEM_PROMPT, chunks_text)