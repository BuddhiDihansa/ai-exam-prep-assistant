"""
Splits parsed page/slide blocks into overlapping chunks small enough to
embed and feed to the LLM, while keeping track of which page each chunk
came from (for citations).
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """
    blocks: [{"text": ..., "page_number": ...}, ...]
    returns: [{"text": ..., "page_number": ..., "chunk_index": ...}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_index = 0
    for block in blocks:
        for piece in splitter.split_text(block["text"]):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "text": piece,
                    "page_number": block["page_number"],
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
    return chunks