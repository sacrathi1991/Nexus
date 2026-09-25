"""
ingestion/chunker.py

Splits a document's full text into overlapping chunks, sized per
config.yaml's chunking.chunk_size / chunking.chunk_overlap.

Uses LangChain's RecursiveCharacterTextSplitter rather than a hand-rolled
splitter — it already tries to break on paragraph/sentence boundaries
before falling back to a hard character cut, which keeps chunks more
coherent than a naive fixed-width slice.

Called from: services/ingestion_service.py, right after extract_pdf_text()
             produces the full document text — chunk_text() is what turns
             that one giant string into the list of pieces that actually
             get embedded and saved as document_chunks rows.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
# These come from config.yaml's chunking section (chunk_size: 1000,
# chunk_overlap: 150 by default) via config.py — same centralized-config
# pattern as GEMINI_MODEL/EMBED_MODEL, so the chunk size can be tuned in
# one place without touching this file.

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    # Roughly how many characters go into each chunk. Not an exact cutoff —
    # the splitter prefers to break at a paragraph/sentence boundary near
    # this size rather than slicing mid-sentence.
    chunk_overlap=CHUNK_OVERLAP,
    # How many characters from the END of one chunk are repeated at the
    # START of the next chunk. This exists so that a sentence or idea
    # sitting right on a chunk boundary doesn't get cut in half and lose
    # meaning — both neighboring chunks get a bit of shared context.
)
# The splitter object is built ONCE, at import time (module load), not
# inside chunk_text() below. That's intentional: building it is cheap but
# there's no reason to redo it on every single function call when the
# settings (CHUNK_SIZE/CHUNK_OVERLAP) never change during a run.


def chunk_text(text: str) -> list[str]:
    """Split full document text into a list of chunk strings.

    `text` is the single combined string returned by
    ingestion/loader.py's extract_pdf_text() — the ENTIRE document's text,
    tables included.

    Returns a plain Python list of strings, e.g.:
        ["chunk 1 text...", "chunk 2 text...", "chunk 3 text...", ...]
    Each element in this list becomes exactly one row in the
    document_chunks table later (see services/ingestion_service.py, where
    this list is looped over with enumerate() to assign chunk_index).
    """
    return _splitter.split_text(text)
