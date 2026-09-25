"""
services/ingestion_service.py

Orchestrates the ingestion pipeline for one PDF: hash-check for duplicates,
load, chunk, embed, and save to the database — updating `documents` and
`ingestion_jobs` throughout so an admin can see status/progress/failures
(Day 5 SOW, Section 1.1 and 5.1).

This is the ONE function every trigger (a manual script now, an API route
later, an S3 event later still) calls. The trigger changes; this doesn't
— that's the entire point of putting this logic in services/ instead of
directly inside ingestion/run_ingestion.py.

Called from: ingestion/run_ingestion.py right now, once per row in
             manifest.csv, as:
             ingest_pdf(file_path=..., filename=..., company=..., year=..., doc_type=...)
"""

import hashlib
from datetime import datetime

from db.database import SessionLocal
from db.models import Document, DocumentChunk, IngestionJob
from ingestion.loader import extract_pdf_text
from ingestion.chunker import chunk_text
from llm.embeddings_client import embeddings
# Notice this one function pulls from every layer we've built so far:
# db/ (to save), ingestion/ (to load+chunk), llm/ (to embed). That's the
# "services orchestrates, the layers below don't call each other directly"
# rule from the folder-structure discussion, made concrete.


def _file_hash(file_path: str) -> str:
    """Compute the SHA-256 hash of a file's raw bytes — this is the
    'fingerprint' used for duplicate detection (Day 5 SOW, Section 1.1:
    "Duplicate check — compute a SHA-256 hash of the file's content").

    Called from: ingest_pdf() below, as the very first step, before
    anything else happens.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Reading the whole file into memory at once would be wasteful/risky
        # for a huge PDF (remember the "500MB, 10,000-page PDF" example from
        # earlier). Instead, this reads it in 8KB blocks and feeds each block
        # into the hash incrementally — same end result, far less memory used.
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()  # the final hash, as a readable hex string


def ingest_pdf(file_path: str, filename: str, company: str, year: int, doc_type: str) -> None:
    """Ingest one PDF: dedup check, load, chunk, embed, store. Prints progress as it goes.

    Parameters map directly to columns on the `documents` table (see
    db/models.py's Document class) — company becomes `department`,
    since that column is being repurposed for this financial-reports
    dataset rather than a literal HR department.
    """
    session = SessionLocal()
    # Opens one database session for this entire file's ingestion. Every
    # session.add()/session.commit() below uses this same session, so all
    # the writes for one document happen through one connection.

    file_hash = _file_hash(file_path)

    # --- Step 0: duplicate check ---------------------------------------
    existing = session.query(Document).filter(
        Document.file_hash == file_hash,
        Document.status == "completed",
    ).first()
    # Translates to: SELECT * FROM documents WHERE file_hash = '<hash>' AND status = 'completed' LIMIT 1
    # Only a COMPLETED prior ingestion counts as "already done" — a stale
    # 'failed' or 'processing' row (e.g. from a crashed or interrupted run)
    # must not silently block every future retry of that same file.
    if existing:
        # A document with this exact content already exists — do nothing
        # further (Day 5 SOW: "stop here; do not re-process a file that's
        # already been ingested under a different filename").
        print(f"  Skipping {filename} — already ingested (document_id={existing.id}).")
        session.close()
        return

    # A stale row (failed, or stuck 'processing' from an interrupted run)
    # for this exact file would otherwise collide with file_hash's UNIQUE
    # constraint below. Clean it up automatically so a retry "just works"
    # without needing manual SQL — a failed attempt should be retryable,
    # not something that permanently blocks that file.
    stale = session.query(Document).filter(Document.file_hash == file_hash).first()
    if stale:
        print(f"  Found a stale '{stale.status}' record for {filename} (document_id={stale.id}) — clearing it before retrying.")
        session.query(IngestionJob).filter(IngestionJob.document_id == stale.id).delete()
        session.query(DocumentChunk).filter(DocumentChunk.document_id == stale.id).delete()
        session.delete(stale)
        session.commit()

    # --- Step 1: create the `documents` row (status = processing) ------
    document = Document(
        filename=filename,
        s3_path=file_path,  # local path for now; becomes an s3:// URI later, same field
        file_hash=file_hash,
        status="processing",
        department=company,   # "department" column repurposed as company/category for this dataset
        year=year,
        doc_type=doc_type,
    )
    session.add(document)   # stages the new row (not written to Postgres yet)
    session.commit()        # actually writes it, and this is also when `document.id` gets assigned

    # --- Step 2: create the matching `ingestion_jobs` row ---------------
    # This is a SEPARATE row from `documents` on purpose — a document can be
    # (re)processed multiple times over its life, and ingestion_jobs keeps
    # a full history of every run, not just the current status.
    job = IngestionJob(
        document_id=document.id,     # links this job back to the document row created above
        triggered_by="manual",       # will become "s3_event" once triggered automatically
        started_at=datetime.utcnow(),
        status="running",
    )
    session.add(job)
    session.commit()

    try:
        # --- Step 3: load -------------------------------------------------
        print(f"  Loading {filename} ...")
        text = extract_pdf_text(file_path)
        # Calls into ingestion/loader.py — returns the WHOLE document as one
        # big string (text + tables combined, page by page).

        # --- Step 4: chunk -------------------------------------------------
        print("  Chunking ...")
        chunks = chunk_text(text)
        # Calls into ingestion/chunker.py — turns that one big string into a
        # list of smaller strings, e.g. ["chunk 1...", "chunk 2...", ...]
        print(f"  {len(chunks)} chunks produced.")

        # --- Step 5: embed (batch) -----------------------------------------
        print("  Embedding + saving chunks ...")
        vectors = embeddings.embed_documents(chunks)
        # One call handles ALL chunks at once (batch), rather than calling
        # the embedding model once per chunk in a loop — fewer network
        # round-trips to Gemini's API, which matters once a document has
        # hundreds of chunks. `vectors` is a list of number-lists, same
        # length and same order as `chunks` (vectors[i] is the embedding
        # for chunks[i]).

        # --- Step 6: save each chunk as its own document_chunks row -------
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # zip() pairs chunks[i] with vectors[i] together; enumerate()
            # gives us i itself, which becomes chunk_index — recording
            # each chunk's position within the document.
            session.add(DocumentChunk(
                document_id=document.id,   # links this chunk back to its parent document
                chunk_index=i,
                chunk_text=chunk,          # the readable text (for admins/audits to inspect later)
                embedding=vector,          # the actual vector, stored in the pgvector column
                department=company,        # metadata copied onto the CHUNK, not just the document —
                year=year,                 # this is what lets retrieval filter at the chunk level
                doc_type=doc_type,         # (Day 5 SOW, Section 1.1: "This metadata travels WITH the chunk")
            ))
        session.commit()
        # One commit for the whole batch of chunks, not one commit per
        # chunk — much faster, and means either ALL chunks for this
        # document get saved, or (if something fails before this point)
        # none of them do.

        # --- Step 7: mark success --------------------------------------
        document.status = "completed"
        document.last_updated_at = datetime.utcnow()
        job.status = "completed"
        job.finished_at = datetime.utcnow()
        session.commit()
        print(f"  Done: {filename} ({len(chunks)} chunks).")

    except Exception as e:
        # If ANYTHING above failed (a bad PDF, a network error calling
        # Gemini, etc.), this catches it so one bad file doesn't crash the
        # entire run_ingestion.py loop over all manifest entries.
        session.rollback()   # undo any uncommitted, half-finished changes from this attempt
        document.status = "failed"
        job.status = "failed"
        job.error_message = str(e)   # this is what an admin sees in the review queue (Day 5 SOW, Section 5.1)
        job.finished_at = datetime.utcnow()
        session.commit()
        print(f"  FAILED: {filename} — {e}")

    finally:
        # Runs whether ingestion succeeded or failed — always release the
        # database connection back to the pool when this function is done
        # with it.
        session.close()
