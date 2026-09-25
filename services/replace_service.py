"""
services/replace_service.py

Handles admin-triggered document REPLACEMENT (Day 5 SOW, Section 1.1.1:
"a full-document replace is only used when the entire source PDF is
genuinely being swapped out... only the previous version's chunks are
deprecated, not deleted outright").

This is deliberately a thin wrapper, not new ingestion logic — it reuses
services/ingestion_service.py's ingest_pdf() for the actual load/chunk/
embed/save work. All this file adds is: "before ingesting the new file,
stop the old one's chunks from being retrievable."

Called from: run_replace.py right now (a manual CLI trigger, same role as
             run_ingestion.py / run_query.py). Later, an admin UI will call
             this same replace_document() function through
             api/documents_router.py — only the trigger changes.
"""

from datetime import datetime

from db.database import SessionLocal
from db.models import Document, DocumentChunk
from services.ingestion_service import ingest_pdf


def replace_document(
    old_filename: str,
    new_file_path: str,
    new_filename: str,
    company: str,
    year: int,
    doc_type: str,
) -> dict:
    """Deprecate an old document's chunks, then ingest its replacement.

    Parameters:
        old_filename   — the filename of the document being replaced (as stored
                          in documents.filename — this is how an admin identifies
                          it, since they think in filenames, not internal ids)
        new_file_path  — local path to the new PDF to ingest
        new_filename   — filename to record for the new document
        company/year/doc_type — metadata for the NEW document (the old one's
                          metadata is untouched — it's a historical record now)

    Returns a dict summarizing what happened, so a UI has something to show
    the admin as confirmation.
    """
    db = SessionLocal()

    # --- Step 1: find the old document by filename ------------------------
    old_document = db.query(Document).filter(Document.filename == old_filename).first()
    if not old_document:
        db.close()
        raise ValueError(f"No document found with filename '{old_filename}' — nothing to replace.")

    # --- Step 2: deprecate the OLD document's chunks -----------------------
    # A bulk UPDATE (not a Python loop over rows) — one SQL statement marks
    # every one of this document's chunks as deprecated in a single query.
    # This is the "fast mitigation" step, done FIRST and on its own, so
    # stale information stops being retrievable immediately — even before
    # the new file has finished being processed.
    deprecated_count = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == old_document.id, DocumentChunk.is_deprecated == False)
        .update({"is_deprecated": True})
    )

    # The document ROW itself is not deleted, and its chunk TEXT is not
    # deleted either — only is_deprecated flips to true. This preserves the
    # audit trail: "this document was ingested, then superseded on this
    # date" remains fully reconstructable later.
    old_document.status = "deprecated"
    old_document.last_updated_at = datetime.utcnow()
    db.commit()

    # Capture the plain int NOW, before db.close() — same DetachedInstanceError
    # pitfall as query_service.py: db.commit() expires old_document's
    # attributes, and accessing old_document.id AFTER the session is closed
    # would try to re-fetch it from a connection that no longer exists.
    old_document_id = old_document.id
    db.close()

    print(f"Deprecated {deprecated_count} chunk(s) from '{old_filename}' (document_id={old_document_id}).")

    # --- Step 3: ingest the new file, exactly like any other upload --------
    # No special-case logic here — this is the SAME ingest_pdf() used for
    # every other document. It runs its own hash-check, load, chunk, embed,
    # save, and documents/ingestion_jobs bookkeeping independently.
    print(f"Ingesting replacement file '{new_filename}' ...")
    ingest_pdf(
        file_path=new_file_path,
        filename=new_filename,
        company=company,
        year=year,
        doc_type=doc_type,
    )

    return {
        "old_document_id": old_document_id,
        "old_filename": old_filename,
        "deprecated_chunk_count": deprecated_count,
        "new_filename": new_filename,
    }
