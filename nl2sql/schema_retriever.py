"""
nl2sql/schema_retriever.py

Step "c" of the schema-retrieval approach: given a user's NL question,
find the handful of tables actually relevant to it, instead of handing the
LLM the entire schema every time.

This is the retrieval half of "RAG over schema" — schema_ingestion_service.py
(embed + store) already ran once, ahead of time. This file runs on EVERY
question, at query time, exactly the same moment _retrieve_top_chunks() in
services/query_service.py runs for document retrieval.

Called from: the NL2SQL query pipeline (services/nl2sql_service.py, not
             built yet) — this file only does retrieval, not SQL generation.
"""

from db.database import SessionLocal
from db.models import SchemaChunk
from llm.embeddings_client import embeddings

TOP_K_TABLES = 5
# How many tables to hand the LLM per question. Same reasoning as query_service.py's
# TOP_K=5 for document chunks: enough to cover a realistic multi-table join
# (e.g. orders + order_details + products + customers), without stuffing in
# tables that have nothing to do with the question.


def retrieve_relevant_tables(question: str, k: int = TOP_K_TABLES) -> list[dict]:
    """Return the k most relevant tables for a question, ranked by embedding
    similarity between the question and each table's schema description.

    Returns a list of dicts: [{"table_name": ..., "schema_text": ...}, ...]
    ordered from most to least relevant — NOT full SchemaChunk ORM objects,
    so the caller never risks a DetachedInstanceError after the session closes
    (same pitfall we hit twice already in query_service.py / replace_service.py).
    """
    db = SessionLocal()

    # embed_query() (singular) — same method query_service.py uses for the
    # user's question, as opposed to embed_documents() (plural, used only
    # during ingestion for a batch of texts).
    question_vector = embeddings.embed_query(question)

    # cosine_distance() ranks schema_chunks by similarity to the question,
    # exactly like _retrieve_top_chunks() ranks document_chunks — smaller
    # distance = more relevant. is_deprecated doesn't apply here (schema
    # rows are upserted, never deprecated), so no extra filter is needed.
    results = (
        db.query(SchemaChunk)
        .order_by(SchemaChunk.embedding.cosine_distance(question_vector))
        .limit(k)
        .all()
    )

    # Capture plain values BEFORE closing the session — same fix as
    # query_service.py's DetachedInstanceError pattern.
    tables = [{"table_name": r.table_name, "schema_text": r.schema_text} for r in results]

    db.close()
    return tables
