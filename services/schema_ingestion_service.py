"""
services/schema_ingestion_service.py

Orchestrates schema-retrieval ingestion (step "a"/"b" of the RAG-over-schema
approach): introspect the Northwind DB's real structure, turn each table
into a text description, embed it, and save it to schema_chunks.

This plays the same orchestration role for NL2SQL that
services/ingestion_service.py plays for RAG — the trigger (a manual script
now, an admin action or a scheduled job later, since schemas change far
less often than documents) calls this ONE function; the function itself
doesn't change when the trigger does.

Called from: run_schema_ingestion.py
"""

from datetime import datetime

from db.database import SessionLocal
from db.models import SchemaChunk
from nl2sql.schema_introspector import introspect_schema, format_table_schema
from llm.embeddings_client import embeddings


def ingest_schema() -> None:
    """Re-embed and re-save every table's schema description.

    Unlike document ingestion, this does NOT need a dedup/hash check or a
    deprecation flow — a schema description isn't a historical record to
    preserve, it's a live snapshot of the current DB structure. Simplest
    correct approach: on every run, replace each table's row entirely
    (upsert by table_name) so schema_chunks always reflects the real,
    current schema.
    """
    session = SessionLocal()

    print("Introspecting Northwind schema ...")
    tables_info = introspect_schema()
    print(f"  Found {len(tables_info)} tables.")

    # --- Step 1: build the text description for every table ---------------
    schema_texts = [format_table_schema(t) for t in tables_info]

    # --- Step 2: embed all table descriptions in one batch call ------------
    # Same reasoning as ingestion_service.py's embed_documents() call — one
    # batch request for all tables, rather than one API call per table.
    print("Embedding table descriptions ...")
    vectors = embeddings.embed_documents(schema_texts)

    # --- Step 3: upsert one schema_chunks row per table ---------------------
    for table_info, schema_text, vector in zip(tables_info, schema_texts, vectors):
        table_name = table_info["table_name"]

        existing = session.query(SchemaChunk).filter(
            SchemaChunk.table_name == table_name,
            SchemaChunk.source_db == "northwind",
        ).first()

        if existing:
            # Table already known — refresh its description/embedding in
            # place, in case a column was added/removed since the last run.
            existing.schema_text = schema_text
            existing.embedding = vector
            existing.updated_at = datetime.utcnow()
        else:
            session.add(SchemaChunk(
                table_name=table_name,
                schema_text=schema_text,
                embedding=vector,
                source_db="northwind",
            ))

    session.commit()
    session.close()
    print(f"Done: {len(tables_info)} table schemas embedded and saved.")
