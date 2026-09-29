"""
run_schema_ingestion.py

Manual trigger for schema ingestion (same role run_ingestion.py plays for
PDFs) — calls services.schema_ingestion_service.ingest_schema() once.

Run this whenever the Northwind schema changes, or just once now to
populate schema_chunks for the first time. Unlike PDF ingestion, there is
no manifest.csv here — introspect_schema() reads every table in the DB
automatically, there's nothing per-table to configure.

Run from the project root:
    python run_schema_ingestion.py
"""

from services.schema_ingestion_service import ingest_schema

ingest_schema()
