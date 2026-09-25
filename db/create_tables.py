"""
db/create_tables.py

One-time (and re-runnable) script: reads every table defined in db/models.py
and issues CREATE TABLE ... IF NOT EXISTS against Postgres.

Run this yourself after the Docker container is up:
    python db/create_tables.py
"""

from sqlalchemy import text

from db.database import engine, Base
from db import models  # noqa: F401  (import registers all table classes with Base)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base.metadata.create_all(bind=engine)

print("All tables created (or already existed).")
