"""
db/models.py

The 7 tables from the RAG SOW (Day 5 playbook doc), as SQLAlchemy ORM
classes. Each class = one table. create_tables.py reads these definitions
and issues the actual CREATE TABLE statements against Postgres.
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, ForeignKey, DateTime, ARRAY
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from db.database import Base
from config import EMBED_DIM


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")  # "user" or "admin"
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    s3_path = Column(String, nullable=False)
    file_hash = Column(String, unique=True, nullable=False, index=True)
    status = Column(String, nullable=False, default="pending")  # pending/processing/completed/failed
    department = Column(String)
    year = Column(Integer)
    doc_type = Column(String)
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    review_due_date = Column(DateTime, nullable=True)

    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBED_DIM), nullable=False)
    department = Column(String)
    year = Column(Integer)
    doc_type = Column(String)
    is_deprecated = Column(Boolean, default=False, nullable=False)
    replaces_chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    triggered_by = Column(String, nullable=False)  # "s3_event" or "manual"
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="running")  # running/completed/failed
    error_message = Column(Text, nullable=True)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    # nullable, same reasoning as documents.uploaded_by — no auth system
    # built yet, so there's no real user to attach a session to. Will
    # become nullable=False once login exists and every session is
    # required to belong to a real user.
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    source_chunk_ids = Column(ARRAY(Integer), nullable=True)  # assistant messages only
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class SchemaChunk(Base):
    """One row per introspected table from the NL2SQL source DB (Northwind).

    This is the "RAG over schema" table: instead of chunk_text being a slice
    of a PDF, schema_text is a plain-English description of one table's
    columns and foreign keys. embedding lets the NL2SQL pipeline retrieve
    only the few relevant tables for a given question, instead of stuffing
    every table into every prompt — the same retrieval mechanism as
    DocumentChunk, applied to schema metadata instead of document text.

    Populated by services/schema_ingestion_service.py, via run_schema_ingestion.py.
    """
    __tablename__ = "schema_chunks"

    id = Column(Integer, primary_key=True)
    table_name = Column(String, nullable=False, unique=True)
    schema_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBED_DIM), nullable=False)
    source_db = Column(String, nullable=False, default="northwind")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class NL2SQLAuditLog(Base):
    """One row per question asked through services.nl2sql_service.ask_db()
    (step 12 of the NL2SQL build sequence). Same reasoning as IngestionJob
    on the RAG side: a permanent, append-only record of what happened,
    independent of whether the underlying question succeeded or failed -
    needed for compliance (who queried what, and when) and for debugging
    (which questions needed retries, which failed, which path answered).

    Lives in the `nexus` database, NOT `northwind` - this is metadata ABOUT
    the app's own usage, not Northwind business data, same reasoning
    ChatSession/Message live in `nexus` rather than the source data's DB.

    Populated by services/nl2sql_service.py's ask_db(), after every
    question - success or failure - right before it returns.
    """
    __tablename__ = "nl2sql_audit_log"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    user_role = Column(String, nullable=True)
    employee_id = Column(Integer, nullable=True)
    source = Column(String, nullable=False)  # "semantic_layer" or "nl2sql_generation"
    metric_used = Column(String, nullable=True)
    sql_text = Column(Text, nullable=True)
    row_count = Column(Integer, nullable=True)
    retries_used = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(String, nullable=False)  # "up" or "down"
    comment = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="open")  # open/reviewed/resolved
    resolved_chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
