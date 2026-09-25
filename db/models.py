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
