"""
db/database.py

The single connection point to Postgres. Every other file that needs the
database imports `engine` or `SessionLocal` from here — nobody else builds
their own connection.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"connect_timeout": 10},
    # Without this, attempting to connect while Postgres/Docker is
    # unreachable can hang INDEFINITELY with no error - observed directly
    # when Docker Desktop itself was down and a v3 graph call silently
    # froze instead of failing. 10s is enough for a healthy local
    # container to respond; anything slower means something is actually
    # wrong and should surface as a clear connection error, not a freeze.
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# All models in db/models.py inherit from this, so create_tables.py can
# find every table definition through one object.
Base = declarative_base()


def get_db():
    """Yields a session, guarantees it closes even if the caller raises.
    FastAPI routes will use this via Depends(get_db) later."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
