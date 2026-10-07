import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# secrets — from .env via OS environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
# Separate connection to the Northwind sample DB (NL2SQL source data) — a
# different database on the same Postgres server as DATABASE_URL, kept apart
# on purpose: DATABASE_URL is the RAG/pgvector store, this is plain
# relational business data. Only read from, never written to by this app.
NORTHWIND_DATABASE_URL = os.getenv("NORTHWIND_DATABASE_URL")
# A SEPARATE, restricted (non-superuser) connection used ONLY for actually
# EXECUTING queries against Northwind (step 11 - row-level access control).
# NORTHWIND_DATABASE_URL above connects as nexus_user, which owns these
# tables and is a Postgres SUPERUSER - Postgres always bypasses row-level
# security for superusers and table owners, no matter what policies exist.
# nl2sql_app has no such bypass, so RLS policies actually apply to it.
NORTHWIND_APP_DATABASE_URL = os.getenv("NORTHWIND_APP_DATABASE_URL")

# Redis connection - the exact-match response cache in front of v3
# (graph/v3_multi_agent). Defaults to the local Docker container
# (docker-compose.yml's "redis" service) if not set in .env - unlike the
# Postgres URLs above, this has a working default since Redis has no
# credentials to configure locally, so it's not required in .env.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# resolve config.yaml relative to this file, not the current working directory
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path) as f:
    _settings = yaml.safe_load(f)

GEMINI_MODEL = _settings["llm"]["model"]
TEMPERATURE = _settings["llm"]["temperature"]
EMBED_MODEL = _settings["embeddings"]["model"]
EMBED_DIM = _settings["embeddings"]["dimension"]
CHUNK_SIZE = _settings["chunking"]["chunk_size"]
CHUNK_OVERLAP = _settings["chunking"]["chunk_overlap"]

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing. Add it to your .env file.")

if not NORTHWIND_DATABASE_URL:
    raise ValueError("NORTHWIND_DATABASE_URL is missing. Add it to your .env file.")

if not NORTHWIND_APP_DATABASE_URL:
    raise ValueError("NORTHWIND_APP_DATABASE_URL is missing. Add it to your .env file.")