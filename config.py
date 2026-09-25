import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# secrets — from .env via OS environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

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