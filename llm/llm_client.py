"""
llm/llm_client.py

One shared CHAT model instance, built once from config.py — the counterpart
to llm/embeddings_client.py, but for generating answers instead of vectors.

This is a completely separate object from `embeddings` in
embeddings_client.py: embeddings turn text INTO a vector (for storing/
searching); this llm turns a prompt (question + retrieved context) INTO a
natural-language answer. Both happen to be Gemini models, but they're used
for different jobs and are never interchangeable.

Called from: services/query_service.py, as: llm.invoke(prompt)
"""

import warnings

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE

# gemini-3.5-flash-lite ignores `temperature` entirely (fixed sampling) -
# this project still passes it (config.yaml keeps it, in case the model
# is ever swapped for one that DOES support it), but the library re-warns
# on EVERY single invoke() call, not once - across a multi-call chain
# (v3's supervisor -> sub-agent -> ... loop) this floods the console with
# duplicate, purely cosmetic noise. Suppressed by exact message text only,
# so any OTHER, genuinely new warning from this library still surfaces.
warnings.filterwarnings(
    "ignore",
    message=".*uses fixed sampling defaults.*",
    category=UserWarning,
)
# Same values already verified working in test_llm.py — this just moves
# the object into its own reusable file instead of being recreated inline
# wherever it's needed, same pattern as embeddings_client.py.

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
    timeout=60,
    # Without this, a stalled or rate-limited request can hang FOREVER
    # with no error and no response to Ctrl+C, since it's blocked inside
    # a native network call rather than regular Python code - this was
    # observed directly (v3's nested supervisor -> sub-agent -> RAG call
    # chain hung indefinitely). 60s is generous for this project's calls,
    # while still guaranteeing a real error eventually instead of a
    # silent, unrecoverable hang.
)
# Built once, at import time. Every file that does
# `from llm.llm_client import llm` gets this SAME object back (Python
# caches modules), not a fresh one each time.
