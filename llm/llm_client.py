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

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE
# Same values already verified working in test_llm.py — this just moves
# the object into its own reusable file instead of being recreated inline
# wherever it's needed, same pattern as embeddings_client.py.

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
)
# Built once, at import time. Every file that does
# `from llm.llm_client import llm` gets this SAME object back (Python
# caches modules), not a fresh one each time.
