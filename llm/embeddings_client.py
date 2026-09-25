"""
llm/embeddings_client.py

One shared embeddings object, built once from config.py, imported by
anything that needs to turn text into a vector — the ingestion pipeline
(embedding chunks) and, later, the query pipeline (embedding the user's
question). Both must use this same instance/model, since a chunk's vector
and a question's vector are only comparable if they came from the same
embedding model — this file is what guarantees that (there's only one
place an embeddings object gets created in the whole project).

Called from: services/ingestion_service.py currently
             (embeddings.embed_documents(chunks))
             The query pipeline (not built yet) will import this same
             `embeddings` object and call embeddings.embed_query(question).
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GOOGLE_API_KEY, EMBED_MODEL
# Same config.py values already verified working back in test_embed.py —
# nothing new here, just centralizing the object instead of each file
# creating its own copy.

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=GOOGLE_API_KEY,
)
# This line runs ONCE, the first time any other file does
# `from llm.embeddings_client import embeddings` (Python caches imported
# modules, so a second import elsewhere in the same run reuses this same
# object instead of recreating it).
#
# This object has two methods worth knowing:
#   .embed_documents(list_of_texts) -> list of vectors   (batch — used in ingestion)
#   .embed_query(single_text)       -> one vector          (single — used at query time)
# Both call the same underlying Gemini embedding model; embed_documents is
# just the batch-friendly version for embedding many chunks at once.
