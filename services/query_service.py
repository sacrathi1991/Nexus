"""
services/query_service.py

Orchestrates the RAG query pipeline: embed the user's question, retrieve
the most relevant chunks from document_chunks, build a prompt, ask the
LLM, and save both the question and answer as `messages` rows — with the
assistant message's source_chunk_ids populated, since that's what makes
the Section 1.3 correction/feedback flow possible at all.

This is a plain function chain, NOT a LangGraph graph — see the discussion
on why: there's only one path right now (retrieve -> answer), so there's
no routing decision for LangGraph to make yet. That changes once NL2SQL
is added as a second path.

Called from: run_query.py right now (a manual CLI trigger, same role
             run_ingestion.py played for the ingestion pipeline). Later,
             api/chat_router.py will call this same ask() function from a
             real HTTP endpoint — only the trigger changes, same pattern
             as ingestion.
"""

from db.database import SessionLocal
from db.models import DocumentChunk, ChatSession, Message
from llm.embeddings_client import embeddings
from llm.llm_client import llm
from llm.prompts import RAG_PROMPT

TOP_K = 5
# How many chunks to retrieve per question. A constant here (not in
# config.yaml) since it's a query-pipeline-specific tuning knob, separate
# from ingestion's CHUNK_SIZE/CHUNK_OVERLAP.

HISTORY_LIMIT = 2
# How many PRIOR messages (not exchanges — individual message rows) to
# feed back to the LLM as short-term memory. Kept deliberately small: this
# is what keeps the prompt from growing longer and longer as a
# conversation goes on — the same constraint a real UI would need, since
# an unbounded prompt eventually gets slow, expensive, and can hit the
# model's context-length limit.


def _get_recent_history(session, session_id: int, limit: int = HISTORY_LIMIT) -> list[Message]:
    """Fetch the most recent `limit` messages already saved in this session.

    Called from: ask() below, BEFORE the current question is saved — so
    "recent history" never accidentally includes the question being asked
    right now.

    Returns them in chronological order (oldest of the two first), since
    that's the order a human reading the conversation would expect —
    the DB query itself fetches newest-first (to correctly get the LAST
    2), then this reverses that just for display/prompt order.
    """
    recent = (
        session.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(recent))


def _format_history(messages: list[Message]) -> str:
    """Turn a list of Message rows into plain text for the prompt's {history} slot.

    Called from: ask() below, right after _get_recent_history().
    """
    if not messages:
        # First question in a session — there's nothing to show yet.
        return "(none — this is the start of the conversation)"
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _retrieve_top_chunks(session, question_embedding, k: int = TOP_K) -> list[DocumentChunk]:
    """Find the k chunks whose embedding is closest to the question's embedding.

    Called from: ask() below, right after the question itself is embedded.

    This is the exact query we ran by hand in SQL earlier
    (ORDER BY embedding <=> ... LIMIT k), just expressed through the ORM
    instead of raw SQL — .cosine_distance() is a method the pgvector
    Python package adds onto the embedding column.
    """
    return (
        session.query(DocumentChunk)
        .filter(DocumentChunk.is_deprecated == False)
        # Deprecated chunks (see Day 5 SOW, Section 1.3.1) are never
        # retrievable — this is the entire enforcement mechanism for the
        # "fast mitigation" step of the stale-info correction flow.
        .order_by(DocumentChunk.embedding.cosine_distance(question_embedding))
        .limit(k)
        .all()
    )


def _extract_text(content) -> str:
    """Normalize an LLM response's .content into a plain string.

    Called from: ask() below, right after llm.invoke() returns.

    Depending on the model/SDK version, LangChain's response.content can be
    EITHER a plain string (older/simpler models) OR a list of structured
    content blocks like [{"type": "text", "text": "..."}, {"type": "..."}]
    (newer Gemini models, which can attach extra metadata blocks alongside
    the actual answer text). Message.content is a plain Text database
    column, so it must always be a string by the time it's saved — this
    function handles both shapes and always returns one.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Pull out just the text from any block whose type is "text",
        # ignoring other block types (e.g. thought-signature metadata),
        # and join them in case the answer is split across multiple blocks.
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()

    return str(content)  # fallback — shouldn't normally happen


def _build_context(chunks: list[DocumentChunk]) -> str:
    """Turn a list of DocumentChunk rows into one plain-text block to hand the LLM.

    Called from: ask() below, right after retrieval.
    """
    # Each chunk is labeled with its id so the LLM's context clearly
    # separates one chunk from the next — this isn't required by the
    # prompt itself, just makes the context more legible if you ever print
    # it while debugging.
    return "\n\n".join(f"[chunk_id={c.id}]\n{c.chunk_text}" for c in chunks)


def ask(question: str, session_id: int | None = None, user_id: int | None = None) -> dict:
    """Answer a question using RAG, and record the conversation in the database.

    Parameters:
        question   — the user's question, as plain text
        session_id — an existing chat_sessions.id to continue, or None to start a new session
        user_id    — which user this session belongs to (nullable for now, same as
                     documents.uploaded_by — no auth system built yet)

    Returns a dict: {"answer": str, "source_chunk_ids": list[int], "session_id": int}
    """
    db = SessionLocal()

    # --- Step 1: get or create the chat session -------------------------
    if session_id:
        session_row = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    else:
        session_row = None

    if not session_row:
        # No existing session was given (or the given id didn't match
        # anything) -> start a new conversation.
        session_row = ChatSession(user_id=user_id, title=question[:50])
        # title is just a display label (like ChatGPT's sidebar) — using
        # the first 50 characters of the first question as a placeholder.
        db.add(session_row)
        db.commit()  # needed now so session_row.id exists for the messages below

    # --- Step 1.5: capture recent history BEFORE saving the new question ---
    # Must happen before Step 2 below — otherwise the question being asked
    # right now would already be in the DB and could get counted as its
    # own "recent history", which makes no sense.
    history_messages = _get_recent_history(db, session_row.id)
    history_text = _format_history(history_messages)

    # --- Step 2: save the user's message ---------------------------------
    db.add(Message(session_id=session_row.id, role="user", content=question))
    db.commit()

    # --- Step 3: embed the question ---------------------------------------
    question_vector = embeddings.embed_query(question)
    # embed_query() (singular) is used here, not embed_documents() (plural,
    # used in ingestion) — same underlying model, but this is the
    # single-text version meant for one-off queries rather than batches.

    # --- Step 4: retrieve the most relevant chunks -------------------------
    chunks = _retrieve_top_chunks(db, question_vector)

    if not chunks:
        # No chunks in the database at all (e.g. nothing ingested yet) —
        # short-circuit instead of sending an empty context to the LLM.
        answer = "I don't have any documents ingested yet to answer that from."
        source_chunk_ids = []
    else:
        # --- Step 5: build the prompt and ask the LLM -----------------------
        context = _build_context(chunks)
        prompt = RAG_PROMPT.format(history=history_text, context=context, question=question)
        response = llm.invoke(prompt)
        answer = _extract_text(response.content)
        source_chunk_ids = [c.id for c in chunks]
        # This list is exactly what gets stored on the assistant message
        # below — it's the traceability link the Day 5 SOW's correction
        # flow depends on entirely.

    # --- Step 6: save the assistant's message, WITH its source chunks ------
    db.add(Message(
        session_id=session_row.id,
        role="assistant",
        content=answer,
        source_chunk_ids=source_chunk_ids,
    ))
    db.commit()

    # Capture the plain int NOW, while session_row is still attached to an
    # open session — db.commit() expires its attributes (so the next
    # access would re-fetch from the DB to guarantee freshness), and
    # db.close() below means that re-fetch would have nowhere to go,
    # raising DetachedInstanceError. Reading it into a plain variable
    # first sidesteps the problem entirely.
    session_id_value = session_row.id

    db.close()

    return {
        "answer": answer,
        "source_chunk_ids": source_chunk_ids,
        "session_id": session_id_value,
    }
