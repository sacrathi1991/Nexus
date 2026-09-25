"""
run_query.py

Manual trigger for the query pipeline (same role ingestion/run_ingestion.py
played for ingestion) — a simple command-line loop for testing
services/query_service.py before any API/UI exists.

Run from the project root:
    python run_query.py

Type a question, see the answer + which chunk ids it came from, and keep
asking follow-ups in the same session. Type "exit" to quit.

Later, api/chat_router.py will call the exact same ask() function from a
real HTTP endpoint — this file gets thrown away at that point; ask() won't
need to change.
"""

from services.query_service import ask

print("Nexus query pipeline — type a question ('exit' to quit)\n")

session_id = None
# Starts as None (a brand-new session). After the first question,
# ask() returns the session_id it created, and every question after that
# reuses it — so this loop behaves like one continuous chat, not a fresh
# session per question.

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    result = ask(question, session_id=session_id)
    session_id = result["session_id"]  # carry the session forward to the next loop iteration

    print(f"\nNexus: {result['answer']}")
    print(f"(source chunk_ids: {result['source_chunk_ids']})\n")
