"""
run_nl2sql_query.py

Manual trigger for the NL2SQL pipeline (same role run_query.py plays for
RAG) — a simple command-line loop for testing services/nl2sql_service.py
before any API/UI exists.

Run from the project root:
    python run_nl2sql_query.py

Type a question about the Northwind data, see the generated SQL, which
tables were retrieved for it, and the result rows. Type "exit" to quit.

Later, api/sql_router.py will call the exact same ask_db() function from a
real HTTP endpoint — this file gets thrown away at that point; ask_db()
won't need to change.
"""

from services.nl2sql_service import ask_db

print("Nexus NL2SQL pipeline — type a question ('exit' to quit)\n")

# Step 11 (row-level access control): since there's no real login system
# yet, this stands in for "who just logged in" - a real API layer would
# get this from an authenticated session instead of asking on the CLI.
print("Log in as: (1) admin  (2) a specific employee")
choice = input("Choose 1 or 2: ").strip()
if choice == "2":
    employee_id = int(input("Employee ID (e.g. 4 for Margaret Peacock): ").strip())
    user_context = {"role": "employee", "employee_id": employee_id}
else:
    user_context = {"role": "admin", "employee_id": None}
print(f"Logged in as: {user_context}\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    result = ask_db(question, user_context=user_context)

    if result["clarification_question"]:
        # Ambiguity handling: the LLM declined to guess. No SQL was run at
        # all - print its question and loop back for the user's reply.
        print(f"\nNexus needs clarification: {result['clarification_question']}\n")
        continue

    print(f"\nSource      : {result['source']}" + (f" (metric: {result['metric_used']})" if result["metric_used"] else ""))
    print(f"Tables used : {result['tables_used']}")
    print(f"Attempts    : {len(result['attempts'])}")
    if len(result["attempts"]) > 1:
        # Only print the full attempt-by-attempt history when a retry
        # actually happened - keeps normal (first-try-succeeds) output
        # exactly as clean as before this feature was added.
        for a in result["attempts"]:
            print(f"  attempt {a['attempt']}: sql={a['sql']!r} error={a['error']!r}")
    if result["assumption"]:
        print(f"Assumption  : {result['assumption']}")
    print(f"SQL         : {result['sql']}")
    if result["error"]:
        print(f"Error       : {result['error']}\n")
    else:
        print(f"Rows        : {result['rows']}")
        print(f"\nNexus: {result['answer']}\n")
