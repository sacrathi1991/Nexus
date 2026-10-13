"""
run_chat_v1.py (was run_chat.py)

WHY THIS FILE EXISTS: the manual CLI trigger for v1 - the baseline
one-shot router (graph/v1_simple_router). Renamed from run_chat.py to
run_chat_v1.py as part of splitting Nexus's RAG+NL2SQL combination logic
into 3 comparable versions (v1 simple router, v2 single agent,
v3 multi-agent) - see graph/v1_simple_router/build_graph.py's docstring
for why this version is limited to answering from only ONE of RAG/NL2SQL
per question.

Run from the project root:
    python run_chat_v1.py

Later, an api/chat_router.py will call the exact same ask_nexus() function
from a real HTTP endpoint - this file gets thrown away at that point;
ask_nexus() won't need to change.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
# Same fix as run_chat_v3.py - Windows' terminal codepage (cp1252) can't
# represent every character an answer might contain (e.g. the Rupee sign
# in TCS revenue figures), which crashes print() outright rather than
# just rendering it oddly.

from graph.v1_simple_router.build_graph import ask_nexus

print("Nexus v1 (simple router) - type a question ('exit' to quit)\n")

# Same "simulate a login" stand-in as run_nl2sql_query.py - only matters if
# a question actually routes to NL2SQL, but asked upfront either way since
# there's no real login system yet.
print("Log in as: (1) admin  (2) a specific employee")
choice = input("Choose 1 or 2: ").strip()
if choice == "2":
    employee_id = int(input("Employee ID (e.g. 4 for Margaret Peacock): ").strip())
    user_context = {"role": "employee", "employee_id": employee_id}
else:
    user_context = {"role": "admin", "employee_id": None}
print(f"Logged in as: {user_context}\n")

session_id = None  # carried forward across turns, same as run_query.py

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    result = ask_nexus(question, session_id=session_id, user_context=user_context)
    session_id = result.get("session_id") or session_id

    print(f"\nRoute  : {result['route']}")
    print(f"Source : {result['source']}")
    print(f"Nexus  : {result['answer']}\n")
