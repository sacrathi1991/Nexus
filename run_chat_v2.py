"""
run_chat_v2.py (was run_chat_agent.py)

WHY THIS FILE EXISTS: the manual CLI trigger for v2 - the single agent,
bound to both RAG and NL2SQL tools directly, looping until it decides
it's done (graph/v2_single_agent). Renamed from run_chat_agent.py as part
of splitting Nexus's RAG+NL2SQL combination logic into 3 comparable
versions (v1 simple router, v2 single agent, v3 multi-agent).

Run from the project root:
    python run_chat_v2.py

Try a compound question, e.g.:
    "What was TCS's Q1 revenue, and how many orders has our top Northwind
    customer placed?"
This needs BOTH tools to answer correctly - watch the printed
"Tool(s) called" line to confirm both actually ran (v1 could never do
this - it only ever calls one tool per question).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
# Same fix as run_chat_v3.py - Windows' terminal codepage (cp1252) can't
# represent every character an answer might contain (e.g. the Rupee sign
# in TCS revenue figures), which crashes print() outright rather than
# just rendering it oddly.

from graph.v2_single_agent.build_agent_graph import ask_nexus_agent

print("Nexus v2 (single agent, both tools) - type a question ('exit' to quit)\n")

print("Log in as: (1) admin  (2) a specific employee")
choice = input("Choose 1 or 2: ").strip()
if choice == "2":
    employee_id = int(input("Employee ID (e.g. 4 for Margaret Peacock): ").strip())
    user_context = {"role": "employee", "employee_id": employee_id}
else:
    user_context = {"role": "admin", "employee_id": None}
print(f"Logged in as: {user_context}\n")

session_id = None

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    result = ask_nexus_agent(question, session_id=session_id, user_context=user_context)
    session_id = result["session_id"]

    print(f"\nTool(s) called: {result['tool_calls_made']}")
    print(f"Nexus: {result['answer']}\n")
