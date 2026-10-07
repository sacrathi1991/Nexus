"""
run_chat_v3.py

WHY THIS FILE EXISTS: the manual CLI trigger for v3 - the multi-agent
version, where each tool is wrapped in its own independent sub-agent and
a supervisor decides which sub-agent(s) to delegate to
(graph/v3_multi_agent). New in this reorganization, alongside v1 and v2.

Run from the project root:
    python run_chat_v3.py

Try the SAME compound question as v2, e.g.:
    "What was TCS's Q1 revenue, and how many orders has our top Northwind
    customer placed?"
Watch the printed "Sub-agent(s) called" line - it will only ever show
"rag_agent_tool" / "nl2sql_agent_tool", NEVER the raw tool names, since
the supervisor has no visibility into rag_tool/nl2sql_tool directly -
only into the 2 wrapped sub-agents.

Step 13 (Redis exact-match cache): ask the SAME question twice in a row
and watch the "Response source" line change from "FULL PIPELINE (slow)"
to "REDIS CACHE (instant)" on the second ask - that's the cache hit,
visibly skipping the entire supervisor/sub-agent chain.

For any question that touches NL2SQL, watch the "NL2SQL path" line(s) -
"semantic_layer" means a pre-built metric answered it (cheap, no SQL
generated); "nl2sql_generation" means the LLM wrote fresh SQL itself,
since nothing in the metric catalog matched. A compound question can
print more than one such line, if NL2SQL was delegated to more than once.
"""

from graph.v3_multi_agent.build_multi_agent_graph import ask_nexus_multiagent

print("Nexus v3 (multi-agent, supervisor + sub-agents) - type a question ('exit' to quit)\n")

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

    result = ask_nexus_multiagent(question, session_id=session_id, user_context=user_context)
    session_id = result["session_id"]

    source_label = "REDIS CACHE (instant)" if result["from_cache"] else "FULL PIPELINE (slow)"
    print(f"\nResponse source: {source_label}")
    print(f"Sub-agent(s) called: {result['sub_agents_called']}")
    for entry in result["nl2sql_source_log"]:
        if entry["source"] == "semantic_layer":
            print(f"NL2SQL path: PICKED FROM SEMANTIC MODEL (metric: {entry['metric_used']}) — subtask: {entry['subtask']!r}")
        else:
            print(f"NL2SQL path: NOT FOUND IN SEMANTIC MODEL — LLM GENERATED ITS OWN SQL — subtask: {entry['subtask']!r}")
        if entry["sql"]:
            print(f"  SQL: {entry['sql']}")
    print(f"Nexus: {result['answer']}\n")
