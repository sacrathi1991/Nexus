"""
eval/integration_eval_harness.py

The eval harness for the RAG+NL2SQL INTEGRATION layer (v3, the
multi-agent version) - a different thing from eval/eval_harness.py (which
assumes RAG is the right tool) and eval/nl2sql_eval_harness.py (which
assumes NL2SQL is the right tool). This one tests the DECISION itself:
did the supervisor delegate to the right sub-agent(s) for each question -
neither too few (an incomplete compound answer) nor too many (wasted
cost/latency on an unnecessary tool call)?

Development-time only, exactly like the other two harnesses: run this
yourself whenever you change a tool's docstring (graph/common/
agent_tools.py), the supervisor's own logic (graph/v3_multi_agent/
build_multi_agent_graph.py), or the underlying LLM model - never called
by a real user's question.

Calls the REAL ask_nexus_multiagent() - not a separate copy of the
routing/tool logic - same reasoning as the other two harnesses: a pass
here must mean "the real integration layer works."

Two independent checks per question:
    1. Routing check - did EXACTLY the expected set of sub-agents get
       called (as a set, ignoring order/duplicates)? Catches both
       under-triggering (a compound question answered from only one
       source) and over-triggering (an unnecessary extra tool call).
    2. Answer check - same keyword-containment technique as the other two
       harnesses, applied to the final combined answer. Skipped (treated
       as automatically passing) when expected_keywords is empty - the
       out-of-scope question in the golden set only cares about routing
       (that NEITHER tool was called), not about specific wording.

Run from the project root:
    python -m eval.integration_eval_harness
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
# Same fix as run_chat_v1/v2/v3.py - this harness prints real RAG/NL2SQL
# answers via the same streaming/reasoning trace, which can contain
# characters (e.g. the Rupee sign in TCS revenue figures) Windows'
# default terminal codepage (cp1252) can't represent - without this,
# print() crashes outright instead of just rendering it oddly.

from graph.v3_multi_agent.build_multi_agent_graph import ask_nexus_multiagent

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "integration_golden_set.json")

# Run as admin so NL2SQL's row-level security (Step 11) never restricts
# the facts below - the golden set's expected numbers are the ADMIN
# (unrestricted) view of the data, same as how those facts were originally
# confirmed against the database.
ADMIN_CONTEXT = {"role": "admin", "employee_id": None}


def _check_routing(expected_tools: list[str], actual_tools: list[str]) -> bool:
    """True if the SET of sub-agents called exactly matches expected -
    order and duplicate calls don't matter, but any missing or extra
    sub-agent does."""
    return set(expected_tools) == set(actual_tools)


def _check_answer(expected_keywords: list[str], answer: str) -> bool:
    """Same keyword-containment technique as the other two harnesses -
    automatically True if there are no keywords to check (the
    out-of-scope question doesn't care about exact wording)."""
    if not expected_keywords:
        return True
    answer_lower = (answer or "").lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def run_eval() -> None:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []

    for item in golden_set:
        print(f"[{item['id']}] {item['question']}")

        response = ask_nexus_multiagent(item["question"], user_context=ADMIN_CONTEXT)
        actual_tools = response["sub_agents_called"]

        routing_ok = _check_routing(item["expected_tools"], actual_tools)
        answer_ok = _check_answer(item["expected_keywords"], response["answer"])
        passed = routing_ok and answer_ok

        results.append({"id": item["id"], "passed": passed, "routing_ok": routing_ok, "answer_ok": answer_ok})

        status = "PASS" if passed else "FAIL"
        print(f"    -> {status}  (routing_ok={routing_ok}, answer_ok={answer_ok})")
        print(f"    Expected tools : {item['expected_tools']}")
        print(f"    Actual tools   : {actual_tools}")
        print(f"    Answer         : {response['answer']}")
        print()

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    routing_fail_count = sum(1 for r in results if not r["routing_ok"])
    answer_fail_count = sum(1 for r in results if not r["answer_ok"])

    print("=" * 50)
    print(f"RESULT: {passed_count}/{total} passed")
    if passed_count < total:
        print(f"  Routing failures : {routing_fail_count}  (wrong set of sub-agents called)")
        print(f"  Answer failures  : {answer_fail_count}  (right sub-agent(s), but answer missed expected facts)")
    print("=" * 50)

    if passed_count < total:
        raise SystemExit(1)


if __name__ == "__main__":
    run_eval()
