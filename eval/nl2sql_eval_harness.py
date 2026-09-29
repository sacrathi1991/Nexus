"""
eval/nl2sql_eval_harness.py

Step 8 of the NL2SQL build sequence — the exact same "report card" concept
as eval/eval_harness.py, applied to the NL2SQL pipeline instead of RAG.
Development-time only: run this yourself after changing sql_prompt.py,
schema_retriever.py, sql_guardrails.py, or the LLM/embedding model, never
called by a real user.

Calls the SAME services.nl2sql_service.ask_db() function a real user's
question would go through — not a separate copy of the retrieval/prompt/
execution logic — same reasoning as the RAG harness: a pass here must mean
"the real pipeline works."

Checks per question (mirrors the RAG harness's two-check design, adapted
for structured rows instead of a text answer):
    1. Execution check — did the pipeline return a result with no error
       and no unresolved clarification request? (Every golden-set question
       here is written to be unambiguous, so hitting the clarify path
       counts as a failure worth investigating, not an expected outcome.)
    2. Value check — do ALL expected_keywords appear, as a case-insensitive
       substring, somewhere in the stringified result rows?

Also reports, per question, whether a retry (self-correction) was needed
and whether the LLM made an assumption — useful diagnostic signal beyond
plain pass/fail, since a "pass that took 2 retries" is a different result
than a "pass on the first try."

Run from the project root:
    python -m eval.nl2sql_eval_harness
"""

import json
import os

from services.nl2sql_service import ask_db

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "nl2sql_golden_set.json")


def _check_values(expected_keywords: list[str], rows: list[dict]) -> bool:
    """True if EVERY expected keyword appears somewhere in the stringified
    rows - same keyword-containment approach as the RAG harness's
    _check_answer(), just applied to structured rows instead of LLM text."""
    rows_text = str(rows).lower()
    return all(keyword.lower() in rows_text for keyword in expected_keywords)


def run_eval() -> None:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []

    for item in golden_set:
        print(f"[{item['id']}] {item['question']}")

        result = ask_db(item["question"])

        if result["clarification_question"]:
            # Every golden-set question is written to be answerable
            # outright - the LLM asking for clarification here means it's
            # being overly cautious on a question that shouldn't need it.
            passed = False
            print(f"    -> FAIL (unexpected clarification request: {result['clarification_question']!r})")
        elif result["error"]:
            passed = False
            print(f"    -> FAIL (pipeline error: {result['error']})")
        else:
            passed = _check_values(item["expected_keywords"], result["rows"])
            status = "PASS" if passed else "FAIL"
            retries = len(result["attempts"]) - 1
            print(f"    -> {status}  (retries_needed={retries}, assumption={result['assumption']!r})")
            print(f"    Expected keywords : {item['expected_keywords']}")
            print(f"    SQL               : {result['sql']}")
            print(f"    Rows              : {result['rows']}")

        results.append({"id": item["id"], "passed": passed})
        print()

    # --- Summary -------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    print("=" * 50)
    print(f"RESULT: {passed_count}/{total} passed")
    print("=" * 50)

    if passed_count < total:
        raise SystemExit(1)


if __name__ == "__main__":
    run_eval()
