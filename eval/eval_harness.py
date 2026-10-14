"""
eval/eval_harness.py

The "report card" for the RAG pipeline (Section 5 of the Day 6 playbook doc).
This is a DEVELOPMENT-time tool, never called by real users — you run it
yourself, from the terminal, whenever you change chunker.py, prompts.py,
loader.py, or the LLM/embedding model in config.yaml, to check whether the
pipeline still answers known questions correctly.

It deliberately calls the SAME ask() function that run_query.py (real user
queries) calls — not a separate copy of the retrieval/prompt logic — so a
pass here actually means "the real pipeline works," not "some similar code
works." See eval/golden_set.json for the fixed question/answer set this
runs against.

Each golden question gets its OWN fresh session (session_id=None), since
these are independent factual questions, not a real back-and-forth
conversation — feeding them through one shared session would leak
irrelevant "recent history" between unrelated questions.

Two independent checks per question (this is the "retrieval bug vs
generation bug" distinction from the playbook doc):
    1. Retrieval check — was the expected source chunk actually retrieved
       (any id in expected_chunk_ids present in source_chunk_ids)?
    2. Answer check — do the expected_keywords all appear in the LLM's
       final answer (case-insensitive substring match)?

This uses simple keyword containment, not semantic similarity or an
LLM-as-judge — cheapest option, no extra API calls, good enough for
short factual answers. Swapping in a smarter check later (e.g. an LLM
grading call) only means changing _check_answer() below; nothing else
in this file would need to change.

Run from the project root:
    python -m eval.eval_harness
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
# Same fix as run_chat_v1/v2/v3.py - this harness prints real RAG answers,
# which can contain characters (e.g. the Rupee sign in TCS revenue
# figures) Windows' default terminal codepage (cp1252) can't represent -
# without this, print() crashes outright instead of just rendering it oddly.

from services.query_service import ask

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")


def _check_retrieval(expected_chunk_ids: list[int], retrieved_chunk_ids: list[int]) -> bool:
    """True if ANY expected chunk was among what got retrieved (recall@k, k=TOP_K)."""
    return any(cid in retrieved_chunk_ids for cid in expected_chunk_ids)


def _check_answer(expected_keywords: list[str], answer: str) -> bool:
    """True if EVERY expected keyword shows up somewhere in the generated answer."""
    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def run_eval() -> None:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []

    for item in golden_set:
        print(f"[{item['id']}] {item['question']}")

        # Fresh session every time (session_id=None) — see module docstring.
        response = ask(question=item["question"], session_id=None, user_id=None)

        retrieval_ok = _check_retrieval(item["expected_chunk_ids"], response["source_chunk_ids"])
        answer_ok = _check_answer(item["expected_keywords"], response["answer"])
        passed = retrieval_ok and answer_ok

        results.append({
            "id": item["id"],
            "passed": passed,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
        })

        status = "PASS" if passed else "FAIL"
        print(f"    -> {status}  (retrieval_ok={retrieval_ok}, answer_ok={answer_ok})")
        print(f"    Expected keywords : {item['expected_keywords']}")
        print(f"    Got answer        : {response['answer']}")
        print(f"    Retrieved chunks  : {response['source_chunk_ids']}  (expected one of {item['expected_chunk_ids']})")
        print()

    # --- Summary -------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    retrieval_fail_count = sum(1 for r in results if not r["retrieval_ok"])
    answer_fail_count = sum(1 for r in results if not r["answer_ok"])

    print("=" * 50)
    print(f"RESULT: {passed_count}/{total} passed")
    if passed_count < total:
        print(f"  Retrieval failures : {retrieval_fail_count}  (right chunk never retrieved)")
        print(f"  Answer failures    : {answer_fail_count}  (chunk retrieved, but answer missed expected facts)")
    print("=" * 50)

    # Non-zero exit code on any failure — this is what lets this script
    # later gate a CI pipeline (like a failing pytest run blocking a merge),
    # even though nothing in this project wires that up yet.
    if passed_count < total:
        raise SystemExit(1)


if __name__ == "__main__":
    run_eval()
