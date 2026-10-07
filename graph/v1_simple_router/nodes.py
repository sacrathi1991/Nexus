"""
graph/v1_simple_router/nodes.py

WHY THIS VERSION (v1) EXISTS: see state.py's docstring in this same
folder - this is the baseline, one-shot-classification version.

What this file does: the two "tool" nodes - RAG and NL2SQL. Neither
underlying function (services.query_service.ask / services.nl2sql_service
.ask_db) is changed AT ALL here. Each node's only job is to call the
existing function and translate its own differently-shaped result into
the one common shape defined in graph/v1_simple_router/state.py (answer,
source, metadata, error), since ask() and ask_db() return different
fields today (chunk ids vs. SQL/metric names) and this version's graph
needs one consistent shape regardless of which node ran.

Note: in v1, EXACTLY ONE of these two nodes ever runs per question - the
router (router.py) picks one, and the graph goes straight to END after.
Compare with v2/v3, where both ask()/ask_db() are instead wrapped as
LangChain tools (graph/common/agent_tools.py) that a looping agent can
call one, the other, or BOTH times in a single conversation turn.

Called from: graph/v1_simple_router/build_graph.py, added as nodes named
             "rag" and "nl2sql".
"""

from services.query_service import ask
from services.nl2sql_service import ask_db


def rag_node(state: dict) -> dict:
    """Call the existing RAG pipeline, unchanged, and normalize its result."""
    result = ask(
        question=state["question"],
        session_id=state.get("session_id"),
    )
    return {
        "answer": result["answer"],
        "source": "rag",
        "metadata": {"source_chunk_ids": result["source_chunk_ids"]},
        "error": None,
        "session_id": result["session_id"],  # carried back so the next turn can reuse it
    }


def nl2sql_node(state: dict) -> dict:
    """Call the existing NL2SQL pipeline, unchanged, and normalize its result.

    ask_db() can return an answer, a clarification_question (nothing was
    executed), or an error (every retry failed) - exactly one of those
    three is ever non-None, so "answer" here picks whichever one applies,
    in that priority order, so the graph's caller always has SOMETHING
    user-facing to show, regardless of which of the three happened.
    """
    result = ask_db(
        question=state["question"],
        user_context=state.get("user_context"),
    )

    final_answer = result["answer"] or result["clarification_question"] or result["error"]

    return {
        "answer": final_answer,
        "source": result["source"],
        "metadata": {
            "sql": result["sql"],
            "metric_used": result["metric_used"],
            "tables_used": result["tables_used"],
        },
        "error": result["error"],
    }
