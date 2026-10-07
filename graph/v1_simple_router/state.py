"""
graph/v1_simple_router/state.py

WHY THIS VERSION (v1) EXISTS: this is the FIRST, simplest way RAG and
NL2SQL were combined - classify a question once, call exactly ONE tool,
done. It cannot handle a question needing BOTH RAG and NL2SQL (e.g.
"what was TCS's revenue, and how many orders has our top customer
placed?") - it will only ever answer from whichever ONE path the router
picked. v2 (single_agent) and v3 (multi_agent) exist specifically to fix
that limitation. Kept here, unchanged, as the baseline to compare against.

What this file does: defines the shared "state" that flows through this
version's graph - every node (router, rag_node, nl2sql_node) reads from
this same dict and writes updates back into it. This is what makes a
LangGraph graph different from just calling functions directly: instead
of passing return values around by hand, one shared state object
accumulates everything as it moves through the graph, node to node.

Called from: graph/v1_simple_router/build_graph.py (defines the graph
             using this shape), graph/v1_simple_router/router.py and
             graph/v1_simple_router/nodes.py (read/write it).
"""

from typing import TypedDict, Optional


class GraphState(TypedDict, total=False):
    # --- input, set once at the start ---------------------------------
    question: str
    user_context: Optional[dict]      # {"role": ..., "employee_id": ...} - for NL2SQL's row-level security
    session_id: Optional[int]         # for RAG's conversation memory (query_service.ask())

    # --- written by the router node -------------------------------------
    route: str                        # "rag" or "nl2sql"

    # --- written by whichever tool node actually ran --------------------
    answer: str                       # the final, user-facing natural-language answer
    source: str                       # "rag" or "semantic_layer" / "nl2sql_generation" (from ask_db())
    metadata: dict                    # RAG's source_chunk_ids, or NL2SQL's sql/metric_used/tables_used
    error: Optional[str]
