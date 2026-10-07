"""
graph/v1_simple_router/build_graph.py

WHY THIS VERSION (v1) EXISTS: this is the FIRST, simplest way RAG and
NL2SQL were combined - built the moment a real routing decision existed
(before this, RAG and NL2SQL each had only one path, so a plain function
was enough and LangGraph wasn't needed at all). It classifies a question
ONCE and calls exactly ONE tool - it CANNOT answer a compound question
needing both RAG and NL2SQL (e.g. "what was TCS's revenue, and how many
orders has our top customer placed?"). graph/v2_single_agent and
graph/v3_multi_agent exist specifically to fix that limitation. Kept
here, unchanged, as the baseline every later version is compared against.

The graph's shape:
    START -> router -> (rag OR nl2sql, based on state["route"]) -> END

Called from: run_chat_v1.py
"""

from langgraph.graph import StateGraph, START, END

from graph.v1_simple_router.state import GraphState
from graph.v1_simple_router.router import router_node
from graph.v1_simple_router.nodes import rag_node, nl2sql_node

_graph_builder = StateGraph(GraphState)

_graph_builder.add_node("router", router_node)
_graph_builder.add_node("rag", rag_node)
_graph_builder.add_node("nl2sql", nl2sql_node)

_graph_builder.add_edge(START, "router")

# The conditional edge - this is the actual routing mechanism. After
# "router" runs, LangGraph looks at state["route"] (set by router_node)
# and sends execution to whichever node name that value maps to below.
# Notice there is NO edge back to "router" - each question gets exactly
# ONE decision, ONE tool call, then END. That's the defining limitation
# of this version.
_graph_builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {"rag": "rag", "nl2sql": "nl2sql"},
)

_graph_builder.add_edge("rag", END)
_graph_builder.add_edge("nl2sql", END)

compiled_graph = _graph_builder.compile()


def ask_nexus(question: str, session_id: int = None, user_context: dict = None) -> dict:
    """The single entry point a trigger calls - runs the whole graph for
    one question and returns the final state (route, answer, source,
    metadata, error). Mirrors ask() and ask_db()'s role: one function,
    trigger-agnostic, that a CLI script or a future API route calls the
    same way.
    """
    initial_state = {
        "question": question,
        "session_id": session_id,
        "user_context": user_context,
    }
    return compiled_graph.invoke(initial_state)
