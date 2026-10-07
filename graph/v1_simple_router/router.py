"""
graph/v1_simple_router/router.py

WHY THIS VERSION (v1) EXISTS: see state.py's docstring in this same
folder - this is the baseline, one-shot-classification version.

What this file does: the router NODE - a plain function that LangGraph
calls with the current state, and which returns the fields to update in
that state. This is the "real multi-path routing decision" that was
always the stated condition for introducing LangGraph in this project
(previously, RAG and NL2SQL each had only one path, so a plain function
was enough).

Called from: graph/v1_simple_router/build_graph.py, added as a node
             named "router".
"""

from graph.v1_simple_router.router_prompt import ROUTER_PROMPT
from graph.common.text_utils import extract_text
from llm.llm_client import llm


def router_node(state: dict) -> dict:
    """Classify state['question'] as needing RAG or NL2SQL, and return
    {"route": "rag" | "nl2sql"} to be merged into the graph's state.

    Defaults to "rag" if the LLM's answer isn't exactly one of the two
    expected words - a wrong route gives a possibly-wrong answer, which is
    recoverable; a crash here would give no answer at all.
    """
    prompt = ROUTER_PROMPT.format(question=state["question"])
    response = llm.invoke(prompt)
    raw = extract_text(response.content).lower()

    if "nl2sql" in raw:
        route = "nl2sql"
    else:
        route = "rag"

    return {"route": route}
