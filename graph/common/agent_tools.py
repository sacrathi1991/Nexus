"""
graph/common/agent_tools.py

WHY THIS FILE EXISTS: this project builds 3 side-by-side versions of "how
should RAG and NL2SQL be combined" (see graph/v1_simple_router,
graph/v2_single_agent, graph/v3_multi_agent), to make the difference
between a one-shot router, a single agent, and a multi-agent system
concrete and comparable. v1 does NOT use this file at all - it classifies
a question and calls ask()/ask_db() directly, with no LangChain
tool-binding involved. v2 AND v3 both genuinely need the SAME thing here:
a way to hand ask() and ask_db() to an LLM as callable tools. That's why
this file lives in common/ instead of inside either version's own folder
- it is truly shared, not duplicated.

What it does: wraps the existing ask() (RAG) and ask_db() (NL2SQL)
pipelines as plain LangChain @tool functions, so a reasoning LLM can be
bound to them and decide, turn by turn, which one(s) to call. Neither
ask() nor ask_db() is changed at all here; each tool function just calls
the existing pipeline and returns its answer as plain text with a
citation footer appended, since that's the shape a tool's return value
needs to be for the LLM to read back into its own reasoning.

Called from:
    graph/v2_single_agent/build_agent_graph.py - binds BOTH tools directly
        to one LLM (the single-agent version).
    graph/v3_multi_agent/build_multi_agent_graph.py - wraps EACH tool
        inside its own separate sub-agent first (the multi-agent version).
"""

from langchain_core.tools import tool

from services.query_service import ask
from services.nl2sql_service import ask_db


def make_rag_tool(session_holder: dict):
    """Returns a rag_tool bound to ONE conversation's session_id, tracked
    via a mutable dict (session_holder) so the session_id ask() creates on
    its first call can be read back by the caller after this tool runs -
    a plain closure over an immutable value couldn't do that, since the
    value needs to be written back out, not just read.
    """

    @tool
    def rag_tool(question: str) -> str:
        """Answer a question using ingested financial documents (earnings
        reports, investor presentations - e.g. TCS, Sherwin-Williams).
        Use this for questions about a company's financial results,
        business narrative, or anything from a report or transcript."""
        result = ask(question=question, session_id=session_holder.get("session_id"))
        session_holder["session_id"] = result["session_id"]
        return f"{result['answer']}\n(Source: RAG document chunks {result['source_chunk_ids']})"

    return rag_tool


def make_nl2sql_tool(user_context: dict, source_log: list = None):
    """Returns an nl2sql_tool bound to ONE user's identity (user_context) -
    needed so ask_db()'s row-level security applies correctly. Fixed for
    the whole conversation (unlike session_id), so a plain closure over it
    is enough - it never needs to be written back out.

    source_log (optional): a mutable list this tool APPENDS to every time
    it runs - {"subtask": ..., "source": "semantic_layer"|"nl2sql_generation",
    "metric_used": str|None}. This is the explicit, non-truncated way the
    caller finds out "picked from the semantic model" vs "LLM generated
    its own SQL" for every NL2SQL call made during a conversation turn -
    the citation text embedded in the tool's own return string (below)
    still exists for the LLM's own reasoning, but that string can get
    paraphrased or truncated by the time a human sees the final answer,
    so source_log is the reliable, structured way to surface this for
    display (used by graph/v3_multi_agent's ask_nexus_multiagent()).
    """

    @tool
    def nl2sql_tool(question: str) -> str:
        """Answer a question using the Northwind business database (orders,
        customers, products, employees, suppliers). Use this for counts,
        totals, rankings, or specific records from structured business
        data."""
        result = ask_db(question=question, user_context=user_context)
        answer = result["answer"] or result["clarification_question"] or result["error"]

        citation = f"(Source: {result['source']}"
        if result["metric_used"]:
            citation += f", metric: {result['metric_used']}"
        if result["sql"]:
            citation += f", SQL: {result['sql']}"
        citation += ")"

        if source_log is not None:
            source_log.append({
                "subtask": question,
                "source": result["source"],
                "metric_used": result["metric_used"],
                "sql": result["sql"],
            })

        return f"{answer}\n{citation}"

    return nl2sql_tool
