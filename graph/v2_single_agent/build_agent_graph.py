"""
graph/v2_single_agent/build_agent_graph.py

WHY THIS VERSION (v2) EXISTS: v1 (graph/v1_simple_router) can only ever
call ONE of RAG/NL2SQL per question, so it fails on a compound question
needing both (e.g. "what was TCS's revenue, and how many orders has our
top customer placed?"). This version fixes that: ONE LLM is bound to
BOTH tools (rag_tool, nl2sql_tool) directly, and it LOOPS - deciding,
turn by turn, "call another tool" or "I'm done" - so it can call one, the
other, or BOTH in sequence before producing one combined final answer.

This is exactly Part 1's pattern from Other_Documents/Learning_Examples/
single_vs_multi_agent_demo.py (single agent, multiple tools, agent<->tools
loop), applied to Nexus's 2 REAL tools instead of mock ones - NOT
multi-agent (see graph/v3_multi_agent for that), since ask()/ask_db() are
still single-shot pipelines with no internal reasoning loop of their own;
only this one top-level LLM reasons.

Called from: run_chat_v2.py
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from llm.llm_client import llm
from graph.common.agent_tools import make_rag_tool, make_nl2sql_tool
from graph.common.text_utils import extract_text


def build_agent_graph(user_context: dict = None):
    """Builds and compiles one fresh agent graph, bound to this specific
    user's context (needed for NL2SQL's row-level security).

    Rebuilt per conversation rather than reused globally - a deliberate
    simplification at this project's current scale; a production system
    would instead pass session/user identity through LangGraph's injected-
    state mechanism and reuse one compiled graph across many requests.

    Returns (compiled_graph, session_holder) - session_holder is the same
    mutable dict rag_tool writes its session_id into, so the caller can
    read the updated value back out after the graph finishes running.
    """
    session_holder = {"session_id": None}

    rag_tool = make_rag_tool(session_holder)
    nl2sql_tool = make_nl2sql_tool(user_context)
    tools = [rag_tool, nl2sql_tool]

    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        # Sends the ENTIRE message history so far - this is what lets the
        # LLM use rag_tool's result to help decide whether it ALSO needs
        # nl2sql_tool (or vice versa) for a compound question.
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # <-- the loop: after either tool, control returns to the agent

    compiled = graph.compile()
    return compiled, session_holder


def ask_nexus_agent(question: str, session_id: int = None, user_context: dict = None) -> dict:
    """The single entry point - runs the looping agent for one question and
    returns:
        {"answer": str, "tool_calls_made": list[str], "session_id": int | None}

    tool_calls_made lists which tool(s) actually ran, in order - this is
    the concrete, visible proof that a compound question can now trigger
    BOTH rag_tool and nl2sql_tool, something v1's one-shot router could
    never do.
    """
    compiled, session_holder = build_agent_graph(user_context)
    session_holder["session_id"] = session_id

    result = compiled.invoke({"messages": [HumanMessage(content=question)]})

    tool_calls_made = [
        tc["name"]
        for m in result["messages"]
        for tc in (getattr(m, "tool_calls", None) or [])
    ]

    return {
        "answer": extract_text(result["messages"][-1].content),
        "tool_calls_made": tool_calls_made,
        "session_id": session_holder["session_id"],
    }
