"""
graph/v3_multi_agent/build_multi_agent_graph.py

WHY THIS VERSION (v3) EXISTS: v2 (graph/v2_single_agent) already fixes v1's
"can only call one tool" limitation - it can call both rag_tool and
nl2sql_tool for one compound question. v3 exists to demonstrate a
DIFFERENT axis of change, not a further capability fix: instead of ONE
LLM reasoning about both raw tools directly, EACH tool gets wrapped in
its OWN independent agent (its own LLM, its own compiled graph, its own
internal reasoning loop), and a SUPERVISOR agent coordinates between
them - the supervisor never sees rag_tool/nl2sql_tool directly, only the
2 wrapped sub-agents.

This is exactly Part 2's pattern from Other_Documents/Learning_Examples/
single_vs_multi_agent_demo.py (multi-agent: N single-tool sub-agents +
1 supervisor), applied to Nexus's 2 real tools instead of mock ones.

For today's 2 simple tools, v2 and v3 will often produce near-identical
answers - the real value of v3 shows up as tools/sub-agents grow more
complex or numerous, where each sub-agent can have its OWN specialized
reasoning (e.g. a future sub-agent that itself uses multiple raw tools)
without the top-level supervisor needing to understand any of that detail.

Called from: run_chat_v3.py
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from graph.common.text_utils import extract_text
from graph.common.response_cache import get_cached_response, set_cached_response
from graph.common.semantic_cache import get_semantically_cached_response, set_semantically_cached_response
from graph.common.mcp_tools import get_tavily_tools

from llm.llm_client import llm
from graph.common.agent_tools import make_rag_tool, make_nl2sql_tool


SUPERVISOR_SYSTEM_PROMPT = (
    "Before calling any tool, briefly state in one short sentence which "
    "tool you are choosing and why it fits what the question needs right "
    "now. If the question has multiple parts, name the plan for covering "
    "all of them before your first tool call, then address one part at a "
    "time. This explanation is for a human watching your reasoning live - "
    "always include it, even when the right tool seems obvious."
)
# WHY THIS EXISTS: without it, Gemini's tool-calling responses contain
# ONLY the function call itself - no accompanying text - so a human
# watching the terminal sees "-> calling tavily_search(...)" with zero
# visibility into WHY that tool, out of everyone available, was picked.
# This is a visibility-only prompt addition - it does not change WHICH
# tool gets called, only whether the model also narrates its own choice
# alongside making it. _stream_and_report() prints this text (see its
# "reasoning:" line) whenever the model includes it.


def _make_one_tool_agent(bound_tool):
    """Builds and compiles ONE single-tool sub-agent - its own LLM, its
    own agent<->tools loop, bound to only ONE tool. This sub-agent has no
    knowledge that any other tool or sub-agent exists; it is fully
    independent and could be invoked entirely on its own.

    Same shape as v2's build_agent_graph(), just with a list of exactly
    1 tool instead of 2 - this is deliberate: it's the same mechanical
    pattern, applied at a smaller scope, that gets nested one level deeper
    for the multi-agent version below.
    """
    llm_with_one_tool = llm.bind_tools([bound_tool])

    def agent_node(state: MessagesState):
        response = llm_with_one_tool.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([bound_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()  # graph.compile() #1 or #2 - one per sub-agent


def _stream_and_report(compiled_graph, label: str, initial_messages: list) -> dict:
    """Runs a compiled graph using .stream() instead of .invoke(), printing
    a live progress line for every step as it happens - which tool is
    being called (and with what argument), and a preview of what came
    back - instead of staying silent until the whole chain finishes.

    This is purely a VISIBILITY layer: it changes nothing about what the
    graph computes, only how much of its progress is visible while it
    runs. Used for BOTH the supervisor's own loop and each sub-agent's
    internal loop (with a different "label" per level), so nested
    reasoning is visible at every level, not just the top one.

    stream_mode="values" means each yielded item is the FULL current
    state (the growing messages list), so the last message at each step
    is always the newest thing that just happened - no manual diffing
    needed to figure out what's new.
    """
    final_state = {"messages": initial_messages}
    for state in compiled_graph.stream({"messages": initial_messages}, stream_mode="values"):
        final_state = state
        last_message = state["messages"][-1]

        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            # The model's own explanation for THIS choice, if it produced
            # one - Gemini can emit plain text alongside a tool call, not
            # just the call itself, but only does so reliably when asked
            # to (see SUPERVISOR_SYSTEM_PROMPT below). Printed before the
            # call itself so a reader sees "why" before "what".
            reasoning_text = extract_text(last_message.content) if last_message.content else ""
            if reasoning_text:
                print(f"  [{label}] reasoning: {reasoning_text}")
            for tc in tool_calls:
                print(f"  [{label}] -> calling {tc['name']}({tc['args']})")
        elif getattr(last_message, "type", None) == "tool":
            preview = extract_text(last_message.content)
            preview = preview[:150] + ("..." if len(preview) > 150 else "")
            print(f"  [{label}] <- result: {preview}")
        elif getattr(last_message, "type", None) == "ai":
            print(f"  [{label}] done reasoning.")

    return final_state


def _run_subagent_and_get_final_text(compiled_subagent, label: str, subtask: str) -> str:
    """Runs a compiled sub-agent to completion on one subtask (streaming
    its own internal progress under the given label), returning only its
    final text answer. The SUPERVISOR never sees how many times the
    sub-agent called its own tool - only this final text crosses back up
    to it - but the TERMINAL now shows it, via _stream_and_report's print
    statements, so a human watching isn't left in the dark.
    """
    print(f"  [{label}] starting on subtask: {subtask!r}")
    result = _stream_and_report(compiled_subagent, label, [HumanMessage(content=subtask)])
    return extract_text(result["messages"][-1].content)


def build_multi_agent_graph(user_context: dict = None):
    """Builds the full multi-agent hierarchy for one conversation:
    2 independent sub-agents (Step A), each wrapped as a plain tool the
    supervisor can call (Step B), and one supervisor bound to those
    wrapped tools (Step C).

    Returns (compiled_supervisor_graph, session_holder, nl2sql_source_log) -
    session_holder is the same session_id-tracking mechanism as v2, and
    nl2sql_source_log is a mutable list nl2sql_tool appends to every time
    it runs, so the caller can report "picked from the semantic model" vs
    "LLM generated its own SQL" explicitly, for every NL2SQL call made
    this turn - see make_nl2sql_tool()'s docstring in agent_tools.py.
    """
    session_holder = {"session_id": None}
    nl2sql_source_log = []

    # --- Step A: 2 separately-compiled, single-tool sub-agents ------------
    rag_tool = make_rag_tool(session_holder)
    nl2sql_tool = make_nl2sql_tool(user_context, nl2sql_source_log)

    rag_agent = _make_one_tool_agent(rag_tool)
    nl2sql_agent = _make_one_tool_agent(nl2sql_tool)

    # --- Step B: wrap each compiled sub-agent as a plain tool --------------
    # This is "agent-as-a-tool" - the supervisor calls these exactly like
    # it would call rag_tool/nl2sql_tool directly, but each one is secretly
    # a full agent with its own internal reasoning loop.
    @tool
    def rag_agent_tool(subtask: str) -> str:
        """Delegate to the RAG agent - it can answer questions from
        ingested financial documents (earnings reports, investor
        presentations - e.g. TCS, Sherwin-Williams)."""
        print("[Supervisor] delegating to RAG agent...")
        return _run_subagent_and_get_final_text(rag_agent, "RAG Agent", subtask)

    @tool
    def nl2sql_agent_tool(subtask: str) -> str:
        """Delegate to the NL2SQL agent - it can answer questions from the
        Northwind business database (orders, customers, products,
        employees, suppliers)."""
        print("[Supervisor] delegating to NL2SQL agent...")
        return _run_subagent_and_get_final_text(nl2sql_agent, "NL2SQL Agent", subtask)

    # --- Step D (Step 15): Tavily's 4 MCP tools, added DIRECTLY - no
    # sub-agent wrapping, unlike rag_agent_tool/nl2sql_agent_tool above.
    # Each Tavily operation (search/extract/map/crawl) is a single,
    # one-shot call with no benefit from its own reasoning loop, so these
    # are bound to the supervisor exactly like a plain tool would be -
    # see mcp_tools.py's docstring for the full reasoning.
    tavily_tools = get_tavily_tools()
    print(f"[Supervisor] MCP tools discovered from Tavily: {[t.name for t in tavily_tools]}")

    supervisor_tools = [rag_agent_tool, nl2sql_agent_tool, *tavily_tools]
    # The FULL roster the supervisor is choosing from, printed once up
    # front - not just the Tavily subset - so a human sees everything
    # available BEFORE any reasoning/tool-picking happens, not after.
    print(f"[Supervisor] Full tool roster: {[t.name for t in supervisor_tools]}")

    # --- Step C: the supervisor - SAME loop shape as v2's agent, bound to
    # SUB-AGENTS + raw MCP tools together ------------------------------------
    supervisor_llm = llm.bind_tools(supervisor_tools)

    def supervisor_node(state: MessagesState):
        # The system prompt is prepended fresh on every invocation rather
        # than stored in graph state - this node only ever RETURNS the AI
        # response (see below), never the system message itself, so
        # nothing would persist it into state across turns anyway. This
        # is the standard LangGraph pattern for a node-level system
        # prompt: cheap to re-send, and guarantees it's always present
        # regardless of how many reasoning steps this question takes.
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        response = supervisor_llm.invoke(messages)
        return {"messages": [response]}

    def supervisor_should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools", ToolNode(supervisor_tools))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", supervisor_should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "supervisor")

    compiled_supervisor = graph.compile()  # graph.compile() #3 - the top-level supervisor
    # Total for this version: 3 separate graph.compile() calls
    # (2 sub-agents + 1 supervisor), versus v2's single graph.compile().

    return compiled_supervisor, session_holder, nl2sql_source_log


def ask_nexus_multiagent(question: str, session_id: int = None, user_context: dict = None) -> dict:
    """The single entry point - checks TWO cache tiers before running the
    full multi-agent hierarchy, in order:
        1. Exact-match cache (response_cache.py) - same question, worded
           identically. Free to check, zero API cost.
        2. Semantic cache (semantic_cache.py) - same question, worded
           differently. Costs one embedding call to check, catches
           paraphrases the exact-match tier can't.
    Only on a miss from BOTH does the full pipeline run. Returns:
        {"answer": str, "sub_agents_called": list[str],
         "session_id": int | None, "from_cache": bool,
         "cache_type": "exact" | "semantic" | None}

    from_cache/cache_type are the explicit, always-present signal for
    whether (and HOW) this answer came from Redis instead of a full
    pipeline run - never left implicit, so a caller (or a human reading
    CLI output) can always tell what happened without guessing from
    latency alone.

    sub_agents_called lists which sub-agent(s) the SUPERVISOR chose to
    delegate to, in order - notice this can only ever contain
    "rag_agent_tool" / "nl2sql_agent_tool", never "rag_tool"/"nl2sql_tool"
    directly, since the supervisor has no visibility into the raw tools at
    all - that isolation is the defining feature of this version.

    Known, deliberately-accepted limitation: both tiers are per-question
    caches, not conversation-aware - a follow-up question that depends on
    earlier chat history (e.g. "what about that?") could, in principle,
    hit a stale cached answer from a completely different, unrelated
    conversation if a similar/identical question was ever asked before.
    Not fixed here - documented as a real tradeoff of keeping this simple.
    """
    exact_hit = get_cached_response(question, user_context)
    if exact_hit is not None:
        return {
            "answer": exact_hit["answer"],
            "sub_agents_called": exact_hit["sub_agents_called"],
            "nl2sql_source_log": exact_hit.get("nl2sql_source_log", []),
            "session_id": session_id,  # THIS caller's session, never overwritten by the cache
            "from_cache": True,
            "cache_type": "exact",
        }

    semantic_hit = get_semantically_cached_response(question, user_context)
    if semantic_hit is not None:
        return {
            "answer": semantic_hit["answer"],
            "sub_agents_called": semantic_hit["sub_agents_called"],
            "nl2sql_source_log": semantic_hit.get("nl2sql_source_log", []),
            "session_id": session_id,
            "from_cache": True,
            "cache_type": "semantic",
            "matched_question": semantic_hit["matched_question"],
            "similarity_score": semantic_hit["similarity_score"],
        }

    compiled_supervisor, session_holder, nl2sql_source_log = build_multi_agent_graph(user_context)
    session_holder["session_id"] = session_id

    print("[Supervisor] starting...")
    result = _stream_and_report(compiled_supervisor, "Supervisor", [HumanMessage(content=question)])

    sub_agents_called = [
        tc["name"]
        for m in result["messages"]
        for tc in (getattr(m, "tool_calls", None) or [])
    ]

    response = {
        "answer": extract_text(result["messages"][-1].content),
        "sub_agents_called": sub_agents_called,
        "nl2sql_source_log": nl2sql_source_log,
        "session_id": session_holder["session_id"],
        "from_cache": False,
        "cache_type": None,
    }

    # Only answer + sub_agents_called + nl2sql_source_log are cached -
    # session_id is specific to THIS conversation and must never leak into
    # a future cache hit for a different caller. Stored in BOTH tiers, so
    # either a later exact repeat or a later paraphrase of this SAME
    # question can hit on the next ask.
    cacheable_result = {
        "answer": response["answer"],
        "sub_agents_called": response["sub_agents_called"],
        "nl2sql_source_log": response["nl2sql_source_log"],
    }
    set_cached_response(question, user_context, cacheable_result)
    set_semantically_cached_response(question, user_context, cacheable_result)

    return response
