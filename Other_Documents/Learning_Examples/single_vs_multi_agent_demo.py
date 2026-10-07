"""
Other_Documents/Learning_Examples/single_vs_multi_agent_demo.py

A standalone, runnable side-by-side comparison: SAME 3 tools (search,
weather, currency), built two different ways -

    PART 1 - SINGLE AGENT: one LLM, bound to all 3 tools directly, loops
             (reason -> call a tool -> observe -> reason again -> ...)
             until IT decides it has everything it needs.

    PART 2 - MULTI AGENT: the same 3 tools, but each wrapped inside its
             OWN separate one-tool agent (its own compiled graph, its own
             LLM, its own internal loop) - and a SUPERVISOR agent that
             does not know the 3 real tools exist at all. From the
             supervisor's point of view, each sub-agent just IS a tool -
             it calls "person_info_agent", not "search_tool" directly.

This is NOT part of the Nexus app itself - it doesn't touch RAG, NL2SQL,
or graph/ at all. It exists purely to make the single-agent vs
multi-agent structural difference concrete and runnable, using the same
shared Gemini client (llm.llm_client) already configured for this
project, so no new API key setup is needed.

Run from the project root:
    python "Other_Documents/Learning_Examples/single_vs_multi_agent_demo.py"

This needs real network access to the Gemini API - it will not run inside
a sandboxed shell with no internet access; run it from your own terminal.
"""

import os
import sys

# This file lives 2 folders below the project root
# (Other_Documents/Learning_Examples/...), but it needs to import llm.llm_client,
# which lives AT the project root. Running a script directly (not as a
# module with `-m`) only puts the SCRIPT'S OWN folder on the import path,
# not the project root - so without this line, `from llm.llm_client import
# llm` fails with "ModuleNotFoundError: No module named 'llm'". This
# inserts the project root onto the path explicitly, so the script works
# no matter how it's run.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from llm.llm_client import llm
# Reusing the SAME shared Gemini client already used by every other part
# of Nexus (RAG, NL2SQL) - proves this pattern works with the model
# already configured for this project, nothing extra to set up.


# ===========================================================================
# THE 3 TOOLS - identical in both PART 1 and PART 2. Mocked with fixed
# lookup data (no real web/weather/currency API keys involved) so this file
# is fully runnable on its own - only the LLM calls need real network access.
# ===========================================================================

@tool
def search_tool(query: str) -> str:
    """Search the web for a fact about a person, e.g. where they live or were born."""
    facts = {
        "where does virat kohli live": "Virat Kohli currently lives in Mumbai, India.",
        "where was virat kohli born": "Virat Kohli was born in Delhi, India.",
    }
    return facts.get(query.lower().strip(), f"No search result found for: {query!r}")


@tool
def weather_tool(city: str) -> str:
    """Get the current weather for a given city."""
    weather = {"delhi": "32 C, hazy", "mumbai": "30 C, humid"}
    return weather.get(city.lower().strip(), f"No weather data for: {city!r}")


@tool
def currency_tool(country: str) -> str:
    """Get the currency exchange rate to USD for a given country."""
    rates = {"india": "1 USD = 83.2 INR"}
    return rates.get(country.lower().strip(), f"No exchange rate data for: {country!r}")


ALL_TOOLS = [search_tool, weather_tool, currency_tool]

TEST_QUESTION = (
    "Give me the exchange rate of the currency where Virat Kohli lives, "
    "and also the weather of the city where he was born."
)


# ===========================================================================
# PART 1 - SINGLE AGENT
#
# ONE LLM, bound to all 3 raw tools. It sees the full conversation so far
# (including every previous tool's result) at every step, and decides:
# "call another tool" or "I have enough - answer now."
#
# Graph shape:  START -> agent -> (loop: tools -> agent) -> ... -> END
# ===========================================================================

def build_single_agent():
    llm_with_all_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        # Sends the ENTIRE message history so far (question + every past
        # tool result) - this is what lets one LLM chain multiple tool
        # calls together, using earlier results to decide later ones.
        response = llm_with_all_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        # If the LLM's last response includes tool_calls, it wants to act
        # again - route to "tools". Otherwise it produced a final answer -
        # route to END. This IS the loop: "tools" always routes back to
        # "agent" below, so this can fire many times in a row.
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # <-- THE LOOP: tools always returns to agent, not to END

    return graph.compile()  # ONE graph.compile() call - single agent, single graph


# ===========================================================================
# PART 2 - MULTI AGENT
#
# Step A: build 3 separate ONE-TOOL agents, each its OWN compiled graph,
#         each with its OWN internal agent<->tools loop (same shape as
#         Part 1, just with only 1 tool each). None of them know the other
#         2 tools - or the other 2 sub-agents - exist.
#
# Step B: wrap each compiled sub-agent as a plain @tool function - from
#         the supervisor's perspective, calling "person_info_agent_tool"
#         looks EXACTLY like calling any other tool. The supervisor never
#         sees search_tool/weather_tool/currency_tool directly - only the
#         3 wrapped sub-agents.
#
# Step C: the supervisor is built with the EXACT SAME loop structure as
#         Part 1's single agent - just bound to 3 sub-agents instead of 3
#         raw tools. This is the point: single-agent and multi-agent use
#         the SAME mechanical pattern (agent <-> tools loop) - what
#         differs is WHAT is being called as a "tool".
# ===========================================================================

def _make_one_tool_agent(bound_tool):
    """Builds and compiles ONE single-tool agent - identical shape to
    build_single_agent() above, just with a list of exactly 1 tool."""
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

    return graph.compile()  # <-- graph.compile() #1, #2, #3 - one per sub-agent


def _run_subagent_and_get_final_text(compiled_subagent, subtask: str) -> str:
    """Runs a compiled sub-agent graph to completion on one subtask, and
    returns only its final text answer - the supervisor never sees HOW
    many tool calls happened inside, or any intermediate reasoning."""
    result = compiled_subagent.invoke({"messages": [HumanMessage(content=subtask)]})
    return result["messages"][-1].content


def build_multi_agent():
    # --- Step A: 3 separately-compiled, single-tool sub-agents -----------
    person_info_agent = _make_one_tool_agent(search_tool)
    finance_agent = _make_one_tool_agent(currency_tool)
    weather_agent = _make_one_tool_agent(weather_tool)

    # --- Step B: wrap each compiled sub-agent as a plain tool -------------
    # This is "agent-as-a-tool" - the supervisor calls these exactly like
    # it would call search_tool/weather_tool/currency_tool directly, but
    # each one is secretly a full agent with its own internal loop.
    @tool
    def person_info_agent_tool(subtask: str) -> str:
        """Delegate to the Person Info agent - it can look up facts about
        a person, such as where they live or where they were born."""
        return _run_subagent_and_get_final_text(person_info_agent, subtask)

    @tool
    def finance_agent_tool(subtask: str) -> str:
        """Delegate to the Finance agent - it can look up the currency
        exchange rate for a given country."""
        return _run_subagent_and_get_final_text(finance_agent, subtask)

    @tool
    def weather_agent_tool(subtask: str) -> str:
        """Delegate to the Weather agent - it can look up the current
        weather for a given city."""
        return _run_subagent_and_get_final_text(weather_agent, subtask)

    supervisor_tools = [person_info_agent_tool, finance_agent_tool, weather_agent_tool]

    # --- Step C: the supervisor - SAME loop shape as Part 1, bound to
    # sub-agents instead of raw tools -------------------------------------
    supervisor_llm = llm.bind_tools(supervisor_tools)

    def supervisor_node(state: MessagesState):
        response = supervisor_llm.invoke(state["messages"])
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

    return graph.compile()  # <-- graph.compile() #4 - the top-level supervisor graph
    # Total for the multi-agent version: 4 separate graph.compile() calls
    # (3 sub-agents + 1 supervisor), versus Part 1's single graph.compile().


# ===========================================================================
# Run both versions on the SAME question, so the transcripts can be
# compared directly.
# ===========================================================================

def _print_transcript(label: str, messages) -> None:
    print(f"\n{'=' * 20} {label} - full message transcript {'=' * 20}")
    for m in messages:
        role = m.__class__.__name__
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            print(f"[{role}] requested tool call(s): {[tc['name'] for tc in tool_calls]}")
        else:
            content = getattr(m, "content", "")
            print(f"[{role}] {content}")


if __name__ == "__main__":
    print("Question:", TEST_QUESTION)

    single_agent = build_single_agent()
    single_result = single_agent.invoke({"messages": [HumanMessage(content=TEST_QUESTION)]})
    _print_transcript("SINGLE AGENT", single_result["messages"])
    print("\nSINGLE AGENT final answer:", single_result["messages"][-1].content)

    multi_agent = build_multi_agent()
    multi_result = multi_agent.invoke({"messages": [HumanMessage(content=TEST_QUESTION)]})
    _print_transcript("MULTI AGENT (supervisor only - sub-agent internals are hidden)", multi_result["messages"])
    print("\nMULTI AGENT final answer:", multi_result["messages"][-1].content)





# python "Other_Documents/Learning_Examples/single_vs_multi_agent_demo.py"
