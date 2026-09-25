"""
langgraph_playground.py

Everything in ONE file, the way you'd build it while first learning LangGraph:
LLM instance -> tools -> agent node -> conditional routing -> compile -> invoke.

Each block below has a comment marking which folder/file it would move into
once this gets split up into the layered structure we discussed
(api/ -> services/ -> llm/{prompts,tools,graphs}/).

Run this file directly:  python langgraph_playground.py
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE


# ---------------------------------------------------------------------------
# 1. LLM INSTANCE
# FUTURE HOME: llm/llm_client.py
# One shared LLM object, built once from config.py, imported everywhere else
# that needs it (services, graphs, tools if they ever need an LLM directly).
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
)


# ---------------------------------------------------------------------------
# 2. TOOLS
# FUTURE HOME: llm/tools/  -> one file per tool (or per closely related group)
#   add + multiply      -> llm/tools/math_tools.py
#   get_exchange_rate    -> llm/tools/exchange_tool.py
#   get_city_temperature -> llm/tools/weather_tool.py
#
# A "tool" is just a normal Python function with a @tool decorator and a
# docstring. The docstring is NOT a comment for humans here — the LLM reads
# it to decide whether/when to call this function. Keep it precise.
# ---------------------------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the product."""
    return a * b


@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Get the exchange rate from base_currency to target_currency (e.g. USD to INR)."""
    # Stub data for learning purposes.
    # Real version (later) would call a live exchange-rate API.
    fake_rates = {
        ("USD", "INR"): 83.2,
        ("EUR", "INR"): 90.1,
        ("USD", "EUR"): 0.92,
    }
    rate = fake_rates.get((base_currency.upper(), target_currency.upper()))
    if rate is None:
        return f"No rate available for {base_currency} to {target_currency}."
    return f"1 {base_currency.upper()} = {rate} {target_currency.upper()}"


@tool
def get_city_temperature(city: str) -> str:
    """Get the current temperature for a given city."""
    # Stub data for learning purposes.
    # Real version (later) would call a live weather API.
    fake_temps = {
        "delhi": "34°C",
        "mumbai": "31°C",
        "bangalore": "26°C",
        "new york": "18°C",
    }
    temp = fake_temps.get(city.lower())
    if temp is None:
        return f"No temperature data available for {city}."
    return f"The current temperature in {city.title()} is {temp}."


# All tools in one list, so the LLM knows every function it's allowed to call.
tools = [add, multiply, get_exchange_rate, get_city_temperature]


# ---------------------------------------------------------------------------
# 3. BIND TOOLS TO THE LLM
# FUTURE HOME: llm/graphs/main_graph.py (this is graph-setup code, not a
# standalone concept of its own — it lives right next to the nodes that use it)
#
# bind_tools() doesn't call anything yet. It just tells the LLM "here is the
# menu of functions you're allowed to ask for, with their names and expected
# arguments." The LLM decides at runtime whether to use one, based on the
# user's question.
# ---------------------------------------------------------------------------
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 4. NODES
# FUTURE HOME: llm/graphs/main_graph.py
#
# A node is just a function that takes the current state and returns an
# update to it. MessagesState is a built-in LangGraph state shape that's
# just {"messages": [...]} — a running conversation list.
# ---------------------------------------------------------------------------

def agent_node(state: MessagesState) -> dict:
    """The 'thinking' node — asks the LLM to respond, possibly by calling a tool."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# We don't hand-write a "tools node" — LangGraph provides a ready-made one
# (ToolNode) that automatically executes whichever tool the LLM asked for
# and feeds the result back into the conversation.
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 5. GRAPH DEFINITION + CONDITIONAL ROUTING
# FUTURE HOME: llm/graphs/main_graph.py
#
# Flow:
#   START -> agent_node -> (did the LLM ask for a tool?)
#       yes -> tool_node -> back to agent_node (so it can use the tool's result)
#       no  -> END (the LLM gave a final answer, nothing left to do)
#
# tools_condition is a built-in LangGraph helper that checks the last message
# for a tool call and routes accordingly — this is the same idea as the
# route_decision() function we wrote by hand in the earlier RAG-vs-SQL
# example, just provided for us since "did the LLM call a tool" is such a
# common check.
# ---------------------------------------------------------------------------
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")


# ---------------------------------------------------------------------------
# 6. COMPILE
# FUTURE HOME: llm/graphs/main_graph.py (this is the object that gets
# imported by services/chat_service.py — everything above this line stays
# "private" to the llm/graphs/ folder)
# ---------------------------------------------------------------------------
compiled_graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# 7. CALL IT
# FUTURE HOME: THIS PART DOES NOT MOVE INTO llm/ AT ALL.
#
# This is exactly the job of services/chat_service.py + api/chat_router.py:
#   - services/chat_service.py would do:
#         from llm.graphs.main_graph import compiled_graph
#         def handle_chat(question):
#             result = compiled_graph.invoke({"messages": [HumanMessage(content=question)]})
#             return result["messages"][-1].content
#
#   - api/chat_router.py would do:
#         @router.post("/chat")
#         def chat(request: ChatRequest):
#             answer = handle_chat(request.question)
#             return ChatResponse(answer=answer)
#
# i.e. this __main__ block is standing in for "a request arriving" — in the
# real app, a FastAPI router calls a service, which calls compiled_graph,
# instead of us calling it directly at the bottom of the file like this.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    question = "What is 12 plus 30, and what's the exchange rate from USD to INR?"
    print(f"Question: {question}\n")

    result = compiled_graph.invoke({"messages": [HumanMessage(content=question)]})

    final_answer = result["messages"][-1].content
    print(f"Answer: {final_answer}")

    print("\n--- Full message trace (see how the tool calls happened) ---")
    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content!r}")
