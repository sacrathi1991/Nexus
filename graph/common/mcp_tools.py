"""
graph/common/mcp_tools.py

WHY THIS FILE EXISTS: everything in agent_tools.py wraps NEXUS'S OWN
pipelines (ask()/ask_db()) as tools. This file is different - it wraps
SOMEONE ELSE'S already-built, already-hosted capability (Tavily's web
search/extract/map/crawl) as tools, by acting as an MCP CLIENT connecting
to Tavily's MCP SERVER. Nexus writes none of the underlying logic here;
Tavily's own infrastructure does the actual work. This is Step 15 -
Nexus's first MCP integration, kept separate from agent_tools.py because
the two have a fundamentally different relationship to the tools they
produce: agent_tools.py tools are Nexus's own code; these are a thin,
reusable bridge to an external system's already-published MCP server.

Transport: remote (streamable HTTP), not a local stdio subprocess - no
Node.js is installed on this machine, and a remote connection needs
nothing beyond an API key and a URL, with no process for Nexus to launch
or manage.

Called from: graph/v3_multi_agent/build_multi_agent_graph.py, which adds
             these tools directly into supervisor_tools - NOT wrapped in
             their own sub-agent/graph.compile(). Per the "does this tool
             need its own internal reasoning loop" test worked through
             for this project: Tavily's operations are each a single,
             one-shot call (search once, extract one URL, etc.) with no
             benefit from an agent loop around them, so they're bound
             directly to the supervisor exactly like a plain tool.
"""

import asyncio
import os

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import TAVILY_API_KEY

# Local stdio transport - Tavily's REMOTE MCP server (streamable HTTP)
# hung indefinitely on tools/list during testing (initialize succeeded,
# every call after it never got a response, with both query-param and
# Bearer auth, reproduced even bypassing the SDK with raw HTTP requests -
# a genuine issue in Tavily's hosted MCP bridge, not this project's code).
# The LOCAL package (tavily-mcp, run via npx) is a SEPARATE, independently
# maintained implementation of the same bridge - still calls Tavily's
# real search API over the network either way (see this file's own
# earlier version/Day 12 doc for why "local" doesn't mean the underlying
# work is local), but it's different code, so it isn't guaranteed to
# share the remote bridge's bug.
_NODE_DIR = r"C:\Program Files\nodejs"
_NPX_PATH = os.path.join(_NODE_DIR, "npx.cmd")

# Cached after the first fetch - the MCP discovery handshake (connect,
# then tools/list) only needs to happen ONCE per process, since Tavily's
# 4 tools and their schemas don't change between questions. Without this,
# build_multi_agent_graph() being called fresh on every cache-miss
# question would repeat the handshake every single time, for no benefit.
_cached_tavily_tools = None


async def _fetch_tavily_tools() -> list:
    """The actual MCP client work: connect to Tavily's remote server,
    run the discovery handshake, get back LangChain-compatible tool
    objects for all 4 of Tavily's tools (search, extract, map, crawl) -
    this is langchain-mcp-adapters doing the translation discussed
    conceptually in Day 12's playbook doc: MCP's tools/list response,
    converted into the exact tool shape llm.bind_tools() already expects,
    so these tools are indistinguishable from rag_agent_tool/
    nl2sql_agent_tool once returned.
    """
    env = dict(os.environ)
    env["PATH"] = _NODE_DIR + os.pathsep + env.get("PATH", "")
    env["TAVILY_API_KEY"] = TAVILY_API_KEY

    client = MultiServerMCPClient({
        "tavily": {
            "transport": "stdio",
            "command": _NPX_PATH,
            "args": ["-y", "tavily-mcp@latest"],
            "env": env,
        }
    })
    return await client.get_tools()


def _make_sync_compatible(async_tool: StructuredTool) -> StructuredTool:
    """langchain-mcp-adapters only implements an ASYNC execution path for
    MCP tools (every call is a real network request to the MCP server) -
    there is no sync _run at all. That's invisible during bind_tools()/
    discovery, but LangGraph's ToolNode calls tools SYNCHRONOUSLY by
    default, which is exactly what rag_agent_tool/nl2sql_agent_tool (both
    plain sync functions) never had a problem with - mixing in an
    async-only tool breaks ToolNode's sync path with
    "StructuredTool does not support sync invocation".

    The fix: wrap the async tool in a new StructuredTool whose sync `func`
    just runs the original async tool to completion via asyncio.run().
    ToolNode's sync tool calls execute inside a worker thread (confirmed
    via the traceback hit during testing), not the main thread's event
    loop, so asyncio.run() creating a fresh event loop there is safe - no
    "event loop already running" conflict. Same async-to-sync boundary-
    crossing principle as get_tavily_tools() itself, just one level
    deeper (execution, not discovery), and kept contained here for the
    same reason: the rest of v3 never has to think about async at all.
    """
    def sync_func(**kwargs):
        return asyncio.run(async_tool.ainvoke(kwargs))

    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=sync_func,
    )


def get_tavily_tools() -> list:
    """Sync wrapper, since build_multi_agent_graph() and everything else
    in v3 is synchronous - MCP's discovery step is async by nature (it's
    a real network call), so this is the one place that boundary gets
    crossed, via asyncio.run(), kept contained to this file so the rest
    of v3's code never has to think about async at all. Each returned
    tool is also sync-compatible (see _make_sync_compatible) so ToolNode
    can execute them the same way it executes every other supervisor tool.
    """
    global _cached_tavily_tools
    if _cached_tavily_tools is None:
        async_tools = asyncio.run(_fetch_tavily_tools())
        _cached_tavily_tools = [_make_sync_compatible(t) for t in async_tools]
    return _cached_tavily_tools
