"""
graph/common/text_utils.py

WHY THIS FILE EXISTS: Gemini's response.content can come back as a plain
string OR as a list of structured content blocks (e.g.
[{"type": "text", "text": "..."}]) - this has already caused a real bug
in this project twice before (services/query_service.py,
services/nl2sql_service.py both had to add their own local copy of this
same normalization). It was missed in graph/v2_single_agent and
graph/v3_multi_agent when they were first built, causing
"AttributeError: 'list' object has no attribute 'lower'" the first time
v3 was run end-to-end for real. Centralizing it here - genuinely shared
by v1, v2, and v3 - means this fix only needs to happen in one place.

What it does: extract_text(content) normalizes an LLM response's content
into a plain string, regardless of which shape Gemini happened to return.

Called from: graph/v1_simple_router/router.py, graph/v2_single_agent/
             build_agent_graph.py, graph/v3_multi_agent/
             build_multi_agent_graph.py - anywhere a raw message's
             .content needs to be read as plain text.
"""


def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()
