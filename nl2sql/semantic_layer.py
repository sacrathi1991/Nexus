"""
nl2sql/semantic_layer.py

Step 9 of the build sequence — the semantic layer's matching + execution
logic. The metrics themselves (their SQL, already written and vetted by a
human) live in nl2sql/metrics_catalog.py; this file is the runtime piece
that (a) asks the LLM which catalog metric fits a question, if any, and
(b) runs that metric's fixed SQL template with the extracted parameters
bound in - never string-interpolated.

Called from: services/nl2sql_service.py, as the FIRST thing tried for
             every question - only if nothing here matches does the
             pipeline fall back to free-form SQL generation
             (nl2sql/sql_prompt.py's NL2SQL_PROMPT).
"""

import json

from sqlalchemy import text

from nl2sql.metrics_catalog import METRICS, get_metric
from nl2sql.db_engine import northwind_query_engine
from nl2sql.access_control import apply_user_context
from llm.llm_client import llm
from nl2sql.sql_prompt import SEMANTIC_LAYER_PROMPT

QUERY_TIMEOUT_MS = 5000


def _extract_text(content) -> str:
    """Same normalization as nl2sql_service.py's _extract_text() - Gemini's
    response.content can be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_catalog_for_prompt() -> str:
    """Turn the METRICS list into plain text the LLM can read - name,
    description, and each parameter's name/type/required-ness."""
    blocks = []
    for metric in METRICS:
        param_lines = []
        for p in metric["parameters"]:
            required = "required" if p.get("required") else "optional"
            param_lines.append(f"  - {p['name']} ({p['type']}, {required}): {p['description']}")
        params_text = "\n".join(param_lines) if param_lines else "  (no parameters)"
        blocks.append(f"{metric['name']}: {metric['description']}\nParameters:\n{params_text}")
    return "\n\n".join(blocks)


def match_metric(question: str) -> dict:
    """Ask the LLM which catalog metric (if any) fits this question.

    Returns {"metric": str | None, "parameters": dict}. Returns
    {"metric": None, "parameters": {}} whenever nothing matches, the LLM
    named a metric that isn't actually in the catalog, or its response
    wasn't valid JSON - every one of those cases degrades to "no match",
    never to a crash, since the caller's job is simply to fall back to
    free-form SQL generation when this returns no match.
    """
    prompt = SEMANTIC_LAYER_PROMPT.format(
        metrics_catalog=_format_catalog_for_prompt(), question=question
    )
    response = llm.invoke(prompt)
    raw = _extract_text(response.content)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"metric": None, "parameters": {}}

    metric_name = parsed.get("metric")
    parameters = parsed.get("parameters") or {}

    if metric_name is None:
        return {"metric": None, "parameters": {}}

    if get_metric(metric_name) is None:
        # The LLM named a metric that doesn't exist in our catalog - treat
        # exactly like "no match" rather than trusting a hallucinated name.
        return {"metric": None, "parameters": {}}

    return {"metric": metric_name, "parameters": parameters}


def missing_required_parameters(metric_name: str, parameters: dict) -> list[str]:
    """Return the names of any REQUIRED parameters this metric needs that
    are missing (or null) in what the LLM extracted - used by the caller
    to decide whether it's actually safe to run this metric, or whether it
    should fall back to free-form SQL generation instead of guessing."""
    metric = get_metric(metric_name)
    missing = []
    for p in metric["parameters"]:
        if p.get("required") and parameters.get(p["name"]) is None:
            missing.append(p["name"])
    return missing


def execute_metric(metric_name: str, parameters: dict, user_context: dict = None) -> list[dict]:
    """Run one catalog metric's fixed SQL template with the given
    parameters bound in (never string-interpolated - see module docstring
    for why this is safe by construction, independent of the SQL
    guardrails used on the free-form generation path).

    Applies defaults for any optional parameter the metric defines but the
    question didn't specify a value for.

    Runs on northwind_query_engine (the restricted nl2sql_app role, NOT
    the superuser nexus_user) and identifies itself via apply_user_context()
    BEFORE running the metric's SQL - this is what makes the orders table's
    row-level security policy (step 11) actually apply to this query,
    exactly like it applies to the free-form path in nl2sql_service.py.
    """
    metric = get_metric(metric_name)

    bind_params = {}
    for p in metric["parameters"]:
        value = parameters.get(p["name"])
        if value is None and "default" in p:
            value = p["default"]
        bind_params[p["name"]] = value

    with northwind_query_engine.connect() as conn:
        conn.execute(text(f"SET statement_timeout = {QUERY_TIMEOUT_MS}"))
        apply_user_context(conn, user_context)
        result = conn.execute(text(metric["sql_template"]), bind_params)
        return [dict(row) for row in result.mappings().all()]


def format_response(metric_name: str, parameters: dict, rows: list[dict]) -> str:
    """Turn a metric's result rows into a natural-language sentence, using
    its response_template - NO extra LLM call needed, since the metric
    (and therefore its result shape) was already known in advance. This is
    step 10 (response shaping) for the semantic-layer path specifically;
    the free-form path (nl2sql_service.py's _ask_db_freeform) still needs
    an LLM call instead, since its result shape is unpredictable.
    """
    metric = get_metric(metric_name)

    if not rows:
        return "No data was found for this."

    template = metric["response_template"]

    if metric.get("multi_row"):
        lines = []
        for i, row in enumerate(rows, start=1):
            context = {**parameters, **row, "rank": i}
            lines.append(template.format(**context))
        return "\n".join(lines)

    row = rows[0]
    context = {**parameters, **row}

    # total_revenue's {year_display} isn't a real column or parameter name -
    # it's computed here so the template can read naturally either way
    # ("Total revenue was..." vs "Total revenue in 1997 was...").
    if metric_name == "total_revenue":
        year = parameters.get("year")
        context["year_display"] = f" in {year}" if year else ""

    return template.format(**context)
