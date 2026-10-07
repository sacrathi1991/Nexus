"""
services/nl2sql_service.py

Orchestrates the NL2SQL query pipeline (steps 3-6 of the build sequence):
retrieve relevant tables -> build the prompt -> LLM generates SQL ->
validate it's safe -> execute against Northwind -> if it fails, feed the
error back to the LLM and retry, up to MAX_RETRIES times -> return the
result.

This plays the same role for NL2SQL that services/query_service.py plays
for RAG - one function, ask_db(), that a trigger (a manual script now, an
API route later) calls. The trigger changes; this doesn't.

Called from: run_nl2sql_query.py
"""

import json

from sqlalchemy import text

from nl2sql.schema_retriever import retrieve_relevant_tables
from nl2sql.sql_prompt import NL2SQL_PROMPT, SQL_CORRECTION_PROMPT, ANSWER_PROMPT
from nl2sql.sql_guardrails import validate_and_prepare_sql, UnsafeSQLError
from nl2sql.response_parser import parse_llm_sql_response
from nl2sql.db_engine import northwind_query_engine
from nl2sql.access_control import apply_user_context
from nl2sql.semantic_layer import match_metric, missing_required_parameters, execute_metric, format_response
from nl2sql.metrics_catalog import get_metric
from nl2sql.audit_log import log_query
from llm.llm_client import llm

# How long, in milliseconds, a generated query is allowed to run before
# Postgres kills it - protects against an accidentally expensive query
# (e.g. a missing join condition causing a cross join) hanging the app.
QUERY_TIMEOUT_MS = 5000

# How many CORRECTION attempts are allowed after the first generation -
# e.g. MAX_RETRIES=2 means up to 3 total tries (1 initial + 2 corrections)
# before giving up. Kept small on purpose: each retry is a full extra LLM
# call, so this bounds both latency and cost for a question that's
# fundamentally unanswerable, rather than looping indefinitely.
MAX_RETRIES = 2


def _extract_text(content) -> str:
    """Normalize the LLM's response into a plain string - same shape issue
    as query_service.py's _extract_text(): Gemini's response.content can be
    a plain string OR a list of {"type": "text", "text": ...} blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()


def _generate_answer(question: str, rows: list[dict]) -> str:
    """Phrase raw result rows as a natural sentence, for the FREE-FORM path
    only - unlike the semantic layer's format_response(), this path's
    result shape is unpredictable (whatever columns the LLM decided to
    SELECT), so a templated string can't be pre-written for it; this needs
    an actual LLM call, same as query_service.py's RAG answer generation.
    """
    prompt = ANSWER_PROMPT.format(question=question, rows=json.dumps(rows, default=str))
    response = llm.invoke(prompt)
    return _extract_text(response.content)


def _run_sql(sql: str, user_context: dict = None) -> list[dict]:
    """Execute one already-guardrail-checked SQL string against Northwind.

    Left as its own small function (rather than inlined in the retry loop)
    so the loop body below reads as "generate, check, RUN, handle failure"
    without the connection/timeout details cluttering it.

    Runs on northwind_query_engine (the restricted nl2sql_app role) and
    identifies itself via apply_user_context() (step 11 - row-level access
    control) BEFORE running the LLM-generated SQL - this is what stops the
    free-form generation path from being able to see rows the current
    user isn't allowed to see, regardless of what SQL the LLM wrote.
    """
    with northwind_query_engine.connect() as conn:
        # SET (not SET LOCAL) applies for this connection's lifetime, which
        # is exactly one `with` block here - it never affects any other
        # query elsewhere, since the connection is closed right after.
        conn.execute(text(f"SET statement_timeout = {QUERY_TIMEOUT_MS}"))
        apply_user_context(conn, user_context)
        result = conn.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]


def ask_db(question: str, user_context: dict = None) -> dict:
    """Answer a question - the semantic layer (step 9) is tried FIRST;
    free-form NL2SQL generation (steps 3-7) is only used as a fallback
    when no catalog metric fits. See _ask_db_semantic_layer() and
    _ask_db_freeform() for each path's own docstring.

    user_context (step 11 - row-level access control):
        {"role": "admin" | "employee", "employee_id": int | None}
    Identifies WHO is asking, so Postgres's row-level security policy on
    `orders` can restrict results accordingly - enforced by the database
    itself, on BOTH paths below, regardless of which one answers. Passing
    None means "no identity" - the policy then default-denies every row.

    Every returned dict includes "source" ("semantic_layer" or
    "nl2sql_generation") and "metric_used" (the matched metric's name, or
    None on the free-form path) - so a caller can always tell which path
    actually answered a given question.
    """
    match = match_metric(question)

    if match["metric"] is not None:
        missing = missing_required_parameters(match["metric"], match["parameters"])
        if not missing:
            result = _ask_db_semantic_layer(question, match["metric"], match["parameters"], user_context)
            log_query(question, user_context, result)
            return result
        # A metric matched, but a REQUIRED parameter couldn't be extracted
        # from the question (e.g. "how many orders for this customer" with
        # no customer named) - falling back to guess the parameter would
        # be worse than falling back to free-form generation, which can at
        # least ask for clarification (step 7) instead.

    result = _ask_db_freeform(question, user_context)
    log_query(question, user_context, result)
    return result


def _ask_db_semantic_layer(question: str, metric_name: str, parameters: dict, user_context: dict = None) -> dict:
    """Run a matched, pre-vetted metric - no LLM-generated SQL involved at
    all past the matching step, so there's nothing to guardrail-check or
    retry: the SQL was already fixed and reviewed in metrics_catalog.py."""
    rows = execute_metric(metric_name, parameters, user_context)
    # Templated, not an LLM call - see format_response()'s docstring for why
    # this path can afford to skip the extra API call the free-form path needs.
    answer = format_response(metric_name, parameters, rows)

    # Surface the metric's OWN sql_template, for the same reason the
    # free-form path surfaces its generated SQL - groundedness. This is
    # the fixed template (with :param placeholders), not the literal
    # substituted query psycopg2 sent - showing the actual parameter
    # values too would need threading them out of execute_metric()
    # separately, not done here to keep this simple.
    sql = get_metric(metric_name)["sql_template"]

    return {
        "question": question, "tables_used": [], "sql": sql, "rows": rows,
        "error": None, "attempts": [], "assumption": None, "clarification_question": None,
        "source": "semantic_layer", "metric_used": metric_name, "answer": answer,
    }


def _ask_db_freeform(question: str, user_context: dict = None) -> dict:
    """Answer a question by generating and running a SQL query, retrying
    with the database's own error message fed back to the LLM if the
    first attempt fails. This is steps 3-7 of the build sequence, called
    only when the semantic layer (step 9) found no matching metric.

    Returns a dict:
        {"question": str, "tables_used": list[str], "sql": str | None,
         "rows": list[dict], "error": str | None, "attempts": list[dict],
         "assumption": str | None, "clarification_question": str | None,
         "source": "nl2sql_generation", "metric_used": None, "answer": str | None}

    "answer" (step 10) is only set on a SUCCESSFUL result - an LLM-phrased
    sentence built from the actual rows returned. It is left None whenever
    there's nothing to phrase: NO_QUERY, a clarification request, or a
    failure after every retry - in each of those cases "error" or
    "clarification_question" already carries the user-facing message.

    "attempts" records every try made (its SQL and outcome), in order -
    useful for debugging *why* a question needed correction, and for a
    future eval harness to measure how often retries were even needed.
    "error" is set (and "rows" is empty) only once every attempt has been
    exhausted, or the LLM declined outright (NO_QUERY) - a caller (e.g. a
    future API route) always gets a normal dict back, never a crash.

    "clarification_question" is set (with no SQL run at all) when the
    question was too ambiguous to safely guess at - the caller should show
    this question to the user and ask again, rather than treat it as a
    failure. "assumption" is set alongside a normal successful result when
    the LLM proceeded on a stated, reasonable interpretation instead of
    asking - the caller should show this to the user so they know what was
    assumed on their behalf.
    """
    # --- Step 1: retrieve the tables relevant to this question -------------
    tables = retrieve_relevant_tables(question)
    table_schemas = "\n\n".join(t["schema_text"] for t in tables)
    tables_used = [t["table_name"] for t in tables]

    attempts = []
    last_sql = None
    last_error = None

    for attempt_num in range(1, MAX_RETRIES + 2):
        # --- Generate: first try uses the normal prompt, every retry uses
        # the correction prompt with the previous SQL + error attached ----
        if attempt_num == 1:
            prompt = NL2SQL_PROMPT.format(table_schemas=table_schemas, question=question)
        else:
            prompt = SQL_CORRECTION_PROMPT.format(
                table_schemas=table_schemas, question=question,
                failed_sql=last_sql, error_message=last_error,
            )
        response = llm.invoke(prompt)
        raw_response = _extract_text(response.content)

        # --- Ambiguity handling (step 7): figure out which of the 4 modes
        # the LLM responded with, BEFORE assuming it returned plain SQL ----
        parsed = parse_llm_sql_response(raw_response)

        if parsed["mode"] == "no_query":
            attempts.append({"attempt": attempt_num, "sql": None, "error": "LLM returned NO_QUERY"})
            return {
                "question": question, "tables_used": tables_used, "sql": None, "rows": [],
                "error": "The retrieved tables don't contain enough information to answer this question.",
                "attempts": attempts, "assumption": None, "clarification_question": None,
                "source": "nl2sql_generation", "metric_used": None, "answer": None,
            }

        if parsed["mode"] == "clarify":
            # Not a failure to retry - the LLM is explicitly saying it
            # cannot safely guess. No SQL was even generated, so there's
            # nothing to correct; return immediately and let the caller
            # show this question to the user.
            attempts.append({"attempt": attempt_num, "sql": None, "error": None})
            return {
                "question": question, "tables_used": tables_used, "sql": None, "rows": [],
                "error": None, "attempts": attempts, "assumption": None,
                "clarification_question": parsed["clarification_question"],
                "source": "nl2sql_generation", "metric_used": None, "answer": None,
            }

        generated_sql = parsed["sql"]
        assumption = parsed["assumption"]  # None unless mode == "assumption"

        # --- Guardrails, BEFORE this SQL ever touches the DB ----------------
        try:
            safe_sql = validate_and_prepare_sql(generated_sql)
        except UnsafeSQLError as e:
            attempts.append({"attempt": attempt_num, "sql": generated_sql, "error": str(e)})
            last_sql, last_error = generated_sql, str(e)
            continue  # try again with the correction prompt, if attempts remain

        # --- Execute -----------------------------------------------------------
        try:
            rows = _run_sql(safe_sql, user_context)
        except Exception as e:
            attempts.append({"attempt": attempt_num, "sql": safe_sql, "error": str(e)})
            last_sql, last_error = safe_sql, str(e)
            continue  # try again with the correction prompt, if attempts remain

        # --- Success -------------------------------------------------------------
        attempts.append({"attempt": attempt_num, "sql": safe_sql, "error": None})
        answer = _generate_answer(question, rows)
        return {
            "question": question, "tables_used": tables_used, "sql": safe_sql,
            "rows": rows, "error": None, "attempts": attempts,
            "assumption": assumption, "clarification_question": None,
            "source": "nl2sql_generation", "metric_used": None, "answer": answer,
        }

    # Every attempt failed - surface the LAST error, but keep the full
    # history in "attempts" so it's clear this wasn't a single failure.
    return {
        "question": question, "tables_used": tables_used, "sql": last_sql, "rows": [],
        "error": f"Failed after {len(attempts)} attempt(s). Last error: {last_error}",
        "attempts": attempts, "assumption": None, "clarification_question": None,
        "source": "nl2sql_generation", "metric_used": None, "answer": None,
    }
