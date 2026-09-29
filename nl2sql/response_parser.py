"""
nl2sql/response_parser.py

Step 7 of the build sequence: ambiguity handling. Parses the LLM's raw
text response (from either NL2SQL_PROMPT or SQL_CORRECTION_PROMPT) into
one of four outcomes, based on the CLARIFY: / ASSUMPTION: protocol those
prompts were told to follow:

    - "no_query"   -> the question can't be answered from the given tables at all
    - "clarify"     -> the question is too ambiguous to safely guess - ask the user
    - "assumption"  -> the LLM picked a reasonable interpretation and says which
    - "sql"         -> the question wasn't ambiguous, plain SQL was returned

Kept in its own file, separate from the prompts and the service, so this
text-parsing logic can be tested and changed independently of both.

Called from: services/nl2sql_service.py, right after every LLM call that
             is expected to return SQL (the first generation AND every
             correction retry).
"""


def parse_llm_sql_response(raw_text: str) -> dict:
    """Returns a dict: {"mode": str, "sql": str | None, "assumption": str | None,
    "clarification_question": str | None}
    """
    text = raw_text.strip()

    if text.upper() == "NO_QUERY":
        return {"mode": "no_query", "sql": None, "assumption": None, "clarification_question": None}

    if text.upper().startswith("CLARIFY:"):
        question_text = text.split(":", 1)[1].strip()
        return {"mode": "clarify", "sql": None, "assumption": None, "clarification_question": question_text}

    if text.upper().startswith("ASSUMPTION:"):
        # First line is the assumption; everything after the first blank
        # line (or just the next lines, if the LLM didn't add one) is SQL.
        first_line, _, rest = text.partition("\n")
        assumption_text = first_line.split(":", 1)[1].strip()
        sql_text = rest.strip()
        return {"mode": "assumption", "sql": sql_text, "assumption": assumption_text, "clarification_question": None}

    # No special prefix - the question wasn't ambiguous, this is plain SQL.
    return {"mode": "sql", "sql": text, "assumption": None, "clarification_question": None}
