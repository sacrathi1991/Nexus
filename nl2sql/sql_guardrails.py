"""
nl2sql/sql_guardrails.py

The safety layer between "the LLM generated some SQL text" and "that SQL
actually runs against the database" (steps 4-5 of the NL2SQL build
sequence). An LLM-generated query must NEVER be trusted to run as-is - this
is the single most important file in the NL2SQL pipeline from a security
standpoint.

Three checks, applied in order:
    1. Must be a read-only statement (SELECT / WITH ... SELECT only).
    2. Must not contain any data-modifying or schema-modifying keyword,
       anywhere in the query (catches a modifying statement smuggled
       inside a subquery or CTE, not just at the start of the string).
    3. Must have a LIMIT clause - one is added automatically if missing,
       so a mistakenly unbounded query can't return millions of rows.

Called from: services/nl2sql_service.py, immediately after the LLM
             generates SQL and before it is ever executed.
"""

import re

# Every keyword here would modify data or schema - none of them should
# EVER appear in a query this pipeline is allowed to run. Checked with word
# boundaries (\b) so this doesn't misfire on a column/table name that merely
# CONTAINS one of these words (e.g. a column called "updated_at").
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "REPLACE", "MERGE",
]

_DEFAULT_ROW_LIMIT = 100


class UnsafeSQLError(ValueError):
    """Raised when generated SQL fails a safety check - caught by the
    caller and turned into a user-facing 'couldn't run that query' message,
    never allowed to propagate into an actual DB execution attempt."""


def validate_and_prepare_sql(sql: str, max_rows: int = _DEFAULT_ROW_LIMIT) -> str:
    """Check generated SQL is safe to run, and add a row limit if missing.

    Returns the (possibly modified) SQL string, ready to execute.
    Raises UnsafeSQLError if the query fails any safety check.
    """
    cleaned = sql.strip().rstrip(";")
    upper = cleaned.upper()

    # --- Check 1: must start with a read-only statement ---------------------
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise UnsafeSQLError(
            f"Generated statement is not a SELECT/WITH query - refusing to run it: {cleaned[:100]}"
        )

    # --- Check 2: no modifying keyword anywhere in the query ----------------
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise UnsafeSQLError(
                f"Generated query contains forbidden keyword '{keyword}' - refusing to run it."
            )

    # --- Check 3: enforce a row limit ----------------------------------------
    # A crude but effective check - if LIMIT doesn't appear at all, append
    # one. This won't catch every edge case (e.g. LIMIT inside a subquery
    # but not on the outer query), but it stops the common failure mode: an
    # unbounded SELECT accidentally returning an entire large table.
    if not re.search(r"\bLIMIT\b", upper):
        cleaned = f"{cleaned} LIMIT {max_rows}"

    return cleaned
