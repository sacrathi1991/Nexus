"""
nl2sql/access_control.py

Step 11 of the build sequence: row-level access control. This file's only
job is to tell Postgres WHO is asking, on the connection about to run a
query - the actual row-filtering decision is made entirely by the RLS
policy in nl2sql/rls_setup.sql, not by any code here or anywhere else in
this app. That's the whole point: enforcement lives in the database, so
neither the semantic layer nor the LLM's free-form SQL generation can
accidentally (or deliberately, via a bad prompt) bypass it.

A "user context" here is just {"role": "admin" | "employee",
"employee_id": int | None} - there's no real auth system in this project
yet (same "no auth built yet" state as everywhere else), so this is
supplied directly by the CLI trigger (run_nl2sql_query.py) as a stand-in
for what a real login session would provide later.

Called from: services/nl2sql_service.py and nl2sql/semantic_layer.py,
             immediately after opening a connection on northwind_query_engine
             and BEFORE running any actual query on it.
"""

from sqlalchemy import text
from sqlalchemy.engine import Connection

ADMIN_ROLE = "admin"
EMPLOYEE_ROLE = "employee"


def apply_user_context(conn: Connection, user_context: dict) -> None:
    """Set this connection's session variables so the orders_access_policy
    RLS policy can read them via current_setting(...).

    user_context: {"role": "admin" | "employee", "employee_id": int | None}
    If user_context is missing entirely (None), BOTH variables are left
    unset - the policy's default-deny behavior then applies, exactly as
    intended for a request that never identified itself.
    """
    if user_context is None:
        return

    role = user_context.get("role", EMPLOYEE_ROLE)
    employee_id = user_context.get("employee_id")

    # set_config(name, value, is_local) - is_local=false means this applies
    # for the rest of THIS connection's session, same lifetime as the
    # statement_timeout already being set elsewhere on this same connection.
    conn.execute(text("SELECT set_config('app.user_role', :role, false)"), {"role": role})
    conn.execute(
        text("SELECT set_config('app.current_employee_id', :employee_id, false)"),
        {"employee_id": str(employee_id) if employee_id is not None else ""},
    )
