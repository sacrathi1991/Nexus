"""
nl2sql/audit_log.py

Step 12 of the build sequence: audit logging. Records every question
asked through the NL2SQL pipeline - who asked it, which path answered it,
what SQL actually ran, how many rows came back, and whether it succeeded
or failed. This is a permanent, append-only record (never updated or
deleted), same principle as ingestion_jobs on the RAG side.

Why this matters beyond just debugging: once row-level access control
(step 11) exists, "who is allowed to see what" is being enforced silently
by Postgres - this log is what lets an admin later ANSWER "who actually
looked at what," which is exactly the kind of question a compliance
review or a security incident asks. Access control decides what CAN be
seen; audit logging records what WAS seen.

Called from: services/nl2sql_service.py's ask_db(), exactly once per
             question, right before it returns - covers both the
             semantic-layer path and the free-form generation path,
             success or failure, from one call site.
"""

from db.database import SessionLocal
from db.models import NL2SQLAuditLog


def log_query(question: str, user_context: dict, result: dict) -> None:
    """Insert one audit row for a completed ask_db() call.

    Never raises - a logging failure must not break the actual answer
    the user is waiting on. Any error here is printed, not propagated.
    """
    user_context = user_context or {}

    try:
        db = SessionLocal()
        db.add(NL2SQLAuditLog(
            question=question,
            user_role=user_context.get("role"),
            employee_id=user_context.get("employee_id"),
            source=result["source"],
            metric_used=result["metric_used"],
            sql_text=result["sql"],
            row_count=len(result["rows"]) if result["rows"] else 0,
            retries_used=max(len(result["attempts"]) - 1, 0),
            error=result["error"],
        ))
        db.commit()
        db.close()
    except Exception as e:
        # Logging is a side effect, not the main task - a broken audit
        # table should never be the reason a user doesn't get their answer.
        print(f"  [audit_log] failed to record query: {e}")
