"""
nl2sql/db_engine.py

TWO SQLAlchemy engines for the Northwind (NL2SQL source) database - kept
deliberately separate:

    northwind_engine     - connects as nexus_user (owns the tables, is a
                            superuser). Used ONLY for schema introspection
                            (schema_introspector.py) - reading table/column
                            metadata is not sensitive, and superuser
                            privileges are irrelevant there.

    northwind_query_engine - connects as nl2sql_app, a restricted,
                            non-superuser role with SELECT-only access.
                            Used for every EXECUTION of an actual query
                            (semantic_layer.py's execute_metric(), and
                            nl2sql_service.py's free-form SQL execution) -
                            this is what makes row-level security
                            (nl2sql/rls_setup.sql) actually take effect,
                            since Postgres always bypasses RLS for a
                            superuser or a table's owner, regardless of
                            any ENABLE/FORCE setting on the table itself.

This is intentionally separate from db/database.py's engine/SessionLocal,
which point at the `nexus` database (RAG/pgvector data) - two different
databases, two different engines, never mixed.
"""

from sqlalchemy import create_engine

from config import NORTHWIND_DATABASE_URL, NORTHWIND_APP_DATABASE_URL

_CONNECT_ARGS = {"connect_timeout": 10}
# Same reasoning as db/database.py's engine - without this, a connection
# attempt while Docker/Postgres is unreachable can hang indefinitely with
# no error instead of failing fast.

northwind_engine = create_engine(NORTHWIND_DATABASE_URL, connect_args=_CONNECT_ARGS)
northwind_query_engine = create_engine(NORTHWIND_APP_DATABASE_URL, connect_args=_CONNECT_ARGS)
