"""
nl2sql/schema_introspector.py

Step "a" and "b" of the schema-retrieval approach (RAG over schema): reads
the ACTUAL table/column/foreign-key structure of the Northwind database
directly from Postgres's own metadata catalog — nothing here is hardcoded
or guessed, so if the schema ever changes, re-running this picks up the
change automatically.

This plays the same role for NL2SQL that ingestion/loader.py plays for
RAG: it is the "load" step, just loading DB schema metadata instead of PDF
text. The output of this file (a list of table descriptions) is what
services/schema_ingestion_service.py embeds and saves next.

Called from: services/schema_ingestion_service.py
"""

from sqlalchemy import inspect

from nl2sql.db_engine import northwind_engine


def introspect_schema() -> list[dict]:
    """Read every table's columns and foreign keys from Postgres's catalog.

    Returns a list of dicts, one per table:
        {
            "table_name": "orders",
            "columns": [{"name": "order_id", "type": "INTEGER", "primary_key": True}, ...],
            "foreign_keys": [{"column": "customer_id", "references_table": "customers",
                               "references_column": "customer_id"}, ...],
        }

    Called from: services/schema_ingestion_service.py, once per ingestion run.
    """
    inspector = inspect(northwind_engine)
    tables_info = []

    for table_name in inspector.get_table_names():
        # get_columns() returns every column's name/type/nullable for this table
        raw_columns = inspector.get_columns(table_name)

        # get_pk_constraint() returns which column(s) form the primary key,
        # so we can mark them in the output below
        pk_columns = set(inspector.get_pk_constraint(table_name)["constrained_columns"])

        columns = [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "primary_key": col["name"] in pk_columns,
            }
            for col in raw_columns
        ]

        # get_foreign_keys() returns this table's relationships to OTHER
        # tables — e.g. orders.customer_id -> customers.customer_id. This is
        # exactly the join information an LLM needs to write correct SQL.
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            # a composite FK could reference multiple columns; zip pairs
            # each local column with its corresponding referenced column
            for local_col, remote_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                foreign_keys.append({
                    "column": local_col,
                    "references_table": fk["referred_table"],
                    "references_column": remote_col,
                })

        tables_info.append({
            "table_name": table_name,
            "columns": columns,
            "foreign_keys": foreign_keys,
        })

    return tables_info


def format_table_schema(table_info: dict) -> str:
    """Turn one table's structured metadata into a plain-English description
    an LLM can read directly — this is the "document" that gets embedded,
    the same role a PDF's extracted text plays in ingestion/loader.py.

    Example output:
        Table: orders
        Columns: order_id (PK, INTEGER), customer_id (INTEGER), order_date (DATE), ...
        Foreign Keys: customer_id references customers.customer_id

    Called from: services/schema_ingestion_service.py, once per table
    returned by introspect_schema().
    """
    table_name = table_info["table_name"]

    column_parts = []
    for col in table_info["columns"]:
        label = f"{col['name']} ({'PK, ' if col['primary_key'] else ''}{col['type']})"
        column_parts.append(label)
    columns_text = ", ".join(column_parts)

    if table_info["foreign_keys"]:
        fk_parts = [
            f"{fk['column']} references {fk['references_table']}.{fk['references_column']}"
            for fk in table_info["foreign_keys"]
        ]
        fk_text = "Foreign Keys: " + ", ".join(fk_parts)
    else:
        fk_text = "Foreign Keys: none"

    return f"Table: {table_name}\nColumns: {columns_text}\n{fk_text}"
