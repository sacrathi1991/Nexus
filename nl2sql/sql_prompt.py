"""
nl2sql/sql_prompt.py

The NL2SQL equivalent of llm/prompts.py's RAG_PROMPT — turns a question +
retrieved table schemas into one prompt asking the LLM for a SQL query.

Called from: services/nl2sql_service.py
"""

from langchain_core.prompts import PromptTemplate

NL2SQL_PROMPT = PromptTemplate.from_template(
    """You are a SQL generator for a PostgreSQL database. Using ONLY the
tables described below, write a single SQL query that answers the question.

The question may be AMBIGUOUS - it could reasonably be interpreted more
than one way (e.g. "sales" could mean total revenue or number of units
sold; a year or date range might not be specified). Handle this as follows:
- If you can pick a single, reasonable, commonly-expected interpretation,
  proceed with it, but output an ASSUMPTION line first (see format below).
- If the question is too ambiguous to safely guess (multiple interpretations
  would give very different answers, and no interpretation is clearly the
  default), do NOT guess - ask for clarification instead (see format below).
- If the question is not ambiguous, just output the SQL query as normal.

Rules:
- Use ONLY the tables and columns listed below - never invent a table or column name.
- Write a SELECT query only - never INSERT, UPDATE, DELETE, DROP, ALTER, or any other statement.
- If the question cannot be answered from the tables given at all, output exactly: NO_QUERY
- If you need to ask the user to clarify, output exactly one line: CLARIFY: <your clarifying question>
- If you are proceeding on an assumption, output exactly one line: ASSUMPTION: <the assumption you made>
  followed by a blank line, then ONLY the raw SQL query (no explanation, no markdown code fences, no comments).
- Otherwise, output ONLY the raw SQL query (no explanation, no markdown code fences, no comments).

Tables:
{table_schemas}

Question:
{question}

Response:"""
)


SQL_CORRECTION_PROMPT = PromptTemplate.from_template(
    """You previously generated a SQL query for the question below, but the
database rejected it. Using ONLY the tables described, fix the query.

Rules:
- Use ONLY the tables and columns listed below - never invent a table or column name.
- Write a SELECT query only - never INSERT, UPDATE, DELETE, DROP, ALTER, or any other statement.
- If it cannot be fixed with the tables given, output exactly: NO_QUERY
- If you need to ask the user to clarify, output exactly one line: CLARIFY: <your clarifying question>
- If you are proceeding on an assumption, output exactly one line: ASSUMPTION: <the assumption you made>
  followed by a blank line, then ONLY the corrected raw SQL query (no explanation, no markdown code fences, no comments).
- Otherwise, output ONLY the corrected raw SQL query (no explanation, no markdown code fences, no comments).

Tables:
{table_schemas}

Question:
{question}

Previous (failed) SQL query:
{failed_sql}

Database error:
{error_message}

Response:"""
)


SEMANTIC_LAYER_PROMPT = PromptTemplate.from_template(
    """You are matching a user's question to a pre-defined metric, if one fits.
Below is a catalog of already-approved metrics. Each has a name, a
description, and the parameters it accepts.

Decide:
- If ONE metric clearly answers the question, output a JSON object:
  {{"metric": "<metric_name>", "parameters": {{"<param_name>": <value>, ...}}}}
  Only include parameters the metric actually defines. Use null for any
  optional parameter the question doesn't specify a value for.
- If NO metric in the catalog fits the question, output exactly:
  {{"metric": null, "parameters": {{}}}}

Output ONLY the JSON object - no explanation, no markdown code fences.

Metric catalog:
{metrics_catalog}

Question:
{question}

JSON:"""
)


ANSWER_PROMPT = PromptTemplate.from_template(
    """Answer the question in one or two short, natural sentences using ONLY
the query result data below - do not add any fact not present in it.

Rules:
- Write plain prose, not a table or a bullet list.
- If the result is empty, say plainly that no matching data was found.
- Do not mention SQL, tables, or column names - the user only cares about the answer.

Question:
{question}

Query result (JSON rows):
{rows}

Answer:"""
)
