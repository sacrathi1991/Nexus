"""
graph/v1_simple_router/router_prompt.py

WHY THIS VERSION (v1) EXISTS: see state.py's docstring in this same
folder - this is the baseline, one-shot-classification version.

What this file does: the classification prompt for the router node
(graph/v1_simple_router/router.py). Its only job is to decide which of
the two existing pipelines a question needs - it knows nothing about how
RAG or NL2SQL work internally, only what KIND of question each one is
good for. This is deliberately a cheap, single-word-output classification
call, not a full answer-generation call, and it is asked ONCE per
question, never in a loop - that one-shot nature is exactly what v2/v3
were built to move past.

Called from: graph/v1_simple_router/router.py
"""

from langchain_core.prompts import PromptTemplate

ROUTER_PROMPT = PromptTemplate.from_template(
    """You are a router deciding which system should answer a question.
There are two systems available:

1. RAG - answers questions from ingested financial documents (earnings
   reports, investor presentations - e.g. TCS, Sherwin-Williams). Use this
   for questions about companies' financial results, business narrative,
   or anything that sounds like it comes from a report or transcript.

2. NL2SQL - answers questions from a business database (Northwind: orders,
   customers, products, employees, suppliers). Use this for questions
   asking for counts, totals, rankings, or specific records from
   structured business/transactional data.

Decide which ONE system should handle this question. Output ONLY one
word: either "rag" or "nl2sql" - no explanation, no punctuation.

Question:
{question}

Answer:"""
)
