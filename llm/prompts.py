"""
llm/prompts.py

The prompt template that turns (recent chat history + retrieved chunks +
user's question) into the actual text sent to the LLM. Kept in its own
file, separate from query_service.py, so the wording of the prompt can be
tuned without touching any retrieval/database logic — same "one concern
per file" idea as everything else in this project.

Called from: services/query_service.py, as:
    prompt = RAG_PROMPT.format(history=history_text, context=context_text, question=question)
"""

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate.from_template(
    """You are Nexus, an assistant answering questions using ONLY the context provided below.
If the answer isn't contained in the context, say you don't have enough information — do not guess or use outside knowledge.

Recent conversation (for follow-up context only — do not treat this as source material):
{history}

Context:
{context}

Question:
{question}

Answer:"""
)
# PromptTemplate.from_template() parses the {history}, {context}, and
# {question} placeholders out of this string. Later,
# .format(history=..., context=..., question=...) fills them in and
# returns a plain string ready to hand to the LLM.
#
# The instruction "using ONLY the context provided" (meaning the retrieved
# CHUNKS, not the history) is deliberate — history exists only so a
# follow-up question like "what about their profit margin?" can be
# understood as referring back to whatever "they" meant a moment ago. It
# must never be treated as a source of facts on its own — that's still
# strictly the job of {context}, otherwise the LLM could start answering
# from its own earlier (possibly wrong) phrasing instead of the documents.


GROUNDEDNESS_PROMPT = PromptTemplate.from_template(
    """You are a fact-checker verifying whether an AI-generated answer is fully supported by its source material.

Source material (the ONLY thing the answer is allowed to be based on):
{context}

Generated answer to verify:
{answer}

Check whether every factual claim in the answer - every number, name, date, or specific detail - actually appears in or is directly implied by the source material above. An answer is NOT grounded if it states a specific fact that does not appear in the source material, even if it sounds plausible or is the kind of thing that COULD be true.

Respond with ONLY valid JSON, no other text, in this exact shape:
{{"grounded": true|false, "unsupported_claims": "<describe any claims not backed by the source, or empty string if fully grounded>"}}"""
)
# Step 17 - the output-side groundedness guardrail (RAG only - NL2SQL's
# answers are generated from actual live query RESULTS, not a document an
# LLM might misread, which is already a stronger form of groundedness by
# construction). Called from services/groundedness_check.py, AFTER an
# answer is generated, using the SAME retrieved chunks as {context} - this
# is a second, independent LLM call whose only job is to catch the
# generation step hallucinating something the source material never said.
