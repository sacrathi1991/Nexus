"""
llm/prompts.py

The prompt template that turns (retrieved chunks + user's question) into
the actual text sent to the LLM. Kept in its own file, separate from
query_service.py, so the wording of the prompt can be tuned without
touching any retrieval/database logic — same "one concern per file" idea
as everything else in this project.

Called from: services/query_service.py, as:
    prompt = RAG_PROMPT.format(context=context_text, question=question)
"""

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate.from_template(
    """You are Nexus, an assistant answering questions using ONLY the context provided below.
If the answer isn't contained in the context, say you don't have enough information — do not guess or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""
)
# PromptTemplate.from_template() parses the {context} and {question}
# placeholders out of this string. Later, .format(context=..., question=...)
# fills them in and returns a plain string ready to hand to the LLM.
#
# The instruction "using ONLY the context provided" is deliberate — this is
# what keeps the chatbot grounded in the actual ingested documents instead
# of the LLM falling back on its own general training knowledge, which
# would defeat the entire point of RAG.
