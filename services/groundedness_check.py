"""
services/groundedness_check.py

Step 17 - the output-side guardrail for RAG. After an answer is generated
from retrieved chunks, this runs ONE MORE LLM call whose only job is to
verify the answer didn't hallucinate a fact the source material never
actually said - a fluent, confident-sounding number or claim an LLM
invented is otherwise indistinguishable from a real one just by reading
the final answer.

Scoped to RAG only, not NL2SQL - NL2SQL's answers are phrased directly
from actual query RESULT ROWS (services/nl2sql_service.py's
_generate_answer()), which is already a stronger, more direct form of
groundedness than "an LLM read some document text and summarized it" -
there's no document for the model to misread, just numbers it's handed.

Fails OPEN (grounded=True, checked=False) if the classifier's response
can't be parsed - same reasoning as graph/common/input_guardrail.py: this
is a SECOND layer of quality assurance, not the only line of defense, so
a parsing glitch should degrade to "show the answer as normal," not
"silently swallow a legitimate answer."

Called from: services/query_service.py's ask(), right after the answer
             is generated, using the SAME retrieved chunks already built
             into {context} for the original answer - no second
             retrieval needed.
"""

import json

from llm.llm_client import llm
from llm.prompts import GROUNDEDNESS_PROMPT


def _extract_text(content) -> str:
    """Same normalization every other LLM-calling file in this project
    needs - response.content can be a plain string or a list of content
    blocks, depending on the model/SDK version."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()


def check_groundedness(answer: str, context: str) -> dict:
    """Returns {"grounded": bool, "unsupported_claims": str, "checked": bool}.

    "checked" is False only when the classifier call itself failed to
    parse - distinct from "grounded": True, which means the check RAN and
    found no issue. A caller that only cares about "should I warn the
    user" can just check `not result["grounded"]`; "checked" exists for
    anyone who wants to tell "verified clean" apart from "couldn't verify".
    """
    prompt = GROUNDEDNESS_PROMPT.format(context=context, answer=answer)

    try:
        response = llm.invoke(prompt)
        raw = _extract_text(response.content)
        parsed = json.loads(raw)
        grounded = bool(parsed.get("grounded", True))
        unsupported_claims = parsed.get("unsupported_claims", "")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"grounded": True, "unsupported_claims": "", "checked": False}

    return {"grounded": grounded, "unsupported_claims": unsupported_claims, "checked": True}
