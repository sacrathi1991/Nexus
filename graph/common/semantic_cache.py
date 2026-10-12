"""
graph/common/semantic_cache.py

A SECOND, broader cache layer in front of v3 (graph/v3_multi_agent),
checked only after the exact-match cache (response_cache.py) misses. The
exact-match cache only recognizes a repeat question if it's worded
identically; this one recognizes a repeat question if it MEANS the same
thing, even when worded completely differently - "What was TCS's Q1
revenue?" vs "Can you tell me TCS's revenue for the first quarter?".

How it decides "same meaning": the incoming question is embedded (same
embeddings model RAG uses for document chunks - llm/embeddings_client.py),
then compared via cosine similarity against the embeddings of every
previously-cached question FOR THIS SAME USER IDENTITY. The closest one
above SIMILARITY_THRESHOLD is treated as a match; anything below it is
treated as a genuinely different question, same as a normal cache miss.

Brute-force by design: similarity is computed in plain Python against
every entry scoped to this identity, with no vector index (RediSearch/
Redis Stack). Correct and simple for a learning-scale cache; a real
production deployment with thousands of cached entries per identity would
swap this scan+compare step for a proper vector index, but the caching
LOGIC (embed -> compare -> threshold -> hit/miss) stays the same either
way - only the search mechanism underneath would change.

WHY user_context still scopes every key here, exactly like
response_cache.py: NL2SQL's row-level security means the same question
text can have a different correct answer depending on who's asking - that
correctness requirement doesn't go away just because matching is now
approximate instead of exact. Semantic similarity is only ever computed
BETWEEN two entries that already belong to the same identity - never
across identities.

Called from: graph/v3_multi_agent/build_multi_agent_graph.py's
             ask_nexus_multiagent(), as the SECOND check (after the
             exact-match cache misses) and as a SECOND store (alongside
             the exact-match store) once a full pipeline run completes.
"""

import hashlib
import json
import math
import uuid

import redis

from config import REDIS_URL
from llm.embeddings_client import embeddings

_redis_client = redis.from_url(REDIS_URL, decode_responses=True)

CACHE_TTL_SECONDS = 600
# Same window as response_cache.py's exact-match cache - kept identical on
# purpose, since these two caches are meant to expire together, not drift
# out of sync with each other.

SEMANTIC_CACHE_KEY_PREFIX = "nexus:v3:semcache:"

SIMILARITY_THRESHOLD = 0.93
# How close two questions' embeddings must be (cosine similarity, 0-1) to
# be treated as "the same question". Set conservatively high on purpose -
# too loose, and genuinely different questions about the same topic (e.g.
# one quarter's revenue vs. a different quarter's) risk being wrongly
# treated as identical, which is a worse failure than a few avoidable
# cache misses on real paraphrases.


def _identity_namespace(user_context: dict) -> str:
    """Scopes every key to one identity, the same correctness requirement
    as response_cache.py - but here it has to be a separate NAMESPACE
    (a key prefix), not folded into one combined hash, because this file
    needs to list/scan every entry belonging to one identity to compare
    against. A single combined hash (question+identity) couldn't be
    scanned by identity alone the way a prefix can.
    """
    user_context = user_context or {}
    identity = f"{user_context.get('role')}:{user_context.get('employee_id')}"
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Plain-Python cosine similarity - no numpy dependency needed at this
    scale. Returns a value in [-1, 1]; in practice, two real sentence
    embeddings from the same model land in [0, 1], with 1 meaning
    identical direction (same meaning), 0 meaning unrelated.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_semantically_cached_response(question: str, user_context: dict) -> dict | None:
    """Embeds `question`, compares it against every previously-cached
    question for this same identity, and returns the matched entry's
    stored result if the best similarity clears SIMILARITY_THRESHOLD.

    Returns None on a miss - either no entries exist yet for this
    identity, or none are close enough - exactly like a normal cache miss,
    not an error.

    The returned dict includes two extra keys beyond the original result -
    "matched_question" (what it matched against) and "similarity_score" -
    purely for transparency (so a caller/terminal can show "matched an
    existing cached question, X% similar" instead of silently returning an
    answer to a question that was never asked verbatim).
    """
    question_embedding = embeddings.embed_query(question)
    namespace = _identity_namespace(user_context)
    scan_pattern = f"{SEMANTIC_CACHE_KEY_PREFIX}{namespace}:*"

    best_score = 0.0
    best_entry = None

    for key in _redis_client.scan_iter(match=scan_pattern):
        raw_value = _redis_client.get(key)
        if raw_value is None:
            continue  # expired between the scan and this GET - skip it
        entry = json.loads(raw_value)
        score = _cosine_similarity(question_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None or best_score < SIMILARITY_THRESHOLD:
        return None

    result = dict(best_entry["result"])
    result["matched_question"] = best_entry["question"]
    result["similarity_score"] = round(best_score, 4)
    return result


def set_semantically_cached_response(question: str, user_context: dict, result: dict) -> None:
    """Stores this question's embedding alongside the result, under a
    NEW, unique key (not a key derived from the question text - unlike
    response_cache.py, there's nothing to deterministically re-derive here,
    since future lookups happen by similarity, not by re-hashing the same
    text) - so this question becomes matchable by any future paraphrase of
    it, for this same identity.
    """
    question_embedding = embeddings.embed_query(question)
    namespace = _identity_namespace(user_context)
    entry_id = uuid.uuid4().hex[:16]
    key = f"{SEMANTIC_CACHE_KEY_PREFIX}{namespace}:{entry_id}"

    entry = {
        "question": question,
        "embedding": question_embedding,
        "result": result,
    }
    _redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(entry))
