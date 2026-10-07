"""
graph/common/response_cache.py

The exact-match response cache in front of v3 (graph/v3_multi_agent).
Deliberately kept simple on purpose: cache key = hash(normalized question
+ user_context), cache value = the full result dict from
ask_nexus_multiagent(), one flat TTL for everything. No metric/parameter
tier, no semantic (embedding-based) matching - those were discussed as
real, legitimate enhancements but explicitly deferred to keep this first
version simple and easy to reason about (see Day 11 playbook doc's
"What could be added in a real-time project" section for that writeup).

WHY user_context is part of the key, not just the question: the same
question text can have a DIFFERENT correct answer depending on who's
asking, because of NL2SQL's row-level security (Step 11) - an admin and
employee 4 asking "how many orders are there?" get genuinely different,
both-correct answers. Leaving user_context out of the key would mean
whichever identity asks first "poisons" the cache for every other
identity asking the same words afterward - a real data leak, not just a
wrong answer.

Called from: graph/v3_multi_agent/build_multi_agent_graph.py's
             ask_nexus_multiagent(), as the very first and very last
             steps (check before running the graph, store after).
"""

import hashlib
import json

import redis

from config import REDIS_URL

_redis_client = redis.from_url(REDIS_URL, decode_responses=True)
# decode_responses=True - Redis stores bytes internally, but this makes
# the client hand back plain Python strings instead of bytes, so the rest
# of this file never has to think about encoding/decoding manually.

CACHE_TTL_SECONDS = 600
# 10 minutes - short enough that NL2SQL answers (which depend on data
# that changes as Northwind's orders/etc. get updated) don't stay stale
# for long, long enough to actually catch repeated questions asked within
# the same short session - the realistic window for a "FAQ-style" repeat.

CACHE_KEY_PREFIX = "nexus:v3:cache:"


def _build_cache_key(question: str, user_context: dict) -> str:
    """hash(normalized question + identity) -> one deterministic key.

    Normalizing (strip + lowercase) means "What is TCS revenue?" and
    "what is tcs revenue? " (trailing space, different case) hit the SAME
    key - still exact-match in spirit (no embedding/similarity involved),
    just not sensitive to whitespace/casing that carries no real meaning.
    """
    normalized_question = question.strip().lower()
    user_context = user_context or {}
    identity = f"{user_context.get('role')}:{user_context.get('employee_id')}"
    raw = f"{normalized_question}|{identity}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}{digest}"


def get_cached_response(question: str, user_context: dict) -> dict | None:
    """Returns the cached result dict if present, else None (cache miss).

    A miss is the NORMAL case the first time any question is asked - this
    is not an error, just "nothing here yet."
    """
    key = _build_cache_key(question, user_context)
    raw_value = _redis_client.get(key)
    if raw_value is None:
        return None
    return json.loads(raw_value)


def set_cached_response(question: str, user_context: dict, result: dict) -> None:
    """Stores one result dict under its key, with CACHE_TTL_SECONDS expiry.

    Stored as a JSON string (Redis's simplest value type - a plain
    string) rather than a Redis hash, since the whole dict is always
    read/written together as one unit - there's no need to update just
    one field of a cached entry independently.
    """
    key = _build_cache_key(question, user_context)
    _redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(result))
