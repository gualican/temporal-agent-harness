"""Pure, deterministic memory scoring — safe to run inside workflow query handlers.

Relevance uses cosine similarity when both the query and the memory have
embeddings, otherwise keyword (token-overlap) matching. A recency term with a
30-day half-life breaks ties toward fresher memories.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from temporal_agent_harness.harness.memory.providers.temporal.models import MemoryItem, RecallParams, ScoredMemory

RELEVANCE_WEIGHT = 0.85
RECENCY_WEIGHT = 0.15
RECENCY_HALF_LIFE_DAYS = 30.0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_relevance(query: str, memory: MemoryItem) -> float:
    query_tokens = _tokens(query)
    memory_tokens = _tokens(memory.text + " " + " ".join(memory.tags))
    if not query_tokens or not memory_tokens:
        return 0.0
    return len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def score_memory(memory: MemoryItem, params: RecallParams, now: datetime) -> float:
    if params.query_vector is not None and memory.vector is not None:
        relevance = max(_cosine(params.query_vector, memory.vector), 0.0)
    else:
        relevance = _keyword_relevance(params.query_text, memory)
    age_days = max((now - memory.created_at).total_seconds() / 86400, 0.0)
    recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return RELEVANCE_WEIGHT * relevance + RECENCY_WEIGHT * recency


def recall_memories(
    memories: list[MemoryItem], params: RecallParams, now: datetime
) -> list[ScoredMemory]:
    scored = sorted(
        (ScoredMemory(memory=m.model_copy(update={"vector": None}), score=score_memory(m, params, now))
         for m in memories),
        key=lambda s: s.score,
        reverse=True,
    )
    return scored[: params.top_k]
