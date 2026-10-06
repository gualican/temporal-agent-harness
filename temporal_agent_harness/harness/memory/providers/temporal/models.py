# ABOUTME: Payload models for the Temporal-workflow memory backend (workflow state, update/query
# params, and the activity params the service takes). All inherit ``Model``, which marks every field
# as set after validation — see its docstring for why that matters under the agents payload converter.

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


class Model(BaseModel):
    """Base for every payload model of the Temporal memory backend.

    The OpenAI Agents plugin's payload converter serializes with
    exclude_unset=True (so OpenAI's drifting response models round-trip), which
    silently drops any field not in __pydantic_fields_set__ — and in-place
    mutations like list.append never mark a field as set. Workflow state that
    is mutated in place and then serialized (queries, update results, and
    especially continue-as-new arguments) would lose those fields. Marking all
    fields as set after validation makes exclude_unset a no-op for our models.
    """

    @model_validator(mode="after")
    def _mark_all_fields_set(self) -> "Model":
        self.__pydantic_fields_set__.update(type(self).model_fields)
        return self


class MemoryItem(Model):
    id: str
    text: str
    tags: list[str] = []
    created_at: datetime
    # Embedding of `text`, set at write time when embedding succeeds. Stripped
    # from all responses leaving the memory workflow to keep payloads small.
    vector: list[float] | None = None


class MemoryState(Model):
    """Full state of one memory workflow; carried across continue-as-new."""

    scope: str
    memories: list[MemoryItem] = []
    # Eviction cap: the serialized state must stay well under Temporal's 2MB
    # payload limit since continue-as-new carries it as an argument.
    max_memories: int = 500


class RememberParams(Model):
    text: str
    tags: list[str] = []


class RecallParams(Model):
    query_text: str
    # Optional caller-side embedding of query_text: query handlers cannot run
    # activities, so semantic search requires the caller to embed the query.
    query_vector: list[float] | None = None
    top_k: int = 5


class ScoredMemory(Model):
    memory: MemoryItem
    score: float


# Params for the client-holding activities that mediate memory access
# (workflows cannot signal/query/update other workflows directly).


class MemoryWrite(Model):
    scope: str
    text: str
    tags: list[str] = []


class MemoryRecall(Model):
    scope: str
    query: str
    top_k: int = 5


class MemoryForget(Model):
    scope: str
    memory_id: str
