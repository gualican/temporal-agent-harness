# ABOUTME: Backend-neutral value types for the harness memory layer: one MemoryRecord shape every
# MemoryProvider returns, so the shared tools and prompt-context code never see a backend's own model.

from datetime import datetime

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """One stored memory, as any backend reports it.

    ``score`` is set only on recall results, and its scale is the backend's own (cosine, rerank,
    ...) — comparable within one result list, not across backends. ``created_at`` is ``None`` when
    the backend doesn't expose it.
    """

    id: str
    text: str
    tags: list[str] = Field(default_factory=list)
    score: float | None = None
    created_at: datetime | None = None
