from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    message: str = Field(
        min_length=1,
        max_length=10_000,
    )
    thread_id: UUID | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)
    thread_id: UUID
    model_used: str = Field(min_length=1)
    cached: bool = False
    processing_time_ms: float = Field(ge=0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    sources: list[str] = Field(default_factory=list)
    agent_route: Literal["rag", "direct"] | None = None
    retrieved_chunks: int = Field(default=0, ge=0)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None