from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

ObservationType = Literal[
    "span",
    "event",
    "generation",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "embedding",
    "guardrail",
]

OBSERVATION_TYPES: tuple[ObservationType, ...] = (
    "span",
    "event",
    "generation",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "embedding",
    "guardrail",
)

AttributeValue = str | int | float | bool | list[str] | None
Attributes = dict[str, AttributeValue]
JsonValue = Any

ScoreDataType = Literal["numeric", "categorical", "boolean", "text"]
ScoreSource = Literal["api", "user", "evaluator", "annotation"]


@dataclass
class SoftprobeConfig:
    public_key: str
    base_url: str
    otlp_endpoint: str
    service_name: str = "softprobe-app"
    service_version: str | None = None
    environment: str | None = None
    release: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] | None = None
    redact_keys: list[str] | None = None
    timeout_ms: int = 10_000
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass
class ScoreRequest:
    score_id: str
    timestamp: str
    trace_id: str | None
    span_id: str | None
    session_id: str | None
    name: str
    data_type: ScoreDataType
    numeric_value: float | None
    string_value: str | None
    boolean_value: bool | None
    source: ScoreSource
    comment: str | None
    config_id: str | None
    author_id: str | None
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_id": self.score_id,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "session_id": self.session_id,
            "name": self.name,
            "data_type": self.data_type,
            "numeric_value": self.numeric_value,
            "string_value": self.string_value,
            "boolean_value": self.boolean_value,
            "source": self.source,
            "comment": self.comment,
            "config_id": self.config_id,
            "author_id": self.author_id,
            "metadata": self.metadata,
        }


class ScoreTransport:
    def create_score(self, request: ScoreRequest) -> None:
        raise NotImplementedError


NormalizedSpanEvent = dict[str, Any]
NormalizedSpan = dict[str, Any]
