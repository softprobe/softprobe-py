"""Softprobe Python instrumentation SDK."""

from typing import Any

from softprobe.client import SoftprobeClient
from softprobe.observation import Generation, Observation
from softprobe.redaction import default_redact_keys, redact_value
from softprobe.scores import build_score_request
from softprobe.tools import (
    accumulate_tool_call_deltas,
    finalize_tool_call_deltas,
    normalize_tool_calls,
    normalize_tool_definitions,
    record_tool_calls,
    record_tool_definitions,
    tool_result_event_payload,
)
from softprobe.types import OBSERVATION_TYPES

__all__ = [
    "OBSERVATION_TYPES",
    "Generation",
    "Observation",
    "SoftprobeClient",
    "accumulate_tool_call_deltas",
    "build_score_request",
    "default_redact_keys",
    "finalize_tool_call_deltas",
    "normalize_tool_calls",
    "normalize_tool_definitions",
    "propagate_attributes",
    "record_tool_calls",
    "record_tool_definitions",
    "redact_value",
    "tool_result_event_payload",
    "observe_openai",
    "CallbackHandler",
]


def __getattr__(name: str) -> Any:
    if name == "observe_openai":
        from softprobe.openai import observe_openai

        return observe_openai
    if name == "propagate_attributes":
        from softprobe.propagation import propagate_attributes

        return propagate_attributes
    if name == "CallbackHandler":
        from softprobe.langchain import CallbackHandler

        return CallbackHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
