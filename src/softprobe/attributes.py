from __future__ import annotations

from typing import Any, Mapping

from softprobe.redaction import serialize_captured
from softprobe.types import Attributes


def apply_metadata(attrs: Attributes, metadata: Mapping[str, Any] | None) -> None:
    if not metadata:
        return
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            attrs[f"sp.metadata.{key}"] = value
        else:
            serialized = serialize_captured(value, [])
            if serialized is not None:
                attrs[f"sp.metadata.{key}"] = serialized


def build_observation_attributes(
    *,
    observation_type: str,
    attributes: Mapping[str, Any] | None = None,
    input: Any | None = None,
    output: Any | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    release: str | None = None,
    tags: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    version: str | None = None,
    trace_name: str | None = None,
    redact_keys: list[str],
) -> Attributes:
    attrs: Attributes = dict(attributes or {})
    attrs["sp.observation.type"] = observation_type
    if session_id:
        attrs["sp.session.id"] = session_id
        attrs["gen_ai.conversation.id"] = session_id
    if user_id:
        attrs["sp.user.id"] = user_id
    if release:
        attrs["sp.release"] = release
    if tags is not None:
        attrs["sp.tags"] = list(tags)
    if version:
        attrs["sp.version"] = version
    if trace_name:
        attrs["sp.trace.name"] = trace_name
    apply_metadata(attrs, metadata)
    serialized_input = serialize_captured(input, redact_keys)
    serialized_output = serialize_captured(output, redact_keys)
    if serialized_input is not None:
        attrs["sp.input"] = serialized_input
    if serialized_output is not None:
        attrs["sp.output"] = serialized_output
    return attrs


def build_generation_attributes(
    *,
    attributes: Mapping[str, Any] | None = None,
    input: Any | None = None,
    output: Any | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    release: str | None = None,
    tags: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    version: str | None = None,
    trace_name: str | None = None,
    redact_keys: list[str],
    model: str | None = None,
    response_model: str | None = None,
    provider: str | None = None,
    operation_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    usage: Mapping[str, int] | None = None,
    cost: Mapping[str, float] | None = None,
    completion_start_time: str | None = None,
    response_id: str | None = None,
    finish_reasons: list[str] | None = None,
    prompt: Mapping[str, Any] | None = None,
) -> Attributes:
    attrs = build_observation_attributes(
        observation_type="generation",
        attributes=attributes,
        input=input,
        output=output,
        session_id=session_id,
        user_id=user_id,
        release=release,
        tags=tags,
        metadata=metadata,
        version=version,
        trace_name=trace_name,
        redact_keys=redact_keys,
    )
    if operation_name:
        attrs["gen_ai.operation.name"] = operation_name
    if provider:
        attrs["gen_ai.provider.name"] = provider
    if model:
        attrs["gen_ai.request.model"] = model
    if response_model:
        attrs["gen_ai.response.model"] = response_model
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if usage:
        if "input_tokens" in usage:
            attrs["gen_ai.usage.input_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            attrs["gen_ai.usage.output_tokens"] = usage["output_tokens"]
        if "total_tokens" in usage:
            attrs["gen_ai.usage.total_tokens"] = usage["total_tokens"]
        elif "input_tokens" in usage or "output_tokens" in usage:
            attrs["gen_ai.usage.total_tokens"] = int(usage.get("input_tokens", 0)) + int(
                usage.get("output_tokens", 0)
            )
    if cost:
        if "input" in cost:
            attrs["sp.cost.input"] = cost["input"]
        if "output" in cost:
            attrs["sp.cost.output"] = cost["output"]
        if "total" in cost:
            attrs["sp.cost.total"] = cost["total"]
    if completion_start_time:
        attrs["sp.generation.completion_start_time"] = completion_start_time
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if finish_reasons is not None:
        attrs["gen_ai.response.finish_reasons"] = list(finish_reasons)
    if prompt:
        if prompt.get("id"):
            attrs["sp.prompt.id"] = str(prompt["id"])
        if prompt.get("name"):
            attrs["sp.prompt.name"] = str(prompt["name"])
        if prompt.get("version") is not None:
            attrs["sp.prompt.version"] = int(prompt["version"])
    return attrs
