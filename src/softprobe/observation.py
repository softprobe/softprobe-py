from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, Mapping, Union

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    Status,
    StatusCode,
    Tracer,
    TraceFlags,
)

from softprobe.attributes import (
    build_generation_attributes,
    build_observation_attributes,
)
from softprobe.propagation import get_propagated_attributes
from softprobe.redaction import serialize_captured
from softprobe.types import Attributes, ObservationType

TimeInput = Union[datetime, float, int]


def _to_unix_nanos(value: TimeInput | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ms = value.timestamp() * 1000
    else:
        ms = float(value)
    return int(ms * 1_000_000)


@dataclass
class ObservationRuntime:
    tracer: Tracer
    redact_keys: list[str]
    session_id: str | None = None
    user_id: str | None = None
    release: str | None = None
    tags: list[str] | None = None


def _otel_attrs(attrs: Attributes) -> dict[str, Any]:
    return {key: value for key, value in attrs.items() if value is not None}


def _resolve_parent_context(
    *,
    parent: "Observation | None" = None,
    trace_context: Mapping[str, str] | None = None,
) -> Any | None:
    if parent is not None:
        return trace.set_span_in_context(parent.otel_span)
    if not trace_context:
        return None
    trace_id = trace_context.get("trace_id")
    if not trace_id:
        return None
    parent_span_id = trace_context.get("parent_span_id") or "0"
    span_context = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(parent_span_id, 16),
        is_remote=True,
        trace_flags=TraceFlags(0x01),
    )
    return trace.set_span_in_context(NonRecordingSpan(span_context))


def _merged_defaults(
    runtime: ObservationRuntime,
    *,
    session_id: str | None,
    user_id: str | None,
    release: str | None,
    tags: list[str] | None,
    metadata: Mapping[str, Any] | None,
    version: str | None,
    trace_name: str | None,
) -> dict[str, Any]:
    propagated = get_propagated_attributes()
    merged_metadata = dict(propagated.get("metadata") or {})
    if metadata:
        merged_metadata.update(dict(metadata))
    return {
        "session_id": (
            session_id
            if session_id is not None
            else propagated.get("session_id", runtime.session_id)
        ),
        "user_id": (
            user_id if user_id is not None else propagated.get("user_id", runtime.user_id)
        ),
        "release": (
            release if release is not None else propagated.get("release", runtime.release)
        ),
        "tags": tags if tags is not None else propagated.get("tags", runtime.tags),
        "metadata": merged_metadata or None,
        "version": version if version is not None else propagated.get("version"),
        "trace_name": (
            trace_name if trace_name is not None else propagated.get("trace_name")
        ),
    }


class Observation:
    def __init__(
        self,
        span: Span,
        runtime: ObservationRuntime,
        observation_type: ObservationType,
    ) -> None:
        self._span = span
        self._runtime = runtime
        self._observation_type = observation_type
        self._ended = False
        self._saw_error = False
        self._token: Any | None = None

    @property
    def span_id(self) -> str:
        return format(self._span.get_span_context().span_id, "016x")

    @property
    def trace_id(self) -> str:
        return format(self._span.get_span_context().trace_id, "032x")

    @property
    def otel_span(self) -> Span:
        return self._span

    def update(
        self,
        *,
        attributes: Mapping[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
    ) -> Observation:
        if self._ended:
            return self
        attrs = build_observation_attributes(
            observation_type=self._observation_type,
            attributes=attributes,
            input=input,
            output=output,
            session_id=session_id if session_id is not None else self._runtime.session_id,
            user_id=user_id if user_id is not None else self._runtime.user_id,
            release=release if release is not None else self._runtime.release,
            tags=tags if tags is not None else self._runtime.tags,
            redact_keys=self._runtime.redact_keys,
        )
        self._span.set_attributes(_otel_attrs(attrs))
        if status_message:
            self._saw_error = True
            self._span.set_status(Status(StatusCode.ERROR, status_message))
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> Observation:
        if self._ended:
            return self
        self._span.set_attributes(dict(attributes))
        return self

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> Observation:
        if self._ended:
            return self
        self._span.add_event(name, attributes=dict(attributes or {}))
        return self

    def add_content_event(self, name: str, content: Any) -> Observation:
        if self._ended:
            return self
        serialized = serialize_captured(content, self._runtime.redact_keys)
        if serialized is not None:
            self._span.add_event(name, attributes={"content": serialized})
        return self

    def record_exception(self, error: BaseException) -> Observation:
        if self._ended:
            return self
        self._span.record_exception(error)
        self._saw_error = True
        self._span.set_status(Status(StatusCode.ERROR, str(error)))
        return self

    def end(
        self,
        *,
        attributes: Mapping[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        end_time: TimeInput | None = None,
    ) -> None:
        if self._ended:
            return
        if attributes is not None or input is not None or output is not None or status_message:
            self.update(
                attributes=attributes,
                input=input,
                output=output,
                status_message=status_message,
            )
        if status_message:
            self._saw_error = True
            self._span.set_status(Status(StatusCode.ERROR, status_message))
        elif not self._saw_error:
            self._span.set_status(Status(StatusCode.OK))
        end_ns = _to_unix_nanos(end_time)
        if end_ns is not None:
            self._span.end(end_time=end_ns)
        else:
            self._span.end()
        self._ended = True
        if self._token is not None:
            otel_context.detach(self._token)
            self._token = None

    def attach(self) -> None:
        if self._token is None:
            self._token = otel_context.attach(trace.set_span_in_context(self._span))


class Generation(Observation):
    def update(
        self,
        *,
        attributes: Mapping[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
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
        prompt_event: Any | None = None,
        completion_event: Any | None = None,
        inference_details: Any | None = None,
    ) -> Generation:
        if self._ended:
            return self
        attrs = build_generation_attributes(
            attributes=attributes,
            input=input,
            output=output,
            session_id=session_id if session_id is not None else self._runtime.session_id,
            user_id=user_id if user_id is not None else self._runtime.user_id,
            release=release if release is not None else self._runtime.release,
            tags=tags if tags is not None else self._runtime.tags,
            redact_keys=self._runtime.redact_keys,
            model=model,
            response_model=response_model,
            provider=provider,
            operation_name=operation_name,
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
            cost=cost,
            completion_start_time=completion_start_time,
            response_id=response_id,
            finish_reasons=finish_reasons,
            prompt=prompt,
        )
        self._span.set_attributes(_otel_attrs(attrs))
        self._emit_generation_events(
            prompt_event=prompt_event,
            completion_event=completion_event,
            inference_details=inference_details,
        )
        if status_message:
            self._saw_error = True
            self._span.set_status(Status(StatusCode.ERROR, status_message))
        return self

    def end(
        self,
        *,
        attributes: Mapping[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        model: str | None = None,
        response_model: str | None = None,
        provider: str | None = None,
        usage: Mapping[str, int] | None = None,
        cost: Mapping[str, float] | None = None,
        response_id: str | None = None,
        finish_reasons: list[str] | None = None,
        prompt_event: Any | None = None,
        completion_event: Any | None = None,
        inference_details: Any | None = None,
        end_time: TimeInput | None = None,
    ) -> None:
        if self._ended:
            return
        if any(
            value is not None
            for value in (
                attributes,
                input,
                output,
                status_message,
                model,
                response_model,
                provider,
                usage,
                cost,
                response_id,
                finish_reasons,
                prompt_event,
                completion_event,
                inference_details,
            )
        ):
            self.update(
                attributes=attributes,
                input=input,
                output=output,
                status_message=status_message,
                model=model,
                response_model=response_model,
                provider=provider,
                usage=usage,
                cost=cost,
                response_id=response_id,
                finish_reasons=finish_reasons,
                prompt_event=prompt_event,
                completion_event=completion_event,
                inference_details=inference_details,
            )
        if status_message:
            self._saw_error = True
            self._span.set_status(Status(StatusCode.ERROR, status_message))
        elif not self._saw_error:
            self._span.set_status(Status(StatusCode.OK))
        end_ns = _to_unix_nanos(end_time)
        if end_ns is not None:
            self._span.end(end_time=end_ns)
        else:
            self._span.end()
        self._ended = True
        if self._token is not None:
            otel_context.detach(self._token)
            self._token = None

    def _emit_generation_events(
        self,
        *,
        prompt_event: Any | None,
        completion_event: Any | None,
        inference_details: Any | None,
    ) -> None:
        if prompt_event is not None:
            self.add_content_event("gen_ai.content.prompt", prompt_event)
        if completion_event is not None:
            self.add_content_event("gen_ai.content.completion", completion_event)
        if inference_details is not None:
            self.add_content_event(
                "gen_ai.client.inference.operation.details",
                inference_details,
            )


def start_observation(
    runtime: ObservationRuntime,
    *,
    name: str,
    as_type: ObservationType = "span",
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
    parent: Observation | None = None,
    trace_context: Mapping[str, str] | None = None,
    start_time: TimeInput | None = None,
) -> Observation:
    defaults = _merged_defaults(
        runtime,
        session_id=session_id,
        user_id=user_id,
        release=release,
        tags=tags,
        metadata=metadata,
        version=version,
        trace_name=trace_name,
    )
    attrs = build_observation_attributes(
        observation_type=as_type,
        attributes=attributes,
        input=input,
        output=output,
        session_id=defaults["session_id"],
        user_id=defaults["user_id"],
        release=defaults["release"],
        tags=defaults["tags"],
        metadata=defaults["metadata"],
        version=defaults["version"],
        trace_name=defaults["trace_name"],
        redact_keys=runtime.redact_keys,
    )
    parent_ctx = _resolve_parent_context(parent=parent, trace_context=trace_context)
    start_ns = _to_unix_nanos(start_time)
    span_kwargs: dict[str, Any] = {
        "name": name,
        "context": parent_ctx,
        "attributes": _otel_attrs(attrs),
    }
    if start_ns is not None:
        span_kwargs["start_time"] = start_ns
    span = runtime.tracer.start_span(**span_kwargs)
    return Observation(span, runtime, as_type)


def start_generation(
    runtime: ObservationRuntime,
    *,
    name: str,
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
    parent: Observation | None = None,
    trace_context: Mapping[str, str] | None = None,
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
    prompt_event: Any | None = None,
    completion_event: Any | None = None,
    inference_details: Any | None = None,
    start_time: TimeInput | None = None,
) -> Generation:
    defaults = _merged_defaults(
        runtime,
        session_id=session_id,
        user_id=user_id,
        release=release,
        tags=tags,
        metadata=metadata,
        version=version,
        trace_name=trace_name,
    )
    attrs = build_generation_attributes(
        attributes=attributes,
        input=input,
        output=output,
        session_id=defaults["session_id"],
        user_id=defaults["user_id"],
        release=defaults["release"],
        tags=defaults["tags"],
        metadata=defaults["metadata"],
        version=defaults["version"],
        trace_name=defaults["trace_name"],
        redact_keys=runtime.redact_keys,
        model=model,
        response_model=response_model,
        provider=provider,
        operation_name=operation_name,
        temperature=temperature,
        max_tokens=max_tokens,
        usage=usage,
        cost=cost,
        completion_start_time=completion_start_time,
        response_id=response_id,
        finish_reasons=finish_reasons,
        prompt=prompt,
    )
    parent_ctx = _resolve_parent_context(parent=parent, trace_context=trace_context)
    start_ns = _to_unix_nanos(start_time)
    span_kwargs: dict[str, Any] = {
        "name": name,
        "context": parent_ctx,
        "attributes": _otel_attrs(attrs),
    }
    if start_ns is not None:
        span_kwargs["start_time"] = start_ns
    span = runtime.tracer.start_span(**span_kwargs)
    generation = Generation(span, runtime, "generation")
    generation.update(
        prompt_event=prompt_event,
        completion_event=completion_event,
        inference_details=inference_details,
    )
    return generation


@contextmanager
def observation_scope(observation: Observation) -> Generator[Observation, None, None]:
    observation.attach()
    try:
        yield observation
        observation.end()
    except BaseException as exc:
        observation.record_exception(exc)
        observation.end()
        raise
