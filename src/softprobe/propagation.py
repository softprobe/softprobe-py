"""Scoped Softprobe attribute and W3C baggage propagation."""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from opentelemetry import baggage as otel_baggage
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import CarrierT, Getter, Setter
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

BAGGAGE_SESSION_KEY = "sp.session.id"
_EXPLICIT_SESSION_ATTR_KEYS = (
    "sp.session.id",
    "session.id",
    "gen_ai.conversation.id",
)
_SESSION_ID_ASCII_RE = re.compile(r"^[!-~]{1,128}$")

_propagation: ContextVar[dict[str, Any] | None] = ContextVar(
    "softprobe_propagation", default=None
)

_w3c_propagator = CompositePropagator(
    [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
)


class _CaseInsensitiveMappingGetter(Getter[Mapping[str, Any]]):
    def get(self, carrier: Mapping[str, Any], key: str) -> list[str] | None:
        direct = carrier.get(key)
        if isinstance(direct, str):
            return [direct]
        if isinstance(direct, (list, tuple)):
            return [item for item in direct if isinstance(item, str)]
        lower = key.lower()
        for entry_key, value in carrier.items():
            if entry_key.lower() != lower:
                continue
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [item for item in value if isinstance(item, str)]
        return None

    def keys(self, carrier: Mapping[str, Any]) -> list[str]:
        return list(carrier.keys())


class _DictSetter(Setter[dict[str, str]]):
    def set(self, carrier: CarrierT, key: str, value: str) -> None:
        if isinstance(carrier, dict):
            carrier[key] = value


_HEADER_GETTER = _CaseInsensitiveMappingGetter()
_HEADER_SETTER = _DictSetter()


def normalize_valid_session_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not _SESSION_ID_ASCII_RE.match(trimmed):
        return None
    return trimmed


def set_session_baggage(
    session_id: str | None,
    context: otel_context.Context | None = None,
) -> otel_context.Context:
    base = context if context is not None else otel_context.get_current()
    valid = normalize_valid_session_id(session_id)
    if valid is None:
        return base
    return otel_baggage.set_baggage(BAGGAGE_SESSION_KEY, valid, context=base)


def get_baggage_session_id(
    context: otel_context.Context | None = None,
) -> str | None:
    current = otel_context.get_current()
    candidates = (
        [current]
        if context is None or context == current
        else [context, current]
    )
    for candidate in candidates:
        raw = otel_baggage.get_baggage(BAGGAGE_SESSION_KEY, context=candidate)
        head = raw.split(";", 1)[0] if isinstance(raw, str) else raw
        valid = normalize_valid_session_id(head)
        if valid is not None:
            return valid
    return None


@contextmanager
def extract_trace_headers(
    headers: Mapping[str, Any] | None,
) -> Generator[otel_context.Context, None, None]:
    """Extract W3C traceparent and baggage headers into the active OTel context."""
    next_ctx = (
        _w3c_propagator.extract(
            carrier=headers,
            context=otel_context.get_current(),
            getter=_HEADER_GETTER,
        )
        if headers
        else otel_context.get_current()
    )
    token = otel_context.attach(next_ctx)
    try:
        yield next_ctx
    finally:
        otel_context.detach(token)


def _extract_baggage_member_key(member: str) -> str | None:
    eq_idx = member.find("=")
    if eq_idx <= 0:
        return None
    key = member[:eq_idx].strip()
    return key or None


def _merge_baggage_header_values(
    existing: str | None,
    injected: str | None,
) -> str | None:
    existing_trimmed = (existing or "").strip()
    injected_trimmed = (injected or "").strip()
    if not existing_trimmed:
        return injected_trimmed or None
    if not injected_trimmed:
        return existing_trimmed

    members: list[str] = []
    seen_keys: set[str] = set()
    for raw in existing_trimmed.split(","):
        item = raw.strip()
        if not item:
            continue
        members.append(item)
        key = _extract_baggage_member_key(item)
        if key:
            seen_keys.add(key)
    for raw in injected_trimmed.split(","):
        item = raw.strip()
        if not item:
            continue
        key = _extract_baggage_member_key(item)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        members.append(item)
    return ",".join(members) if members else None


def _take_case_insensitive_header(headers: dict[str, str], target: str) -> str | None:
    lower = target.lower()
    values: list[str] = []
    for key in list(headers.keys()):
        if key.lower() == lower:
            val = headers.pop(key)
            if isinstance(val, str) and val.strip():
                values.append(val.strip())
    return ",".join(values) if values else None


def inject_trace_headers(
    headers: Mapping[str, str] | None = None,
    *,
    context: otel_context.Context | None = None,
) -> dict[str, str]:
    """Inject active W3C traceparent and sp.session.id baggage into a headers dict."""
    out: dict[str, str] = dict(headers or {})
    prior_baggage = _take_case_insensitive_header(out, "baggage")
    effective_ctx = context if context is not None else otel_context.get_current()
    if get_baggage_session_id(effective_ctx) is None:
        active_span = otel_trace.get_current_span(effective_ctx)
        span_attrs = getattr(active_span, "attributes", None)
        fallback_sid = normalize_valid_session_id(
            (_propagation.get() or {}).get("session_id")
        ) or _read_explicit_session_attribute(span_attrs)
        if fallback_sid:
            effective_ctx = set_session_baggage(fallback_sid, context=effective_ctx)
    injected: dict[str, str] = {}
    _w3c_propagator.inject(
        carrier=injected,
        context=effective_ctx,
        setter=_HEADER_SETTER,
    )
    if "traceparent" in injected:
        _take_case_insensitive_header(out, "traceparent")
        out["traceparent"] = injected["traceparent"]
    if "tracestate" in injected:
        _take_case_insensitive_header(out, "tracestate")
        out["tracestate"] = injected["tracestate"]
    merged_baggage = _merge_baggage_header_values(
        prior_baggage,
        injected.get("baggage"),
    )
    if merged_baggage:
        out["baggage"] = merged_baggage
    return out


def get_propagated_attributes() -> dict[str, Any]:
    return dict(_propagation.get() or {})


@contextmanager
def propagate_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    version: str | None = None,
    release: str | None = None,
    trace_name: str | None = None,
) -> Generator[None, None, None]:
    """Temporarily override Softprobe defaults and W3C session baggage for nested work."""
    current = get_propagated_attributes()
    merged = dict(current)
    if session_id is not None:
        merged["session_id"] = session_id
    if user_id is not None:
        merged["user_id"] = user_id
    if tags is not None:
        merged["tags"] = list(tags)
    if metadata is not None:
        base_meta = dict(merged.get("metadata") or {})
        base_meta.update(dict(metadata))
        merged["metadata"] = base_meta
    if version is not None:
        merged["version"] = version
    if release is not None:
        merged["release"] = release
    if trace_name is not None:
        merged["trace_name"] = trace_name
    prop_token: Token[dict[str, Any] | None] = _propagation.set(merged)
    baggage_ctx = set_session_baggage(merged.get("session_id"))
    otel_token = otel_context.attach(baggage_ctx)
    try:
        yield
    finally:
        otel_context.detach(otel_token)
        _propagation.reset(prop_token)


def _read_explicit_session_attribute(attrs: Mapping[str, Any] | None) -> str | None:
    if not attrs:
        return None
    for key in _EXPLICIT_SESSION_ATTR_KEYS:
        valid = normalize_valid_session_id(attrs.get(key))
        if valid is not None:
            return valid
    return None


def _has_explicit_session_attribute(attrs: Mapping[str, Any] | None) -> bool:
    if not attrs:
        return False
    for key in _EXPLICIT_SESSION_ATTR_KEYS:
        val = attrs.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


class BaggageSessionSpanProcessor(SpanProcessor):
    """Stamps sp.session.id from active W3C baggage onto spans lacking an explicit session."""

    def on_start(
        self,
        span: Span,
        parent_context: otel_context.Context | None = None,
    ) -> None:
        existing_attrs = getattr(span, "attributes", None)
        if _has_explicit_session_attribute(existing_attrs):
            return
        session_id = normalize_valid_session_id(
            (_propagation.get() or {}).get("session_id")
        ) or get_baggage_session_id(parent_context)
        if session_id is None:
            return
        span.set_attribute("sp.session.id", session_id)
        span.set_attribute("gen_ai.conversation.id", session_id)

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
