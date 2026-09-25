from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from softprobe.types import NormalizedSpan


def normalize_readable_spans(spans: list[ReadableSpan]) -> list[NormalizedSpan]:
    by_id = {format(span.context.span_id, "016x"): span for span in spans if span.context}
    normalized = [_normalize_one(span, by_id) for span in spans]
    return sorted(normalized, key=lambda item: item["name"])


def _normalize_one(span: ReadableSpan, by_id: dict[str, ReadableSpan]) -> NormalizedSpan:
    attributes: dict[str, Any] = {}
    for key, value in (span.attributes or {}).items():
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            attributes[key] = list(value)
        else:
            attributes[key] = value
    observation_type = attributes.get("sp.observation.type", "span")
    parent_name = None
    if span.parent and span.parent.span_id:
        parent = by_id.get(format(span.parent.span_id, "016x"))
        if parent is not None:
            parent_name = parent.name
    status_code = "UNSET"
    if span.status.status_code == StatusCode.OK:
        status_code = "OK"
    elif span.status.status_code == StatusCode.ERROR:
        status_code = "ERROR"
    events = []
    for event in span.events:
        event_attrs: dict[str, Any] = {}
        for key, value in (event.attributes or {}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                import json

                event_attrs[key] = json.dumps(list(value))
            else:
                event_attrs[key] = value
        events.append({"name": event.name, "attributes": event_attrs})
    return {
        "name": span.name,
        "observation_type": observation_type,
        "parent_name": parent_name,
        "attributes": attributes,
        "events": events,
        "status_code": status_code,
        "status_message": span.status.description,
    }
