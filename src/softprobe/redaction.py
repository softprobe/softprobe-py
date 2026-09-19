from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_REDACT_KEYS = ("password", "api_key", "authorization", "secret", "token")


def default_redact_keys() -> list[str]:
    return list(DEFAULT_REDACT_KEYS)


def redact_value(
    value: Any,
    redact_keys: list[str],
    placeholder: str = "[REDACTED]",
) -> Any:
    key_set = {key.lower() for key in redact_keys}
    return _walk(value, key_set, placeholder)


def _to_jsonable(value: Any) -> Any:
    """Coerce LangChain/LangGraph objects (messages, State, ToolMessage) for JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # LangChain BaseMessage and similar
    msg_type = getattr(value, "type", None)
    if msg_type is not None and hasattr(value, "content"):
        payload: dict[str, Any] = {
            "role": str(msg_type),
            "content": _to_jsonable(getattr(value, "content", None)),
        }
        for key in ("name", "tool_call_id", "id"):
            attr = getattr(value, key, None)
            if attr is not None:
                payload[key] = str(attr)
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = _to_jsonable(tool_calls)
        return payload
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json"))
        except Exception:
            try:
                return _to_jsonable(value.model_dump())
            except Exception:
                return str(value)
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return _to_jsonable(value.dict())
        except Exception:
            return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _walk(value: Any, key_set: set[str], placeholder: str) -> Any:
    value = _to_jsonable(value)
    if isinstance(value, list):
        return [_walk(item, key_set, placeholder) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            out[key] = (
                placeholder
                if str(key).lower() in key_set
                else _walk(nested, key_set, placeholder)
            )
        return out
    return value


def serialize_captured(
    value: Any | None,
    redact_keys: list[str],
) -> str | None:
    if value is None:
        return None
    return json.dumps(redact_value(value, redact_keys), separators=(",", ":"))
