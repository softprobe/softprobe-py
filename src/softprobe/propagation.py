"""Scoped Softprobe attribute propagation via contextvars."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Generator, Mapping

_propagation: ContextVar[dict[str, Any] | None] = ContextVar(
    "softprobe_propagation", default=None
)


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
    """Temporarily override Softprobe defaults for nested observations."""
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
    token: Token[dict[str, Any] | None] = _propagation.set(merged)
    try:
        yield
    finally:
        _propagation.reset(token)
