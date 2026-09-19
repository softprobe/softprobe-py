"""Resolve product conversation / user identity from the builder's world.

Priority: LangChain/LangGraph run metadata (incl. configurable) →
explicit Softprobe fallbacks → SOFTPROBE_* env.
Softprobe never invents a session id.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

SESSION_KEYS = (
    "thread_id",
    "threadId",
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
    "chat_id",
    "chatId",
)

USER_KEYS = ("user_id", "userId", "user")


def _as_non_empty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _first_key(sources: list[Mapping[str, Any] | None], keys: tuple[str, ...]) -> str | None:
    for source in sources:
        if not source:
            continue
        for key in keys:
            found = _as_non_empty(source.get(key))
            if found:
                return found
    return None


def identity_sources_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> list[Mapping[str, Any] | None]:
    if not metadata:
        return []
    configurable = metadata.get("configurable")
    if isinstance(configurable, Mapping):
        return [configurable, metadata]
    return [metadata]


def resolve_run_identity(
    *,
    metadata: Mapping[str, Any] | None = None,
    fallback_session_id: str | None = None,
    fallback_user_id: str | None = None,
    env: Mapping[str, str | None] | None = None,
) -> tuple[str | None, str | None]:
    source_env = env if env is not None else os.environ
    from_meta = identity_sources_from_metadata(metadata)
    session_id = (
        _first_key(from_meta, SESSION_KEYS)
        or _as_non_empty(fallback_session_id)
        or _as_non_empty(source_env.get("SOFTPROBE_SESSION_ID"))
    )
    user_id = (
        _first_key(from_meta, USER_KEYS)
        or _as_non_empty(fallback_user_id)
        or _as_non_empty(source_env.get("SOFTPROBE_USER_ID"))
    )
    return session_id, user_id
