"""Zero-setup LangChain instrumentation (env auto + ``instrument()``).

When ``SOFTPROBE_PUBLIC_KEY`` and ``SOFTPROBE_BASE_URL`` are set, Softprobe
registers into LangChain's callback configure hooks so every run is traced
without editing ``callbacks``.

Opt out: ``SOFTPROBE_LANGCHAIN=0``.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from softprobe.config import resolve_softprobe_config_from_env

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers.context import register_configure_hook
except ImportError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Install langchain-core to use softprobe.langchain: "
        "pip install 'softprobe[langchain]'"
    ) from exc

if TYPE_CHECKING:
    from softprobe.langchain import CallbackHandler

_CONTEXT_VAR: ContextVar[BaseCallbackHandler | None] = ContextVar(
    "softprobe_langchain_handler",
    default=None,
)
_hook_registered = False
_active_handler: CallbackHandler | None = None


def _env_flag_disabled(name: str = "SOFTPROBE_LANGCHAIN") -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"0", "false", "off", "no"}


def is_instrumented() -> bool:
    return _CONTEXT_VAR.get() is not None


def get_handler() -> CallbackHandler | None:
    """Return the active auto/instrument handler, if any."""
    return _active_handler


def instrument(**handler_kwargs: Any) -> CallbackHandler:
    """Enable Softprobe on all LangChain / LangGraph runs in this process.

    Registers a LangChain configure hook (additive — does not replace other
    handlers). Safe to call more than once; returns the active handler.
    """
    global _hook_registered, _active_handler
    from softprobe.langchain import CallbackHandler as _CallbackHandler

    if not _hook_registered:
        register_configure_hook(_CONTEXT_VAR, True)
        _hook_registered = True
    if _active_handler is None or handler_kwargs:
        _active_handler = _CallbackHandler(**handler_kwargs)
    _CONTEXT_VAR.set(_active_handler)
    return _active_handler


def uninstrument() -> None:
    """Stop injecting Softprobe into new LangChain runs."""
    global _active_handler
    _CONTEXT_VAR.set(None)
    _active_handler = None


def auto_instrument_from_env(**handler_kwargs: Any) -> CallbackHandler | None:
    """Instrument when Softprobe credentials are present (unless disabled)."""
    if _env_flag_disabled():
        return None
    if resolve_softprobe_config_from_env() is None:
        return None
    return instrument(**handler_kwargs)
