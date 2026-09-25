"""Langfuse-style OpenAI instrumentation for Softprobe.

Usage::

    from softprobe import SoftprobeClient
    from softprobe.openai import observe_openai
    from openai import OpenAI

    sp = SoftprobeClient(...)
    client = observe_openai(OpenAI(), softprobe_client=sp)
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        name="my-generation",       # Softprobe-only kwarg
        session_id=chat_id,         # your app conversation id (or omit → env / defaults)
    )

Tool definitions and assistant ``tool_calls`` are recorded on the generation
span. Execution spans are app-owned — wrap each tool run yourself::

    with sp.start_tool(
        name=call.name,
        tool_name=call.name,
        tool_call_id=call.id,
        parent=generation,
    ) as tool:
        result = run_tool(call)
        tool.update(output=result)
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from types import SimpleNamespace
from typing import Any

from softprobe.client import SoftprobeClient
from softprobe.observation import Generation
from softprobe.tools import normalize_tool_calls, record_tool_calls, record_tool_definitions

SOFTPROBE_KWARGS = frozenset(
    {
        "name",
        "session_id",
        "user_id",
        "metadata",
        "softprobe_prompt",
        "tags",
        "release",
    }
)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            try:
                return value.model_dump()
            except Exception:
                return str(value)
    if isinstance(value, SimpleNamespace):
        return {key: _serialize(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        data = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if data:
            return {key: _serialize(item) for key, item in data.items()}
    return value


def _split_kwargs(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    softprobe_args: dict[str, Any] = {}
    openai_args: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in SOFTPROBE_KWARGS:
            softprobe_args[key] = value
        else:
            openai_args[key] = value
    return softprobe_args, openai_args


def _usage_from_response(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    result: dict[str, int] = {}
    if input_tokens is not None:
        result["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        result["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        result["total_tokens"] = int(total_tokens)
    elif result:
        result["total_tokens"] = int(result.get("input_tokens", 0)) + int(
            result.get("output_tokens", 0)
        )
    return result


def _completion_output(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return _serialize(response)
    message = getattr(choices[0], "message", None)
    if message is None:
        return _serialize(choices[0])
    return _serialize(message)


def _message_tool_calls(response: Any) -> list[Any]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    if message is None:
        return []
    raw = getattr(message, "tool_calls", None)
    if raw is None and isinstance(message, Mapping):
        raw = message.get("tool_calls")
    return list(raw) if raw else []


def _request_input(
    messages: Any,
    tools: Any,
    tool_choice: Any,
) -> dict[str, Any] | None:
    if messages is None and tools is None and tool_choice is None:
        return None
    payload: dict[str, Any] = {}
    if messages is not None:
        payload["messages"] = _serialize(messages)
    if tools is not None:
        payload["tools"] = _serialize(tools)
    if tool_choice is not None:
        payload["tool_choice"] = _serialize(tool_choice)
    return payload


def _finish_reasons(response: Any) -> list[str] | None:
    choices = getattr(response, "choices", None) or []
    reasons = []
    for choice in choices:
        reason = getattr(choice, "finish_reason", None)
        if reason:
            reasons.append(str(reason))
    return reasons or None


def _provider_from_client(client: Any) -> str:
    base_url = str(getattr(client, "base_url", "") or "")
    if "generativelanguage.googleapis.com" in base_url:
        return "google"
    if "azure" in base_url.lower():
        return "azure"
    return "openai"


def _trace_chat_completion(
    *,
    softprobe_client: SoftprobeClient,
    openai_client: Any,
    method: Any,
    args: tuple[Any, ...],
    kwargs: MutableMapping[str, Any],
    defaults: Mapping[str, Any],
    id_sink: MutableMapping[str, str] | None = None,
) -> Any:
    softprobe_args, openai_args = _split_kwargs(kwargs)
    if openai_args.get("stream"):
        raise NotImplementedError(
            "softprobe.openai streaming is not supported yet; call without stream=True"
        )

    name = softprobe_args.get("name") or defaults.get("generation_name") or "OpenAI-generation"
    session_id = (
        softprobe_args.get("session_id")
        or defaults.get("session_id")
        or os.environ.get("SOFTPROBE_SESSION_ID")
    )
    user_id = (
        softprobe_args.get("user_id")
        or defaults.get("user_id")
        or os.environ.get("SOFTPROBE_USER_ID")
    )
    tags = softprobe_args.get("tags") or defaults.get("tags")
    release = softprobe_args.get("release") or defaults.get("release")
    model = openai_args.get("model")
    messages = openai_args.get("messages")
    tools = openai_args.get("tools")
    tool_choice = openai_args.get("tool_choice")
    temperature = openai_args.get("temperature")
    max_tokens = openai_args.get("max_tokens") or openai_args.get("max_completion_tokens")

    generation: Generation = softprobe_client.start_generation(
        name=str(name),
        session_id=session_id,
        user_id=user_id,
        tags=list(tags) if tags is not None else None,
        release=release,
        model=str(model) if model is not None else None,
        provider=_provider_from_client(openai_client),
        operation_name="chat",
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        input=_request_input(messages, tools, tool_choice),
        prompt_event=_serialize(messages) if messages is not None else None,
        inference_details={
            "provider": _provider_from_client(openai_client),
            "model": model,
            **({"tool_choice": _serialize(tool_choice)} if tool_choice is not None else {}),
        },
    )
    if tools is not None:
        record_tool_definitions(generation, tools)
    if id_sink is not None:
        id_sink["span_id"] = generation.span_id
        id_sink["trace_id"] = generation.trace_id

    try:
        response = method(*args, **openai_args)
        response_model = getattr(response, "model", None) or model
        response_id = getattr(response, "id", None)
        output = _completion_output(response)
        raw_tool_calls = _serialize(_message_tool_calls(response))
        if not isinstance(raw_tool_calls, list):
            raw_tool_calls = []
        normalized_calls = normalize_tool_calls(raw_tool_calls)
        if normalized_calls and isinstance(output, dict):
            output = {**output, "tool_calls": normalized_calls}
        if normalized_calls:
            record_tool_calls(generation, raw_tool_calls)
        generation.update(
            response_model=str(response_model) if response_model is not None else None,
            response_id=str(response_id) if response_id is not None else None,
            usage=_usage_from_response(response),
            finish_reasons=_finish_reasons(response),
            output=output,
            completion_event=output,
        )
        generation.end()
        return response
    except Exception as exc:
        generation.record_exception(exc)
        generation.end()
        raise


def observe_openai(
    client: Any,
    *,
    softprobe_client: SoftprobeClient,
    generation_name: str = "OpenAI-generation",
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    release: str | None = None,
) -> Any:
    """Return a proxied OpenAI client that traces chat.completions.create."""

    defaults = {
        "generation_name": generation_name,
        "session_id": session_id,
        "user_id": user_id,
        "tags": tags,
        "release": release,
    }
    id_sink: dict[str, str] = {}

    class _CompletionsProxy:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def create(self, *args: Any, **kwargs: Any) -> Any:
            return _trace_chat_completion(
                softprobe_client=softprobe_client,
                openai_client=client,
                method=self._wrapped.create,
                args=args,
                kwargs=kwargs,
                defaults=defaults,
                id_sink=id_sink,
            )

        def __getattr__(self, item: str) -> Any:
            return getattr(self._wrapped, item)

    class _ChatProxy:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        @property
        def completions(self) -> _CompletionsProxy:
            return _CompletionsProxy(self._wrapped.completions)

        def __getattr__(self, item: str) -> Any:
            return getattr(self._wrapped, item)

    class _ClientProxy:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        @property
        def chat(self) -> _ChatProxy:
            return _ChatProxy(self._wrapped.chat)

        @property
        def last_generation_span_id(self) -> str | None:
            return id_sink.get("span_id")

        @property
        def last_generation_trace_id(self) -> str | None:
            return id_sink.get("trace_id")

        def __getattr__(self, item: str) -> Any:
            return getattr(self._wrapped, item)

    return _ClientProxy(client)


def create_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a plain OpenAI client (requires the optional `openai` extra)."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Install the OpenAI extra: pip install 'softprobe[openai]'"
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url, **kwargs)


def create_gemini_openai_client(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    **kwargs: Any,
) -> Any:
    """Create an OpenAI-compatible Gemini client.

    `model` is unused for client construction but documented for callers.
    """
    _ = model
    return create_openai_client(
        api_key=api_key,
        base_url=GEMINI_OPENAI_BASE_URL,
        **kwargs,
    )
