"""Tool-call normalization and attribute helpers (Issue #7 Part A)."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from softprobe.types import Attributes

TOOL_KINDS = ("function", "mcp", "shell", "file", "hook", "other")
TOOL_STATUSES = ("ok", "error", "cancelled")


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _arguments_value(raw: Any) -> Any:
    if raw is None:
        return {}
    if isinstance(raw, (dict, list, str, int, float, bool)):
        return raw
    return str(raw)


def normalize_tool_definitions(tools: Any) -> list[dict[str, Any]]:
    """Flatten provider tool definition shapes to Softprobe definitions."""
    if not tools:
        return []
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for item in tools:
        mapping = _as_mapping(item)
        if mapping is None:
            continue
        # OpenAI Chat Completions: {type, function: {name, description, parameters}}
        function = _as_mapping(mapping.get("function"))
        if function and function.get("name"):
            out.append(
                {
                    "name": str(function["name"]),
                    "description": function.get("description"),
                    "parameters": function.get("parameters"),
                }
            )
            continue
        # Flat / AI SDK / Anthropic-ish
        name = mapping.get("name") or mapping.get("toolName")
        if name:
            out.append(
                {
                    "name": str(name),
                    "description": mapping.get("description"),
                    "parameters": mapping.get("parameters")
                    or mapping.get("input_schema")
                    or mapping.get("inputSchema"),
                }
            )
    return out


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Flatten provider tool_call shapes to {id?, name, arguments}."""
    if not tool_calls:
        return []
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(tool_calls):
        mapping = _as_mapping(item)
        if mapping is None:
            continue
        function = _as_mapping(mapping.get("function"))
        name = None
        arguments: Any = {}
        call_id = mapping.get("id") or mapping.get("toolCallId") or mapping.get("tool_call_id")
        if function:
            name = function.get("name")
            arguments = _arguments_value(function.get("arguments"))
        else:
            name = mapping.get("name") or mapping.get("toolName")
            arguments = _arguments_value(
                mapping.get("arguments")
                if "arguments" in mapping
                else mapping.get("args")
                if "args" in mapping
                else mapping.get("input")
            )
        if not name:
            continue
        normalized: dict[str, Any] = {
            "name": str(name),
            "arguments": arguments,
            "index": mapping.get("index", index),
        }
        if call_id:
            normalized["id"] = str(call_id)
        if mapping.get("type"):
            normalized["type"] = mapping["type"]
        out.append(normalized)
    return out


def tool_definition_attributes(definitions: Sequence[Mapping[str, Any]]) -> Attributes:
    names = [str(item["name"]) for item in definitions if item.get("name")]
    return {
        "sp.tool.available_names": names,
        "sp.tool.available_count": len(names),
    }


def tool_call_attributes(calls: Sequence[Mapping[str, Any]]) -> Attributes:
    names = [str(item["name"]) for item in calls if item.get("name")]
    ids = [str(item["id"]) for item in calls if item.get("id")]
    attrs: Attributes = {
        "sp.tool.call_names": names,
        "sp.tool.call_count": len(names),
    }
    if ids:
        attrs["sp.tool.call_ids"] = ids
    return attrs


def tool_span_attributes(
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    index: int | None = None,
    mcp_server: str | None = None,
    mcp_tool: str | None = None,
) -> Attributes:
    attrs: Attributes = {}
    if tool_name:
        attrs["gen_ai.tool.name"] = tool_name
    if tool_call_id:
        attrs["gen_ai.tool.call.id"] = tool_call_id
    if kind:
        attrs["sp.tool.kind"] = kind if kind in TOOL_KINDS else "other"
    if status:
        attrs["sp.tool.status"] = status if status in TOOL_STATUSES else status
    if index is not None:
        attrs["sp.tool.index"] = int(index)
    if mcp_server:
        attrs["sp.mcp.server"] = mcp_server
    if mcp_tool:
        attrs["sp.mcp.tool"] = mcp_tool
    return attrs


def record_tool_definitions(observation: Any, tools: Any) -> list[dict[str, Any]]:
    """Attach available-tool summary attributes to a generation (or any observation)."""
    definitions = normalize_tool_definitions(tools)
    attrs = tool_definition_attributes(definitions)
    if attrs:
        observation.update(attributes=attrs)
    return definitions


def record_tool_calls(observation: Any, tool_calls: Any) -> list[dict[str, Any]]:
    """Attach requested tool-call summary attributes to a generation."""
    calls = normalize_tool_calls(tool_calls)
    attrs = tool_call_attributes(calls)
    if attrs:
        observation.update(attributes=attrs)
    return calls


def tool_result_event_payload(
    *,
    name: str,
    content: Any,
    tool_call_id: str | None = None,
    role: str = "tool",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": role, "name": name, "content": content}
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def accumulate_tool_call_deltas(
    state: MutableMapping[int, dict[str, Any]],
    deltas: Any,
) -> MutableMapping[int, dict[str, Any]]:
    """Buffer streaming tool_call fragments keyed by index (Langfuse Python pattern)."""
    if not deltas:
        return state
    if not isinstance(deltas, Sequence) or isinstance(deltas, (str, bytes)):
        return state
    for delta in deltas:
        mapping = _as_mapping(delta)
        if mapping is None:
            continue
        index = int(mapping.get("index", 0))
        entry = state.setdefault(
            index,
            {"id": None, "type": None, "name": None, "arguments": ""},
        )
        if mapping.get("id") and not entry.get("id"):
            entry["id"] = str(mapping["id"])
        if mapping.get("type") and not entry.get("type"):
            entry["type"] = mapping["type"]
        function = _as_mapping(mapping.get("function")) or {}
        if function.get("name") and not entry.get("name"):
            entry["name"] = str(function["name"])
        if function.get("arguments"):
            entry["arguments"] = str(entry.get("arguments") or "") + str(
                function["arguments"]
            )
        # Flat delta shapes
        if mapping.get("name") and not entry.get("name"):
            entry["name"] = str(mapping["name"])
        if mapping.get("arguments") is not None and not function:
            entry["arguments"] = str(entry.get("arguments") or "") + str(
                mapping["arguments"]
            )
    return state


def finalize_tool_call_deltas(
    state: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert buffered deltas into normalized tool calls ordered by index."""
    calls: list[dict[str, Any]] = []
    for index in sorted(state.keys()):
        entry = state[index]
        name = entry.get("name")
        if not name:
            continue
        call: dict[str, Any] = {
            "name": str(name),
            "arguments": entry.get("arguments") or "",
            "index": index,
        }
        if entry.get("id"):
            call["id"] = str(entry["id"])
        if entry.get("type"):
            call["type"] = entry["type"]
        calls.append(call)
    return calls


def dumps_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
