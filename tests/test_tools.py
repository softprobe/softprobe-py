from softprobe.tools import (
    accumulate_tool_call_deltas,
    finalize_tool_call_deltas,
    normalize_tool_calls,
    normalize_tool_definitions,
    record_tool_calls,
    record_tool_definitions,
    tool_result_event_payload,
)


def test_normalize_openai_tools_and_calls() -> None:
    definitions = normalize_tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "find",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    assert definitions == [
        {
            "name": "lookup",
            "description": "find",
            "parameters": {"type": "object"},
        }
    ]
    calls = normalize_tool_calls(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":1}'},
            }
        ]
    )
    assert calls[0]["id"] == "call_1"
    assert calls[0]["name"] == "lookup"
    assert calls[0]["arguments"] == '{"q":1}'


def test_accumulate_parallel_tool_call_deltas() -> None:
    state: dict[int, dict] = {}
    accumulate_tool_call_deltas(
        state,
        [
            {"index": 0, "id": "call_a", "function": {"name": "a", "arguments": "{"}},
            {"index": 1, "id": "call_b", "function": {"name": "b", "arguments": "{"}},
        ],
    )
    accumulate_tool_call_deltas(
        state,
        [
            {"index": 0, "function": {"arguments": '"x":1}'}},
            {"index": 1, "function": {"arguments": '"y":2}'}},
        ],
    )
    calls = finalize_tool_call_deltas(state)
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert calls[0]["arguments"] == '{"x":1}'
    assert calls[1]["arguments"] == '{"y":2}'


def test_record_helpers_update_observation(monkeypatch) -> None:
    updates: list[dict] = []

    class Fake:
        def update(self, **kwargs):
            updates.append(kwargs)

    obs = Fake()
    record_tool_definitions(
        obs,
        [{"type": "function", "function": {"name": "lookup"}}],
    )
    record_tool_calls(
        obs,
        [{"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}],
    )
    assert updates[0]["attributes"]["sp.tool.available_count"] == 1
    assert updates[1]["attributes"]["sp.tool.call_ids"] == ["c1"]
    payload = tool_result_event_payload(
        name="lookup", content="ok", tool_call_id="c1"
    )
    assert payload["tool_call_id"] == "c1"
