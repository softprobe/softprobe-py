#!/usr/bin/env python3
"""Issue #7 Part A: agent → generation → tool with tool-call attributes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from softprobe import SoftprobeClient
from softprobe.tools import record_tool_calls, record_tool_definitions

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(
    os.environ.get("TOOL_CALL_IDS_PATH", ROOT / "examples" / ".tool-call-ids-py.json")
)
SESSION_ID = os.environ.get("TOOL_CALL_SESSION_ID", "sess-tool-call-py")
BASE_URL = os.environ.get("SOFTPROBE_BASE_URL", "http://127.0.0.1:8090")
PUBLIC_KEY = os.environ.get("SOFTPROBE_PUBLIC_KEY", "e2e-token")
OTLP_ENDPOINT = os.environ.get("SOFTPROBE_OTLP_ENDPOINT", f"{BASE_URL}/v1/traces")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup a document",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        },
    }
]
TOOL_CALLS = [
    {
        "id": "call_lookup_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"docs"}'},
    }
]


def main() -> None:
    client = SoftprobeClient(
        public_key=PUBLIC_KEY,
        base_url=BASE_URL,
        otlp_endpoint=OTLP_ENDPOINT,
        service_name="tool-call-example-py",
        service_version="0.1.0",
        environment="e2e",
    )

    trace_id = ""
    agent_span_id = ""
    generation_span_id = ""
    tool_span_id = ""

    with client.observation(
        name="toolcall.agent",
        as_type="agent",
        session_id=SESSION_ID,
        user_id="user-tool-call-py",
        tags=["tool-call", "python"],
        input={"goal": "answer with tools"},
    ) as agent:
        trace_id = agent.trace_id
        agent_span_id = agent.span_id

        with client.generation(
            name="toolcall.generation",
            session_id=SESSION_ID,
            model="gpt-4o-mini",
            provider="openai",
            operation_name="chat",
            usage={"input_tokens": 40, "output_tokens": 12, "total_tokens": 52},
            input={
                "messages": [{"role": "user", "content": "find docs"}],
                "tools": TOOLS,
            },
            output={
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_lookup_1",
                        "name": "lookup",
                        "arguments": '{"q":"docs"}',
                    }
                ],
            },
            prompt_event=[{"role": "user", "content": "find docs"}],
            completion_event=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_lookup_1",
                            "name": "lookup",
                            "arguments": '{"q":"docs"}',
                        }
                    ],
                }
            ],
        ) as generation:
            generation_span_id = generation.span_id
            record_tool_definitions(generation, TOOLS)
            record_tool_calls(generation, TOOL_CALLS)
            generation.update(
                response_model="gpt-4o-mini-2024-07-18",
                finish_reasons=["tool_calls"],
            )

            tool = client.start_tool(
                name="toolcall.lookup",
                tool_name="lookup",
                tool_call_id="call_lookup_1",
                kind="function",
                status="ok",
                index=0,
                parent=generation,
                session_id=SESSION_ID,
                input={"q": "docs"},
                output={"ok": True, "hits": ["a"]},
            )
            tool_span_id = tool.span_id
            tool.add_content_event(
                "gen_ai.tool.message",
                {
                    "role": "tool",
                    "name": "lookup",
                    "tool_call_id": "call_lookup_1",
                    "content": "ok",
                },
            )
            tool.end()

    payload = {
        "language": "python",
        "serviceName": "tool-call-example-py",
        "sessionId": SESSION_ID,
        "traceId": trace_id,
        "agentSpanId": agent_span_id,
        "generationSpanId": generation_span_id,
        "toolSpanId": tool_span_id,
        "toolCallId": "call_lookup_1",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    client.shutdown()


if __name__ == "__main__":
    main()
