#!/usr/bin/env python3
"""Phase 2 nested Softprobe Python sample app."""

from __future__ import annotations

import json
import os
from pathlib import Path

from softprobe import SoftprobeClient

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(os.environ.get("PHASE2_IDS_PATH", ROOT / "examples" / ".phase2-ids-py.json"))
SESSION_ID = os.environ.get("PHASE2_SESSION_ID", "sess-phase2-py")
BASE_URL = os.environ.get("SOFTPROBE_BASE_URL", "http://127.0.0.1:8090")
PUBLIC_KEY = os.environ.get("SOFTPROBE_PUBLIC_KEY", "e2e-token")
OTLP_ENDPOINT = os.environ.get("SOFTPROBE_OTLP_ENDPOINT", f"{BASE_URL}/v1/traces")


def main() -> None:
    client = SoftprobeClient(
        public_key=PUBLIC_KEY,
        base_url=BASE_URL,
        otlp_endpoint=OTLP_ENDPOINT,
        service_name="phase2-example-py",
        service_version="0.1.0",
        environment="e2e",
    )

    trace_id = ""
    agent_span_id = ""
    generation_span_id = ""

    with client.observation(
        name="phase2.agent",
        as_type="agent",
        session_id=SESSION_ID,
        user_id="user-phase2-py",
        tags=["phase2", "python"],
        input={"goal": "phase2"},
    ) as agent:
        trace_id = agent.trace_id
        agent_span_id = agent.span_id

        with client.observation(
            name="phase2.chain",
            as_type="chain",
            session_id=SESSION_ID,
        ):
            with client.observation(
                name="phase2.retriever",
                as_type="retriever",
                session_id=SESSION_ID,
                input={"query": "docs"},
                output={"docs": ["a"]},
            ):
                client.start_embedding(
                    name="phase2.embedding",
                    session_id=SESSION_ID,
                    attributes={
                        "gen_ai.request.model": "text-embedding-3-small",
                        "gen_ai.usage.input_tokens": 8,
                        "gen_ai.usage.total_tokens": 8,
                    },
                ).end()

        with client.observation(
            name="phase2.tool",
            as_type="tool",
            session_id=SESSION_ID,
            input={"name": "lookup"},
            output={"ok": True},
        ) as tool:
            tool.add_content_event(
                "gen_ai.tool.message",
                {"role": "tool", "name": "lookup", "content": "ok"},
            )

        with client.generation(
            name="phase2.generation",
            session_id=SESSION_ID,
            model="gpt-4o-mini",
            provider="openai",
            operation_name="chat",
            usage={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
            cost={"input": 0.0001, "output": 0.0002, "total": 0.0003},
            input={"messages": [{"role": "user", "content": "hi"}]},
            output={"content": "hello from py"},
            prompt_event=[{"role": "user", "content": "hi"}],
            completion_event=[{"role": "assistant", "content": "hello from py"}],
        ) as generation:
            generation_span_id = generation.span_id
            generation.update(
                response_model="gpt-4o-mini-2024-07-18",
                finish_reasons=["stop"],
            )

        client.start_evaluator(
            name="phase2.evaluator",
            session_id=SESSION_ID,
            output={"score": 0.95},
        ).end()
        client.start_guardrail(
            name="phase2.guardrail",
            session_id=SESSION_ID,
            output={"passed": True},
        ).end()
        client.start_observation(
            name="phase2.span",
            as_type="span",
            session_id=SESSION_ID,
        ).end()

    score_ids = {
        "span": f"score-phase2-py-span-{generation_span_id}",
        "trace": f"score-phase2-py-trace-{trace_id[:16]}",
        "session": f"score-phase2-py-session-{SESSION_ID}",
    }
    client.create_score(
        score_id=score_ids["span"],
        name="faithfulness",
        data_type="numeric",
        source="evaluator",
        numeric_value=0.91,
        trace_id=trace_id,
        span_id=generation_span_id,
    )
    client.create_score(
        score_id=score_ids["trace"],
        name="quality",
        data_type="categorical",
        source="api",
        string_value="good",
        trace_id=trace_id,
    )
    client.create_score(
        score_id=score_ids["session"],
        name="thumbs",
        data_type="boolean",
        source="user",
        boolean_value=True,
        session_id=SESSION_ID,
    )

    client.force_flush()
    client.shutdown()

    payload = {
        "language": "python",
        "sessionId": SESSION_ID,
        "traceId": trace_id,
        "agentSpanId": agent_span_id,
        "generationSpanId": generation_span_id,
        "scoreIds": score_ids,
        "serviceName": "phase2-example-py",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
