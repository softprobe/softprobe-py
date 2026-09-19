from __future__ import annotations

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from softprobe.client import SoftprobeClient
from softprobe.normalize import normalize_readable_spans
from softprobe.redaction import redact_value
from softprobe.scores import build_score_request
from softprobe.types import ScoreRequest, ScoreTransport

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "contracts" / "fixtures"


class MemoryScoreTransport(ScoreTransport):
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create_score(self, request: ScoreRequest) -> None:
        self.requests.append(request.to_dict())


def make_client(
    *,
    redact_keys: list[str] | None = None,
) -> tuple[SoftprobeClient, InMemorySpanExporter, MemoryScoreTransport]:
    exporter = InMemorySpanExporter()
    scores = MemoryScoreTransport()
    client = SoftprobeClient(
        public_key="test-key",
        base_url="http://127.0.0.1:8091",
        otlp_endpoint="http://127.0.0.1:8091/v1/traces",
        service_name="softprobe-sdk-contract",
        service_version="0.1.0",
        environment="test",
        redact_keys=redact_keys,
        span_exporter=exporter,
        score_transport=scores,
        use_simple_processor=True,
    )
    return client, exporter, scores


def test_rejects_missing_public_key() -> None:
    with pytest.raises(ValueError, match="public_key"):
        SoftprobeClient(
            public_key="",
            base_url="http://localhost",
            otlp_endpoint="http://localhost/v1/traces",
        )


def test_derives_otlp_endpoint_from_base_url() -> None:
    exporter = InMemorySpanExporter()
    client = SoftprobeClient(
        public_key="test-key",
        base_url="http://127.0.0.1:8091",
        span_exporter=exporter,
        use_simple_processor=True,
    )
    client.start_observation(name="ping", as_type="span").end()
    client.force_flush()
    assert len(exporter.get_finished_spans()) == 1
    client.shutdown()


def test_historical_timestamps() -> None:
    client, exporter, _ = make_client()
    start_ms = 1_721_390_400_000  # 2024-07-19T12:00:00Z
    end_ms = start_ms + 5_000
    agent = client.start_observation(
        name="opencode.turn",
        as_type="agent",
        start_time=start_ms,
    )
    generation = client.start_generation(
        name="opencode.generation",
        parent=agent,
        start_time=start_ms + 100,
    )
    generation.end(end_time=end_ms)
    agent.end(end_time=end_ms + 50)
    client.force_flush()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["opencode.turn"].start_time // 1_000_000_000 == start_ms // 1000
    assert spans["opencode.generation"].end_time // 1_000_000_000 == end_ms // 1000
    client.shutdown()


def test_parent_child_context() -> None:
    client, exporter, _ = make_client()
    with client.observation(name="parent", as_type="agent"):
        with client.observation(name="child", as_type="tool"):
            pass
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    child = next(span for span in spans if span["name"] == "child")
    assert child["parent_name"] == "parent"
    client.shutdown()


def test_scoped_exception_sets_error_status() -> None:
    client, exporter, _ = make_client()
    with pytest.raises(RuntimeError, match="fail"):
        with client.observation(name="boom", as_type="span"):
            raise RuntimeError("fail")
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    assert spans[0]["status_code"] == "ERROR"
    assert spans[0]["status_message"] == "fail"
    client.shutdown()


def test_contract_nested_spans_match_fixture() -> None:
    expected = json.loads((FIXTURES / "expected-nested-spans.json").read_text())
    client, exporter, _ = make_client()

    with client.observation(
        name="agent.run",
        as_type="agent",
        session_id="sess-contract-1",
        user_id="user-contract-1",
        tags=["contract", "nested"],
        input={"goal": "answer question"},
    ):
        with client.observation(
            name="chain.plan",
            as_type="chain",
            session_id="sess-contract-1",
        ):
            with client.observation(
                name="retriever.search",
                as_type="retriever",
                input={"query": "docs"},
                output={"docs": ["a", "b"]},
            ):
                emb = client.start_embedding(
                    name="embedding.encode",
                    attributes={
                        "gen_ai.request.model": "text-embedding-3-small",
                        "gen_ai.usage.input_tokens": 12,
                        "gen_ai.usage.total_tokens": 12,
                    },
                )
                emb.end()

        with client.generation(
            name="generation.answer",
            session_id="sess-contract-1",
            model="gpt-4o-mini",
            provider="openai",
            operation_name="chat",
            temperature=0.2,
            max_tokens=128,
            usage={"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
            cost={"input": 0.0001, "output": 0.0002, "total": 0.0003},
            prompt={"name": "answer", "version": 1},
            input={"messages": [{"role": "user", "content": "hi"}]},
            output={
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_lookup_1",
                        "name": "lookup",
                        "arguments": '{"name":"lookup"}',
                    }
                ],
            },
            prompt_event=[{"role": "user", "content": "hi"}],
            completion_event=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_lookup_1",
                            "name": "lookup",
                            "arguments": '{"name":"lookup"}',
                        }
                    ],
                }
            ],
            inference_details={"provider": "openai", "model": "gpt-4o-mini"},
            attributes={
                "sp.tool.available_names": ["lookup"],
                "sp.tool.available_count": 1,
                "sp.tool.call_names": ["lookup"],
                "sp.tool.call_count": 1,
                "sp.tool.call_ids": ["call_lookup_1"],
            },
        ) as generation:
            generation.update(
                response_model="gpt-4o-mini-2024-07-18",
                response_id="chatcmpl-contract-1",
                finish_reasons=["tool_calls"],
            )
            tool = client.start_tool(
                name="tool.lookup",
                tool_name="lookup",
                tool_call_id="call_lookup_1",
                kind="function",
                status="ok",
                index=0,
                parent=generation,
                input={"name": "lookup"},
                output={"ok": True},
            )
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

        client.start_evaluator(
            name="evaluator.quality",
            output={"score": 0.9},
        ).end()
        client.start_guardrail(
            name="guardrail.check",
            output={"passed": True},
        ).end()
        client.start_observation(name="span.helper", as_type="span").end()

    client.force_flush()
    actual = normalize_readable_spans(list(exporter.get_finished_spans()))
    expected_sorted = sorted(expected, key=lambda item: item["name"])
    assert actual == expected_sorted
    client.shutdown()


def test_redaction_fixture() -> None:
    privacy = json.loads((FIXTURES / "privacy-redaction.json").read_text())
    assert redact_value(privacy["exampleInput"], privacy["redactKeys"]) == privacy[
        "exampleOutputCaptured"
    ]


def test_scores_match_fixture() -> None:
    expected = json.loads((FIXTURES / "expected-scores.json").read_text())
    client, _, scores = make_client()
    for score in expected:
        client.create_score(
            score_id=score["score_id"],
            timestamp=score["timestamp"],
            trace_id=score["trace_id"],
            span_id=score["span_id"],
            session_id=score["session_id"],
            name=score["name"],
            data_type=score["data_type"],
            numeric_value=score["numeric_value"],
            string_value=score["string_value"],
            boolean_value=score["boolean_value"],
            source=score["source"],
            comment=score["comment"],
            config_id=score["config_id"],
            author_id=score["author_id"],
            metadata=score["metadata"],
        )
    assert scores.requests == expected
    client.shutdown()


def test_invalid_score_rejected() -> None:
    with pytest.raises(ValueError, match="target"):
        build_score_request(
            score_id="x",
            name="n",
            data_type="numeric",
            source="api",
            numeric_value=1.0,
        )


def test_lifecycle() -> None:
    client, exporter, _ = make_client()
    obs = client.start_observation(name="life", as_type="span")
    obs.end()
    obs.end()
    client.force_flush()
    assert len(exporter.get_finished_spans()) == 1
    client.shutdown()
    client.shutdown()


@pytest.mark.asyncio
async def test_async_generation_context() -> None:
    client, exporter, _ = make_client()
    async with client.async_generation(name="async.gen", model="m") as generation:
        generation.update(output={"ok": True})
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    assert spans[0]["name"] == "async.gen"
    assert spans[0]["observation_type"] == "generation"
    client.shutdown()
