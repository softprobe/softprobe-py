from __future__ import annotations

from types import SimpleNamespace

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from softprobe.client import SoftprobeClient
from softprobe.normalize import normalize_readable_spans
from softprobe.openai import observe_openai
from softprobe.types import ScoreRequest, ScoreTransport


class MemoryScoreTransport(ScoreTransport):
    def create_score(self, request: ScoreRequest) -> None:
        return None


def make_client() -> tuple[SoftprobeClient, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    client = SoftprobeClient(
        public_key="test-key",
        base_url="http://127.0.0.1:8091",
        otlp_endpoint="http://127.0.0.1:8091/v1/traces",
        span_exporter=exporter,
        score_transport=MemoryScoreTransport(),
        use_simple_processor=True,
    )
    return client, exporter


class FakeCompletions:
    def create(self, **kwargs):
        assert "name" not in kwargs
        assert "session_id" not in kwargs
        return SimpleNamespace(
            id="chatcmpl-test",
            model="gpt-4o-mini-2024-07-18",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content="2"),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=1,
                total_tokens=11,
            ),
        )


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.base_url = "https://api.openai.com/v1"


def test_observe_openai_maps_generation_attributes() -> None:
    client, exporter = make_client()
    wrapped = observe_openai(
        FakeOpenAI(),
        softprobe_client=client,
        session_id="sess-openai-unit",
    )
    response = wrapped.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1+1="}],
        temperature=0,
        name="math-gen",
        session_id="sess-openai-unit",
    )
    assert response.choices[0].message.content == "2"
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    assert len(spans) == 1
    span = spans[0]
    assert span["name"] == "math-gen"
    assert span["observation_type"] == "generation"
    attrs = span["attributes"]
    assert attrs["sp.observation.type"] == "generation"
    assert attrs["sp.session.id"] == "sess-openai-unit"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 1
    assert attrs["gen_ai.usage.total_tokens"] == 11
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    assert "sp.input" in attrs
    assert "sp.output" in attrs
    event_names = {event["name"] for event in span["events"]}
    assert "gen_ai.content.prompt" in event_names
    assert "gen_ai.content.completion" in event_names
    client.shutdown()


def test_observe_openai_records_errors() -> None:
    class BoomCompletions:
        def create(self, **kwargs):
            raise RuntimeError("provider down")

    class BoomChat:
        def __init__(self) -> None:
            self.completions = BoomCompletions()

    class BoomOpenAI:
        def __init__(self) -> None:
            self.chat = BoomChat()
            self.base_url = "https://api.openai.com/v1"

    client, exporter = make_client()
    wrapped = observe_openai(BoomOpenAI(), softprobe_client=client)
    try:
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        raise AssertionError("expected failure")
    except RuntimeError as exc:
        assert "provider down" in str(exc)
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    assert spans[0]["status_code"] == "ERROR"
    client.shutdown()


def test_observe_openai_records_tools_and_tool_calls() -> None:
    class ToolCompletions:
        def create(self, **kwargs):
            assert "tools" in kwargs
            return SimpleNamespace(
                id="chatcmpl-tools",
                model="gpt-4o-mini-2024-07-18",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_lookup_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="lookup",
                                        arguments='{"q":"docs"}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                ),
            )

    class ToolChat:
        def __init__(self) -> None:
            self.completions = ToolCompletions()

    class ToolOpenAI:
        def __init__(self) -> None:
            self.chat = ToolChat()
            self.base_url = "https://api.openai.com/v1"

    client, exporter = make_client()
    wrapped = observe_openai(ToolOpenAI(), softprobe_client=client)
    wrapped.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "find docs"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup docs",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
        name="tool-gen",
    )
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    attrs = spans[0]["attributes"]
    assert attrs["sp.tool.available_names"] == ["lookup"]
    assert attrs["sp.tool.available_count"] == 1
    assert attrs["sp.tool.call_names"] == ["lookup"]
    assert attrs["sp.tool.call_ids"] == ["call_lookup_1"]
    assert attrs["gen_ai.response.finish_reasons"] == ["tool_calls"]
    assert "tools" in attrs["sp.input"]
    client.shutdown()
