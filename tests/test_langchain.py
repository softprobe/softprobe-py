from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from softprobe.client import SoftprobeClient
from softprobe.identity import resolve_run_identity
from softprobe.langchain import CallbackHandler
from softprobe.langchain_instrument import (
    auto_instrument_from_env,
    instrument,
    is_instrumented,
    uninstrument,
)
from softprobe.normalize import normalize_readable_spans
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


def test_langchain_chain_tool_generation_parenting() -> None:
    client, exporter = make_client()
    handler = CallbackHandler(
        softprobe_client=client,
        session_id="sess-lc",
        user_id="user-lc",
        tags=["langchain"],
        metadata={"feature": "math"},
    )

    chain_id = uuid4()
    tool_id = uuid4()
    llm_id = uuid4()

    handler.on_chain_start(
        {"name": "agent", "id": ["agent"]},
        {"input": "2+2"},
        run_id=chain_id,
        name="agent",
    )
    handler.on_tool_start(
        {"name": "calculator"},
        "2+2",
        run_id=tool_id,
        parent_run_id=chain_id,
        name="calculator",
        metadata={"tool_call_id": "call_calc_1"},
    )
    handler.on_tool_end("4", run_id=tool_id)
    handler.on_chat_model_start(
        {"id": ["ChatOpenAI"], "kwargs": {"model": "gpt-4o-mini"}},
        [[HumanMessage(content="summarize")]],
        run_id=llm_id,
        parent_run_id=chain_id,
        name="ChatOpenAI",
        invocation_params={
            "temperature": 0,
            "model": "gpt-4o-mini",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "calculator", "parameters": {"type": "object"}},
                }
            ],
        },
        metadata={"ls_model_name": "gpt-4o-mini"},
    )
    handler.on_llm_new_token("ok", run_id=llm_id)
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="4"))]],
            llm_output={"token_usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
        ),
        run_id=llm_id,
    )
    handler.on_chain_end({"output": "4"}, run_id=chain_id)

    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    by_name = {span["name"]: span for span in spans}
    assert by_name["calculator"]["parent_name"] == "agent"
    assert by_name["ChatOpenAI"]["parent_name"] == "agent"
    assert by_name["ChatOpenAI"]["observation_type"] == "generation"
    assert by_name["calculator"]["observation_type"] == "tool"
    assert by_name["calculator"]["attributes"]["gen_ai.tool.name"] == "calculator"
    assert by_name["calculator"]["attributes"]["gen_ai.tool.call.id"] == "call_calc_1"
    assert by_name["ChatOpenAI"]["attributes"]["sp.tool.available_names"] == ["calculator"]
    assert by_name["agent"]["attributes"].get("sp.session.id") == "sess-lc"
    assert by_name["agent"]["attributes"].get("sp.metadata.feature") == "math"
    event_names = {e["name"] for e in by_name["calculator"]["events"]}
    assert "gen_ai.tool.message" in event_names
    assert handler.last_trace_id
    client.shutdown()


def test_tool_without_provider_id_omits_call_id() -> None:
    client, exporter = make_client()
    handler = CallbackHandler(softprobe_client=client)
    tool_id = uuid4()
    handler.on_tool_start({"name": "calculator"}, "2+2", run_id=tool_id, name="calculator")
    handler.on_tool_end("4", run_id=tool_id)
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    attrs = spans[0]["attributes"]
    assert attrs["gen_ai.tool.name"] == "calculator"
    assert "gen_ai.tool.call.id" not in attrs
    client.shutdown()


def test_langchain_retriever() -> None:
    client, exporter = make_client()
    handler = CallbackHandler(softprobe_client=client)
    run_id = uuid4()
    handler.on_retriever_start({"name": "docs"}, "refund rules", run_id=run_id, name="docs")
    handler.on_retriever_end(
        [Document(page_content="BASIC-INTL-7", metadata={"source": "fare"})],
        run_id=run_id,
    )
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    assert spans[0]["observation_type"] == "retriever"
    client.shutdown()


def test_langchain_uses_thread_id_from_metadata() -> None:
    client, exporter = make_client()
    handler = CallbackHandler(softprobe_client=client)
    chain_id = uuid4()
    handler.on_chain_start(
        {"name": "agent", "id": ["agent"]},
        {"input": "hi"},
        run_id=chain_id,
        name="agent",
        metadata={"thread_id": "chat-42", "user_id": "u-9"},
    )
    handler.on_chain_end({"output": "ok"}, run_id=chain_id)
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    attrs = spans[0]["attributes"]
    assert attrs.get("sp.session.id") == "chat-42"
    assert attrs.get("sp.user.id") == "u-9"
    client.shutdown()


def test_resolve_run_identity_prefers_app_ids() -> None:
    session_id, user_id = resolve_run_identity(
        metadata={"configurable": {"thread_id": "t1", "user_id": "u1"}},
        fallback_session_id="fallback",
        env={"SOFTPROBE_SESSION_ID": "env-sess"},
    )
    assert session_id == "t1"
    assert user_id == "u1"


def test_instrument_injects_into_callback_manager() -> None:
    uninstrument()
    client, _exporter = make_client()
    handler = instrument(softprobe_client=client)
    assert is_instrumented()
    manager = CallbackManager.configure()
    assert any(h.name == "SoftprobeCallbackHandler" for h in manager.handlers)
    assert handler in manager.handlers or any(
        getattr(h, "name", None) == "SoftprobeCallbackHandler" for h in manager.handlers
    )
    uninstrument()
    assert not is_instrumented()
    client.shutdown()


def test_auto_instrument_from_env(monkeypatch) -> None:
    uninstrument()
    monkeypatch.setenv("SOFTPROBE_PUBLIC_KEY", "pk-auto")
    monkeypatch.setenv("SOFTPROBE_BASE_URL", "http://127.0.0.1:8091")
    monkeypatch.delenv("SOFTPROBE_LANGCHAIN", raising=False)

    # Avoid real OTLP: inject a no-export client via instrument kwargs after auto check.
    # auto_instrument_from_env creates CallbackHandler.from_env — use simple processor via
    # monkeypatch SoftprobeClient.from_env.
    from softprobe import client as client_mod

    exporter = InMemorySpanExporter()

    def _from_env(**overrides):
        return SoftprobeClient(
            public_key="pk-auto",
            base_url="http://127.0.0.1:8091",
            otlp_endpoint="http://127.0.0.1:8091/v1/traces",
            span_exporter=exporter,
            score_transport=MemoryScoreTransport(),
            use_simple_processor=True,
            **{k: v for k, v in overrides.items() if k not in ("public_key", "base_url")},
        )

    monkeypatch.setattr(client_mod.SoftprobeClient, "from_env", classmethod(lambda cls, **kw: _from_env(**kw)))

    h = auto_instrument_from_env()
    assert h is not None
    assert is_instrumented()
    manager = CallbackManager.configure()
    assert any(getattr(x, "name", None) == "SoftprobeCallbackHandler" for x in manager.handlers)

    monkeypatch.setenv("SOFTPROBE_LANGCHAIN", "0")
    uninstrument()
    assert auto_instrument_from_env() is None
    uninstrument()


def test_explicit_parent_and_metadata() -> None:
    client, exporter = make_client()
    parent = client.start_observation(name="parent", as_type="agent")
    child = client.start_observation(
        name="child",
        as_type="tool",
        parent=parent,
        metadata={"k": "v"},
    )
    child.end(output={"ok": True})
    parent.end()
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    child_span = next(span for span in spans if span["name"] == "child")
    assert child_span["parent_name"] == "parent"
    assert child_span["attributes"]["sp.metadata.k"] == "v"
    client.shutdown()


def test_langgraph_state_and_tool_message_serialize() -> None:
    """LangGraph passes State / ToolMessage objects — must not raise on JSON."""
    from langchain_core.messages import ToolMessage

    from softprobe.redaction import serialize_captured

    class FakeState:
        def __init__(self) -> None:
            self.messages = [HumanMessage(content="hi"), ToolMessage(content="21", tool_call_id="c1")]

        def model_dump(self, mode: str = "python") -> dict:
            return {"messages": self.messages}

    raw = serialize_captured(
        {"messages": [HumanMessage(content="3*7"), ToolMessage(content="21", tool_call_id="c1")]},
        [],
    )
    assert raw is not None
    parsed = json.loads(raw)
    assert parsed["messages"][0]["content"] == "3*7"
    assert parsed["messages"][1]["tool_call_id"] == "c1"

    state_json = serialize_captured(FakeState(), [])
    assert state_json is not None
    assert "21" in state_json

    client, exporter = make_client()
    handler = CallbackHandler(softprobe_client=client, session_id="sess-ser")
    chain_id = uuid4()
    tool_id = uuid4()
    handler.on_chain_start(
        {"name": "agent"},
        FakeState(),
        run_id=chain_id,
        name="ReAct Agent",
    )
    handler.on_tool_start(
        {"name": "multiply"},
        "a=3,b=7",
        run_id=tool_id,
        parent_run_id=chain_id,
        name="multiply",
    )
    handler.on_tool_end(ToolMessage(content="21", tool_call_id="c1"), run_id=tool_id)
    handler.on_chain_end({"messages": [HumanMessage(content="done")]}, run_id=chain_id)
    client.force_flush()
    spans = normalize_readable_spans(list(exporter.get_finished_spans()))
    assert any(s["name"] == "multiply" for s in spans)
    client.shutdown()
