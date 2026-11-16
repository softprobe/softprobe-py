from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from softprobe import (
    BaggageSessionSpanProcessor,
    SoftprobeClient,
    extract_trace_headers,
    inject_trace_headers,
    propagate_attributes,
)


def _make_client() -> tuple[SoftprobeClient, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    client = SoftprobeClient(
        ingestion_key="spk_test",
        base_url="http://127.0.0.1:8091",
        span_exporter=exporter,
        use_simple_processor=True,
    )
    return client, exporter


def test_stamps_session_id_from_incoming_w3c_baggage_across_child_spans() -> None:
    client, exporter = _make_client()
    incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    incoming_span_id = "00f067aa0ba902b7"

    with (
        extract_trace_headers(
            {
                "traceparent": f"00-{incoming_trace_id}-{incoming_span_id}-01",
                "Baggage": "tenant=acme, sp.session.id=sess_py_baggage_1;ttl=30",
            }
        ),
        client.observation(name="http-handler"),
        client.tracer.start_as_current_span("sqlalchemy.select"),
    ):
        pass

    client.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.context is not None
        assert format(span.context.trace_id, "032x") == incoming_trace_id
        assert dict(span.attributes or {})["sp.session.id"] == "sess_py_baggage_1"
        assert (
            dict(span.attributes or {})["gen_ai.conversation.id"]
            == "sess_py_baggage_1"
        )
    client.shutdown()


def test_explicit_session_id_takes_precedence_over_baggage() -> None:
    client, exporter = _make_client()

    with extract_trace_headers({"baggage": "sp.session.id=sess_from_baggage"}):
        obs = client.start_observation(
            name="explicit-obs",
            session_id="sess_explicit_py",
        )
        obs.end()

        raw = client.tracer.start_span(
            "raw-explicit",
            attributes={"sp.session.id": "sess_raw_py"},
        )
        raw.end()

    client.force_flush()
    by_name = {s.name: dict(s.attributes or {}) for s in exporter.get_finished_spans()}
    assert by_name["explicit-obs"]["sp.session.id"] == "sess_explicit_py"
    assert by_name["raw-explicit"]["sp.session.id"] == "sess_raw_py"
    client.shutdown()


def test_ignores_invalid_baggage_and_omits_session_when_absent() -> None:
    assert BaggageSessionSpanProcessor is not None
    client, exporter = _make_client()

    with extract_trace_headers({"baggage": f"sp.session.id={'z' * 129}"}):
        client.start_observation(name="oversized").end()

    with extract_trace_headers({"baggage": "sp.session.id=bad%20space"}):
        client.start_observation(name="whitespace").end()

    client.start_observation(name="no-session").end()
    client.force_flush()

    for span in exporter.get_finished_spans():
        attrs = dict(span.attributes or {})
        assert "sp.session.id" not in attrs
        assert "gen_ai.conversation.id" not in attrs
    client.shutdown()


def test_injects_outbound_traceparent_and_session_baggage_headers() -> None:
    client, exporter = _make_client()

    with propagate_attributes(session_id="sess_prop_py"):
        raw_child = client.tracer.start_span("raw.child")
        raw_child.end()
        prop_headers = inject_trace_headers()
    assert "sp.session.id=sess_prop_py" in prop_headers.get("baggage", "")

    with client.observation(name="outbound-root", session_id="sess_obs_py") as obs:
        obs_headers = inject_trace_headers(
            {"x-keep": "1", "Baggage": "tenant=acme;ttl=30"}
        )
        assert obs.trace_id in obs_headers.get("traceparent", "")

    assert obs_headers["x-keep"] == "1"
    assert "Baggage" not in obs_headers
    assert "tenant=acme;ttl=30" in obs_headers.get("baggage", "")
    assert "sp.session.id=sess_obs_py" in obs_headers.get("baggage", "")
    client.force_flush()
    raw_finished = next(
        s for s in exporter.get_finished_spans() if s.name == "raw.child"
    )
    assert dict(raw_finished.attributes or {})["sp.session.id"] == "sess_prop_py"
    client.shutdown()
