from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone

import pytest

from softprobe.scores import HttpScoreTransport, build_score_request


def test_build_score_request_success_with_datetime() -> None:
    dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    req = build_score_request(
        score_id="s1",
        name="quality",
        data_type="numeric",
        source="human",
        timestamp=dt,
        trace_id="t1",
        numeric_value=5.0,
    )
    assert req.score_id == "s1"
    assert req.name == "quality"
    assert req.data_type == "numeric"
    assert req.source == "human"
    assert req.timestamp == "2023-01-01T12:00:00Z"
    assert req.trace_id == "t1"
    assert req.numeric_value == 5.0

def test_build_score_request_success_with_string_timestamp() -> None:
    req = build_score_request(
        score_id="s2",
        name="quality",
        data_type="numeric",
        source="human",
        timestamp="2023-01-01T12:00:00Z",
        trace_id="t1",
        numeric_value=5.0,
    )
    assert req.timestamp == "2023-01-01T12:00:00Z"

def test_build_score_request_success_without_timestamp() -> None:
    req = build_score_request(
        score_id="s1",
        name="quality",
        data_type="boolean",
        source="human",
        session_id="sess1",
        boolean_value=True,
    )
    assert req.timestamp is not None
    assert req.boolean_value is True
    assert req.session_id == "sess1"

def test_build_score_request_validation_errors() -> None:
    # missing score_id
    with pytest.raises(ValueError, match="score_id is required"):
        build_score_request(score_id="", name="test", data_type="numeric", source="human", trace_id="t1")

    # missing name
    with pytest.raises(ValueError, match="name is required"):
        build_score_request(score_id="s1", name="   ", data_type="numeric", source="human", trace_id="t1")

    # missing target
    with pytest.raises(ValueError, match="score must target span_id, trace_id, and/or session_id"):
        build_score_request(score_id="s1", name="n1", data_type="numeric", source="human")

    # numeric data missing value
    with pytest.raises(ValueError, match="numeric scores require numeric_value"):
        build_score_request(score_id="s1", name="n1", data_type="numeric", source="human", trace_id="t1")

    # categorical missing string_value
    with pytest.raises(ValueError, match="categorical scores require string_value"):
        build_score_request(score_id="s1", name="n1", data_type="categorical", source="human", trace_id="t1")

    # text missing string_value
    with pytest.raises(ValueError, match="text scores require string_value"):
        build_score_request(score_id="s1", name="n1", data_type="text", source="human", trace_id="t1")

    # boolean missing boolean_value
    with pytest.raises(ValueError, match="boolean scores require boolean_value"):
        build_score_request(score_id="s1", name="n1", data_type="boolean", source="human", trace_id="t1")

def test_http_score_transport_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b"{}"

    def mock_urlopen(req, timeout):
        assert req.full_url == "http://example.com/v1/llm/scores"
        assert req.method == "POST"
        assert req.get_header("Authorization") == "Bearer test_key"
        assert req.get_header("X-test") == "foo"
        assert timeout == 2.5
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["score_id"] == "s1"
        return MockResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    transport = HttpScoreTransport(
        base_url="http://example.com/",
        public_key="test_key",
        headers={"X-Test": "foo"},
        timeout_ms=2500,
    )
    req = build_score_request(
        score_id="s1",
        name="test",
        data_type="numeric",
        source="human",
        trace_id="t1",
        numeric_value=1.0,
    )
    transport.create_score(req)

def test_http_score_transport_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_urlopen_err(req, timeout):
        err = urllib.error.HTTPError(
            url=req.full_url,
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )
        # Mocking read method for HTTPError
        err.read = lambda: b"Invalid payload"
        raise err

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_err)

    transport = HttpScoreTransport("http://example.com", "pk", {}, 1000)
    req = build_score_request(
        score_id="s1", name="test", data_type="numeric", source="human", trace_id="t1", numeric_value=1.0
    )
    
    with pytest.raises(RuntimeError, match=r"score create failed \(400\): Invalid payload"):
        transport.create_score(req)

def test_http_score_transport_non_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockResponseError:
        status = 500
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b"Server Error"
        
    def mock_urlopen_status(req, timeout):
        return MockResponseError()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_status)

    transport = HttpScoreTransport("http://example.com", "pk", {}, 1000)
    req = build_score_request(
        score_id="s1", name="test", data_type="numeric", source="human", trace_id="t1", numeric_value=1.0
    )
    
    with pytest.raises(RuntimeError, match=r"score create failed \(500\): Server Error"):
        transport.create_score(req)
