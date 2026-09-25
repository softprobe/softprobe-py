from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone

from softprobe.types import ScoreDataType, ScoreRequest, ScoreSource, ScoreTransport


def build_score_request(
    *,
    score_id: str,
    name: str,
    data_type: ScoreDataType,
    source: ScoreSource,
    timestamp: str | datetime | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    session_id: str | None = None,
    numeric_value: float | None = None,
    string_value: str | None = None,
    boolean_value: bool | None = None,
    comment: str | None = None,
    config_id: str | None = None,
    author_id: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> ScoreRequest:
    if not score_id or not score_id.strip():
        raise ValueError("score_id is required")
    if not name or not name.strip():
        raise ValueError("name is required")
    if not (span_id or trace_id or session_id):
        raise ValueError("score must target span_id, trace_id, and/or session_id")

    if data_type == "numeric" and numeric_value is None:
        raise ValueError("numeric scores require numeric_value")
    if data_type in {"categorical", "text"} and string_value is None:
        raise ValueError(f"{data_type} scores require string_value")
    if data_type == "boolean" and boolean_value is None:
        raise ValueError("boolean scores require boolean_value")

    if isinstance(timestamp, datetime):
        ts = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    elif timestamp is None:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        ts = timestamp

    return ScoreRequest(
        score_id=score_id,
        timestamp=ts,
        trace_id=trace_id,
        span_id=span_id,
        session_id=session_id,
        name=name,
        data_type=data_type,
        numeric_value=numeric_value,
        string_value=string_value,
        boolean_value=boolean_value,
        source=source,
        comment=comment,
        config_id=config_id,
        author_id=author_id,
        metadata=dict(metadata or {}),
    )


class HttpScoreTransport(ScoreTransport):
    def __init__(
        self,
        base_url: str,
        public_key: str,
        headers: Mapping[str, str],
        timeout_ms: int,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/llm/scores"
        self._public_key = public_key
        self._headers = dict(headers)
        self._timeout_s = timeout_ms / 1000.0

    def create_score(self, request: ScoreRequest) -> None:
        payload = json.dumps(request.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._public_key}",
                "Content-Type": "application/json",
                **self._headers,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as response:
                if response.status >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"score create failed ({response.status}): {body}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"score create failed ({exc.code}): {body}") from exc
