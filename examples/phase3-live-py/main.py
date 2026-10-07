#!/usr/bin/env python3
"""Phase 3 live OpenAI/Gemini sample using softprobe.openai.observe_openai."""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from softprobe import SoftprobeClient
from softprobe.openai import observe_openai

OUT_PATH = Path(os.environ["PHASE3_IDS_PATH"])
SESSION_ID = os.environ["PHASE3_SESSION_ID"]
PROVIDER = os.environ.get("PHASE3_PROVIDER", "openai")
MODEL = os.environ.get("PHASE3_MODEL", "gpt-4o-mini")
API_KEY = os.environ["PHASE3_API_KEY"]
BASE_URL = os.environ.get("PHASE3_BASE_URL")
BASE = os.environ.get("SOFTPROBE_BASE_URL", "http://127.0.0.1:8090")
PUBLIC_KEY = os.environ.get("SOFTPROBE_PUBLIC_KEY", "e2e-token")
OTLP = os.environ.get("SOFTPROBE_OTLP_ENDPOINT", f"{BASE}/v1/traces")


def main() -> None:
    sp = SoftprobeClient(
        public_key=PUBLIC_KEY,
        base_url=BASE,
        otlp_endpoint=OTLP,
        service_name=f"phase3-live-py-{PROVIDER}",
        service_version="0.1.0",
        environment="e2e",
    )
    raw = OpenAI(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else OpenAI(api_key=API_KEY)
    client = observe_openai(
        raw,
        softprobe_client=sp,
        session_id=SESSION_ID,
        generation_name=f"phase3.{PROVIDER}.chat",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Reply with a single digit only."},
            {"role": "user", "content": "What is 1 + 1?"},
        ],
        temperature=0,
        name=f"phase3-{PROVIDER}-math",
        session_id=SESSION_ID,
    )
    content = response.choices[0].message.content
    span_id = client.last_generation_span_id
    trace_id = client.last_generation_trace_id
    sp.force_flush()
    sp.shutdown()

    payload = {
        "language": "python",
        "provider": PROVIDER,
        "sessionId": SESSION_ID,
        "traceId": trace_id,
        "generationSpanId": span_id,
        "model": MODEL,
        "responseId": getattr(response, "id", None),
        "content": content,
        "serviceName": f"phase3-live-py-{PROVIDER}",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
