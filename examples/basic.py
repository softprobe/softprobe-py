"""Softprobe basic example — SoftprobeClient.from_env().

Requires:

  export SOFTPROBE_PUBLIC_KEY=…
  export SOFTPROBE_BASE_URL=…
"""

from __future__ import annotations

import json
import os

from softprobe import SoftprobeClient


def main() -> None:
    client = SoftprobeClient.from_env(service_name="softprobe-python-example-basic")
    session_id = os.environ.get("SOFTPROBE_SESSION_ID", "example-basic-thread").strip()

    with client.observation(
        name="example.agent",
        as_type="agent",
        session_id=session_id,
        input={"goal": "say hello"},
    ) as agent, client.generation(
        name="example.generation",
        parent=agent,
        session_id=session_id,
        model="example-model",
        provider="example",
        input={"messages": [{"role": "user", "content": "hello"}]},
        usage={"input_tokens": 4, "output_tokens": 6},
        output={"content": "hello from Softprobe"},
    ):
        pass

    client.force_flush()
    client.shutdown()
    print(
        json.dumps(
            {
                "ok": True,
                "sessionId": session_id,
                "message": "flushed agent + generation — check Explorer Sessions",
            }
        )
    )


if __name__ == "__main__":
    main()
