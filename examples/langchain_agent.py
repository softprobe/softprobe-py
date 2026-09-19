"""Softprobe LangChain example — docs best path (zero invoke callbacks edits).

Requires:

  export SOFTPROBE_PUBLIC_KEY=…
  export SOFTPROBE_BASE_URL=…
  export GEMINI_KEY=…   # or OPENAI_API_KEY

Softprobe reads configurable.thread_id — it does not mint session UUIDs.
"""

from __future__ import annotations

import json
import os
import sys

# Best path: auto-instrument when SOFTPROBE_* credentials are set.
import softprobe.langchain  # noqa: F401
from softprobe.langchain import get_handler, instrument


def _require_model() -> tuple[str, object]:
    gemini = (os.environ.get("GEMINI_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if gemini:
        if not os.environ.get("GOOGLE_API_KEY", "").strip() and os.environ.get(
            "GEMINI_KEY", ""
        ).strip():
            os.environ["GOOGLE_API_KEY"] = gemini
        from langchain_google_genai import ChatGoogleGenerativeAI

        return "google/gemini-2.5-flash", ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    openai = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai:
        from langchain_openai import ChatOpenAI

        return "openai/gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini")

    print(
        "Set GEMINI_KEY (or GOOGLE_API_KEY) or OPENAI_API_KEY for this example",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    handler = get_handler() or instrument()

    try:
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print(
            "Install example deps: pip install langgraph langchain-google-genai "
            "langchain-openai",
            file=sys.stderr,
        )
        return 2

    label, llm = _require_model()

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    agent = create_react_agent(llm, [multiply])
    thread_id = os.environ.get("SOFTPROBE_SESSION_ID", "example-langchain-thread").strip()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 3 times 7? Use the multiply tool.",
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id, "user_id": "example-user"}},
    )
    handler.flush()

    messages = result.get("messages") or []
    print(
        json.dumps(
            {
                "ok": True,
                "sessionId": thread_id,
                "model": label,
                "messageCount": len(messages),
                "message": "check Explorer Sessions for this thread_id",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
