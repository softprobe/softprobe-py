# Softprobe Python examples

Docs-faithful samples for [`softprobe`](https://pypi.org/project/softprobe/).

Install steps match [Agent QA → LangChain](https://docs.softprobe.ai/en/agent-qa/langchain).

| File | What it shows |
|------|----------------|
| [`basic.py`](./basic.py) | `SoftprobeClient.from_env()` — agent + generation |
| [`langchain_agent.py`](./langchain_agent.py) | Env auto / `instrument()` + app `thread_id` |
| [`phase2-py/`](./phase2-py/) | Nested agent/session propagation and score writes |
| [`phase3-live-py/`](./phase3-live-py/) | Live OpenAI generation capture |
| [`tool-call-py/`](./tool-call-py/) | Agent → generation → correlated tool calls |

## Credentials

```bash
export SOFTPROBE_PUBLIC_KEY="spk_…"
export SOFTPROBE_BASE_URL="https://explorer.softprobe.ai/api/thelake"
export SOFTPROBE_ENVIRONMENT="Production"   # optional
```

For a local thelake started with `SOFTPROBE_LOCAL_ANONYMOUS=1`, set
`SOFTPROBE_BASE_URL=http://127.0.0.1:8090` and any non-empty
`SOFTPROBE_PUBLIC_KEY`; thelake ignores credentials in that explicit mode.

## Run

```bash
uv sync --extra langchain --extra openai
uv run python examples/basic.py

# LangChain agent (needs GEMINI_KEY / GOOGLE_API_KEY or OPENAI_API_KEY):
uv pip install 'langgraph>=0.2' 'langchain-google-genai>=2' 'langchain-openai>=0.2'
uv run python examples/langchain_agent.py

# Integration samples (start thelake first; optional API keys vary by sample):
uv run python examples/phase2-py/main.py
uv run --extra openai python examples/phase3-live-py/main.py
uv run python examples/tool-call-py/main.py
```
