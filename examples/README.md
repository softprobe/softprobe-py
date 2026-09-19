# Softprobe Python examples

Docs-faithful samples for [`softprobe`](https://pypi.org/project/softprobe/).

Install steps match [Agent QA → LangChain](https://docs.softprobe.ai/en/agent-qa/langchain).

| File | What it shows |
|------|----------------|
| [`basic.py`](./basic.py) | `SoftprobeClient.from_env()` — agent + generation |
| [`langchain_agent.py`](./langchain_agent.py) | Env auto / `instrument()` + app `thread_id` |

## Credentials

```bash
export SOFTPROBE_PUBLIC_KEY="spk_…"
export SOFTPROBE_BASE_URL="https://explorer.softprobe.ai/api/thelake"
export SOFTPROBE_ENVIRONMENT="Production"   # optional
```

## Run

```bash
uv sync --extra langchain --extra openai
uv run python examples/basic.py

# LangChain agent (needs GEMINI_KEY / GOOGLE_API_KEY or OPENAI_API_KEY):
uv pip install 'langgraph>=0.2' 'langchain-google-genai>=2' 'langchain-openai>=0.2'
uv run python examples/langchain_agent.py
```
