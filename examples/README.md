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
export INGESTION_KEY="<Ingestion Key>"
export SOFTPROBE_BASE_URL="https://explorer-staging.softprobe.ai/api/thelake"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer ${INGESTION_KEY}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment.name=staging"
```

Set `INGESTION_KEY` to the Ingestion Key from Explorer before running these examples. For a local thelake started with `SOFTPROBE_LOCAL_ANONYMOUS=1`, set `SOFTPROBE_BASE_URL=http://127.0.0.1:8090` and use a non-empty `Authorization` value; thelake ignores credentials in that explicit mode.

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
