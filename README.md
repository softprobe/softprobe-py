# softprobe (Python)

Softprobe **Agent QA** Python SDK — LLM / agent observability via OTLP to Softprobe Explorer.

```bash
pip install softprobe
# LangChain:
pip install 'softprobe[langchain]'
```

Docs: [Agent QA → LangChain](https://docs.softprobe.ai/en/agent-qa/langchain) · [Quick start](https://docs.softprobe.ai/en/agent-qa/getting-started)

## Examples

| Script | Path |
|--------|------|
| Basic `from_env()` | [`examples/basic.py`](examples/basic.py) |
| LangChain best path | [`examples/langchain_agent.py`](examples/langchain_agent.py) |

See [`examples/README.md`](examples/README.md).

```bash
export INGESTION_KEY="<Ingestion Key>"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer ${INGESTION_KEY}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment.name=staging"
python examples/basic.py
```

The Ingestion Key is the bearer value in the standard OTLP headers. The full
trace endpoint selects where traces go. Set `SOFTPROBE_BASE_URL` only when
explicit score writes should use a non-production API; it does not affect trace
export. Without an OTLP endpoint, the SDK uses the production trace URL. You can
pass the trace settings directly:

Without a custom endpoint, the SDK exports traces to
`https://explorer.softprobe.ai/api/thelake/v1/traces`.

```python
import os

from softprobe import SoftprobeClient

client = SoftprobeClient(
    ingestion_key=os.environ["INGESTION_KEY"],
    otlp={
        "traces_endpoint": "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
        "protocol": "http/protobuf",
    },
    resource_attributes={"deployment.environment.name": "staging"},
)
```

`base_url` is optional here; it defaults to the production score API. Set it
separately if score writes should use staging. It never changes the trace
destination. Without an OTLP endpoint, traces use the production trace URL.

## Develop

```bash
uv sync --extra langchain --extra openai --extra dev
uv run pytest -q
uv run mypy src/softprobe
```

## Publish

Create a GitHub Release with tag `vX.Y.Z`. The [release workflow](.github/workflows/release.yml) publishes to PyPI via Trusted Publishing (OIDC).

> **Note:** The older public repo [`softprobe/softprobe-python`](https://github.com/softprobe/softprobe-python) (Hybrid/replay SDK) is **archived**. This repo is the Agent QA / LLM package.
