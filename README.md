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
export SOFTPROBE_BASE_URL="https://explorer-staging.softprobe.ai/api/thelake"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer ${INGESTION_KEY}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment.name=staging"
python examples/basic.py
```

The Ingestion Key is the bearer value in the standard OTLP headers. `SOFTPROBE_BASE_URL` also selects the API base for score writes. You can pass the same trace settings directly:

```python
import os

from softprobe import SoftprobeClient

client = SoftprobeClient(
    ingestion_key=os.environ["INGESTION_KEY"],
    base_url="https://explorer-staging.softprobe.ai/api/thelake",
    otlp={
        "traces_endpoint": "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
        "protocol": "http/protobuf",
    },
    resource_attributes={"deployment.environment.name": "staging"},
)
```

## Develop

```bash
uv sync --extra langchain --extra openai --extra dev
uv run pytest -q
uv run mypy src/softprobe
```

## Publish

Create a GitHub Release with tag `vX.Y.Z`. The [release workflow](.github/workflows/release.yml) publishes to PyPI via Trusted Publishing (OIDC).

> **Note:** The older public repo [`softprobe/softprobe-python`](https://github.com/softprobe/softprobe-python) (Hybrid/replay SDK) is **archived**. This repo is the Agent QA / LLM package.
