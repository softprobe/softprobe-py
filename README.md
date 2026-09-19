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
export SOFTPROBE_PUBLIC_KEY="spk_…"
export SOFTPROBE_BASE_URL="https://explorer.softprobe.ai/api/thelake"
python examples/basic.py
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
