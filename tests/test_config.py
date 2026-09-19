from __future__ import annotations

import pytest

from softprobe.config import (
    MissingSoftprobeCredentialsError,
    derive_otlp_endpoint,
    resolve_softprobe_config_from_env,
    resolve_softprobe_config_from_mapping,
)


def test_derive_otlp_endpoint_defaults() -> None:
    assert derive_otlp_endpoint("http://127.0.0.1:8091") == (
        "http://127.0.0.1:8091/v1/traces"
    )


def test_resolve_softprobe_config_from_env_requires_both_keys() -> None:
    assert resolve_softprobe_config_from_env({}) is None
    assert resolve_softprobe_config_from_env({"SOFTPROBE_PUBLIC_KEY": "pk"}) is None


def test_resolve_softprobe_config_from_mapping_validates() -> None:
    with pytest.raises(MissingSoftprobeCredentialsError):
        resolve_softprobe_config_from_mapping({"public_key": "pk"})


def test_softprobe_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from softprobe.client import SoftprobeClient
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _NullExporter(SpanExporter):
        def export(self, spans):  # type: ignore[no-untyped-def]
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    monkeypatch.setenv("SOFTPROBE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("SOFTPROBE_BASE_URL", "http://127.0.0.1:8091")
    monkeypatch.setenv("SOFTPROBE_ENVIRONMENT", "development")
    client = SoftprobeClient.from_env(
        span_exporter=_NullExporter(),
        use_simple_processor=True,
    )
    assert client is not None
    client.shutdown()


def test_softprobe_client_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from softprobe.client import SoftprobeClient

    monkeypatch.delenv("SOFTPROBE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("SOFTPROBE_BASE_URL", raising=False)
    with pytest.raises(MissingSoftprobeCredentialsError):
        SoftprobeClient.from_env()
