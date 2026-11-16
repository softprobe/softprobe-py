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


def test_derive_otlp_endpoint_explicit() -> None:
    assert derive_otlp_endpoint("http://127.0.0.1:8091", "http://custom/v1/traces") == (
        "http://custom/v1/traces"
    )

def test_resolve_softprobe_config_from_env_all_keys() -> None:
    env = {
        "SOFTPROBE_PUBLIC_KEY": "pk",
        "SOFTPROBE_BASE_URL": "http://test",
        "SOFTPROBE_OTLP_ENDPOINT": "http://test/traces",
        "SOFTPROBE_ENVIRONMENT": "dev",
        "SOFTPROBE_SESSION_ID": "sess-123",
        "SOFTPROBE_USER_ID": "user-456",
        "SOFTPROBE_SERVICE_NAME": "my-service",
    }
    cfg = resolve_softprobe_config_from_env(env)
    assert cfg is not None
    assert cfg["public_key"] == "pk"
    assert cfg["base_url"] == "http://test"
    assert cfg["otlp_endpoint"] == "http://test/traces"
    assert cfg["environment"] == "dev"
    assert cfg["session_id"] == "sess-123"
    assert cfg["user_id"] == "user-456"
    assert cfg["service_name"] == "my-service"


def test_resolve_softprobe_config_from_standard_otel_environment() -> None:
    cfg = resolve_softprobe_config_from_env({
        "SOFTPROBE_BASE_URL": "https://explorer-staging.softprobe.ai/api/thelake",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example/base",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer common,x-common=yes",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "authorization=Bearer staging,x-traces=yes",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment.name=staging,service.name=from-attributes",
        "OTEL_SERVICE_NAME": "staging-agent",
    })

    assert cfg is not None
    assert cfg["ingestion_key"] == "staging"
    assert cfg["public_key"] == "staging"
    assert cfg["base_url"] == "https://explorer-staging.softprobe.ai/api/thelake"
    assert cfg["otlp_endpoint"] == "https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
    assert cfg["otlp"] == {
        "endpoint": "https://collector.example/base",
        "traces_endpoint": "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
        "protocol": "http/protobuf",
        "headers": {
            "authorization": "Bearer staging",
            "x-common": "yes",
            "x-traces": "yes",
        },
    }
    assert cfg["resource_attributes"] == {
        "deployment.environment.name": "staging",
        "service.name": "staging-agent",
    }
    assert cfg["service_name"] == "staging-agent"
    assert cfg["environment"] == "staging"


def test_resolve_generic_otel_endpoint_appends_trace_path() -> None:
    cfg = resolve_softprobe_config_from_env({
        "SOFTPROBE_BASE_URL": "https://explorer-staging.softprobe.ai/api/thelake",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example/base",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer staging,x-common=yes",
    })

    assert cfg is not None
    assert cfg["otlp_endpoint"] == "https://collector.example/base/v1/traces"
    assert cfg["otlp"]["headers"]["x-common"] == "yes"

def test_resolve_softprobe_config_from_env_missing_both() -> None:
    assert resolve_softprobe_config_from_env({}) is None
    cfg = resolve_softprobe_config_from_env({"SOFTPROBE_PUBLIC_KEY": "pk"})
    assert cfg is not None
    assert cfg["base_url"] == "https://explorer.softprobe.ai/api/thelake"
    assert cfg["otlp_endpoint"] == "https://explorer.softprobe.ai/api/thelake/v1/traces"


def test_resolve_softprobe_config_from_env_otel_without_api_base() -> None:
    cfg = resolve_softprobe_config_from_env({
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT":
            "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "Authorization=Bearer staging-key",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment.name=staging",
    })
    assert cfg is not None
    assert cfg["ingestion_key"] == "staging-key"
    assert cfg["base_url"] == "https://explorer.softprobe.ai/api/thelake"
    assert cfg["otlp_endpoint"] == "https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
    assert cfg["resource_attributes"] == {"deployment.environment.name": "staging"}

def test_resolve_softprobe_config_from_mapping_validates() -> None:
    with pytest.raises(MissingSoftprobeCredentialsError):
        resolve_softprobe_config_from_mapping({})

def test_resolve_softprobe_config_from_mapping_all_keys() -> None:
    cfg = resolve_softprobe_config_from_mapping({
        "public_key": "pk",
        "baseUrl": "http://test",
        "otlpEndpoint": "http://test/v2/traces",
        "environment": "prod",
        "sessionId": "sess-123",
        "userId": "user-456",
        "serviceName": "svc-789",
    })
    assert cfg["public_key"] == "pk"
    assert cfg["base_url"] == "http://test"
    assert cfg["otlp_endpoint"] == "http://test/v2/traces"
    assert cfg["environment"] == "prod"
    assert cfg["session_id"] == "sess-123"
    assert cfg["user_id"] == "user-456"
    assert cfg["service_name"] == "svc-789"


def test_resolve_softprobe_config_from_mapping_accepts_ingestion_and_otlp_options() -> None:
    cfg = resolve_softprobe_config_from_mapping({
        "ingestion_key": "staging-key",
        "base_url": "https://scores.example.test/api",
        "otlp": {
            "endpoint": "https://collector.example/base",
            "traces_endpoint": "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
            "protocol": "http/protobuf",
            "headers": {"x-direct": "yes"},
        },
        "resource_attributes": {"deployment.environment.name": "staging"},
    })

    assert cfg["ingestion_key"] == "staging-key"
    assert cfg["base_url"] == "https://scores.example.test/api"
    assert cfg["otlp_endpoint"] == "https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
    assert cfg["otlp"] == {
        "endpoint": "https://collector.example/base",
        "traces_endpoint": "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
        "protocol": "http/protobuf",
        "headers": {"x-direct": "yes"},
    }
    assert cfg["resource_attributes"] == {"deployment.environment.name": "staging"}


def test_resolve_softprobe_config_from_mapping_needs_no_second_api_base() -> None:
    cfg = resolve_softprobe_config_from_mapping({
        "ingestion_key": "staging-key",
        "otlp": {
            "traces_endpoint":
                "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
            "protocol": "http/protobuf",
        },
    })
    assert cfg["base_url"] == "https://explorer.softprobe.ai/api/thelake"
    assert cfg["otlp_endpoint"] == "https://explorer-staging.softprobe.ai/api/thelake/v1/traces"


def test_resolve_softprobe_config_from_mapping_defaults_to_production_endpoint() -> None:
    cfg = resolve_softprobe_config_from_mapping({"ingestion_key": "pk"})
    assert cfg["base_url"] == "https://explorer.softprobe.ai/api/thelake"
    assert cfg["otlp_endpoint"] == "https://explorer.softprobe.ai/api/thelake/v1/traces"


def test_resolve_softprobe_config_from_mapping_rejects_blank_api_base() -> None:
    with pytest.raises(MissingSoftprobeCredentialsError, match="base_url"):
        resolve_softprobe_config_from_mapping({
            "ingestion_key": "pk",
            "base_url": "",
        })


def test_resolve_softprobe_config_rejects_unsupported_protocol() -> None:
    with pytest.raises(ValueError, match="http/protobuf"):
        resolve_softprobe_config_from_mapping({
            "ingestion_key": "pk",
            "base_url": "https://collector.example",
            "otlp": {"protocol": "grpc"},
        })


def test_softprobe_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    from softprobe.client import SoftprobeClient

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


def test_from_env_exports_standard_otel_settings_without_softprobe_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    import softprobe.client as client_module

    captured: dict[str, object] = {}

    class _CaptureExporter(SpanExporter):
        def __init__(self, *, endpoint: str, headers: dict[str, str], timeout: float):
            captured.update(endpoint=endpoint, headers=headers)

        def export(self, spans):  # type: ignore[no-untyped-def]
            captured["resource_attributes"] = dict(spans[0].resource.attributes)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(client_module, "OTLPSpanExporter", _CaptureExporter)
    monkeypatch.delenv("SOFTPROBE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("SOFTPROBE_BASE_URL", raising=False)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://explorer-staging.softprobe.ai/api/thelake/v1/traces",
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "Authorization=Bearer staging-key",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", "deployment.environment.name=staging"
    )

    client = client_module.SoftprobeClient.from_env(use_simple_processor=True)
    client.start_observation(name="otel-env", as_type="span").end()
    client.force_flush()

    assert captured["endpoint"] == (
        "https://explorer-staging.softprobe.ai/api/thelake/v1/traces"
    )
    assert captured["headers"]["Authorization"] == "Bearer staging-key"
    assert captured["resource_attributes"]["deployment.environment.name"] == "staging"
    client.shutdown()
