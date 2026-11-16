from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, TypedDict


DEFAULT_THELAKE_BASE_URL = "https://explorer.softprobe.ai/api/thelake"


class ResolvedSoftprobeConfig(TypedDict, total=False):
    ingestion_key: str
    public_key: str
    base_url: str
    otlp_endpoint: str
    otlp: dict[str, Any]
    resource_attributes: dict[str, Any]
    environment: str
    session_id: str
    user_id: str
    service_name: str


class MissingSoftprobeCredentialsError(ValueError):
    """Raised when Softprobe credentials cannot be resolved."""


def as_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def derive_otlp_endpoint(base_url: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return f"{base_url.rstrip('/')}/v1/traces"


def _parse_key_value_list(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    result: dict[str, str] = {}
    for pair in value.split(","):
        name, separator, item = pair.partition("=")
        if separator and name.strip():
            result[name.strip()] = item.strip()
    return result


def merge_otlp_headers(
    *sources: Mapping[str, str] | None,
) -> dict[str, str]:
    result: dict[str, tuple[str, str]] = {}
    for source in sources:
        for name, value in (source or {}).items():
            result[name.lower()] = (name, value)
    return {name: value for name, value in result.values()}


def _get_header(headers: Mapping[str, str], name: str) -> str | None:
    for header_name, value in headers.items():
        if header_name.lower() == name.lower():
            return value
    return None


def _bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    scheme, separator, token = value.strip().partition(" ")
    if separator and scheme.lower() == "bearer":
        return as_non_empty_string(token)
    return None


def validate_otlp_protocol(value: Any) -> str:
    protocol = (as_non_empty_string(value) or "http/protobuf").lower()
    if protocol != "http/protobuf":
        raise ValueError(
            f'Unsupported OTLP protocol "{protocol}". '
            "The Softprobe exporter supports http/protobuf."
        )
    return protocol


def _resolve_otlp(
    source: Mapping[str, str | None], base_url: str
) -> dict[str, Any]:
    endpoint = as_non_empty_string(source.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
    traces_endpoint = as_non_empty_string(
        source.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )
    legacy_endpoint = as_non_empty_string(source.get("SOFTPROBE_OTLP_ENDPOINT"))
    protocol = validate_otlp_protocol(
        as_non_empty_string(source.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"))
        or as_non_empty_string(source.get("OTEL_EXPORTER_OTLP_PROTOCOL"))
    )
    headers = merge_otlp_headers(
        _parse_key_value_list(source.get("OTEL_EXPORTER_OTLP_HEADERS")),
        _parse_key_value_list(source.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")),
    )
    resolved_endpoint = (
        traces_endpoint
        or (derive_otlp_endpoint(endpoint) if endpoint else None)
        or legacy_endpoint
        or derive_otlp_endpoint(base_url)
    )
    otlp: dict[str, Any] = {
        "traces_endpoint": resolved_endpoint,
        "protocol": protocol,
    }
    if endpoint:
        otlp["endpoint"] = endpoint
    if headers:
        otlp["headers"] = headers
    return otlp


def _resource_attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): attribute
        for name, attribute in value.items()
        if isinstance(attribute, (str, int, float, bool))
        or (
            isinstance(attribute, (list, tuple))
            and all(isinstance(item, (str, int, float, bool)) for item in attribute)
        )
    }


def resolve_softprobe_config_from_env(
    env: Mapping[str, str | None] | None = None,
) -> ResolvedSoftprobeConfig | None:
    source = env if env is not None else os.environ
    headers = merge_otlp_headers(
        _parse_key_value_list(source.get("OTEL_EXPORTER_OTLP_HEADERS")),
        _parse_key_value_list(source.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")),
    )
    ingestion_key = _bearer_token(_get_header(headers, "authorization")) or (
        as_non_empty_string(source.get("SOFTPROBE_PUBLIC_KEY"))
    )
    base_url = as_non_empty_string(source.get("SOFTPROBE_BASE_URL"))
    if not ingestion_key:
        return None
    base_url = base_url or DEFAULT_THELAKE_BASE_URL

    attributes = _parse_key_value_list(source.get("OTEL_RESOURCE_ATTRIBUTES"))
    service_name = (
        as_non_empty_string(source.get("OTEL_SERVICE_NAME"))
        or as_non_empty_string(attributes.get("service.name"))
        or as_non_empty_string(source.get("SOFTPROBE_SERVICE_NAME"))
    )
    if service_name:
        attributes["service.name"] = service_name
    environment = (
        as_non_empty_string(attributes.get("deployment.environment.name"))
        or as_non_empty_string(source.get("SOFTPROBE_ENVIRONMENT"))
    )
    otlp = _resolve_otlp(source, base_url)
    config: ResolvedSoftprobeConfig = {
        "ingestion_key": ingestion_key,
        "public_key": ingestion_key,
        "base_url": base_url,
        "otlp_endpoint": otlp["traces_endpoint"],
        "otlp": otlp,
    }
    if attributes:
        config["resource_attributes"] = attributes
    if environment:
        config["environment"] = environment
    if service_name:
        config["service_name"] = service_name
    for name in ("session_id", "user_id"):
        value = as_non_empty_string(source.get(f"SOFTPROBE_{name.upper()}"))
        if value:
            config[name] = value
    return config


def resolve_softprobe_config_from_mapping(
    raw: Mapping[str, Any],
) -> ResolvedSoftprobeConfig:
    ingestion_key = as_non_empty_string(
        raw.get("ingestion_key")
        or raw.get("ingestionKey")
        or raw.get("api_key")
        or raw.get("apiKey")
        or raw.get("public_key")
        or raw.get("publicKey")
    )
    raw_base_url = (
        raw.get("base_url")
        if raw.get("base_url") is not None
        else raw.get("baseUrl")
    )
    base_url = (
        DEFAULT_THELAKE_BASE_URL
        if raw_base_url is None
        else as_non_empty_string(raw_base_url)
    )
    if not ingestion_key:
        raise MissingSoftprobeCredentialsError(
            "ingestion_key is required"
        )
    if not base_url:
        raise MissingSoftprobeCredentialsError(
            "base_url must be non-empty when provided"
        )

    raw_otlp = raw.get("otlp")
    otlp_values = raw_otlp if isinstance(raw_otlp, Mapping) else {}
    endpoint = as_non_empty_string(otlp_values.get("endpoint"))
    traces_endpoint = as_non_empty_string(
        otlp_values.get("traces_endpoint") or otlp_values.get("tracesEndpoint")
    )
    legacy_endpoint = as_non_empty_string(
        raw.get("otlp_endpoint") or raw.get("otlpEndpoint")
    )
    resolved_endpoint = (
        traces_endpoint
        or legacy_endpoint
        or derive_otlp_endpoint(endpoint or base_url)
    )
    protocol = validate_otlp_protocol(otlp_values.get("protocol"))
    extra_headers = otlp_values.get("headers")
    headers = (
        {str(name): value for name, value in extra_headers.items() if isinstance(value, str)}
        if isinstance(extra_headers, Mapping)
        else {}
    )
    otlp: dict[str, Any] = {
        "traces_endpoint": resolved_endpoint,
        "protocol": protocol,
    }
    if endpoint:
        otlp["endpoint"] = endpoint
    if headers:
        otlp["headers"] = headers

    attributes = _resource_attributes(
        raw.get("resource_attributes") or raw.get("resourceAttributes")
    )
    environment = as_non_empty_string(raw.get("environment"))
    service_name = as_non_empty_string(
        raw.get("service_name") or raw.get("serviceName")
    ) or as_non_empty_string(attributes.get("service.name"))
    if environment and "deployment.environment.name" not in attributes:
        attributes["deployment.environment.name"] = environment
    if service_name and "service.name" not in attributes:
        attributes["service.name"] = service_name

    config: ResolvedSoftprobeConfig = {
        "ingestion_key": ingestion_key,
        "public_key": ingestion_key,
        "base_url": base_url,
        "otlp_endpoint": resolved_endpoint,
        "otlp": otlp,
    }
    if attributes:
        config["resource_attributes"] = attributes
    if environment:
        config["environment"] = environment
    if service_name:
        config["service_name"] = service_name
    session_id = as_non_empty_string(raw.get("session_id") or raw.get("sessionId"))
    user_id = as_non_empty_string(raw.get("user_id") or raw.get("userId"))
    if session_id:
        config["session_id"] = session_id
    if user_id:
        config["user_id"] = user_id
    return config
