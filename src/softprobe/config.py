from __future__ import annotations

import os
from typing import Any, Mapping, TypedDict


class ResolvedSoftprobeConfig(TypedDict, total=False):
    public_key: str
    base_url: str
    otlp_endpoint: str
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


def resolve_softprobe_config_from_env(
    env: Mapping[str, str | None] | None = None,
) -> ResolvedSoftprobeConfig | None:
    source = env if env is not None else os.environ
    public_key = as_non_empty_string(source.get("SOFTPROBE_PUBLIC_KEY"))
    base_url = as_non_empty_string(source.get("SOFTPROBE_BASE_URL"))
    if not public_key or not base_url:
        return None
    cfg: ResolvedSoftprobeConfig = {
        "public_key": public_key,
        "base_url": base_url,
        "otlp_endpoint": derive_otlp_endpoint(
            base_url,
            as_non_empty_string(source.get("SOFTPROBE_OTLP_ENDPOINT")),
        ),
    }
    environment = as_non_empty_string(source.get("SOFTPROBE_ENVIRONMENT"))
    if environment is not None:
        cfg["environment"] = environment
    session_id = as_non_empty_string(source.get("SOFTPROBE_SESSION_ID"))
    if session_id is not None:
        cfg["session_id"] = session_id
    user_id = as_non_empty_string(source.get("SOFTPROBE_USER_ID"))
    if user_id is not None:
        cfg["user_id"] = user_id
    service_name = as_non_empty_string(source.get("SOFTPROBE_SERVICE_NAME"))
    if service_name is not None:
        cfg["service_name"] = service_name
    return cfg


def resolve_softprobe_config_from_mapping(raw: Mapping[str, Any]) -> ResolvedSoftprobeConfig:
    public_key = as_non_empty_string(raw.get("public_key") or raw.get("publicKey"))
    base_url = as_non_empty_string(raw.get("base_url") or raw.get("baseUrl"))
    if not public_key or not base_url:
        raise MissingSoftprobeCredentialsError("public_key and base_url are required")
    explicit = as_non_empty_string(raw.get("otlp_endpoint") or raw.get("otlpEndpoint"))
    cfg: ResolvedSoftprobeConfig = {
        "public_key": public_key,
        "base_url": base_url,
        "otlp_endpoint": derive_otlp_endpoint(base_url, explicit),
    }
    environment = as_non_empty_string(raw.get("environment"))
    if environment is not None:
        cfg["environment"] = environment
    session_id = as_non_empty_string(raw.get("session_id") or raw.get("sessionId"))
    if session_id is not None:
        cfg["session_id"] = session_id
    user_id = as_non_empty_string(raw.get("user_id") or raw.get("userId"))
    if user_id is not None:
        cfg["user_id"] = user_id
    service_name = as_non_empty_string(raw.get("service_name") or raw.get("serviceName"))
    if service_name is not None:
        cfg["service_name"] = service_name
    return cfg
