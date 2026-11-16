from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.semconv.resource import ResourceAttributes

from softprobe.config import (
    DEFAULT_THELAKE_BASE_URL,
    MissingSoftprobeCredentialsError,
    as_non_empty_string,
    derive_otlp_endpoint,
    merge_otlp_headers,
    resolve_softprobe_config_from_env,
    validate_otlp_protocol,
)
from softprobe.observation import (
    Generation,
    Observation,
    ObservationRuntime,
    observation_scope,
    start_generation,
    start_observation,
)
from softprobe.propagation import BaggageSessionSpanProcessor, propagate_attributes
from softprobe.redaction import default_redact_keys
from softprobe.scores import HttpScoreTransport, build_score_request
from softprobe.tools import tool_span_attributes
from softprobe.types import ObservationType, ScoreDataType, ScoreSource, ScoreTransport


class SoftprobeClient:
    def __init__(
        self,
        *,
        ingestion_key: str | None = None,
        public_key: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_THELAKE_BASE_URL,
        otlp_endpoint: str | None = None,
        otlp: Mapping[str, Any] | None = None,
        resource_attributes: Mapping[str, Any] | None = None,
        service_name: str = "softprobe-app",
        service_version: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        redact_keys: list[str] | None = None,
        timeout_ms: int = 10_000,
        headers: Mapping[str, str] | None = None,
        span_exporter: SpanExporter | None = None,
        score_transport: ScoreTransport | None = None,
        use_simple_processor: bool = False,
    ) -> None:
        key = (ingestion_key or api_key or public_key or "").strip()
        if not key:
            raise ValueError("ingestion_key is required")
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        base_url = base_url.strip()
        otlp_options = otlp or {}
        generic_endpoint = as_non_empty_string(otlp_options.get("endpoint"))
        traces_endpoint = as_non_empty_string(
            otlp_options.get("traces_endpoint") or otlp_options.get("tracesEndpoint")
        )
        resolved_otlp = (
            traces_endpoint
            or as_non_empty_string(otlp_endpoint)
            or derive_otlp_endpoint(generic_endpoint or base_url)
        )
        validate_otlp_protocol(otlp_options.get("protocol"))
        if span_exporter is None and not resolved_otlp:
            raise ValueError("otlp_endpoint is required")

        resource_attrs: dict[str, str] = {
            ResourceAttributes.SERVICE_NAME: service_name,
            "telemetry.sdk.name": "softprobe",
        }
        if service_version:
            resource_attrs[ResourceAttributes.SERVICE_VERSION] = service_version
        if environment:
            resource_attrs["deployment.environment.name"] = environment
        resource_attrs.update(dict(resource_attributes or {}))

        otlp_headers = otlp_options.get("headers")
        exporter_headers = merge_otlp_headers(
            otlp_headers if isinstance(otlp_headers, Mapping) else None,
            headers,
            {"Authorization": f"Bearer {key}"},
        )

        exporter = span_exporter or OTLPSpanExporter(
            endpoint=resolved_otlp,
            headers=exporter_headers,
            timeout=timeout_ms / 1000.0,
        )
        processor = (
            SimpleSpanProcessor(exporter)
            if use_simple_processor
            else BatchSpanProcessor(exporter)
        )
        self._provider = TracerProvider(resource=Resource.create(resource_attrs))
        self._provider.add_span_processor(BaggageSessionSpanProcessor())
        self._provider.add_span_processor(processor)
        # Avoid set_tracer_provider so multiple clients/tests can coexist.
        # Parent/child linking uses explicit context attach on observations.

        self._runtime = ObservationRuntime(
            tracer=self._provider.get_tracer("softprobe", "0.1.0"),
            redact_keys=redact_keys or default_redact_keys(),
            session_id=session_id,
            user_id=user_id,
            release=release,
            tags=tags,
        )
        self._score_transport = score_transport or HttpScoreTransport(
            base_url,
            key,
            headers or {},
            timeout_ms,
        )
        self._shut_down = False

    @classmethod
    def from_env(cls, **overrides: Any) -> SoftprobeClient:
        """Build a client from standard OTLP variables and Softprobe settings."""
        source = dict(os.environ)
        explicit_key = (
            overrides.get("ingestion_key")
            or overrides.get("api_key")
            or overrides.get("public_key")
        )
        explicit_base_url = overrides.get("base_url")
        if explicit_base_url:
            source["SOFTPROBE_BASE_URL"] = explicit_base_url
        else:
            source.setdefault("SOFTPROBE_BASE_URL", DEFAULT_THELAKE_BASE_URL)
        if explicit_key:
            source["SOFTPROBE_PUBLIC_KEY"] = explicit_key
        cfg = resolve_softprobe_config_from_env(source)
        if cfg is None:
            raise MissingSoftprobeCredentialsError(
                "Set OTEL_EXPORTER_OTLP_HEADERS or OTEL_EXPORTER_OTLP_TRACES_HEADERS "
                "with the Ingestion Key, or pass ingestion_key"
            )
        direct_otlp = overrides.get("otlp") or {}
        env_otlp = cfg["otlp"]
        explicit_endpoint = (
            as_non_empty_string(direct_otlp.get("traces_endpoint"))
            or as_non_empty_string(direct_otlp.get("tracesEndpoint"))
            or as_non_empty_string(overrides.get("otlp_endpoint"))
        )
        direct_generic_endpoint = as_non_empty_string(direct_otlp.get("endpoint"))
        resolved_endpoint = explicit_endpoint or (
            derive_otlp_endpoint(direct_generic_endpoint)
            if direct_generic_endpoint
            else env_otlp["traces_endpoint"]
        )
        env_headers = env_otlp.get("headers")
        direct_headers = direct_otlp.get("headers")
        otlp_options: dict[str, Any] = {
            "traces_endpoint": resolved_endpoint,
            "protocol": direct_otlp.get("protocol") or env_otlp.get("protocol"),
            "headers": merge_otlp_headers(
                env_headers if isinstance(env_headers, Mapping) else None,
                direct_headers if isinstance(direct_headers, Mapping) else None,
                overrides.get("headers"),
            ),
        }
        generic_endpoint = direct_generic_endpoint or env_otlp.get("endpoint")
        if generic_endpoint:
            otlp_options["endpoint"] = generic_endpoint
        attributes = dict(cfg.get("resource_attributes", {}))
        explicit_attributes = dict(overrides.get("resource_attributes") or {})
        attributes.update(explicit_attributes)
        if overrides.get("service_name") and "service.name" not in explicit_attributes:
            attributes["service.name"] = overrides["service_name"]
        if (
            overrides.get("environment")
            and "deployment.environment.name" not in explicit_attributes
        ):
            attributes["deployment.environment.name"] = overrides["environment"]
        kwargs: dict[str, Any] = {
            "ingestion_key": explicit_key or cfg["ingestion_key"],
            "base_url": cfg["base_url"],
            "otlp_endpoint": resolved_endpoint,
            "otlp": otlp_options,
            "resource_attributes": attributes,
        }
        if cfg.get("environment"):
            kwargs["environment"] = cfg["environment"]
        if cfg.get("session_id"):
            kwargs["session_id"] = cfg["session_id"]
        if cfg.get("user_id"):
            kwargs["user_id"] = cfg["user_id"]
        if cfg.get("service_name"):
            kwargs["service_name"] = cfg["service_name"]
        kwargs.update({
            name: value
            for name, value in overrides.items()
            if name not in {
                "ingestion_key", "api_key", "public_key", "otlp", "otlp_endpoint",
                "resource_attributes",
            }
        })
        return cls(**kwargs)

    def start_observation(
        self,
        *,
        name: str,
        as_type: ObservationType = "span",
        attributes: Mapping[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str | None = None,
        trace_name: str | None = None,
        parent: Observation | None = None,
        trace_context: Mapping[str, str] | None = None,
        start_time: datetime | float | None = None,
    ) -> Observation:
        self._assert_active()
        return start_observation(
            self._runtime,
            name=name,
            as_type=as_type,
            attributes=attributes,
            input=input,
            output=output,
            session_id=session_id,
            user_id=user_id,
            release=release,
            tags=tags,
            metadata=metadata,
            version=version,
            trace_name=trace_name,
            parent=parent,
            trace_context=trace_context,
            start_time=start_time,
        )

    def start_generation(self, *, name: str, **kwargs: Any) -> Generation:
        self._assert_active()
        return start_generation(self._runtime, name=name, **kwargs)

    def propagate_attributes(self, **kwargs: Any) -> Any:
        """Context manager that scopes session/user/tags/metadata for nested work."""
        return propagate_attributes(**kwargs)

    @property
    def tracer(self) -> Any:
        return self._runtime.tracer

    def start_agent(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="agent", **kwargs)

    def start_tool(
        self,
        *,
        name: str,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        index: int | None = None,
        mcp_server: str | None = None,
        mcp_tool: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Observation:
        merged = dict(attributes or {})
        merged.update(
            tool_span_attributes(
                tool_name=tool_name or name,
                tool_call_id=tool_call_id,
                kind=kind or "function",
                status=status or "ok",
                index=index,
                mcp_server=mcp_server,
                mcp_tool=mcp_tool,
            )
        )
        return self.start_observation(
            name=name, as_type="tool", attributes=merged, **kwargs
        )

    def start_chain(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="chain", **kwargs)

    def start_retriever(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="retriever", **kwargs)

    def start_evaluator(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="evaluator", **kwargs)

    def start_embedding(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="embedding", **kwargs)

    def start_guardrail(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="guardrail", **kwargs)

    def start_event(self, *, name: str, **kwargs: Any) -> Observation:
        return self.start_observation(name=name, as_type="event", **kwargs)

    @contextmanager
    def observation(self, *, name: str, **kwargs: Any) -> Iterator[Observation]:
        obs = self.start_observation(name=name, **kwargs)
        with observation_scope(obs) as scoped:
            yield scoped

    @contextmanager
    def generation(self, *, name: str, **kwargs: Any) -> Iterator[Generation]:
        gen = self.start_generation(name=name, **kwargs)
        with observation_scope(gen) as scoped:
            yield scoped  # type: ignore[misc]

    @asynccontextmanager
    async def async_observation(
        self, *, name: str, **kwargs: Any
    ) -> AsyncIterator[Observation]:
        with self.observation(name=name, **kwargs) as obs:
            yield obs

    @asynccontextmanager
    async def async_generation(
        self, *, name: str, **kwargs: Any
    ) -> AsyncIterator[Generation]:
        with self.generation(name=name, **kwargs) as gen:
            yield gen

    def create_score(
        self,
        *,
        score_id: str,
        name: str,
        data_type: ScoreDataType,
        source: ScoreSource,
        timestamp: str | datetime | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        session_id: str | None = None,
        numeric_value: float | None = None,
        string_value: str | None = None,
        boolean_value: bool | None = None,
        comment: str | None = None,
        config_id: str | None = None,
        author_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._assert_active()
        request = build_score_request(
            score_id=score_id,
            name=name,
            data_type=data_type,
            source=source,
            timestamp=timestamp,
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            numeric_value=numeric_value,
            string_value=string_value,
            boolean_value=boolean_value,
            comment=comment,
            config_id=config_id,
            author_id=author_id,
            metadata=metadata,
        )
        self._score_transport.create_score(request)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return bool(self._provider.force_flush(timeout_millis))

    def flush(self, timeout_millis: int = 30_000) -> bool:
        """Alias for :meth:`force_flush`."""
        return self.force_flush(timeout_millis)

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self._provider.shutdown()

    def _assert_active(self) -> None:
        if self._shut_down:
            raise RuntimeError("SoftprobeClient has been shut down")
