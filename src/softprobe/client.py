from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Mapping

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
    MissingSoftprobeCredentialsError,
    derive_otlp_endpoint,
    resolve_softprobe_config_from_env,
)
from softprobe.observation import (
    Generation,
    Observation,
    ObservationRuntime,
    observation_scope,
    start_generation,
    start_observation,
)
from softprobe.propagation import propagate_attributes
from softprobe.redaction import default_redact_keys
from softprobe.scores import HttpScoreTransport, build_score_request
from softprobe.tools import tool_span_attributes
from softprobe.types import ObservationType, ScoreDataType, ScoreSource, ScoreTransport


class SoftprobeClient:
    def __init__(
        self,
        *,
        public_key: str,
        base_url: str,
        otlp_endpoint: str | None = None,
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
        if not public_key or not public_key.strip():
            raise ValueError("public_key is required")
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        base_url = base_url.strip()
        resolved_otlp = (
            otlp_endpoint.strip()
            if otlp_endpoint and otlp_endpoint.strip()
            else derive_otlp_endpoint(base_url)
        )
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

        exporter = span_exporter or OTLPSpanExporter(
            endpoint=resolved_otlp,
            headers={
                "Authorization": f"Bearer {public_key}",
                **dict(headers or {}),
            },
            timeout=timeout_ms / 1000.0,
        )
        processor = (
            SimpleSpanProcessor(exporter)
            if use_simple_processor
            else BatchSpanProcessor(exporter)
        )
        self._provider = TracerProvider(resource=Resource.create(resource_attrs))
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
            public_key.strip(),
            headers or {},
            timeout_ms,
        )
        self._shut_down = False

    @classmethod
    def from_env(cls, **overrides: Any) -> SoftprobeClient:
        """Build a client from ``SOFTPROBE_*`` environment variables.

        Requires ``SOFTPROBE_PUBLIC_KEY`` and ``SOFTPROBE_BASE_URL``.
        Keyword overrides are merged on top of the resolved env config.
        """
        cfg = resolve_softprobe_config_from_env()
        if cfg is None:
            raise MissingSoftprobeCredentialsError(
                "Set SOFTPROBE_PUBLIC_KEY and SOFTPROBE_BASE_URL"
            )
        kwargs: dict[str, Any] = {
            "public_key": cfg["public_key"],
            "base_url": cfg["base_url"],
            "otlp_endpoint": cfg.get("otlp_endpoint"),
        }
        if cfg.get("environment"):
            kwargs["environment"] = cfg["environment"]
        if cfg.get("session_id"):
            kwargs["session_id"] = cfg["session_id"]
        if cfg.get("user_id"):
            kwargs["user_id"] = cfg["user_id"]
        if cfg.get("service_name"):
            kwargs["service_name"] = cfg["service_name"]
        kwargs.update(overrides)
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
        start_time: datetime | float | int | None = None,
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
