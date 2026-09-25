"""LangChain CallbackHandler for Softprobe (Langfuse-style MVP)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from uuid import UUID

from softprobe.client import SoftprobeClient
from softprobe.identity import resolve_run_identity
from softprobe.observation import Generation, Observation
from softprobe.propagation import propagate_attributes
from softprobe.tools import (
    normalize_tool_calls,
    record_tool_calls,
    record_tool_definitions,
    tool_result_event_payload,
    tool_span_attributes,
)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
except ImportError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Install langchain-core to use softprobe.langchain: "
        "pip install 'softprobe[langchain]'"
    ) from exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    role = getattr(message, "type", None) or message.__class__.__name__
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "function",
    }
    payload: dict[str, Any] = {
        "role": role_map.get(str(role), str(role)),
        "content": getattr(message, "content", str(message)),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        normalized = normalize_tool_calls(tool_calls)
        payload["tool_calls"] = normalized or _serialize_tool_calls(tool_calls)
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = str(tool_call_id)
    name = getattr(message, "name", None)
    if name:
        payload["name"] = str(name)
    return payload


def _serialize_tool_calls(tool_calls: Any) -> list[Any]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        return []
    out: list[Any] = []
    for item in tool_calls:
        if isinstance(item, Mapping):
            out.append(dict(item))
        else:
            out.append(str(item))
    return out


def _extract_usage(response: LLMResult) -> dict[str, int] | None:
    llm_output = response.llm_output or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if not isinstance(token_usage, Mapping):
        return None
    result: dict[str, int] = {}
    if "prompt_tokens" in token_usage:
        result["input_tokens"] = int(token_usage["prompt_tokens"])
    if "completion_tokens" in token_usage:
        result["output_tokens"] = int(token_usage["completion_tokens"])
    if "total_tokens" in token_usage:
        result["total_tokens"] = int(token_usage["total_tokens"])
    return result or None


def _extract_model_name(
    serialized: dict[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    invocation_params: Mapping[str, Any] | None,
) -> str | None:
    if metadata:
        ls_model_name = metadata.get("ls_model_name")
        if isinstance(ls_model_name, str):
            return ls_model_name
    if invocation_params:
        for key in ("model", "model_name", "model_id"):
            value = invocation_params.get(key)
            if isinstance(value, str) and value:
                return value
    if serialized:
        kwargs = serialized.get("kwargs") or {}
        for key in ("model", "model_name", "model_id"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
        ids = serialized.get("id")
        if isinstance(ids, list) and ids:
            return str(ids[-1])
    return None


class CallbackHandler(BaseCallbackHandler):
    """Softprobe LangChain callback handler.

    Zero-setup: ``CallbackHandler()`` builds a client from ``SOFTPROBE_*`` env.
    Session/user ids come from LangChain metadata / ``configurable`` (e.g.
    LangGraph ``thread_id``), not Softprobe-invented UUIDs.
    """

    name = "SoftprobeCallbackHandler"

    def __init__(
        self,
        *,
        softprobe_client: SoftprobeClient | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str | None = None,
        trace_context: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._client = softprobe_client or SoftprobeClient.from_env()
        self._session_id = session_id
        self._user_id = user_id
        self._tags = list(tags or [])
        self._metadata = dict(metadata or {})
        self._version = version
        self._trace_context = dict(trace_context or {})
        self._runs: Dict[UUID, Union[Observation, Generation]] = {}
        self._completion_start: Dict[UUID, str] = {}
        self._tool_meta: Dict[UUID, dict[str, Any]] = {}
        self.last_trace_id: str | None = None

    def flush(self, timeout_millis: int = 30_000) -> bool:
        return self._client.flush(timeout_millis)

    def _parent(self, parent_run_id: Optional[UUID]) -> Observation | None:
        if parent_run_id is None:
            return None
        return self._runs.get(parent_run_id)

    def _start(
        self,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        as_type: str,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Observation | Generation:
        parent = self._parent(parent_run_id)
        merged_meta = dict(self._metadata)
        if metadata:
            merged_meta.update(dict(metadata))
        session_id, user_id = resolve_run_identity(
            metadata=metadata,
            fallback_session_id=self._session_id,
            fallback_user_id=self._user_id,
        )
        start_kwargs: dict[str, Any] = {
            "name": name,
            "as_type": as_type,
            "input": input,
            "session_id": session_id,
            "user_id": user_id,
            "tags": list({*(tags or []), *self._tags}) or None,
            "metadata": merged_meta or None,
            "version": self._version,
            "parent": parent,
        }
        if parent is None and self._trace_context:
            start_kwargs["trace_context"] = self._trace_context
        start_kwargs.update(kwargs)
        obs: Observation | Generation
        if as_type == "generation":
            gen_kwargs = {k: v for k, v in start_kwargs.items() if k != "as_type"}
            obs = self._client.start_generation(**gen_kwargs)
        else:
            obs = self._client.start_observation(**start_kwargs)
        self._runs[run_id] = obs
        if parent is None:
            self.last_trace_id = obs.trace_id
        return obs

    def _end(
        self,
        run_id: UUID,
        *,
        output: Any = None,
        status_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        obs = self._runs.pop(run_id, None)
        if obs is None:
            return
        end_kwargs: dict[str, Any] = {"output": output, "status_message": status_message}
        end_kwargs.update(kwargs)
        # Filter Nones for cleaner ends
        end_kwargs = {k: v for k, v in end_kwargs.items() if v is not None}
        obs.end(**end_kwargs)
        self._completion_start.pop(run_id, None)
        self._tool_meta.pop(run_id, None)

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: Any = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        if run_id not in self._completion_start and run_id in self._runs:
            self._completion_start[run_id] = _now_iso()
            obs = self._runs[run_id]
            if isinstance(obs, Generation):
                obs.update(completion_start_time=self._completion_start[run_id])

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        run_name = name or (
            serialized.get("name")
            or (serialized.get("id") or ["chain"])[-1]
            if isinstance(serialized, dict)
            else "chain"
        )
        as_type = "agent" if "agent" in str(run_name).lower() else "chain"
        session_id, user_id = resolve_run_identity(
            metadata=metadata,
            fallback_session_id=self._session_id,
            fallback_user_id=self._user_id,
        )
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=self._tags,
            metadata=self._metadata,
            version=self._version,
        ):
            self._start(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=str(run_name),
                as_type=as_type,
                input=inputs,
                metadata=metadata,
                tags=tags,
            )

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        self._end(run_id, output=outputs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> Any:
        self._end(run_id, status_message=str(error))

    def on_agent_action(self, action: AgentAction, *, run_id: UUID, **kwargs: Any) -> Any:
        # LangChain reuses the agent run_id here; do not overwrite the agent span.
        if run_id in self._runs:
            return
        parent_run_id = kwargs.get("parent_run_id")
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=action.tool,
            as_type="tool",
            input=action.tool_input,
            attributes=tool_span_attributes(
                tool_name=action.tool,
                kind="function",
                status="ok",
            ),
        )

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, **kwargs: Any) -> Any:
        self._end(run_id, output=finish.return_values)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        flat = [_message_to_dict(m) for batch in messages for m in batch]
        model = _extract_model_name(serialized, metadata, invocation_params)
        params = invocation_params or {}
        run_name = name or (
            (serialized.get("id") or ["ChatModel"])[-1]
            if isinstance(serialized, dict)
            else "ChatModel"
        )
        provider = None
        if model:
            lower = model.lower()
            if "gpt" in lower or "openai" in lower:
                provider = "openai"
            elif "gemini" in lower or "google" in lower:
                provider = "google"
        obs = self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=str(run_name),
            as_type="generation",
            input={"messages": flat},
            metadata=metadata,
            tags=tags,
            model=model,
            provider=provider,
            operation_name="chat",
            temperature=params.get("temperature"),
            max_tokens=params.get("max_tokens"),
            prompt_event=flat,
        )
        tools = params.get("tools")
        if tools is not None:
            record_tool_definitions(obs, tools)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        model = _extract_model_name(serialized, metadata, invocation_params)
        params = invocation_params or {}
        obs = self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name or "LLM",
            as_type="generation",
            input={"prompts": prompts},
            metadata=metadata,
            tags=tags,
            model=model,
            operation_name="completion",
            temperature=params.get("temperature"),
            max_tokens=params.get("max_tokens"),
            prompt_event=prompts,
        )
        tools = params.get("tools")
        if tools is not None:
            record_tool_definitions(obs, tools)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Any:
        generations = []
        tool_calls: list[Any] = []
        finish_reasons: list[str] = []
        for batch in response.generations:
            for gen in batch:
                if isinstance(gen, ChatGeneration) and gen.message is not None:
                    payload = _message_to_dict(gen.message)
                    generations.append(payload)
                    if payload.get("tool_calls"):
                        tool_calls.extend(payload["tool_calls"])
                        finish_reasons.append("tool_calls")
                    else:
                        finish_reasons.append("stop")
                else:
                    generations.append({"content": gen.text})
                    finish_reasons.append("stop")
        usage = _extract_usage(response)
        model = None
        if response.llm_output and isinstance(response.llm_output.get("model_name"), str):
            model = response.llm_output["model_name"]
        obs = self._runs.get(run_id)
        if obs is not None and tool_calls:
            record_tool_calls(obs, tool_calls)
        self._end(
            run_id,
            output={"generations": generations},
            usage=usage,
            response_model=model,
            completion_event=generations,
            finish_reasons=finish_reasons or ["stop"],
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> Any:
        self._end(run_id, status_message=str(error))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        run_name = name or serialized.get("name") or "tool"
        # Only correlate with a provider-supplied id (matches generation
        # sp.tool.call_ids); never fabricate one from the LangChain run_id.
        tool_call_id = kwargs.get("tool_call_id") or (metadata or {}).get("tool_call_id")
        self._tool_meta[run_id] = {
            "name": str(run_name),
            "tool_call_id": str(tool_call_id) if tool_call_id else None,
        }
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=str(run_name),
            as_type="tool",
            input=inputs if inputs is not None else input_str,
            metadata=metadata,
            tags=tags,
            attributes=tool_span_attributes(
                tool_name=str(run_name),
                tool_call_id=str(tool_call_id) if tool_call_id else None,
                kind="function",
                status="ok",
            ),
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Any:
        obs = self._runs.get(run_id)
        meta = self._tool_meta.get(run_id) or {}
        if obs is not None:
            obs.add_content_event(
                "gen_ai.tool.message",
                tool_result_event_payload(
                    name=str(meta.get("name") or "tool"),
                    content=output,
                    tool_call_id=meta.get("tool_call_id"),
                ),
            )
        self._end(
            run_id,
            output=output,
            attributes=tool_span_attributes(status="ok"),
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> Any:
        self._end(
            run_id,
            status_message=str(error),
            attributes=tool_span_attributes(status="error"),
        )
    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        run_name = name or serialized.get("name") or "retriever"
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=str(run_name),
            as_type="retriever",
            input={"query": query},
            metadata=metadata,
            tags=tags,
        )

    def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        docs = []
        for doc in documents:
            if isinstance(doc, Document):
                docs.append(
                    {
                        "page_content": doc.page_content,
                        "metadata": dict(doc.metadata or {}),
                    }
                )
            else:
                docs.append(str(doc))
        self._end(run_id, output={"documents": docs})

    def on_retriever_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> Any:
        self._end(run_id, status_message=str(error))


# Zero-setup: env auto-instrument + explicit instrument() / uninstrument().
from softprobe.langchain_instrument import (  # noqa: E402
    auto_instrument_from_env,
    get_handler,
    instrument,
    is_instrumented,
    uninstrument,
)

auto_instrument_from_env()
