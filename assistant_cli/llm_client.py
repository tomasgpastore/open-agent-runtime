from __future__ import annotations

import asyncio
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


LOGGER = logging.getLogger(__name__)
INVALID_FUNCTION_NAME_PATTERN = re.compile(r"function ['\"]([^'\"]+)['\"]")
INVALID_TOOL_INDEX_PATTERN = re.compile(r"tools\[(\d+)\]")
NUMERIC_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "multipleOf",
    "minProperties",
    "maxProperties",
}


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails after retries."""


class LLMToolUnsupportedError(LLMCallError):
    """Raised when the configured model does not support tool calling."""


class OpenAIModelListError(LLMCallError):
    """Raised when listing OpenAI models fails."""


class LLMClient(Protocol):
    @property
    def model_name(self) -> str: ...

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AIMessage: ...


@dataclass(slots=True)
class OllamaLLMConfig:
    base_url: str
    model: str
    temperature: float = 0.1
    context_window: int = 8000
    timeout_seconds: int = 90
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.5


@dataclass(slots=True)
class OpenAILLMConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    timeout_seconds: int = 90
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.5


class OllamaLLMClient:
    """LangChain LLM wrapper around Ollama chat API with retry controls."""

    def __init__(self, config: OllamaLLMConfig) -> None:
        self._config = config
        self._tool_support_known: bool | None = None
        self._base_model = ChatOllama(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            num_ctx=config.context_window,
            client_kwargs={"timeout": config.timeout_seconds},
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AIMessage:
        if tools and self._tool_support_known is False:
            raise LLMToolUnsupportedError(
                f"Model '{self._config.model}' does not support tool calling."
            )

        model = self._base_model.bind_tools(tools) if tools else self._base_model
        last_error: Exception | None = None

        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                if on_token is None:
                    response = await model.ainvoke(list(messages))
                else:
                    response = await self._ainvoke_streaming(
                        model=model,
                        messages=messages,
                        on_token=on_token,
                    )
                if not isinstance(response, AIMessage):
                    raise LLMCallError(
                        f"Unexpected response type from LLM: {type(response).__name__}"
                    )
                if tools:
                    self._tool_support_known = True
                return response
            except Exception as exc:  # noqa: BLE001
                if tools and self._is_tool_unsupported_error(exc):
                    self._tool_support_known = False
                    raise LLMToolUnsupportedError(str(exc)) from exc
                last_error = exc
                if attempt >= self._config.retry_attempts:
                    break
                delay = self._config.retry_backoff_seconds * attempt
                LOGGER.debug(
                    "LLM call attempt %s/%s failed (%s). Retrying in %.1fs",
                    attempt,
                    self._config.retry_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise LLMCallError(f"LLM call failed after retries: {last_error}")

    async def _ainvoke_streaming(
        self,
        model: ChatOllama,
        messages: Sequence[BaseMessage],
        on_token: Callable[[str], None],
    ) -> AIMessage:
        merged_chunk: AIMessageChunk | None = None

        async for chunk in model.astream(list(messages)):
            if not isinstance(chunk, AIMessageChunk):
                continue

            merged_chunk = chunk if merged_chunk is None else merged_chunk + chunk

            text = self._extract_text_content(chunk.content)
            if text:
                try:
                    on_token(text)
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Streaming callback failed", exc_info=True)

        if merged_chunk is None:
            fallback = await model.ainvoke(list(messages))
            if isinstance(fallback, AIMessage):
                return fallback
            raise LLMCallError(
                f"Unexpected response type from streaming fallback: {type(fallback).__name__}"
            )

        message = message_chunk_to_message(merged_chunk)
        if isinstance(message, AIMessage):
            return message
        raise LLMCallError(
            f"Unexpected chunk aggregation type: {type(message).__name__}"
        )

    def _extract_text_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    def _is_tool_unsupported_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "does not support tools" in text or "tool calling is not supported" in text


class OpenAILLMClient:
    """LangChain LLM wrapper around OpenAI chat API with retry controls."""

    def __init__(self, config: OpenAILLMConfig) -> None:
        self._config = config
        self._base_model = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AIMessage:
        tool_definitions = self._prepare_openai_tools(tools or []) if tools else []
        last_error: Exception | None = None

        for attempt in range(1, self._config.retry_attempts + 1):
            model = (
                self._base_model.bind_tools(tool_definitions)
                if tool_definitions
                else self._base_model
            )
            try:
                if on_token is None:
                    response = await model.ainvoke(list(messages))
                else:
                    response = await self._ainvoke_streaming(
                        model=model,
                        messages=messages,
                        on_token=on_token,
                    )
                if not isinstance(response, AIMessage):
                    raise LLMCallError(
                        f"Unexpected response type from LLM: {type(response).__name__}"
                    )
                return response
            except Exception as exc:  # noqa: BLE001
                if tool_definitions and self._is_invalid_tool_schema_error(exc):
                    reduced = self._drop_incompatible_tool(tool_definitions, exc)
                    if reduced is not None and len(reduced) < len(tool_definitions):
                        tool_definitions = reduced
                        last_error = exc
                        continue
                last_error = exc
                if attempt >= self._config.retry_attempts:
                    break
                delay = self._config.retry_backoff_seconds * attempt
                LOGGER.debug(
                    "LLM call attempt %s/%s failed (%s). Retrying in %.1fs",
                    attempt,
                    self._config.retry_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise LLMCallError(f"LLM call failed after retries: {last_error}")

    async def _ainvoke_streaming(
        self,
        model: ChatOpenAI,
        messages: Sequence[BaseMessage],
        on_token: Callable[[str], None],
    ) -> AIMessage:
        merged_chunk: AIMessageChunk | None = None

        async for chunk in model.astream(list(messages)):
            if not isinstance(chunk, AIMessageChunk):
                continue

            merged_chunk = chunk if merged_chunk is None else merged_chunk + chunk

            text = self._extract_text_content(chunk.content)
            if text:
                try:
                    on_token(text)
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Streaming callback failed", exc_info=True)

        if merged_chunk is None:
            fallback = await model.ainvoke(list(messages))
            if isinstance(fallback, AIMessage):
                return fallback
            raise LLMCallError(
                f"Unexpected response type from streaming fallback: {type(fallback).__name__}"
            )

        message = message_chunk_to_message(merged_chunk)
        if isinstance(message, AIMessage):
            return message
        raise LLMCallError(
            f"Unexpected chunk aggregation type: {type(message).__name__}"
        )

    def _extract_text_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    def _prepare_openai_tools(self, tools: Sequence[BaseTool]) -> list[dict]:
        prepared: list[dict] = []
        for tool in tools:
            try:
                definition = convert_to_openai_tool(tool)
            except Exception:  # noqa: BLE001
                continue

            if not isinstance(definition, dict):
                continue

            normalized = deepcopy(definition)
            function_data = normalized.get("function")
            if isinstance(function_data, dict):
                parameters = function_data.get("parameters")
                if isinstance(parameters, dict):
                    function_data["parameters"] = self._migrate_json_schema(parameters)
            prepared.append(normalized)
        return prepared

    def _migrate_json_schema(self, node: object) -> object:
        if isinstance(node, dict):
            updated = {key: self._migrate_json_schema(value) for key, value in node.items()}

            for key in NUMERIC_SCHEMA_KEYS:
                value = updated.get(key)
                if isinstance(value, bool):
                    updated.pop(key, None)

            exclusive_minimum = updated.get("exclusiveMinimum")
            minimum = updated.get("minimum")
            if isinstance(exclusive_minimum, bool):
                if exclusive_minimum and self._is_json_number(minimum):
                    updated["exclusiveMinimum"] = minimum
                    updated.pop("minimum", None)
                else:
                    updated.pop("exclusiveMinimum", None)

            exclusive_maximum = updated.get("exclusiveMaximum")
            maximum = updated.get("maximum")
            if isinstance(exclusive_maximum, bool):
                if exclusive_maximum and self._is_json_number(maximum):
                    updated["exclusiveMaximum"] = maximum
                    updated.pop("maximum", None)
                else:
                    updated.pop("exclusiveMaximum", None)

            return updated

        if isinstance(node, list):
            return [self._migrate_json_schema(item) for item in node]

        return node

    def _is_json_number(self, value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _is_invalid_tool_schema_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "invalid_function_parameters" in text or "invalid schema for function" in text

    def _drop_incompatible_tool(self, definitions: list[dict], exc: Exception) -> list[dict] | None:
        message = str(exc)
        name_match = INVALID_FUNCTION_NAME_PATTERN.search(message)
        if name_match:
            bad_name = name_match.group(1)
            reduced = [
                tool
                for tool in definitions
                if isinstance(tool.get("function"), dict)
                and tool["function"].get("name") != bad_name
            ]
            if len(reduced) < len(definitions):
                return reduced

        index_match = INVALID_TOOL_INDEX_PATTERN.search(message)
        if index_match:
            index = int(index_match.group(1))
            if 0 <= index < len(definitions):
                return [tool for i, tool in enumerate(definitions) if i != index]

        return None


async def list_openai_models(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    timeout_seconds: int = 30,
) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise OpenAIModelListError(f"Failed to list OpenAI models: {exc}") from exc

    raw_items = payload.get("data", [])
    model_ids = sorted(
        {
            item["id"]
            for item in raw_items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    )

    preferred_prefixes = ("gpt", "o", "chatgpt")
    filtered = [
        model_id
        for model_id in model_ids
        if model_id.startswith(preferred_prefixes)
        and "embedding" not in model_id
        and "audio" not in model_id
        and "tts" not in model_id
        and "moderation" not in model_id
        and "whisper" not in model_id
    ]
    return filtered or model_ids
