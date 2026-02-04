from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama


LOGGER = logging.getLogger(__name__)


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails after retries."""


class LLMToolUnsupportedError(LLMCallError):
    """Raised when the configured model does not support tool calling."""


@dataclass(slots=True)
class OllamaLLMConfig:
    base_url: str
    model: str
    temperature: float = 0.1
    context_window: int = 8000
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
                LOGGER.warning(
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
                    LOGGER.exception("Streaming callback failed")

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
