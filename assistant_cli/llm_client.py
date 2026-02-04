from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama


LOGGER = logging.getLogger(__name__)


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails after retries."""


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
        self._base_model = ChatOllama(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            num_ctx=config.context_window,
            client_kwargs={"timeout": config.timeout_seconds},
        )

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
    ) -> AIMessage:
        model = self._base_model.bind_tools(tools) if tools else self._base_model
        last_error: Exception | None = None

        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                response = await model.ainvoke(list(messages))
                if not isinstance(response, AIMessage):
                    raise LLMCallError(
                        f"Unexpected response type from LLM: {type(response).__name__}"
                    )
                return response
            except Exception as exc:  # noqa: BLE001
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
