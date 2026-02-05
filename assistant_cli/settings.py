from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_OLLAMA_BASE_URL = "http://100.126.228.118:11434"
DEFAULT_MODEL = "ministral-3:8b"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_SKILL_DIRS = ["skills", "~/.codex/skills"]


@dataclass(slots=True)
class AppSettings:
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_MODEL
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_model: str = DEFAULT_OPENAI_MODEL
    openai_max_completion_tokens: int = 4096
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_max_completion_tokens: int = 4096
    ollama_temperature: float = 0.1
    model_context_window: int = 20000
    short_term_token_limit: int = 20000
    max_iterations: int = 100
    request_timeout_seconds: int = 120
    llm_request_timeout_seconds: int = 90
    llm_retry_attempts: int = 3
    llm_retry_backoff_seconds: float = 1.5
    tool_timeout_seconds: int = 45
    sqlite_path: Path = Path("data/assistant.db")
    runtime_state_path: Path = Path("data/runtime_state.json")
    mcp_config_path: Path = Path("config/mcp_servers.json")
    mcp_fallback_config_path: Path = Path("config/mcp_servers.sample.json")
    skill_dirs: list[Path] = field(default_factory=list)
    skill_max_per_turn: int = 3
    skill_max_chars: int = 8000



def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default



def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _split_paths(raw: str) -> list[str]:
    if not raw:
        return []
    if os.pathsep in raw:
        parts = [part.strip() for part in raw.split(os.pathsep)]
    else:
        parts = [part.strip() for part in raw.split(",")]
    return [part for part in parts if part]


def _get_path_list(primary: str, fallback: str | None, default: list[str]) -> list[Path]:
    raw = os.getenv(primary)
    if not raw and fallback:
        raw = os.getenv(fallback)
    if raw:
        return [Path(path).expanduser() for path in _split_paths(raw)]
    return [Path(path).expanduser() for path in default]



def load_settings() -> AppSettings:
    load_dotenv()

    sqlite_path = Path(os.getenv("ASSISTANT_SQLITE_PATH", "data/assistant.db")).expanduser()
    runtime_state_path = Path(
        os.getenv("ASSISTANT_RUNTIME_STATE_PATH", "data/runtime_state.json")
    ).expanduser()
    mcp_config_path = Path(os.getenv("MCP_CONFIG_PATH", "config/mcp_servers.json")).expanduser()
    skill_dirs = _get_path_list("ANTON_SKILL_DIRS", "SKILL_DIRS", DEFAULT_SKILL_DIRS)

    settings = AppSettings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        openai_max_completion_tokens=_get_int("OPENAI_MAX_COMPLETION_TOKENS", 4096),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        openrouter_max_completion_tokens=_get_int("OPENROUTER_MAX_COMPLETION_TOKENS", 4096),
        ollama_temperature=_get_float("OLLAMA_TEMPERATURE", 0.1),
        model_context_window=_get_int("MODEL_CONTEXT_WINDOW", 20000),
        short_term_token_limit=_get_int("SHORT_TERM_TOKEN_LIMIT", 20000),
        max_iterations=_get_int("MAX_ITERATIONS", 100),
        request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 120),
        llm_request_timeout_seconds=_get_int("LLM_REQUEST_TIMEOUT_SECONDS", 90),
        llm_retry_attempts=_get_int("LLM_RETRY_ATTEMPTS", 3),
        llm_retry_backoff_seconds=_get_float("LLM_RETRY_BACKOFF_SECONDS", 1.5),
        tool_timeout_seconds=_get_int("TOOL_TIMEOUT_SECONDS", 45),
        sqlite_path=sqlite_path,
        runtime_state_path=runtime_state_path,
        mcp_config_path=mcp_config_path,
        skill_dirs=skill_dirs,
        skill_max_per_turn=_get_int("SKILL_MAX_PER_TURN", 3),
        skill_max_chars=_get_int("SKILL_MAX_CHARS", 8000),
    )

    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
