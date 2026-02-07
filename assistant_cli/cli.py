from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Callable, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import print_formatted_text
from rich.console import Console
from rich.markdown import Markdown

from assistant_cli.agent_graph import LangGraphAgent
from assistant_cli.approval import ApprovalManager
from assistant_cli.llm_client import (
    OpenAILLMClient,
    OpenAILLMConfig,
    OpenAIModelListError,
    OllamaLLMClient,
    OllamaLLMConfig,
    list_openai_models,
)
from assistant_cli.logging_utils import configure_logging
from assistant_cli.mcp_manager import MCPManager
from assistant_cli.memory_store import SQLiteMemoryStore
from assistant_cli.memory_tools import wipe_memory_graph
from assistant_cli.settings import load_settings
from assistant_cli.skills_manager import SkillManager


LOGGER = logging.getLogger(__name__)
CSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
IMAGE_TAG_RE = re.compile(r"\[Image #\d+\]")
MARKDOWN_BLOCK_RE = re.compile(r"(^|\n)\s{0,3}(#{1,6}\s|[-*+]\s|>\s|\d+\.\s|```|~~~)")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
TABLE_SEPARATOR_RE = re.compile(r"(^|\n)\s*\|?[:\- ]+\|[:\-| ]*(\n|$)")
PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n+")


def _strip_ansi_codes(text: str) -> str:
    text = OSC_ESCAPE_RE.sub("", text)
    return CSI_ESCAPE_RE.sub("", text)


@contextmanager
def _suppress_output() -> None:
    with open(os.devnull, "w") as devnull:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)

ANTON_ASCII = r"""

 █████╗ ███╗   ██╗████████╗ ██████╗ ███╗   ██╗
██╔══██╗████╗  ██║╚══██╔══╝██╔═══██╗████╗  ██║
███████║██╔██╗ ██║   ██║   ██║   ██║██╔██╗ ██║
██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╗██║
██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚████║
╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝
"""

class SlashCommandCompleter(Completer):
    def __init__(
        self,
        root_commands_provider: Callable[[], Sequence[str]],
        subcommands_provider: Callable[[str], Sequence[str]],
    ) -> None:
        self._root_commands_provider = root_commands_provider
        self._subcommands_provider = subcommands_provider

    def get_completions(self, document, complete_event):  # type: ignore[override]
        text_before_cursor = document.text_before_cursor
        if not text_before_cursor.startswith("/"):
            return
        if "\n" in text_before_cursor:
            return

        stripped = text_before_cursor.strip()
        trailing_space = text_before_cursor.endswith(" ")
        parts = stripped.split()
        if not parts:
            return

        root = parts[0]
        root_commands = sorted(set(self._root_commands_provider()))
        if len(parts) == 1 and not trailing_space:
            for command in root_commands:
                if command.startswith(root):
                    yield Completion(command, start_position=-len(root))
            return

        if root not in root_commands:
            return

        subcommands = sorted(set(self._subcommands_provider(root)))
        if not subcommands:
            return

        for option in subcommands:
            candidate = f"{root} {option}"
            if candidate.startswith(text_before_cursor):
                yield Completion(candidate, start_position=-len(text_before_cursor))


class ResponseStreamPrinter:
    def __init__(self) -> None:
        self._line_open = False
        self._saw_visible_token = False
        self._line_started = False

    def on_token(self, token: str) -> None:
        if not token:
            return
        cleaned = _strip_ansi_codes(token)
        if not cleaned:
            return
        if not self._line_open:
            sys.stdout.write("> ")
            sys.stdout.flush()
            self._line_open = True
            self._line_started = False
        if not self._line_started:
            cleaned = cleaned.lstrip()
            self._line_started = True
        sys.stdout.write(cleaned)
        sys.stdout.flush()
        if cleaned.strip():
            self._saw_visible_token = True

    def on_tool(self, tool_name: str) -> None:
        self._close_line_if_needed()
        sys.stdout.write(f"tool> {tool_name}\n")
        sys.stdout.flush()

    def finish(self) -> None:
        self._close_line_if_needed()

    @property
    def saw_token(self) -> bool:
        return self._saw_visible_token

    def _close_line_if_needed(self) -> None:
        if self._line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_open = False
            self._line_started = False
            sys.stdout.write("\n")
            sys.stdout.flush()


class AssistantCLI:
    def __init__(self) -> None:
        self.console = Console(no_color=True, highlight=False, markup=False)
        self.settings = load_settings()
        self.current_provider = "local"
        self.current_model = self.settings.ollama_model
        self._pending_images: list[str] = []
        self._markdown_enabled = self._supports_ansi()

        self.memory_store = SQLiteMemoryStore(
            db_path=self.settings.sqlite_path,
            session_id="default",
            token_limit=self.settings.short_term_token_limit,
            context_window=self.settings.model_context_window,
        )

        self.approval_manager = ApprovalManager()
        self.mcp_manager = MCPManager(
            config_path=self.settings.mcp_config_path,
            fallback_config_path=self.settings.mcp_fallback_config_path,
        )
        self.skill_manager = SkillManager(
            skill_dirs=self.settings.skill_dirs,
            max_per_turn=self.settings.skill_max_per_turn,
            max_chars=self.settings.skill_max_chars,
        )

        persisted_selection = self._load_runtime_llm_selection()
        if persisted_selection is not None:
            self.current_provider, self.current_model = persisted_selection
        llm_client = self._build_initial_llm_client()

        self.agent = LangGraphAgent(
            db_path=self.settings.sqlite_path,
            llm_client=llm_client,
            max_iterations=self.settings.max_iterations,
            request_timeout_seconds=self.settings.request_timeout_seconds,
            tool_timeout_seconds=self.settings.tool_timeout_seconds,
            skill_manager=self.skill_manager,
        )
        self._prompt_session = self._build_prompt_session()

    def _build_initial_llm_client(self):
        provider = self.current_provider
        model = self.current_model
        try:
            if provider == "openai":
                return self._build_openai_llm_client(model=model)
            if provider == "openrouter":
                return self._build_openrouter_llm_client(model=model)
            return self._build_local_llm_client(model=model)
        except Exception as exc:  # noqa: BLE001
            self.console.print(
                f"Could not restore persisted LLM selection ({provider}/{model}): {exc}. "
                f"Falling back to local ({self.settings.ollama_model})."
            )
            self.current_provider = "local"
            self.current_model = self.settings.ollama_model
            self._save_runtime_llm_selection()
            return self._build_local_llm_client(model=self.current_model)

    def _load_runtime_llm_selection(self) -> tuple[str, str] | None:
        path = self.settings.runtime_state_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

        llm_data = payload.get("llm")
        if not isinstance(llm_data, dict):
            return None
        provider = llm_data.get("provider")
        model = llm_data.get("model")
        if provider not in {"local", "openai", "openrouter"}:
            return None
        if not isinstance(model, str) or not model.strip():
            return None
        return provider, model.strip()

    def _save_runtime_llm_selection(self) -> None:
        path = self.settings.runtime_state_path
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            else:
                payload = {}
        except Exception:  # noqa: BLE001
            payload = {}

        payload["llm"] = {
            "provider": self.current_provider,
            "model": self.current_model,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _build_prompt_session(self) -> PromptSession[str]:
        key_bindings = KeyBindings()

        @key_bindings.add("enter")
        def _(event) -> None:
            event.current_buffer.validate_and_handle()

        @key_bindings.add("escape", "enter")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-j")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-v")
        def _(event) -> None:
            if self._try_paste_clipboard_image(event):
                return
            data = event.app.clipboard.get_data()
            if data and data.text:
                event.current_buffer.insert_text(data.text)
            else:
                event.current_buffer.insert_text("")

        # --- Enhanced Keybindings ---
        
        # Word Navigation (Ctrl+Left/Right)
        @key_bindings.add("c-left")
        def _(event) -> None:
            event.current_buffer.cursor_left(count=event.arg, mode="WORD")

        @key_bindings.add("c-right")
        def _(event) -> None:
            event.current_buffer.cursor_right(count=event.arg, mode="WORD")

        # Selection (Shift+Arrows)
        # Note: prompt_toolkit selection handling often requires specific terminal support
        # or emacs/vi mode activation. Basic shift+arrow might be caught by terminal escape codes.
        
        @key_bindings.add("home")
        def _(event) -> None:
            event.current_buffer.cursor_position = 0

        @key_bindings.add("end")
        def _(event) -> None:
            event.current_buffer.cursor_position = len(event.current_buffer.text)

        return PromptSession(
            multiline=True,
            completer=SlashCommandCompleter(
                root_commands_provider=self._root_commands,
                subcommands_provider=self._subcommand_options,
            ),
            complete_while_typing=True,
            complete_in_thread=True,
            key_bindings=key_bindings,
            history=InMemoryHistory(),
            prompt_continuation=lambda width, line_number, is_soft_wrap: "... ",
            rprompt=None,
            bottom_toolbar=self._get_footer,
        )

    async def _prompt(self, prompt: str, multiline: bool = False) -> str:
        return await self._prompt_session.prompt_async(
            prompt,
            multiline=multiline,
        )

    def _print_kv_lines(self, title: str, rows: list[tuple[str, str]]) -> None:
        self.console.print(title)
        for key, value in rows:
            self.console.print(f"- {key}: {value}")
        self.console.print()

    def _print_list(self, title: str, items: list[str]) -> None:
        self.console.print(title)
        if not items:
            self.console.print("- (none)")
        else:
            for item in items:
                self.console.print(f"- {item}")
        self.console.print()

    def _build_local_llm_client(self, model: str) -> OllamaLLMClient:
        return OllamaLLMClient(
            OllamaLLMConfig(
                base_url=self.settings.ollama_base_url,
                model=model,
                temperature=self.settings.ollama_temperature,
                context_window=self.settings.model_context_window,
                timeout_seconds=self.settings.llm_request_timeout_seconds,
                retry_attempts=self.settings.llm_retry_attempts,
                retry_backoff_seconds=self.settings.llm_retry_backoff_seconds,
            )
        )

    def _build_openai_llm_client(self, model: str) -> OpenAILLMClient:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        return OpenAILLMClient(
            OpenAILLMConfig(
                api_key=api_key,
                model=model,
                base_url=self.settings.openai_base_url,
                temperature=self.settings.ollama_temperature,
                timeout_seconds=self.settings.llm_request_timeout_seconds,
                retry_attempts=self.settings.llm_retry_attempts,
                retry_backoff_seconds=self.settings.llm_retry_backoff_seconds,
                max_completion_tokens=self.settings.openai_max_completion_tokens,
            )
        )

    def _build_openrouter_llm_client(self, model: str) -> OpenAILLMClient:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        app_url = os.getenv("OPENROUTER_APP_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "LangGraph MCP Assistant").strip()
        headers: dict[str, str] = {}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        return OpenAILLMClient(
            OpenAILLMConfig(
                api_key=api_key,
                model=model,
                base_url=self.settings.openrouter_base_url,
                temperature=self.settings.ollama_temperature,
                timeout_seconds=self.settings.llm_request_timeout_seconds,
                retry_attempts=self.settings.llm_retry_attempts,
                retry_backoff_seconds=self.settings.llm_retry_backoff_seconds,
                max_completion_tokens=self.settings.openrouter_max_completion_tokens,
                default_headers=headers,
                # Kimi 2.5 supports reasoning controls through OpenRouter.
                reasoning={"enabled": True},
            )
        )

    def _switch_provider(self, provider: str, model: str) -> None:
        provider_changed = self.current_provider != provider
        if provider == "local":
            client = self._build_local_llm_client(model=model)
        elif provider == "openai":
            client = self._build_openai_llm_client(model=model)
        elif provider == "openrouter":
            client = self._build_openrouter_llm_client(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.agent.set_llm_client(client)
        self.current_provider = provider
        self.current_model = model
        self._save_runtime_llm_selection()

        # Avoid cross-provider message-format mismatches (especially tool-call transcripts).
        if provider_changed:
            self.memory_store.clear_session()
            self.console.print("Short-term session history was reset after provider switch.")

    async def _handle_command(self, command_line: str) -> bool:
        parts = command_line.strip().split()
        command = parts[0].lower()

        if command == "/quit":
            self.console.print("Goodbye.")
            return True

        if command == "/mcp":
            await self._handle_mcp_command(parts[1:])
            return False

        if command == "/approval":
            self._handle_approval_command(parts[1:])
            return False

        if command == "/memory":
            self._handle_memory_command()
            return False

        if command == "/skills":
            self._handle_skills_command(parts[1:])
            return False

        if command == "/paths":
            await self._handle_paths_command(parts[1:])
            return False

        if command == "/llm":
            await self._handle_llm_command(parts[1:])
            return False

        if command == "/new":
            await self._handle_new_command()
            return False

        if command == "/help":
            self._print_help()
            return False

        self.console.print("Unknown command. Use /help for available commands.")
        return False

    async def _handle_mcp_command(self, args: Sequence[str]) -> None:
        if not args:
            self._print_mcp_status()
            return

        if len(args) == 1 and args[0].lower() == "refresh":
            await self._safe_refresh_mcp("mcp refresh command")
            self._print_mcp_status()
            return

        if len(args) == 2 and args[0].lower() in {"on", "off"}:
            server_name = args[1]
            if not self.mcp_manager.is_server_known(server_name):
                self.console.print(f"Unknown MCP server '{server_name}'.")
                return

            enabled = args[0].lower() == "on"
            self.mcp_manager.set_server_enabled(server_name, enabled)
            await self._safe_refresh_mcp("mcp toggle command")
            self.console.print(
                f"MCP server '{server_name}' set to {'enabled' if enabled else 'disabled'}."
            )
            self._print_mcp_status()
            return

        self.console.print("Usage: /mcp | /mcp refresh | /mcp on <server> | /mcp off <server>")

    async def _handle_paths_command(self, args: Sequence[str]) -> None:
        if not args or args[0].lower() == "list":
            self._print_allowed_paths()
            return

        action = args[0].lower()
        if action not in {"add", "remove"}:
            self.console.print("Usage: /paths | /paths add <path|downloads|desktop|documents> | /paths remove <path>")
            return

        if len(args) < 2:
            self.console.print("Path is required.")
            return

        raw_path = " ".join(args[1:]).strip()
        target_path = self._resolve_directory_alias(raw_path)

        try:
            if action == "add":
                added = self.mcp_manager.add_filesystem_allowed_directory(target_path)
                self.console.print(
                    f"Added filesystem allowed path: {added} (saved to MCP config)."
                )
            else:
                removed = self.mcp_manager.remove_filesystem_allowed_directory(target_path)
                self.console.print(
                    f"Removed filesystem allowed path: {removed} (saved to MCP config)."
                )
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"Failed to update allowed paths: {exc}")
            return

        await self._safe_refresh_mcp("paths update")
        self._print_allowed_paths()

    def _resolve_directory_alias(self, raw: str) -> str:
        aliases = {
            "downloads": "~/Downloads",
            "desktop": "~/Desktop",
            "documents": "~/Documents",
        }
        return aliases.get(raw.lower(), raw)

    def _print_allowed_paths(self) -> None:
        try:
            paths = self.mcp_manager.list_filesystem_allowed_directories()
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"Failed to read filesystem allowed paths: {exc}")
            return
        rows: list[tuple[str, str]] = []
        if not paths:
            rows.append(("(none)", "n/a"))
        else:
            for path in paths:
                exists = "yes" if os.path.exists(path) else "no"
                rows.append((path, exists))
        self._print_kv_lines("Filesystem Allowed Paths", rows)

    def _print_mcp_status(self) -> None:
        self.console.print("MCP Servers")
        for status in self.mcp_manager.list_statuses():
            self.console.print(f"- {status.name}")
            self.console.print(f"  enabled: {'true' if status.enabled else 'false'}")
            self.console.print(f"  connected: {'true' if status.connected else 'false'}")
            if status.tools:
                for tool_name in status.tools:
                    approval = "yes" if self.approval_manager.tool_enabled(tool_name) else "no"
                    self.console.print(f"  tool: {tool_name} (approval: {approval})")
            else:
                self.console.print("  tool: (none)")
            if status.last_error:
                self.console.print(f"  error: {status.last_error}")
        self.console.print()

    def _handle_approval_command(self, args: Sequence[str]) -> None:
        if not args:
            self._print_approval_status()
            return

        if len(args) == 2 and args[0].lower() == "global":
            enabled = self._parse_on_off(args[1])
            if enabled is None:
                self.console.print("Usage: /approval global on|off")
                return
            self.approval_manager.set_global(enabled)
            self.console.print(f"Global approval {'enabled' if enabled else 'disabled'}.")
            return

        if len(args) == 3 and args[0].lower() == "tool":
            tool_name = args[1]
            enabled = self._parse_on_off(args[2])
            if enabled is None:
                self.console.print("Usage: /approval tool <tool_name> on|off")
                return

            self.approval_manager.set_tool(tool_name, enabled)
            self.console.print(f"Approval for '{tool_name}' {'enabled' if enabled else 'disabled'}.")
            return

        self.console.print("Usage: /approval | /approval global on|off | /approval tool <name> on|off")

    def _print_approval_status(self) -> None:
        self.console.print(
            f"Global approval: {'enabled' if self.approval_manager.global_enabled() else 'disabled'}"
        )
        all_tool_names = sorted(self.mcp_manager.tool_names())
        if not all_tool_names:
            self.console.print("Tool-specific approvals: none (no active tools).")
            return
        for status in self.approval_manager.list_statuses(all_tool_names):
            required = "required" if status.approval_required else "not required"
            self.console.print(f"- {status.tool_name}: {required}")
        self.console.print()

    def _handle_memory_command(self) -> None:
        stats = self.memory_store.stats()
        body = (
            f"Estimated tokens in memory: {stats.estimated_tokens}\n"
            f"Rolling memory limit: {stats.token_limit}\n"
            f"Model context window target: {stats.context_window_target}\n"
            f"Recent turns kept: {stats.recent_turns_kept}\n"
            f"Truncation occurred last turn: {'yes' if stats.truncation_occurred_last_turn else 'no'}"
        )
        self.console.print("Short-term Memory")
        for line in body.splitlines():
            self.console.print(f"- {line}")
        self.console.print()

    def _handle_skills_command(self, args: Sequence[str]) -> None:
        if not args or args[0].lower() == "list":
            self._print_skills_summary()
            self._print_skills_list()
            return

        action = args[0].lower()
        if action == "refresh":
            self.skill_manager.refresh()
            self.console.print("Skills refreshed.")
            errors = self.skill_manager.refresh_errors()
            if errors:
                self.console.print("Some skills could not be loaded:")
                for err in errors:
                    self.console.print(f"- {err}")
            self._print_skills_summary()
            self._print_skills_list()
            return

        if action == "show":
            if len(args) < 2:
                self.console.print("Usage: /skills show <name>")
                return
            query = " ".join(args[1:]).strip()
            skill = self.skill_manager.get_skill(query)
            if not skill:
                self.console.print(f"No skill found for '{query}'.")
                return
            self.console.print(f"Skill: {skill.metadata.name}")
            self.console.print(skill.content.strip())
            self.console.print()
            return

        if action == "paths":
            self._print_skill_paths()
            return

        self.console.print("Usage: /skills | /skills list | /skills refresh | /skills show <name> | /skills paths")
        return

    def _print_skills_summary(self) -> None:
        rows = [
            ("Skill dirs", ", ".join(str(path) for path in self.skill_manager.list_skill_dirs()) or "(none)"),
            ("Max skills per turn", str(self.settings.skill_max_per_turn)),
            ("Max skill chars", str(self.settings.skill_max_chars)),
            ("Discovered skills", str(len(self.skill_manager.list_skills()))),
        ]
        self._print_kv_lines("Skill Configuration", rows)

    def _print_skills_list(self) -> None:
        skills = self.skill_manager.list_skills()
        if not skills:
            self._print_list("Available Skills", ["(none)"])
            return
        self.console.print("Available Skills")
        for meta in skills:
            desc = meta.description or ""
            self.console.print(f"- {meta.name}")
            if desc:
                self.console.print(f"  desc: {desc}")
            self.console.print(f"  path: {meta.skill_md_path}")
        self.console.print()

    def _print_skill_paths(self) -> None:
        rows = [(str(path), "yes" if path.exists() else "no") for path in self.skill_manager.list_skill_dirs()]
        self._print_kv_lines("Skill Search Paths", rows)

    def _vision_supported(self) -> bool:
        model = (self.current_model or "").lower()
        vision_markers = (
            "vision",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "llava",
            "kimi",
            "qwen2.5-vl",
            "qwen3-vl",
            "vl",
        )
        return any(marker in model for marker in vision_markers)

    def _read_clipboard_image_base64(self) -> str | None:
        script = (
            'set png_data to the clipboard as «class PNGf»\n'
            'set outFile to POSIX file "{path}"\n'
            'set outHandle to open for access outFile with write permission\n'
            'set eof outHandle to 0\n'
            'write png_data to outHandle\n'
            'close access outHandle\n'
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["osascript", "-e", script.format(path=tmp_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(tmp_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
            return encoded
        except Exception:
            return None
        finally:
            with suppress(Exception):
                os.remove(tmp_path)

    def _try_paste_clipboard_image(self, event) -> bool:
        encoded = self._read_clipboard_image_base64()
        if not encoded:
            return False
        if not self._vision_supported():
            self.console.print("Current model does not support vision. Switch to a vision-capable model.")
            return True
        self._pending_images.append(encoded)
        event.current_buffer.insert_text(f"[Image #{len(self._pending_images)}]")
        return True

    async def _handle_llm_command(self, args: Sequence[str]) -> None:
        if not args:
            self.console.print(f"Current provider: {self.current_provider}")
            self.console.print(f"Current model: {self.current_model}")
            self.console.print("Usage: /llm local [model] | /llm openai [model] | /llm openrouter [model]")
            return

        provider = args[0].lower()
        if provider == "local":
            model = args[1] if len(args) > 1 else self.settings.ollama_model
            self._switch_provider(provider="local", model=model)
            self.console.print(f"Switched LLM provider to local ({model}).")
            return

        if provider == "openai":
            self.console.print(
                "Note: OpenAI API uses API billing; a ChatGPT app subscription does not include API credits."
            )
            explicit_model = args[1] if len(args) > 1 else None
            if explicit_model:
                try:
                    self._switch_provider(provider="openai", model=explicit_model)
                except RuntimeError as exc:
                    self.console.print(str(exc))
                    return
                self.console.print(f"Switched LLM provider to openai ({explicit_model}).")
                return

            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                self.console.print("OPENAI_API_KEY is not set.")
                return

            try:
                model_ids = await list_openai_models(
                    api_key=api_key,
                    base_url=self.settings.openai_base_url,
                    timeout_seconds=self.settings.llm_request_timeout_seconds,
                )
            except OpenAIModelListError as exc:
                self.console.print(str(exc))
                return

            if not model_ids:
                self.console.print("No OpenAI models were returned for this API key.")
                return

            max_display = 30
            self.console.print("Available OpenAI Models")
            for index, model_name in enumerate(model_ids[:max_display], start=1):
                self.console.print(f"- {index}. {model_name}")
            if len(model_ids) > max_display:
                self.console.print(f"... and {len(model_ids) - max_display} more")

            choice = await self._prompt("Choose model by number or enter model id: ", multiline=False)
            choice = choice.strip()
            if not choice:
                self.console.print("OpenAI model selection canceled.")
                return

            selected_model: str | None = None
            if choice.isdigit():
                index = int(choice)
                if 1 <= index <= len(model_ids):
                    selected_model = model_ids[index - 1]
            else:
                if choice in model_ids:
                    selected_model = choice
                else:
                    selected_model = choice

            if not selected_model:
                self.console.print("Invalid selection.")
                return

            try:
                self._switch_provider(provider="openai", model=selected_model)
            except RuntimeError as exc:
                self.console.print(str(exc))
                return
            self.console.print(f"Switched LLM provider to openai ({selected_model}).")
            return

        if provider == "openrouter":
            self.console.print(
                "Note: OpenRouter uses API billing and requires OPENROUTER_API_KEY.",
            )
            model = args[1] if len(args) > 1 else self.settings.openrouter_model
            try:
                self._switch_provider(provider="openrouter", model=model)
            except RuntimeError as exc:
                self.console.print(str(exc))
                return
            self.console.print(f"Switched LLM provider to openrouter ({model}).")
            return

        self.console.print("Usage: /llm local [model] | /llm openai [model] | /llm openrouter [model]")

    async def _handle_new_command(self) -> None:
        self.memory_store.clear_session()
        self.console.print("Short-term memory cleared. Session restarted.")

        answer = await self._prompt("Also clear long-term memory, yes or no? ", multiline=False)

        if answer.strip().lower() not in {"yes", "y"}:
            self.console.print("Kept long-term memory.")
            return

        tools = {tool.name: tool for tool in self.mcp_manager.active_tools()}
        if not tools:
            self.console.print(
                "No MCP tools are currently active. Run /mcp refresh and retry /new if needed."
            )
            return
        try:
            success, message = await wipe_memory_graph(tools)
            self.console.print(message)
            if not success:
                self.console.print("Long-term memory wipe was not completed.")
        except asyncio.CancelledError:
            self._clear_current_task_cancellation()
            self.console.print("Long-term memory wipe was interrupted internally. You can continue.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to wipe long-term memory: %s", exc)
            self.console.print(f"Failed to wipe long-term memory: {exc}")

    def _print_welcome(self) -> None:
        model = self.current_model
        provider = self.current_provider
        cwd = str(Path.cwd())
        lines = [
            "Anton v0.1",
            f"Model: {model} ({provider})",
            f"Dir: {cwd}",
        ]
        width = max(len(line) for line in lines) + 2
        top = "┌" + ("─" * width) + "┐"
        bottom = "└" + ("─" * width) + "┘"
        self.console.print(top)
        for line in lines:
            padded = line.ljust(width)
            self.console.print(f"│{padded}│")
        self.console.print(bottom)
        self.console.print(ANTON_ASCII)
        self.console.print()

    def _get_footer(self) -> str:
        return f"{self.current_model} | {self.current_provider}"

    def _supports_ansi(self) -> bool:
        if not sys.stdout.isatty():
            return False
        if os.getenv("NO_COLOR"):
            return False
        term = os.getenv("TERM", "")
        if term.lower() in {"dumb", ""}:
            return False
        return True

    def _render_markdown_ansi(self, text: str) -> str:
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            force_terminal=True,
            color_system="truecolor",
            highlight=False,
            markup=False,
        )
        console.print(Markdown(text))
        return buffer.getvalue()

    def _looks_like_markdown(self, text: str) -> bool:
        if not text:
            return False
        if MARKDOWN_BLOCK_RE.search(text):
            return True
        if MARKDOWN_LINK_RE.search(text):
            return True
        if "|" in text and TABLE_SEPARATOR_RE.search(text):
            return True
        return False

    def _normalize_plain_response(self, text: str) -> str:
        normalized = _strip_ansi_codes(text).replace("\r\n", "\n").replace("\r", "\n")
        normalized = PARAGRAPH_BREAK_RE.sub("\n", normalized)
        return normalized.rstrip()

    def _print_plain_response(self, text: str) -> None:
        normalized = self._normalize_plain_response(text).lstrip()
        if not normalized:
            sys.stdout.write("> \n\n")
            sys.stdout.flush()
            return
        sys.stdout.write(f"> {normalized}\n\n")
        sys.stdout.flush()

    def _print_markdown_response(self, text: str) -> None:
        ansi_text = self._render_markdown_ansi(text).rstrip("\n")
        if not ansi_text:
            sys.stdout.write("> \n\n")
            sys.stdout.flush()
            return
        print_formatted_text(ANSI("> " + ansi_text), end="\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _print_help(self) -> None:
        commands = [
            ("/mcp", "Show MCP status"),
            ("/mcp refresh", "Reconnect MCP servers"),
            ("/mcp on <server>", "Enable one MCP server"),
            ("/mcp off <server>", "Disable one MCP server"),
            ("/approval", "Show approval settings"),
            ("/approval global on|off", "Toggle global approval"),
            ("/approval tool <tool> on|off", "Toggle approval for one tool"),
            ("/memory", "Show short-term memory stats"),
            ("/skills", "List available skills"),
            ("/skills refresh", "Rescan skill directories"),
            ("/skills show <name>", "Show full SKILL.md instructions"),
            ("/skills paths", "Show configured skill directories"),
            ("/paths", "List filesystem allowed paths"),
            ("/paths add <path>", "Allow filesystem access to a path"),
            ("/paths remove <path>", "Remove an allowed filesystem path"),
            ("/llm local [model]", "Switch to local Ollama model"),
            ("/llm openai [model]", "Switch to OpenAI model"),
            ("/llm openrouter [model]", "Switch to OpenRouter model (default Kimi 2.5)"),
            ("/new", "Reset short-term memory"),
            ("/quit", "Exit"),
        ]
        self.console.print("Commands")
        for command, desc in commands:
            self.console.print(f"- {command}: {desc}")
        self.console.print()
        self.console.print("Editor")
        self.console.print("- Enter sends")
        self.console.print("- Alt+Enter/Ctrl+J inserts newline")
        self.console.print("- Arrow keys navigate text/history")
        self.console.print("- Selection/editing follows prompt-toolkit native behavior")
        self.console.print("- Typing / triggers contextual command autocomplete")
        self.console.print()

    def _root_commands(self) -> list[str]:
        return [
            "/mcp",
            "/approval",
            "/memory",
            "/skills",
            "/paths",
            "/llm",
            "/new",
            "/quit",
            "/help",
        ]

    def _subcommand_options(self, root_command: str) -> list[str]:
        if root_command == "/mcp":
            return ["refresh", "on <server>", "off <server>"]
        if root_command == "/approval":
            return ["global on", "global off", "tool <tool_name> on", "tool <tool_name> off"]
        if root_command == "/paths":
            return ["list", "add <path>", "add downloads", "add desktop", "add documents", "remove <path>"]
        if root_command == "/llm":
            return ["local <model>", "openai <model>", "openrouter <model>"]
        if root_command == "/skills":
            return ["list", "refresh", "show <name>", "paths"]
        return []

    async def _handle_user_message(self, user_input: str) -> None:
        if self._pending_images:
            cleaned = IMAGE_TAG_RE.sub("", user_input).strip()
            content: list[dict[str, object]] = []
            if cleaned:
                content.append({"type": "text", "text": cleaned})
            for encoded in self._pending_images:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                )
            self._pending_images.clear()
            await self._handle_user_message_with_content(HumanMessage(content=content))
            return
        await self._handle_user_message_with_content(HumanMessage(content=user_input))

    async def _handle_user_message_with_content(self, message: HumanMessage) -> None:
        history = self.memory_store.load_messages()
        initial_messages, pre_truncation = self.memory_store.enforce_token_limit([*history, message])
        streamer = ResponseStreamPrinter()

        tool_map = {tool.name: tool for tool in self.mcp_manager.active_tools()}

        try:
            if self._markdown_enabled:
                result = await self.agent.run(
                    messages=initial_messages,
                    tools=tool_map,
                    approval_manager=self.approval_manager,
                    tool_event_callback=streamer.on_tool,
                )
            else:
                result = await self.agent.run(
                    messages=initial_messages,
                    tools=tool_map,
                    approval_manager=self.approval_manager,
                    stream_callback=streamer.on_token,
                    tool_event_callback=streamer.on_tool,
                )
        except TimeoutError:
            streamer.finish()
            self.console.print("Assistant request timed out.")
            self.memory_store.save_messages(initial_messages, truncation_occurred=pre_truncation)
            return
        except Exception as exc:  # noqa: BLE001
            streamer.finish()
            LOGGER.error("Agent run failed: %s", exc)
            self.console.print(f"Assistant error: {exc}")
            self.memory_store.save_messages(initial_messages, truncation_occurred=pre_truncation)
            return

        final_messages, post_truncation = self.memory_store.enforce_token_limit(result.messages)
        self.memory_store.save_messages(
            final_messages,
            truncation_occurred=pre_truncation or post_truncation,
        )

        if result.stop_reason == "tool_rejected":
            streamer.finish()
            return

        streamer.finish()
        if self._markdown_enabled:
            final_answer = result.final_answer or ""
            if self._looks_like_markdown(final_answer):
                self._print_markdown_response(final_answer)
            else:
                self._print_plain_response(final_answer)
        elif not streamer.saw_token:
            self._print_plain_response(result.final_answer or "")

    def _parse_on_off(self, value: str) -> bool | None:
        normalized = value.lower()
        if normalized == "on":
            return True
        if normalized == "off":
            return False
        return None

    def _clear_current_task_cancellation(self) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        while task.cancelling():
            task.uncancel()

    async def _safe_refresh_mcp(self, reason: str) -> bool:
        try:
            with _suppress_output():
                await self.mcp_manager.refresh_connections()
            return True
        except asyncio.CancelledError:
            self._clear_current_task_cancellation()
            self.console.print(f"MCP refresh was interrupted during {reason}. You can continue.")
            return False
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MCP refresh failed during %s: %s", reason, exc)
            self.console.print(f"MCP refresh failed during {reason}: {exc}")
            return False

    async def aclose(self) -> None:
        await self.agent.aclose()
        await self.mcp_manager.aclose()

    async def run(self) -> None:
        await self._safe_refresh_mcp("startup")
        self._print_welcome()

        with patch_stdout():
            while True:
                self._clear_current_task_cancellation()
                try:
                    # Minimal Prompt
                    raw = await self._prompt("• ", multiline=True)
                except EOFError:
                    self.console.print("Goodbye.")
                    break
                except KeyboardInterrupt:
                    continue

                user_input = raw.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    try:
                        should_exit = await self._handle_command(user_input)
                    except asyncio.CancelledError:
                        self._clear_current_task_cancellation()
                        self.console.print("Command was interrupted internally. You can continue.")
                        should_exit = False
                    if should_exit:
                        break
                    continue

                await self._handle_user_message(user_input)


async def _run_with_cleanup(app: AssistantCLI) -> None:
    try:
        await app.run()
    finally:
        await app.aclose()


def run() -> None:
    configure_logging()
    app = AssistantCLI()
    try:
        asyncio.run(_run_with_cleanup(app))
    except asyncio.CancelledError:
        app.console.print("Session interruption handled. You can restart safely.")
    except KeyboardInterrupt:
        app.console.print("\nInterrupted. Goodbye.")
