from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
    def __init__(self, console: Console) -> None:
        self._console = console
        self._line_open = False
        self._saw_token = False

    def on_token(self, token: str) -> None:
        if not token:
            return
        if not self._line_open:
            self._console.print("assistant> ", style="bold cyan", end="")
            self._line_open = True
        self._console.print(token, end="", markup=False, highlight=False, soft_wrap=True)
        self._saw_token = True

    def on_tool(self, tool_name: str) -> None:
        self._close_line_if_needed()
        self._console.print(f"tool> {tool_name}", style="bold magenta")

    def finish(self) -> None:
        self._close_line_if_needed()

    @property
    def saw_token(self) -> bool:
        return self._saw_token

    def _close_line_if_needed(self) -> None:
        if self._line_open:
            self._console.print()
            self._line_open = False


class AssistantCLI:
    def __init__(self) -> None:
        use_color = os.getenv("ASSISTANT_COLOR", "").lower() in {"1", "true", "yes", "on"}
        self.console = Console(no_color=not use_color, highlight=False, markup=False)
        self.settings = load_settings()
        self.current_provider = "local"
        self.current_model = self.settings.ollama_model

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
                f"Falling back to local ({self.settings.ollama_model}).",
                style="yellow",
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

        # Most terminals don't emit a distinct keycode for Shift+Enter.
        # Provide robust newline fallbacks: Ctrl+J and Alt/Meta+Enter (escape+enter).
        @key_bindings.add("escape", "enter")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-j")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

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
        )

    async def _prompt(self, prompt: str, multiline: bool = False) -> str:
        return await self._prompt_session.prompt_async(
            HTML(prompt),
            multiline=multiline,
        )

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
            self.console.print(
                "Short-term session history was reset after provider switch.",
                style="dim",
            )

    async def run(self) -> None:
        await self._safe_refresh_mcp("startup")
        self._print_welcome()

        with patch_stdout():
            while True:
                self._clear_current_task_cancellation()
                try:
                    raw = await self._prompt("<ansicyan>you&gt; </ansicyan>", multiline=True)
                except EOFError:
                    self.console.print("Goodbye.", style="bold green")
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
                        self.console.print(
                            "Command was interrupted internally. You can continue.",
                            style="yellow",
                        )
                        should_exit = False
                    if should_exit:
                        break
                    continue

                await self._handle_user_message(user_input)

    def _clear_current_task_cancellation(self) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        while task.cancelling():
            task.uncancel()

    async def _safe_refresh_mcp(self, reason: str) -> bool:
        try:
            await self.mcp_manager.refresh_connections()
            return True
        except asyncio.CancelledError:
            self._clear_current_task_cancellation()
            self.console.print(
                f"MCP refresh was interrupted during {reason}. You can continue.",
                style="yellow",
            )
            return False
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MCP refresh failed during %s: %s", reason, exc)
            self.console.print(
                f"MCP refresh failed during {reason}: {exc}",
                style="bold red",
            )
            return False

    async def aclose(self) -> None:
        await self.agent.aclose()
        await self.mcp_manager.aclose()

    async def _handle_user_message(self, user_input: str) -> None:
        history = self.memory_store.load_messages()
        initial_messages, pre_truncation = self.memory_store.enforce_token_limit(
            [*history, HumanMessage(content=user_input)]
        )
        streamer = ResponseStreamPrinter(self.console)

        tool_map = {tool.name: tool for tool in self.mcp_manager.active_tools()}

        try:
            result = await self.agent.run(
                messages=initial_messages,
                tools=tool_map,
                approval_manager=self.approval_manager,
                stream_callback=streamer.on_token,
                tool_event_callback=streamer.on_tool,
            )
        except TimeoutError:
            streamer.finish()
            self.console.print("Assistant request timed out.", style="bold red")
            self.memory_store.save_messages(initial_messages, truncation_occurred=pre_truncation)
            return
        except Exception as exc:  # noqa: BLE001
            streamer.finish()
            LOGGER.error("Agent run failed: %s", exc)
            self.console.print(f"Assistant error: {exc}", style="bold red")
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
        if not streamer.saw_token:
            self.console.print("assistant> ", style="bold cyan", end="")
            self.console.print(result.final_answer, markup=False, highlight=False)

    async def _handle_command(self, command_line: str) -> bool:
        parts = command_line.strip().split()
        command = parts[0].lower()

        if command == "/quit":
            self.console.print("Goodbye.", style="bold green")
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

        self.console.print("Unknown command. Use /help for available commands.", style="yellow")
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
                self.console.print(f"Unknown MCP server '{server_name}'.", style="yellow")
                return

            enabled = args[0].lower() == "on"
            self.mcp_manager.set_server_enabled(server_name, enabled)
            await self._safe_refresh_mcp("mcp toggle command")
            self.console.print(
                f"MCP server '{server_name}' set to {'enabled' if enabled else 'disabled'}.",
                style="green" if enabled else "yellow",
            )
            self._print_mcp_status()
            return

        self.console.print("Usage: /mcp | /mcp refresh | /mcp on <server> | /mcp off <server>", style="yellow")

    async def _handle_paths_command(self, args: Sequence[str]) -> None:
        if not args or args[0].lower() == "list":
            self._print_allowed_paths()
            return

        action = args[0].lower()
        if action not in {"add", "remove"}:
            self.console.print("Usage: /paths | /paths add <path|downloads|desktop|documents> | /paths remove <path>", style="yellow")
            return

        if len(args) < 2:
            self.console.print("Path is required.", style="yellow")
            return

        raw_path = " ".join(args[1:]).strip()
        target_path = self._resolve_directory_alias(raw_path)

        try:
            if action == "add":
                added = self.mcp_manager.add_filesystem_allowed_directory(target_path)
                self.console.print(
                    f"Added filesystem allowed path: {added} (saved to MCP config).",
                    style="green",
                )
            else:
                removed = self.mcp_manager.remove_filesystem_allowed_directory(target_path)
                self.console.print(
                    f"Removed filesystem allowed path: {removed} (saved to MCP config).",
                    style="green",
                )
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"Failed to update allowed paths: {exc}", style="bold red")
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
            self.console.print(f"Failed to read filesystem allowed paths: {exc}", style="bold red")
            return

        table = Table(title="Filesystem Allowed Paths")
        table.add_column("Path", style="bold")
        table.add_column("Exists")
        if not paths:
            table.add_row("(none)", "n/a")
        else:
            for path in paths:
                exists = "yes" if os.path.exists(path) else "no"
                table.add_row(path, exists)
        self.console.print(table)

    def _print_mcp_status(self) -> None:
        table = Table(title="MCP Servers", show_lines=True)
        table.add_column("Server", style="bold")
        table.add_column("Enabled")
        table.add_column("Connected")
        table.add_column("Tools")
        table.add_column("Error")
        for status in self.mcp_manager.list_statuses():
            if status.tools:
                tool_lines = []
                for tool_name in status.tools:
                    approval = "yes" if self.approval_manager.tool_enabled(tool_name) else "no"
                    tool_lines.append(f"{tool_name} (approval: {approval})")
                tools_value = "\n".join(tool_lines)
            else:
                tools_value = "(none)"
            table.add_row(
                status.name,
                "true" if status.enabled else "false",
                "true" if status.connected else "false",
                tools_value,
                status.last_error or "",
            )
        self.console.print()
        self.console.print(table)

    def _handle_approval_command(self, args: Sequence[str]) -> None:
        if not args:
            self._print_approval_status()
            return

        if len(args) == 2 and args[0].lower() == "global":
            enabled = self._parse_on_off(args[1])
            if enabled is None:
                self.console.print("Usage: /approval global on|off", style="yellow")
                return
            self.approval_manager.set_global(enabled)
            self.console.print(
                f"Global approval {'enabled' if enabled else 'disabled'}.",
                style="green" if enabled else "yellow",
            )
            return

        if len(args) == 3 and args[0].lower() == "tool":
            tool_name = args[1]
            enabled = self._parse_on_off(args[2])
            if enabled is None:
                self.console.print("Usage: /approval tool <tool_name> on|off", style="yellow")
                return

            self.approval_manager.set_tool(tool_name, enabled)
            self.console.print(
                f"Approval for '{tool_name}' {'enabled' if enabled else 'disabled'}.",
                style="green" if enabled else "yellow",
            )
            return

        self.console.print(
            "Usage: /approval | /approval global on|off | /approval tool <name> on|off",
            style="yellow",
        )

    def _print_approval_status(self) -> None:
        self.console.print(
            f"Global approval: {'enabled' if self.approval_manager.global_enabled() else 'disabled'}",
            style="bold",
        )
        all_tool_names = sorted(self.mcp_manager.tool_names())
        if not all_tool_names:
            self.console.print("Tool-specific approvals: none (no active tools).", style="dim")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Tool")
        table.add_column("Approval required")
        for status in self.approval_manager.list_statuses(all_tool_names):
            table.add_row(
                status.tool_name,
                "required" if status.approval_required else "not required",
            )
        self.console.print(table)

    def _handle_memory_command(self) -> None:
        stats = self.memory_store.stats()
        body = (
            f"Estimated tokens in memory: {stats.estimated_tokens}\n"
            f"Rolling memory limit: {stats.token_limit}\n"
            f"Model context window target: {stats.context_window_target}\n"
            f"Recent turns kept: {stats.recent_turns_kept}\n"
            f"Truncation occurred last turn: {'yes' if stats.truncation_occurred_last_turn else 'no'}"
        )
        self.console.print(Panel.fit(body, title="Short-term Memory", border_style="blue"))

    def _handle_skills_command(self, args: Sequence[str]) -> None:
        if not args or args[0].lower() == "list":
            self._print_skills_summary()
            self._print_skills_list()
            return

        action = args[0].lower()
        if action == "refresh":
            self.skill_manager.refresh()
            self.console.print("Skills refreshed.", style="green")
            errors = self.skill_manager.refresh_errors()
            if errors:
                self.console.print("Some skills could not be loaded:", style="yellow")
                for err in errors:
                    self.console.print(f"- {err}", style="yellow")
            self._print_skills_summary()
            self._print_skills_list()
            return

        if action == "show":
            if len(args) < 2:
                self.console.print("Usage: /skills show <name>", style="yellow")
                return
            query = " ".join(args[1:]).strip()
            skill = self.skill_manager.get_skill(query)
            if not skill:
                self.console.print(f"No skill found for '{query}'.", style="yellow")
                return
            panel = Panel.fit(
                skill.content.strip(),
                title=f"Skill: {skill.metadata.name}",
                border_style="blue",
            )
            self.console.print(panel)
            return

        if action == "paths":
            self._print_skill_paths()
            return

        self.console.print(
            "Usage: /skills | /skills list | /skills refresh | /skills show <name> | /skills paths",
            style="yellow",
        )
        return

    def _print_skills_summary(self) -> None:
        table = Table(title="Skill Configuration")
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        table.add_row(
            "Skill dirs",
            ", ".join(str(path) for path in self.skill_manager.list_skill_dirs()) or "(none)",
        )
        table.add_row("Max skills per turn", str(self.settings.skill_max_per_turn))
        table.add_row("Max skill chars", str(self.settings.skill_max_chars))
        table.add_row("Discovered skills", str(len(self.skill_manager.list_skills())))
        self.console.print(table)

    def _print_skills_list(self) -> None:
        skills = self.skill_manager.list_skills()
        table = Table(title="Available Skills")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        table.add_column("Location")
        if not skills:
            table.add_row("(none)", "", "")
        else:
            for meta in skills:
                table.add_row(meta.name, meta.description or "", str(meta.skill_md_path))
        self.console.print(table)

    def _print_skill_paths(self) -> None:
        table = Table(title="Skill Search Paths")
        table.add_column("Path", style="bold")
        table.add_column("Exists")
        for path in self.skill_manager.list_skill_dirs():
            table.add_row(str(path), "yes" if path.exists() else "no")
        self.console.print(table)

    async def _handle_llm_command(self, args: Sequence[str]) -> None:
        if not args:
            self.console.print(f"Current provider: {self.current_provider}", style="bold")
            self.console.print(f"Current model: {self.current_model}", style="bold")
            self.console.print(
                "Usage: /llm local [model] | /llm openai [model] | /llm openrouter [model]",
                style="yellow",
            )
            return

        provider = args[0].lower()
        if provider == "local":
            model = args[1] if len(args) > 1 else self.settings.ollama_model
            self._switch_provider(provider="local", model=model)
            self.console.print(f"Switched LLM provider to local ({model}).", style="green")
            return

        if provider == "openai":
            self.console.print(
                "Note: OpenAI API uses API billing; a ChatGPT app subscription does not include API credits."
                ,
                style="yellow",
            )
            explicit_model = args[1] if len(args) > 1 else None
            if explicit_model:
                try:
                    self._switch_provider(provider="openai", model=explicit_model)
                except RuntimeError as exc:
                    self.console.print(str(exc), style="bold red")
                    return
                self.console.print(f"Switched LLM provider to openai ({explicit_model}).", style="green")
                return

            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                self.console.print("OPENAI_API_KEY is not set.", style="bold red")
                return

            try:
                model_ids = await list_openai_models(
                    api_key=api_key,
                    base_url=self.settings.openai_base_url,
                    timeout_seconds=self.settings.llm_request_timeout_seconds,
                )
            except OpenAIModelListError as exc:
                self.console.print(str(exc), style="bold red")
                return

            if not model_ids:
                self.console.print("No OpenAI models were returned for this API key.", style="yellow")
                return

            max_display = 30
            table = Table(title="Available OpenAI Models")
            table.add_column("#", justify="right")
            table.add_column("Model")
            for index, model_name in enumerate(model_ids[:max_display], start=1):
                table.add_row(str(index), model_name)
            self.console.print(table)
            if len(model_ids) > max_display:
                self.console.print(f"... and {len(model_ids) - max_display} more", style="dim")

            choice = await self._prompt(
                "<ansiblue>Choose model by number or enter model id: </ansiblue>",
                multiline=False,
            )
            choice = choice.strip()
            if not choice:
                self.console.print("OpenAI model selection canceled.", style="yellow")
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
                self.console.print("Invalid selection.", style="yellow")
                return

            try:
                self._switch_provider(provider="openai", model=selected_model)
            except RuntimeError as exc:
                self.console.print(str(exc), style="bold red")
                return
            self.console.print(f"Switched LLM provider to openai ({selected_model}).", style="green")
            return

        if provider == "openrouter":
            self.console.print(
                "Note: OpenRouter uses API billing and requires OPENROUTER_API_KEY.",
                style="yellow",
            )
            model = args[1] if len(args) > 1 else self.settings.openrouter_model
            try:
                self._switch_provider(provider="openrouter", model=model)
            except RuntimeError as exc:
                self.console.print(str(exc), style="bold red")
                return
            self.console.print(f"Switched LLM provider to openrouter ({model}).", style="green")
            return

        self.console.print(
            "Usage: /llm local [model] | /llm openai [model] | /llm openrouter [model]",
            style="yellow",
        )

    async def _handle_new_command(self) -> None:
        self.memory_store.clear_session()
        self.console.print("Short-term memory cleared. Session restarted.", style="green")

        answer = await self._prompt(
            "<ansiblue>Also clear long-term memory, yes or no? </ansiblue>",
            multiline=False,
        )

        if answer.strip().lower() not in {"yes", "y"}:
            self.console.print("Kept long-term memory.", style="dim")
            return

        tools = {tool.name: tool for tool in self.mcp_manager.active_tools()}
        if not tools:
            self.console.print(
                "No MCP tools are currently active. Run /mcp refresh and retry /new if needed.",
                style="yellow",
            )
            return
        try:
            success, message = await wipe_memory_graph(tools)
            self.console.print(message, style="green" if success else "yellow")
            if not success:
                self.console.print("Long-term memory wipe was not completed.", style="yellow")
        except asyncio.CancelledError:
            self._clear_current_task_cancellation()
            self.console.print(
                "Long-term memory wipe was interrupted internally. You can continue.",
                style="yellow",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to wipe long-term memory: %s", exc)
            self.console.print(f"Failed to wipe long-term memory: {exc}", style="bold red")

    def _print_welcome(self) -> None:
        if self.current_provider == "local":
            provider_endpoint = f"Ollama base URL: {self.settings.ollama_base_url}"
        elif self.current_provider == "openai":
            provider_endpoint = f"OpenAI base URL: {self.settings.openai_base_url}"
        else:
            provider_endpoint = f"OpenRouter base URL: {self.settings.openrouter_base_url}"
        body = (
            f"Model: {self.current_model}\n"
            f"LLM provider: {self.current_provider}\n"
            f"{provider_endpoint}\n"
            "Tip: Type /help for available commands."
        )
        self.console.print(Panel.fit(body, title="LangGraph MCP Assistant", border_style="cyan"))

    def _print_help(self) -> None:
        table = Table(title="Commands")
        table.add_column("Command", style="bold")
        table.add_column("Description")
        table.add_row("/mcp", "Show MCP status")
        table.add_row("/mcp refresh", "Reconnect MCP servers")
        table.add_row("/mcp on <server>", "Enable one MCP server")
        table.add_row("/mcp off <server>", "Disable one MCP server")
        table.add_row("/approval", "Show approval settings")
        table.add_row("/approval global on|off", "Toggle global approval")
        table.add_row("/approval tool <tool> on|off", "Toggle approval for one tool")
        table.add_row("/memory", "Show short-term memory stats")
        table.add_row("/skills", "List available skills")
        table.add_row("/skills refresh", "Rescan skill directories")
        table.add_row("/skills show <name>", "Show full SKILL.md instructions")
        table.add_row("/skills paths", "Show configured skill directories")
        table.add_row("/paths", "List filesystem allowed paths")
        table.add_row("/paths add <path>", "Allow filesystem access to a path")
        table.add_row("/paths remove <path>", "Remove an allowed filesystem path")
        table.add_row("/llm local [model]", "Switch to local Ollama model")
        table.add_row("/llm openai [model]", "Switch to OpenAI model")
        table.add_row("/llm openrouter [model]", "Switch to OpenRouter model (default Kimi 2.5)")
        table.add_row("/new", "Reset short-term memory")
        table.add_row("/quit", "Exit")
        self.console.print(table)
        self.console.print(
            Panel.fit(
                "Enter sends\nAlt+Enter/Ctrl+J inserts newline\n"
                "Arrow keys navigate text/history\n"
                "Selection/editing follows prompt-toolkit native behavior\n"
                "Typing / triggers contextual command autocomplete",
                title="Editor",
                border_style="blue",
            )
        )

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

    def _parse_on_off(self, value: str) -> bool | None:
        normalized = value.lower()
        if normalized == "on":
            return True
        if normalized == "off":
            return False
        return None



def run() -> None:
    configure_logging()
    app = AssistantCLI()
    try:
        asyncio.run(app.run())
    except asyncio.CancelledError:
        app.console.print("Session interruption handled. You can restart safely.", style="yellow")
    except KeyboardInterrupt:
        app.console.print("\nInterrupted. Goodbye.", style="bold yellow")
    finally:
        asyncio.run(app.aclose())
