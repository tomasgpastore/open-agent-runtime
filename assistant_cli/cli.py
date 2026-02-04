from __future__ import annotations

import asyncio
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


LOGGER = logging.getLogger(__name__)


class SlashCommandCompleter(Completer):
    def __init__(self, commands_provider: Callable[[], Sequence[str]]) -> None:
        self._commands_provider = commands_provider

    def get_completions(self, document, complete_event):  # type: ignore[override]
        text_before_cursor = document.text_before_cursor
        if not text_before_cursor.startswith("/"):
            return
        if " " in text_before_cursor or "\n" in text_before_cursor:
            return

        for command in sorted(set(self._commands_provider())):
            if command.startswith(text_before_cursor):
                yield Completion(
                    command,
                    start_position=-len(text_before_cursor),
                )


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
        self.console = Console()
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

        llm_client = self._build_local_llm_client(model=self.settings.ollama_model)

        self.agent = LangGraphAgent(
            db_path=self.settings.sqlite_path,
            llm_client=llm_client,
            max_iterations=self.settings.max_iterations,
            request_timeout_seconds=self.settings.request_timeout_seconds,
            tool_timeout_seconds=self.settings.tool_timeout_seconds,
        )
        self._prompt_session = self._build_prompt_session()

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
            completer=SlashCommandCompleter(self._available_commands),
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
            )
        )

    def _switch_provider(self, provider: str, model: str) -> None:
        if provider == "local":
            client = self._build_local_llm_client(model=model)
        elif provider == "openai":
            client = self._build_openai_llm_client(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.agent.set_llm_client(client)
        self.current_provider = provider
        self.current_model = model

    async def run(self) -> None:
        await self.mcp_manager.refresh_connections()
        self._print_welcome()

        with patch_stdout():
            while True:
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
                    should_exit = await self._handle_command(user_input)
                    if should_exit:
                        break
                    continue

                await self._handle_user_message(user_input)

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
            LOGGER.exception("Agent run failed")
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
            await self.mcp_manager.refresh_connections()
            self._print_mcp_status()
            return

        if len(args) == 2 and args[0].lower() in {"on", "off"}:
            server_name = args[1]
            if not self.mcp_manager.is_server_known(server_name):
                self.console.print(f"Unknown MCP server '{server_name}'.", style="yellow")
                return

            enabled = args[0].lower() == "on"
            self.mcp_manager.set_server_enabled(server_name, enabled)
            await self.mcp_manager.refresh_connections()
            self.console.print(
                f"MCP server '{server_name}' set to {'enabled' if enabled else 'disabled'}.",
                style="green" if enabled else "yellow",
            )
            self._print_mcp_status()
            return

        self.console.print("Usage: /mcp | /mcp refresh | /mcp on <server> | /mcp off <server>", style="yellow")

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
                "[green]true[/green]" if status.connected else "[red]false[/red]",
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

    async def _handle_llm_command(self, args: Sequence[str]) -> None:
        if not args:
            self.console.print(f"Current provider: {self.current_provider}", style="bold")
            self.console.print(f"Current model: {self.current_model}", style="bold")
            self.console.print("Usage: /llm local [model] | /llm openai [model]", style="yellow")
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

        self.console.print("Usage: /llm local [model] | /llm openai [model]", style="yellow")

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

        await self.mcp_manager.refresh_connections()
        tools = {tool.name: tool for tool in self.mcp_manager.active_tools()}
        try:
            success, message = await wipe_memory_graph(tools)
            self.console.print(message, style="green" if success else "yellow")
            if not success:
                self.console.print("Long-term memory wipe was not completed.", style="yellow")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to wipe long-term memory")
            self.console.print(f"Failed to wipe long-term memory: {exc}", style="bold red")

    def _print_welcome(self) -> None:
        body = (
            f"[bold]Model:[/bold] {self.current_model}\n"
            f"[bold]LLM provider:[/bold] {self.current_provider}\n"
            f"[bold]Ollama base URL:[/bold] {self.settings.ollama_base_url}\n"
            "[bold]Tip:[/bold] Type /help for available commands."
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
        table.add_row("/llm local [model]", "Switch to local Ollama model")
        table.add_row("/llm openai [model]", "Switch to OpenAI model")
        table.add_row("/new", "Reset short-term memory")
        table.add_row("/quit", "Exit")
        self.console.print(table)
        self.console.print(
            Panel.fit(
                "Enter sends\nAlt+Enter or Ctrl+J inserts newline\n"
                "Arrow keys navigate text/history\n"
                "Typing / triggers command autocomplete",
                title="Editor",
                border_style="blue",
            )
        )

    def _available_commands(self) -> list[str]:
        return [
            "/mcp",
            "/approval",
            "/memory",
            "/llm",
            "/new",
            "/quit",
            "/help",
        ]

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
    except KeyboardInterrupt:
        app.console.print("\nInterrupted. Goodbye.", style="bold yellow")
    finally:
        asyncio.run(app.aclose())
