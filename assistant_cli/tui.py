from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

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


class InputSubmitted(Message):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class ChatInput(TextArea):
    async def _on_key(self, event: events.Key) -> None:  # type: ignore[override]
        if event.key == "enter" and not event.shift:
            event.stop()
            event.prevent_default()
            value = self.text.strip()
            if value:
                self.post_message(InputSubmitted(value))
                self.text = ""
            return
        if event.key in {"shift+enter"} or (event.key == "enter" and event.shift):
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return
        await super()._on_key(event)


class MessageBubble(Static):
    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self.role = role
        self.content = content
        self.add_class(f"role-{role}")

    def append(self, chunk: str) -> None:
        self.content += chunk
        self.update(self._render())

    def set_content(self, content: str) -> None:
        self.content = content
        self.update(self._render())

    def _render(self) -> Group:
        label = "You" if self.role == "user" else "Anton" if self.role == "assistant" else "System"
        header = Text(label, style="bold cyan" if self.role == "assistant" else "bold green")
        body = Text(self.content)
        return Group(header, body)

    def render(self) -> Group:  # type: ignore[override]
        return self._render()


class ToolCard(Static):
    def __init__(self, tool_name: str, status: str = "running") -> None:
        super().__init__()
        self.tool_name = tool_name
        self.status = status
        self.details: str | None = None
        self.expanded = False

    def set_result(self, result: str, status: str) -> None:
        self.details = result
        self.status = status
        self.update(self._render())

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.update(self._render())

    def _render(self) -> Panel:
        status_icon = "✓" if self.status == "success" else "✗" if self.status == "error" else "⏳"
        lines = [f"Status: {status_icon} {self.status}"]
        if self.expanded and self.details:
            lines.append("")
            lines.append("Result:")
            lines.append(self.details)
        elif self.details:
            lines.append("[Ctrl+O to toggle details]")
        body = "\n".join(lines)
        return Panel(body, title=f"Tool Call: {self.tool_name}", border_style="yellow")

    def render(self) -> Panel:  # type: ignore[override]
        return self._render()


class ApprovalScreen(ModalScreen[str]):
    def __init__(self, tool_name: str, payload: dict) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.payload = payload

    def compose(self) -> ComposeResult:
        payload_text = json.dumps(self.payload, indent=2, ensure_ascii=False, sort_keys=True)
        body = Panel(
            f"Tool: {self.tool_name}\n\nArguments:\n{payload_text}",
            title="Approve Tool Call?",
            border_style="yellow",
        )
        yield Vertical(
            Static(body),
            Horizontal(
                Button("Allow", id="allow", variant="success"),
                Button("Reject", id="reject", variant="error"),
            ),
            id="approval-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "allow":
            self.dismiss("allow")
        else:
            self.dismiss("reject")


class AssistantTUI(App):
    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }

    #header {
        padding: 1 2;
        background: transparent;
    }

    #message-area {
        height: 1fr;
        padding: 1 2;
        background: transparent;
        border-top: solid $primary;
        border-bottom: solid $primary;
        border-left: none;
        border-right: none;
    }

    MessageBubble.role-user {
        margin: 0 0 1 0;
    }

    MessageBubble.role-assistant {
        margin: 0 0 1 0;
    }

    ToolCard {
        margin: 0 0 1 0;
    }

    #input-area {
        height: 5;
        padding: 0 2 0 2;
        background: transparent;
        border: none;
    }

    #footer {
        height: 1;
        padding: 0 2;
        background: transparent;
        border-top: solid $primary;
    }

    #approval-dialog {
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+o", "toggle_tools", "Toggle Tools"),
        Binding("ctrl+l", "scroll_bottom", "Scroll Bottom"),
        Binding("ctrl+r", "new_session", "New Session"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.current_provider = "local"
        self.current_model = self.settings.ollama_model
        self._load_persisted_llm()

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

        self.agent = LangGraphAgent(
            db_path=self.settings.sqlite_path,
            llm_client=self._build_initial_llm_client(),
            max_iterations=self.settings.max_iterations,
            request_timeout_seconds=self.settings.request_timeout_seconds,
            tool_timeout_seconds=self.settings.tool_timeout_seconds,
            skill_manager=self.skill_manager,
        )

        self._current_tool_cards: list[ToolCard] = []

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield ScrollableContainer(id="message-area")
        yield ChatInput(id="input-area")
        yield Static("Ctrl+C quit · Ctrl+O toggle tools · Ctrl+R new", id="footer")

    async def on_mount(self) -> None:
        await self._safe_refresh_mcp("startup")
        self._render_header()
        self.query_one(ChatInput).focus()
        self.query_one("#message-area", ScrollableContainer).can_focus = False

    async def on_key(self, event: events.Key) -> None:  # type: ignore[override]
        if (
            event.key == "enter"
            and not event.shift
            and self.query_one(ChatInput).has_focus
        ):
            event.prevent_default()
            event.stop()
            value = self.query_one(ChatInput).text.strip()
            if value:
                self.post_message(InputSubmitted(value))
                self.query_one(ChatInput).text = ""
                self.query_one(ChatInput).focus()
            return
        return

    async def _safe_refresh_mcp(self, reason: str) -> None:
        try:
            await self.mcp_manager.refresh_connections()
        except Exception as exc:  # noqa: BLE001
            self._post_system_message(f"MCP refresh failed during {reason}: {exc}")

    def _render_header(self) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        status = "connected" if any(s.connected for s in self.mcp_manager.list_statuses()) else "offline"
        header = Panel(
            f"Model: {self.current_model} | Provider: {self.current_provider} | MCP: {status} | {now}",
            title="LangGraph MCP Assistant",
            border_style="cyan",
        )
        self.query_one("#header", Static).update(header)

    def _post_system_message(self, content: str | Table | Panel) -> None:
        area = self.query_one("#message-area", ScrollableContainer)
        if isinstance(content, (Table, Panel)):
            area.mount(Static(content))
        else:
            bubble = MessageBubble("system", content)
            area.mount(bubble)

    def _post_user_message(self, content: str) -> None:
        area = self.query_one("#message-area", ScrollableContainer)
        area.mount(MessageBubble("user", content))

    def _post_assistant_message(self) -> MessageBubble:
        bubble = MessageBubble("assistant", "")
        self.query_one("#message-area", ScrollableContainer).mount(bubble)
        return bubble

    async def _approval_prompt(self, tool_name: str, payload: dict) -> bool:
        result = await self.push_screen(ApprovalScreen(tool_name, payload))
        if result in {"reject", "no"}:
            self._post_system_message("Tool call rejected, stopping.")
            return False
        return True

    async def on_input_submitted(self, message: InputSubmitted) -> None:
        content = message.value.strip()
        if not content:
            return
        if content.startswith("/"):
            await self._handle_command(content)
            return
        self._post_user_message(content)
        self.run_worker(self._handle_user_message(content), exclusive=True)

    async def _handle_user_message(self, content: str) -> None:
        history = self.memory_store.load_messages()
        initial_messages, pre_truncation = self.memory_store.enforce_token_limit(
            [*history, HumanMessage(content=content)]
        )

        assistant_bubble = self._post_assistant_message()

        def on_token(token: str) -> None:
            self.call_from_thread(assistant_bubble.append, token)

        def on_tool(tool_name: str) -> None:
            card = ToolCard(tool_name)
            self._current_tool_cards.append(card)
            self.call_from_thread(
                self.query_one("#message-area", ScrollableContainer).mount, card
            )

        tool_map = {tool.name: tool for tool in self.mcp_manager.active_tools()}

        try:
            result = await self.agent.run(
                messages=initial_messages,
                tools=tool_map,
                approval_manager=self.approval_manager,
                stream_callback=on_token,
                tool_event_callback=on_tool,
                approval_prompt=self._approval_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(assistant_bubble.set_content, f"Assistant error: {exc}")
            return

        self._finalize_tool_cards(result.messages)
        final_messages, post_truncation = self.memory_store.enforce_token_limit(result.messages)
        self.memory_store.save_messages(
            final_messages,
            truncation_occurred=pre_truncation or post_truncation,
        )

        if not assistant_bubble.content.strip():
            self.call_from_thread(assistant_bubble.set_content, result.final_answer)

    def _finalize_tool_cards(self, messages: Iterable[BaseMessage]) -> None:
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        for card, tool_msg in zip(self._current_tool_cards, tool_messages):
            content = str(tool_msg.content)
            status = "error" if "failed" in content.lower() or "timed out" in content.lower() else "success"
            card.set_result(content, status)
        self._current_tool_cards.clear()

    async def _handle_command(self, command_line: str) -> None:
        parts = command_line.split()
        command = parts[0].lower()

        if command == "/help":
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
            table.add_row("/skills", "Show Anton's capabilities")
            table.add_row("/paths", "List filesystem allowed paths")
            table.add_row("/paths add <path>", "Allow filesystem access to a path")
            table.add_row("/paths remove <path>", "Remove an allowed filesystem path")
            table.add_row("/llm local [model]", "Switch to local Ollama model")
            table.add_row("/llm openai [model]", "Switch to OpenAI model")
            table.add_row("/llm openrouter [model]", "Switch to OpenRouter model")
            table.add_row("/new", "Reset short-term memory")
            table.add_row("/quit", "Exit")
            self._post_system_message(table)
            return

        if command == "/quit":
            await self.action_quit()
            return

        if command == "/mcp":
            await self._handle_mcp_command(parts[1:])
            return

        if command == "/approval":
            self._handle_approval_command(parts[1:])
            return

        if command == "/memory":
            self._handle_memory_command()
            return

        if command == "/skills":
            self._handle_skills_command(parts[1:])
            return

        if command == "/paths":
            await self._handle_paths_command(parts[1:])
            return

        if command == "/llm":
            await self._handle_llm_command(parts[1:])
            return

        if command == "/new":
            await self._handle_new_command()
            return

        self._post_system_message("Unknown command. Use /help for available commands.")

    async def _handle_mcp_command(self, args: list[str]) -> None:
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
                self._post_system_message(f"Unknown MCP server '{server_name}'.")
                return
            enabled = args[0].lower() == "on"
            self.mcp_manager.set_server_enabled(server_name, enabled)
            await self._safe_refresh_mcp("mcp toggle command")
            self._post_system_message(
                f"MCP server '{server_name}' set to {'enabled' if enabled else 'disabled'}."
            )
            self._print_mcp_status()
            return
        self._post_system_message("Usage: /mcp | /mcp refresh | /mcp on <server> | /mcp off <server>")

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
        self._post_system_message(table)

    def _handle_approval_command(self, args: list[str]) -> None:
        if not args:
            self._print_approval_status()
            return
        if len(args) == 2 and args[0].lower() == "global":
            enabled = args[1].lower() == "on"
            self.approval_manager.set_global(enabled)
            self._post_system_message(f"Global approval {'enabled' if enabled else 'disabled'}.")
            return
        if len(args) == 3 and args[0].lower() == "tool":
            tool_name = args[1]
            enabled = args[2].lower() == "on"
            self.approval_manager.set_tool(tool_name, enabled)
            self._post_system_message(
                f"Approval for '{tool_name}' {'enabled' if enabled else 'disabled'}."
            )
            return
        self._post_system_message("Usage: /approval | /approval global on|off | /approval tool <name> on|off")

    def _print_approval_status(self) -> None:
        table = Table(title="Tool Approvals")
        table.add_column("Tool")
        table.add_column("Approval required")
        for status in self.approval_manager.list_statuses(self.mcp_manager.tool_names()):
            table.add_row(status.tool_name, "required" if status.approval_required else "not required")
        self._post_system_message(table)

    def _handle_memory_command(self) -> None:
        stats = self.memory_store.stats()
        body = (
            f"Estimated tokens in memory: {stats.estimated_tokens}\n"
            f"Rolling memory limit: {stats.token_limit}\n"
            f"Model context window target: {stats.context_window_target}\n"
            f"Recent turns kept: {stats.recent_turns_kept}\n"
            f"Truncation occurred last turn: {'yes' if stats.truncation_occurred_last_turn else 'no'}"
        )
        self._post_system_message(Panel.fit(body, title="Short-term Memory", border_style="blue"))

    def _handle_skills_command(self) -> None:
        self._handle_skills_command([])

    def _handle_skills_command(self, args: list[str]) -> None:
        if not args or args[0].lower() == "list":
            self._print_skills_summary()
            self._print_skills_list()
            return

        action = args[0].lower()
        if action == "refresh":
            self.skill_manager.refresh()
            self._post_system_message("Skills refreshed.")
            errors = self.skill_manager.refresh_errors()
            if errors:
                self._post_system_message("\n".join(["Some skills could not be loaded:", *errors]))
            self._print_skills_summary()
            self._print_skills_list()
            return

        if action == "show":
            if len(args) < 2:
                self._post_system_message("Usage: /skills show <name>")
                return
            query = " ".join(args[1:]).strip()
            skill = self.skill_manager.get_skill(query)
            if not skill:
                self._post_system_message(f"No skill found for '{query}'.")
                return
            panel = Panel.fit(
                skill.content.strip(),
                title=f"Skill: {skill.metadata.name}",
                border_style="blue",
            )
            self._post_system_message(panel)
            return

        if action == "paths":
            self._print_skill_paths()
            return

        self._post_system_message(
            "Usage: /skills | /skills list | /skills refresh | /skills show <name> | /skills paths"
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
        self._post_system_message(table)

    def _print_skills_list(self) -> None:
        skills = self.skill_manager.list_skills()
        table = Table(title="Available Skills")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        table.add_column("Location")
        if not skills:
            table.add_row("(none)", "", "")
        else:
            for skill in skills:
                table.add_row(skill.name, skill.description, str(skill.skill_md_path))
        self._post_system_message(table)

    def _print_skill_paths(self) -> None:
        table = Table(title="Skill Paths")
        table.add_column("Path", style="bold")
        for path in self.skill_manager.list_skill_dirs():
            table.add_row(str(path))
        self._post_system_message(table)

    async def _handle_paths_command(self, args: list[str]) -> None:
        if not args or args[0].lower() == "list":
            self._print_allowed_paths()
            return
        action = args[0].lower()
        if action not in {"add", "remove"}:
            self._post_system_message("Usage: /paths | /paths add <path> | /paths remove <path>")
            return
        if len(args) < 2:
            self._post_system_message("Path is required.")
            return
        raw_path = " ".join(args[1:]).strip()
        target_path = self._resolve_directory_alias(raw_path)
        try:
            if action == "add":
                added = self.mcp_manager.add_filesystem_allowed_directory(target_path)
                self._post_system_message(f"Added filesystem allowed path: {added}.")
            else:
                removed = self.mcp_manager.remove_filesystem_allowed_directory(target_path)
                self._post_system_message(f"Removed filesystem allowed path: {removed}.")
        except Exception as exc:  # noqa: BLE001
            self._post_system_message(f"Failed to update allowed paths: {exc}")
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
        paths = self.mcp_manager.list_filesystem_allowed_directories()
        table = Table(title="Filesystem Allowed Paths")
        table.add_column("Path", style="bold")
        table.add_column("Exists")
        if not paths:
            table.add_row("(none)", "n/a")
        else:
            for path in paths:
                exists = "yes" if os.path.exists(path) else "no"
                table.add_row(path, exists)
        self._post_system_message(table)

    async def _handle_llm_command(self, args: list[str]) -> None:
        if not args:
            self._post_system_message(
                f"Current provider: {self.current_provider}\nCurrent model: {self.current_model}"
            )
            return
        provider = args[0].lower()
        if provider == "local":
            model = args[1] if len(args) > 1 else self.settings.ollama_model
            self._switch_provider(provider="local", model=model)
            self._post_system_message(f"Switched LLM provider to local ({model}).")
            return
        if provider == "openrouter":
            model = args[1] if len(args) > 1 else self.settings.openrouter_model
            try:
                self._switch_provider(provider="openrouter", model=model)
            except RuntimeError as exc:
                self._post_system_message(str(exc))
                return
            self._post_system_message(f"Switched LLM provider to openrouter ({model}).")
            return
        if provider == "openai":
            explicit_model = args[1] if len(args) > 1 else None
            if explicit_model:
                try:
                    self._switch_provider(provider="openai", model=explicit_model)
                except RuntimeError as exc:
                    self._post_system_message(str(exc))
                    return
                self._post_system_message(f"Switched LLM provider to openai ({explicit_model}).")
                return
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                self._post_system_message("OPENAI_API_KEY is not set.")
                return
            try:
                model_ids = await list_openai_models(
                    api_key=api_key,
                    base_url=self.settings.openai_base_url,
                    timeout_seconds=self.settings.llm_request_timeout_seconds,
                )
            except OpenAIModelListError as exc:
                self._post_system_message(str(exc))
                return
            if not model_ids:
                self._post_system_message("No OpenAI models were returned for this API key.")
                return
            table = Table(title="Available OpenAI Models")
            table.add_column("#", justify="right")
            table.add_column("Model")
            for index, model_name in enumerate(model_ids[:30], start=1):
                table.add_row(str(index), model_name)
            self._post_system_message(table)
            return
        self._post_system_message("Usage: /llm local [model] | /llm openai [model] | /llm openrouter [model]")

    async def _handle_new_command(self) -> None:
        self.memory_store.clear_session()
        self._post_system_message("Short-term memory cleared. Session restarted.")
        answer = await self._ask_user("Also clear long-term memory, yes or no?")
        if answer.strip().lower() not in {"yes", "y"}:
            self._post_system_message("Kept long-term memory.")
            return
        tools = {tool.name: tool for tool in self.mcp_manager.active_tools()}
        if not tools:
            self._post_system_message("No MCP tools are currently active. Run /mcp refresh and retry /new.")
            return
        success, message = await wipe_memory_graph(tools)
        self._post_system_message(message if success else f"{message}\nLong-term memory wipe was not completed.")

    async def _ask_user(self, prompt: str) -> str:
        result = await self.push_screen(ApprovalScreen("Anton", {"prompt": prompt}))
        return "yes" if result == "allow" else "no"

    def action_toggle_tools(self) -> None:
        for card in self.query(ToolCard):
            card.toggle()

    def action_scroll_bottom(self) -> None:
        self.query_one("#message-area", ScrollableContainer).scroll_end(animate=False)

    def action_new_session(self) -> None:
        self.run_worker(self._handle_new_command(), exclusive=True)

    def _load_persisted_llm(self) -> None:
        path = self.settings.runtime_state_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        llm_data = payload.get("llm")
        if not isinstance(llm_data, dict):
            return
        provider = llm_data.get("provider")
        model = llm_data.get("model")
        if provider in {"local", "openai", "openrouter"} and isinstance(model, str):
            self.current_provider = provider
            self.current_model = model

    def _build_initial_llm_client(self):
        try:
            if self.current_provider == "openai":
                return self._build_openai_llm_client(model=self.current_model)
            if self.current_provider == "openrouter":
                return self._build_openrouter_llm_client(model=self.current_model)
            return self._build_local_llm_client(model=self.current_model)
        except Exception:
            self.current_provider = "local"
            self.current_model = self.settings.ollama_model
            return self._build_local_llm_client(model=self.current_model)

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
        self._render_header()
        if provider_changed:
            self.memory_store.clear_session()


def run_tui() -> None:
    configure_logging()
    app = AssistantTUI()
    app.run()
