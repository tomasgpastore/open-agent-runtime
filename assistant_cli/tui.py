from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import RenderableType

from textual.app import App, ComposeResult
from textual import events, on
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from assistant_cli.agent_graph import AgentRunResult, LangGraphAgent
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


# --- Constants ---

ANTON_ASCII = r"""
______          __                      
/\  _  \        /\ \__                   
\ \ \L\ \    ___\ \ ,_\   ___     ___    
 \ \  __ \ /' _ `\ \ \/  / __`\ /' _ `\  
  \ \ \/\ \/\ \/\ \ \ \_/\ \L\ \/\ \/\ \
   \ \_\ \_\ \_\ \_\ \__\ \____/\ \_\_\
    \/_/\/_/\/_/\/_/\/__/\___/  \/_/\/_/
"""

ANTON_TIPS = (
    "Tips for getting started:\n"
    "• Ask questions, edit files, or run commands.\n"
    "• Be specific for best results.\n"
    "• Use /help for command details."
)


# --- Messages ---

class InputSubmitted(Message):
    """Sent when the user submits a message from the input area."""
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


# --- Components ---

class MessageBubble(Static):
    """Displays a single message in the chat feed."""
    
    DEFAULT_CSS = """
    MessageBubble {
        padding: 0 1;
        margin-bottom: 1;
        background: transparent;
    }
    """

    def __init__(self, role: str, content: str = "") -> None:
        super().__init__("")
        self.role = role
        self.content = content
        self._refresh_content()

    def append(self, chunk: str) -> None:
        self.content += chunk
        self._refresh_content()

    def set_content(self, content: str) -> None:
        self.content = content
        self._refresh_content()

    def _refresh_content(self) -> None:
        if self.role == "user":
            prefix = Text("You: ", style="bold green")
        elif self.role == "assistant":
            prefix = Text("Anton: ", style="bold blue")
        else:
            prefix = Text("System: ", style="bold yellow")
        
        text = Text()
        text.append(prefix)
        text.append(self.content)
        
        self.update(text)


class ToolCard(Static):
    """Displays the status and output of a tool execution."""
    
    DEFAULT_CSS = """
    ToolCard {
        margin-bottom: 1;
        padding: 0 1;
    }
    """

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

    def _render(self) -> RenderableType:
        # Clean, minimal tool card
        status_color = "green" if self.status == "success" else "red" if self.status == "error" else "yellow"
        title = f"🛠️  {self.tool_name} "
        
        if self.expanded and self.details:
            return Panel(self.details, title=title, border_style=status_color, expand=False)
        else:
            # Minimal one-liner or collapsed view
            summary = f"[{status_color}]{self.status}[/]"
            if self.details:
                summary += " (ctrl+o to expand)"
            return Text.from_markup(f"{title} - {summary}")


class ChatInput(TextArea):
    """Custom input area with specific keybindings."""

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 1;
        max-height: 10;
        border: none;
        padding: 0 1;
        background: transparent;
        color: $text;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_line_numbers = False

    async def _on_key(self, event: events.Key) -> None:
        # Standardize "Submit" on Enter
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            val = self.text.strip()
            if val:
                self.post_message(InputSubmitted(val))
                self.text = ""
            return
        
        # Newline on Shift+Enter
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return

        # --- Enhanced Navigation & Selection ---
        # Note: TextArea in Textual has decent defaults, but we enforce specific
        # behaviors if they aren't standard.
        
        # Word Navigation (Ctrl+Left/Right)
        if event.key == "ctrl+left":
            event.stop()
            self.action_cursor_word_left()
            return
        if event.key == "ctrl+right":
            event.stop()
            self.action_cursor_word_right()
            return
            
        # Selection (Shift+Home/End)
        # Note: Textual calls might need to be chained for selection.
        # Currently Textual's public API for granular selection manipulation via keys
        # is evolving. We'll rely on defaults for shift+arrows which usually work,
        # but explicit shift+home/end might need custom actions if not supported.

        await super()._on_key(event)


class StatusBar(Static):
    """Bottom status bar with dynamic info."""
    
    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: transparent;
        color: $text-muted;
        content-align: center middle;
        text-opacity: 50%;
    }
    """
    
    def update_info(self, model: str, provider: str) -> None:
        cwd = os.getcwd()
        # Truncate CWD if too long
        if len(cwd) > 50:
            cwd = "..." + cwd[-47:]
            
        text = Text.assemble(
            " 📁 ", (cwd, "bold"),
            "  |  🤖 ", (model, "bold"),
            "  |  ☁️  ", (provider, "bold"),
            "  |  ❓ ", ("/help", "italic dim"),
        )
        self.update(text)


class ApprovalScreen(ModalScreen[str]):
    """Modal for tool approval."""
    
    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 60%;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    #approval-buttons {
        margin-top: 1;
        align: center middle;
    }
    Button {
        margin: 0 1;
    }
    """

    def __init__(self, tool_name: str, payload: dict) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.payload = payload

    def compose(self) -> ComposeResult:
        payload_text = json.dumps(self.payload, indent=2, ensure_ascii=False)
        with Vertical(id="approval-dialog"):
            yield Static(f"Allow execution of tool: [bold]{self.tool_name}[/bold]?", id="approval-title")
            yield TextArea(payload_text, read_only=True, classes="payload-view")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow", id="allow", variant="success")
                yield Button("Reject", id="reject", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "allow":
            self.dismiss("allow")
        else:
            self.dismiss("reject")


class _TuiBaselineAgent:
    """Mock agent for testing."""
    def set_llm_client(self, client): pass
    async def aclose(self): pass
    async def run(self, messages, tools, approval_manager, **kwargs):
        return AgentRunResult(
            messages=[*messages, AIMessage(content="Pong")],
            final_answer="Pong",
            stop_reason=None
        )


# --- Main Application ---

class AssistantTUI(App):
    """Main Textual Application for Anton."""

    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }

    #chat-container {
        height: 1fr;
        padding: 1;
        scrollbar-gutter: stable;
        overflow-y: scroll;
    }
    
    #ascii-art {
        color: blue;
        margin-bottom: 1;
        text-align: center;
    }
    
    #input-container {
        height: auto;
        padding: 1;
        background: transparent;
        border-top: solid $primary-darken-2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+o", "toggle_tools", "Toggle Tool Details"),
        Binding("ctrl+l", "clear_screen", "Clear Screen"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self._test_mode = os.getenv("ASSISTANT_TEST_MODE", "").lower() in {"1", "true", "yes", "on"}
        
        # State
        self.current_provider = "local"
        self.current_model = self.settings.ollama_model
        self._load_persisted_llm()
        
        # Core Components
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
            connect_timeout_seconds=self.settings.mcp_connect_timeout_seconds,
        )
        self.skill_manager = SkillManager(
            skill_dirs=self.settings.skill_dirs,
            max_per_turn=self.settings.skill_max_per_turn,
            max_chars=self.settings.skill_max_chars,
        )

        if self._test_mode:
            self.agent = _TuiBaselineAgent()
        else:
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
        # Chat Area (Top)
        with ScrollableContainer(id="chat-container"):
            yield Static(ANTON_ASCII, id="ascii-art")
            yield Static(ANTON_TIPS, classes="tips")
            # Messages will be mounted here
            
        # Input Area (Middle)
        with Vertical(id="input-container"):
            yield ChatInput(placeholder="Message Anton...")
            
        # Status Bar (Bottom)
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        if not self._test_mode:
            try:
                await self.mcp_manager.refresh_connections()
            except Exception as e:
                self._post_system_message(f"Startup MCP Refresh Failed: {e}")
        
        self.query_one(ChatInput).focus()
        self._update_status_bar()

    async def on_unmount(self) -> None:
        try:
            await self.agent.aclose()
            await self.mcp_manager.aclose()
        except Exception:
            pass

    # --- Actions & Handlers ---

    async def on_input_submitted(self, message: InputSubmitted) -> None:
        """Handle user input."""
        content = message.value.strip()
        if not content:
            return

        # Slash Commands
        if content.startswith("/"):
            await self._handle_command(content)
            return

        # Chat Message
        self._add_chat_message("user", content)
        
        # Run Agent in background worker
        self.run_worker(self._handle_agent_interaction(content), exclusive=True)

    def action_toggle_tools(self) -> None:
        for card in self.query(ToolCard):
            card.toggle()

    def action_clear_screen(self) -> None:
        # Clear chat container but keep ASCII/Tips? Or just clear everything?
        # Let's clear just messages.
        container = self.query_one("#chat-container", ScrollableContainer)
        for child in container.query(MessageBubble):
            child.remove()
        for child in container.query(ToolCard):
            child.remove()
        for child in container.query(".system-message"):
            child.remove()

    # --- Core Logic ---

    async def _handle_agent_interaction(self, user_input: str) -> None:
        history = self.memory_store.load_messages()
        # Enforce limits before running
        initial_messages, pre_trunc = self.memory_store.enforce_token_limit(
            [*history, HumanMessage(content=user_input)]
        )
        if pre_trunc:
             self._post_system_message("Memory truncated to fit context window.")

        # Prepare UI for streaming
        container = self.query_one("#chat-container", ScrollableContainer)
        
        # Placeholder for Assistant Message
        assistant_bubble = MessageBubble("assistant", "")
        self.call_from_thread(container.mount, assistant_bubble)
        self.call_from_thread(container.scroll_end, animate=False)

        # Callbacks for streaming
        loop = asyncio.get_running_loop()

        def on_token(token: str):
            loop.call_soon_threadsafe(assistant_bubble.append, token)
            # Auto-scroll on new content? Textual's ScrollableContainer *should* stick to bottom if configured
            # but explicit scroll often helps.
            loop.call_soon_threadsafe(container.scroll_end, animate=False)

        def on_tool(event: object):
            tool_name = "tool"
            if isinstance(event, dict):
                raw_name = event.get("tool_name")
                if isinstance(raw_name, str) and raw_name:
                    tool_name = raw_name
            elif isinstance(event, str) and event:
                tool_name = event

            card = ToolCard(tool_name)
            self._current_tool_cards.append(card)
            loop.call_soon_threadsafe(container.mount, card)
            loop.call_soon_threadsafe(container.scroll_end, animate=False)

        # Run Agent
        tool_map = {tool.name: tool for tool in self.mcp_manager.active_tools()}
        
        try:
            result = await self.agent.run(
                messages=initial_messages,
                tools=tool_map,
                approval_manager=self.approval_manager,
                stream_callback=on_token,
                tool_event_callback=on_tool,
                approval_prompt=self._approval_prompt
            )
            
            # Finalize
            self._finalize_tool_cards(result.messages)
            
            # Save Memory
            final_messages, post_trunc = self.memory_store.enforce_token_limit(result.messages)
            self.memory_store.save_messages(final_messages, truncation_occurred=pre_trunc or post_trunc)
            
            # Ensure final answer is displayed if streaming missed it (rare with this setup)
            if not assistant_bubble.content and result.final_answer:
                assistant_bubble.set_content(result.final_answer)

        except Exception as e:
            assistant_bubble.set_content(f"Error: {e}")

    async def _approval_prompt(self, tool_name: str, payload: dict) -> bool:
        """Show modal for approval."""
        res = await self.push_screen(ApprovalScreen(tool_name, payload))
        if res == "allow":
            return True
        self._post_system_message(f"Tool '{tool_name}' rejected.")
        return False

    def _finalize_tool_cards(self, messages: Iterable[BaseMessage]) -> None:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        for card, msg in zip(self._current_tool_cards, tool_msgs):
            content = str(msg.content)
            status = "error" if "failed" in content.lower() or "timed out" in content.lower() else "success"
            card.set_result(content, status)
        self._current_tool_cards.clear()

    # --- Helpers ---

    def _add_chat_message(self, role: str, content: str) -> None:
        bubble = MessageBubble(role, content)
        container = self.query_one("#chat-container", ScrollableContainer)
        container.mount(bubble)
        container.scroll_end(animate=False)

    def _post_system_message(self, content: str | RenderableType) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        if isinstance(content, str):
            container.mount(MessageBubble("system", content))
        else:
            container.mount(Static(content, classes="system-message"))
        container.scroll_end(animate=False)

    def _update_status_bar(self) -> None:
        self.query_one(StatusBar).update_info(self.current_model, self.current_provider)

    # --- Command Handling (Simplified for brevity, logic copied from CLI) ---

    async def _handle_command(self, cmd: str) -> None:
        parts = cmd.strip().split()
        base = parts[0].lower()
        
        if base == "/quit":
            self.exit()
        elif base == "/help":
            self._show_help()
        elif base == "/llm":
            await self._handle_llm_cmd(parts[1:])
        elif base == "/mcp":
            await self._handle_mcp_cmd(parts[1:])
        elif base == "/new":
            await self._handle_new_cmd()
        else:
            self._post_system_message(f"Unknown command: {base}")

    def _show_help(self):
        help_text = """
        [bold]Available Commands:[/bold]
        /llm [local|openai|openrouter] [model] - Switch LLM
        /mcp [refresh|on|off] - Manage Tools
        /new - Start new session
        /quit - Exit
        """
        self._post_system_message(Panel(help_text, title="Help"))

    async def _handle_llm_cmd(self, args: list[str]):
        if not args:
            self._post_system_message(f"Current: {self.current_provider} / {self.current_model}")
            return
        
        provider = args[0].lower()
        model = args[1] if len(args) > 1 else None
        
        try:
            if provider == "local":
                target = model or self.settings.ollama_model
                self.agent.set_llm_client(self._build_local_llm_client(target))
            elif provider == "openai":
                target = model or self.settings.openai_model
                self.agent.set_llm_client(self._build_openai_llm_client(target))
            elif provider == "openrouter":
                target = model or self.settings.openrouter_model
                self.agent.set_llm_client(self._build_openrouter_llm_client(target))
            else:
                self._post_system_message(f"Unknown provider: {provider}")
                return
                
            self.current_provider = provider
            self.current_model = target
            self._update_status_bar()
            self._post_system_message(f"Switched to {provider} ({target})")
            
        except Exception as e:
            self._post_system_message(f"Failed to switch LLM: {e}")

    async def _handle_mcp_cmd(self, args: list[str]):
        # Simplified implementation
        if not args or args[0] == "refresh":
            try:
                await self.mcp_manager.refresh_connections()
                self._post_system_message("MCP Tools Refreshed.")
            except Exception as e:
                self._post_system_message(f"Error: {e}")
        # Add on/off logic as needed...

    async def _handle_new_cmd(self):
        self.memory_store.clear_session()
        self._post_system_message("Short-term memory cleared.")

    # --- LLM Client Builders (Copied) ---

    def _load_persisted_llm(self) -> None:
        path = self.settings.runtime_state_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            llm = payload.get("llm", {})
            if llm.get("provider") and llm.get("model"):
                self.current_provider = llm["provider"]
                self.current_model = llm["model"]
        except Exception:
            pass

    def _build_initial_llm_client(self):
        # ... logic to restore client ...
        if self.current_provider == "openai": return self._build_openai_llm_client(self.current_model)
        if self.current_provider == "openrouter": return self._build_openrouter_llm_client(self.current_model)
        return self._build_local_llm_client(self.current_model)

    def _build_local_llm_client(self, model: str) -> OllamaLLMClient:
        return OllamaLLMClient(
            OllamaLLMConfig(
                base_url=self.settings.ollama_base_url,
                model=model,
                temperature=self.settings.ollama_temperature,
                context_window=self.settings.model_context_window,
                timeout_seconds=self.settings.llm_request_timeout_seconds,
            )
        )

    def _build_openai_llm_client(self, model: str) -> OpenAILLMClient:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key: raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAILLMClient(
            OpenAILLMConfig(
                api_key=api_key,
                model=model,
                base_url=self.settings.openai_base_url,
                temperature=self.settings.ollama_temperature,
                timeout_seconds=self.settings.llm_request_timeout_seconds,
            )
        )

    def _build_openrouter_llm_client(self, model: str) -> OpenAILLMClient:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key: raise RuntimeError("OPENROUTER_API_KEY missing")
        return OpenAILLMClient(
            OpenAILLMConfig(
                api_key=api_key,
                model=model,
                base_url=self.settings.openrouter_base_url,
                temperature=self.settings.ollama_temperature,
                timeout_seconds=self.settings.llm_request_timeout_seconds,
            )
        )


def run_tui() -> None:
    configure_logging()
    app = AssistantTUI()
    app.run(inline=False)
