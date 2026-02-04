# LangGraph MCP Assistant

Production-quality Python CLI personal assistant using:
- **LangGraph** for controlled multi-tool agent orchestration.
- **LangChain** wrappers for Ollama and MCP tools.
- **Remote Ollama over HTTP** as the LLM backend.
- **MCP servers** as pluggable tools.
- **SQLite** for short-term memory + LangGraph checkpoints.

## Features

- CLI REPL with commands:
  - `/mcp`
  - `/approval`
  - `/memory`
  - `/paths`
  - `/llm`
  - `/new`
  - `/quit`
- Prompt-toolkit terminal UI:
  - multiline compose
  - slash command autocomplete/filtering
  - line editing with arrow keys/history
- Rich rendering for structured output (panels/tables/status views)
- Dynamic MCP server enable/disable at runtime.
- Per-tool and global approval gates before tool execution.
- Multi-step agent loop with up to 10 tool-iteration cycles by default.
- Rolling short-term memory budget (3000 token estimate).
- 8k context target with explicit budget reporting.
- Optional long-term memory wipe via MCP memory tools on `/new`.

## Project structure

```
.
├── assistant_cli
│   ├── agent_graph.py
│   ├── approval.py
│   ├── cli.py
│   ├── llm_client.py
│   ├── logging_utils.py
│   ├── mcp_manager.py
│   ├── memory_store.py
│   ├── memory_tools.py
│   └── settings.py
├── config
│   └── mcp_servers.sample.json
├── tests
│   └── smoke_test.py
├── main.py
└── requirements.txt
```

## Requirements

- Python **3.12.x** (recommended)
- Node.js + `npx` (for stdio MCP servers)
- Reachable remote Ollama server

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/mcp_servers.sample.json config/mcp_servers.json
```

## Dependency troubleshooting

If install fails with an `mcp` resolver conflict:

1. Confirm interpreter version:
   ```bash
   python --version
   ```
2. Recreate the venv with Python 3.12 specifically:
   ```bash
   rm -rf .venv
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

## MCP connection troubleshooting

If `/mcp` shows `connected=False`:

1. Ensure npm registry access works:
   ```bash
   npm ping
   ```
2. If npm prints `Access token expired or revoked`, refresh auth:
   ```bash
   npm logout
   npm login
   ```
3. Fetch server runs from Python package `mcp-server-fetch` via:
   - command: `python`
   - args: `-m mcp_server_fetch`
4. Reinstall dependencies after updates:
   ```bash
   pip install -r requirements.txt
   ```
5. For `web_search`, export `BRAVE_API_KEY` before launching the app.

## Environment variables

```bash
export OLLAMA_BASE_URL="http://100.126.228.118:11434"
export OLLAMA_MODEL="ministral-3:8b"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="..."

# Optional tuning
export OLLAMA_TEMPERATURE="0.1"
export SHORT_TERM_TOKEN_LIMIT="3000"
export MODEL_CONTEXT_WINDOW="8000"
export MAX_ITERATIONS="10"
export REQUEST_TIMEOUT_SECONDS="120"
export MCP_CONFIG_PATH="config/mcp_servers.json"
export ASSISTANT_SQLITE_PATH="data/assistant.db"

# If using Brave MCP web search
export BRAVE_API_KEY="..."
```

Note: some Ollama tags (including certain `llama3:8b` builds) do **not** support tool calling.
If tool calls fail with `does not support tools`, switch to a tool-capable model tag on your server.
OpenAI usage requires API credentials and API billing. A ChatGPT app subscription does not include API credits.

## Run

```bash
python main.py
```

## MCP configuration

`config/mcp_servers.json` is auto-copied from `config/mcp_servers.sample.json` on first run.

Example server definitions include:
- `filesystem`
- `fetch`
- `web_search`
- `memory`
- `sequential_thinking`

Each server has `enabled` and transport settings. You can toggle at runtime:
- `/mcp on <server>`
- `/mcp off <server>`

### Filesystem allowed paths

Use `/paths` to manage directories exposed by the filesystem MCP server:
- `/paths` or `/paths list` shows current allowed paths.
- `/paths add <path>` adds a directory.
- `/paths add downloads|desktop|documents` adds common macOS folders quickly.
- `/paths remove <path>` removes a directory (at least one must remain).
- Path changes are persisted to `config/mcp_servers.json`.

## LLM provider switching

Use `/llm` to inspect or switch providers at runtime:
- `/llm` shows current provider and model.
- `/llm local [model]` switches to local Ollama.
- `/llm openai` lists available OpenAI models and prompts selection.
- `/llm openai <model>` switches directly to a specific OpenAI model.
- The selected provider/model is persisted to `data/runtime_state.json` and restored on restart.

## Terminal editing

- `Enter` sends the message.
- `Alt+Enter` or `Ctrl+J` inserts newline.
- Arrow keys navigate text and history.
- Selection/editing behavior uses prompt-toolkit native controls (terminal-dependent).
- Typing `/` shows only root commands first (for example `/llm`), then contextual subcommands after a root is selected (`/llm local <model>`, `/llm openai <model>`).

Note: `Cmd+Arrow` usually does not reach terminal apps on macOS unless remapped by your terminal profile.

## Logging and output

- Default log level is `ERROR` for a cleaner CLI experience.
- Set `LOG_LEVEL=INFO` (or `DEBUG`) if you want verbose diagnostics.
- Rich colors are off by default to avoid ANSI artifacts in terminals that don't render them well.
- Set `ASSISTANT_COLOR=1` to enable colored Rich output.

## Short-term memory design

- Stored in SQLite table `conversation_state` inside `data/assistant.db` by default.
- Token estimate heuristic: serialized char length / 4.
- Rolling cap: **3000** estimated tokens.
- On overflow, oldest messages are truncated first.
- `/memory` prints:
  - estimated tokens in memory
  - token limit (3000)
  - model context target (8000)
  - recent turns kept
  - whether truncation happened in the last turn

## Long-term memory behavior

- Long-term memory is provided by MCP memory tools.
- It is **not automatically injected** into the prompt.
- Agent prompt instructs the model to explicitly retrieve needed memory via tools.
- `/new` clears short-term memory and asks:
  - `Also clear long-term memory, yes or no?`
- If yes, assistant tries to wipe entities and relations from memory graph safely.

## Agent loop details

LangGraph loop executes:
1. **Router node**: decide final answer vs tool call(s).
2. **Tool node**: execute approved tool calls, append tool outputs.
3. Return to router until stop condition.

Stop conditions:
- model returns final answer (no tool calls)
- tool call is rejected by user
- max iterations reached (default 10)
- hard request timeout

## Approval mode

`/approval` supports:
- global on/off
- per-tool on/off

When required, each tool call shows:
- tool name
- JSON payload
- prompt: `Allow or Reject? [allow/reject]`

Rejecting stops the current request immediately with:
- `Tool call rejected, stopping`

## Smoke test

```bash
python -m unittest tests/smoke_test.py
```
