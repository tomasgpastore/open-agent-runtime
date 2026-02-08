# Anton v0.2 Runtime

Production-quality Python CLI coworker agent runtime using:
- **LangGraph** for controlled multi-tool agent orchestration.
- **LangChain** wrappers for Ollama and MCP tools.
- **Remote Ollama over HTTP** as the LLM backend.
- **MCP servers** as pluggable tools.
- **SQLite** for short-term memory, long-term memory, and graph checkpoints/state.

## Features

- CLI REPL with commands:
  - `/mcp`
  - `/approval`
  - `/memory`
  - `/memory daily ...`
  - `/memory fact ...`
  - `/memory retrieve ...`
  - `/graph ...`
  - `/skills`
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
- Filesystem skill discovery via `SKILL.md` metadata (Agent Skills).
- Per-tool and global approval gates before tool execution.
- Multi-step agent loop with up to 100 tool-iteration cycles by default.
- Rolling short-term memory budget (20,000 token estimate).
- 20k context target with explicit budget reporting.
- Graph Builder Anton (`/graph build`) for intent-to-graph generation.
- Graph execution modes: `strict`, `bounded`, `flex`.
- Graph replay/resume with persisted checkpoints.
- Graph scheduler with cron-like persisted schedules.
- Guard against repeated identical tool-call loops in a single turn.
- Optional long-term memory wipe via MCP memory tools on `/new`.

## Project structure

```
.
├── assistant_cli
│   ├── agent_graph.py
│   ├── approval.py
│   ├── cli.py
│   ├── daily_memory.py
│   ├── graph/
│   ├── cloud/
│   ├── llm_client.py
│   ├── long_term_memory.py
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
export OPENAI_MAX_COMPLETION_TOKENS="4096"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_MODEL="moonshotai/kimi-k2.5"
export OPENROUTER_API_KEY="..."
export OPENROUTER_MAX_COMPLETION_TOKENS="4096"

# Optional tuning
export OLLAMA_TEMPERATURE="0.1"
export SHORT_TERM_TOKEN_LIMIT="20000"
export MODEL_CONTEXT_WINDOW="20000"
export MAX_ITERATIONS="100"
export REQUEST_TIMEOUT_SECONDS="600"
export MCP_CONNECT_TIMEOUT_SECONDS="12"
export MCP_CONFIG_PATH="config/mcp_servers.json"
export ASSISTANT_SQLITE_PATH="data/assistant.db"
export ANTON_DAILY_MEMORY_DIR="data/memory/daily"
export ANTON_SKILL_DIRS="skills:~/.codex/skills"
export SKILL_MAX_PER_TURN="3"
export SKILL_MAX_CHARS="8000"

# If using Brave MCP web search
export BRAVE_API_KEY="..."
```

Note: some Ollama tags (including certain `llama3:8b` builds) do **not** support tool calling.
If tool calls fail with `does not support tools`, switch to a tool-capable model tag on your server.
OpenAI usage requires API credentials and API billing. A ChatGPT app subscription does not include API credits.

## Skills

Skills live in folders containing a `SKILL.md` with YAML frontmatter (`name`, `description`). Configure
skill directories with `ANTON_SKILL_DIRS` (or `SKILL_DIRS`) and manage them via:
- `/skills` or `/skills list`
- `/skills refresh`
- `/skills show <name>`
- `/skills paths`

## Run

```bash
python main.py
```

Optional CLI wrappers (from repo root):

```bash
bin/anton .
bin/anton /path/to/project
```

## MCP configuration

`config/mcp_servers.json` is auto-copied from `config/mcp_servers.sample.json` on first run.

Example server definitions include:
- `filesystem`
- `fetch`
- `web_search`
- `memory`
- `pdf_reader`
- `playwright`
- `sequential_thinking`

Each server has `enabled` and transport settings. You can toggle at runtime:
- `/mcp on <server>`
- `/mcp off <server>`

Notes:
- `pdf_reader` uses `@sylphlab/pdf-reader-mcp` for PDF extraction.
- `playwright` uses `@playwright/mcp` for browser automation workflows.

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
- `/llm openrouter [model]` switches to OpenRouter (default: `moonshotai/kimi-k2.5`).
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
- Rolling cap: **20,000** estimated tokens.
- On overflow, oldest messages are truncated first.
- `/memory` prints:
  - estimated tokens in memory
  - token limit (20,000)
  - model context target (20,000)
  - recent turns kept
  - whether truncation happened in the last turn
- `/memory daily days [limit]` lists archived day files.
- `/memory daily search <query> [--day YYYY-MM-DD] [--limit N]` searches indexed daily chunks.
- Daily archive format:
  - append-only `.md` files by day under `data/memory/daily`
  - chunk indexing with token-based overlap for retrieval
- `/skills` shows Anton's current capabilities, providers, and active tool list.

## Long-term memory behavior

- Local long-term facts are stored in SQLite (`long_term_facts`) and managed via:
  - `/memory fact add|search|get|delete|list|prune`
- Combined retrieval across daily + long-term memory is available via:
  - `/memory retrieve <query>`
- MCP memory tools remain optional for external graph-memory workflows.
- `/new` clears short-term memory and asks:
  - `Also clear long-term memory, yes or no?`
- If yes, assistant tries to wipe entities and relations from memory graph safely.
  - Wipe uses currently active MCP memory tools (no forced reconnect).

## Graph runtime v0.2

- Graph definitions are validated and persisted in SQLite.
- Node types include:
  - `start`, `end`, `transform`, `tool`, `ai_template`, `condition`
  - `read_state`, `write_state`, `read_prior_runs`
- Every node takes input and emits output; output is available to downstream nodes.
- State nodes persist and retrieve graph-level execution state across runs.
- Reliability controls:
  - checkpoint per step
  - replay from prior runs (`/graph replay`)
  - resume from failure checkpoints (`/graph resume`)
  - scheduler for recurring runs (`/graph schedule ...`)

## Agent loop details

LangGraph loop executes:
1. **Router node**: decide final answer vs tool call(s).
2. **Tool node**: execute approved tool calls, append tool outputs.
3. Return to router until stop condition.

Stop conditions:
- model returns final answer (no tool calls)
- tool call is rejected by user
- max iterations reached (default 100)
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
