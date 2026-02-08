# Anton v0.2 Specification (Implemented)

## Summary
Anton v0.2 is implemented as a local-first coworker agent runtime with three coordinated systems:
- General Anton: conversational agent with MCP tools, skills, approvals, and streaming output.
- Graph Builder Anton: intent-to-graph generator that outputs validated Anton graph JSON.
- Error Handler Anton: bounded runtime remediation (retry/fallback/fail policy).

The v0.2 differentiator is controlled automation through deterministic graph execution with persistent run state, checkpoints, replay/resume, scheduling, and memory-backed state nodes.

## Scope Completed
- MCP + skills + hooks integrated in runtime.
- Graph execution engine with validation, policy gates, and typed node runtime.
- Graph state memory persisted in SQLite (run history + key/value state + checkpoints + events).
- Guarantee modes: `strict`, `bounded`, `flex`.
- Graph scheduler (cron-style) with persisted schedules and manual/automatic triggers.
- Long-term memory store and daily archive retrieval planner.
- CLI control surface for graph build/run/replay/resume/state/schedule and memory operations.

## Memory Architecture

### Hot State (short-term)
- System prompt + Anton persona.
- Rolling conversation memory in SQLite.
- Active execution context for current turn/graph run.

### Cold State
- Daily chat archive (`data/memory/daily/YYYY-MM-DD.md`) with chunk indexing.
- Long-term fact store (`long_term_facts`) with namespace, importance, TTL, prune, and retrieval.

### Retrieval Model
- Combined retrieval path (`/memory retrieve`) merges:
  - daily archive hits
  - long-term fact hits
- Ranking favors high-importance long-term facts while still returning daily-context hits.

## Graph Runtime Contract

### Node I/O
Yes: every node consumes input and produces output.
- Default node input: previous node output (`last`).
- Node input can be explicitly templated from context (`{{input.foo}}`, `{{node_id.bar}}`, etc.).
- Node output is stored in context under its `node_id` and becomes `last` for next node.

### State Nodes
State nodes follow the same input/output contract and are persisted across executions:
- `read_state`: reads durable graph key/value state.
- `write_state`: writes durable graph key/value state.
- `read_prior_runs`: reads prior run metadata for the same graph.

This enables conditional flows based on prior graph state and historical executions.

## Guarantee Modes

### `strict`
- Highest determinism.
- For `tool` and `ai_template` nodes requires:
  - `idempotency_key`
  - `timeout_seconds`
  - `max_retries`
- `llm_condition` is constrained and must provide valid branch options.
- Best for repeatable automation with minimal runtime ambiguity.

### `bounded`
- Balanced mode.
- Requires bounded runtime controls (`timeout_seconds`, `max_retries`) on executable nodes.
- Allows constrained LLM branching with safe fallbacks.
- Recommended default.

### `flex`
- Highest autonomy.
- Looser constraints; allows explicit node fallback outputs where configured.
- Useful for exploratory automations where determinism is less strict.

## Branching Model
Branching is configured by condition nodes:
- `typed_condition`: operator-based deterministic branching (`eq`, `gt`, `contains`, etc.).
- `llm_condition`: schema-constrained branch selection from `branch_options`.

Interpretation by mode:
- `strict`: prefer deterministic/validated branching, tighter enforcement.
- `bounded`: deterministic + constrained LLM branching.
- `flex`: allows broader LLM-mediated behavior with fallback pathways.

## Execution Reliability
- Runtime validation before execution.
- Checkpoint written at each node transition.
- Error checkpoints persisted for resume.
- Replay support from prior run inputs.
- Resume support from last failed checkpoint context.
- Hook lifecycle events:
  - `before_run`
  - `after_run`
  - `before_node`
  - `after_node`
  - `on_error`

## CLI Surface (v0.2)

### Graph
- `/graph build <intent>`
- `/graph validate <path|graph_id>`
- `/graph show <path|graph_id>`
- `/graph render <path|graph_id>`
- `/graph patch <graph_id> <node_id> <field> <json_value>`
- `/graph run <path|graph_id> [--mode strict|bounded|flex] [--input payload.json]`
- `/graph replay <run_id>`
- `/graph resume <run_id>`
- `/graph runs <graph_id> [limit]`
- `/graph state show|get|history|checkpoints ...`
- `/graph schedule add|list|on|off|delete|trigger|tick|start|stop ...`

### Memory
- `/memory` (short-term stats)
- `/memory daily days [limit]`
- `/memory daily search <query> [--day YYYY-MM-DD] [--limit N]`
- `/memory fact add|search|get|delete|list|prune ...`
- `/memory retrieve <query> [--day YYYY-MM-DD] [--limit N]`

## Cloud-Ready Boundary (Implemented in v0.2)
- Execution backend protocol abstraction.
- Local backend implementation (`LocalExecutionBackend`) wired to graph executor.
- Credential/provider protocol contracts for future hosted execution.

## Completion Status
v0.2 implementation is complete for the local runtime scope described above.
All core v0.2 capabilities (general agent, graph builder, bounded error handling, stateful graph execution, memory architecture, and scheduling) are implemented and test-covered in this repository.
