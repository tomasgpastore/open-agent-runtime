# Anton v0.2 Plan

## Vision
Anton v0.2 is a graph-centric coworker agent platform with three coordinated agents:
- General Anton for day-to-day assistance
- Graph Builder Anton for automation graph generation
- Error Handler Anton for policy-bounded remediation

The core differentiator is controlled tool orchestration through graph execution with strong reliability, explicit state, and auditable runs.

## Scope (Milestone 1)
- Local-first runtime with cloud-ready interfaces
- Internal graph JSON schema and runtime validation
- Graph state memory for prior-run and key-value state access
- Guarantee modes per run: `strict`, `bounded`, `flex`
- Explicit state nodes only: `read_state`, `write_state`, `read_prior_runs`
- Condition nodes supporting:
  - typed operators
  - schema-constrained enum LLM branching
- Hybrid scheduling design (in-app scheduler + external trigger contracts)
- CLI/TUI-first control surface

## Memory Architecture
### Hot State
- System prompt
- Persona
- Rolling short-term conversation memory
- Active execution context

### Cold State
- Daily chat archives as `.md` files per day
- Chunking/indexing target:
  - 400-token chunks
  - 80-token overlap
- Long-term fact memory with retrieval by keyword and semantic lookup

## Graph State Memory
- Scope: same-graph prior state only
- Default: prior-state access disabled unless explicit node enables it
- State reads and writes must happen through dedicated state nodes
- Every node keeps typed input/output contracts

## Execution Guarantee Modes
### strict
- Highest predictability
- Blocks unsafe constructs
- Requires strong schema and policy compliance

### bounded
- Controlled flexibility
- Allows constrained LLM branching and bounded remediations

### flex
- Maximum autonomy
- Broader behavior allowed with weaker reproducibility guarantees

## Data Contracts (Initial)
- `graph.schema.json` for graph definition validation
- Run/audit persistence:
  - `graph_runs`
  - `graph_checkpoints`
  - `graph_state_kv`
  - `graph_events`

## Initial Delivery Steps
1. Add graph package scaffold (`schema`, `state_store`, `executor`)
2. Add CLI `/graph` commands for validate/run/state inspection
3. Add guarantee mode enforcement and state-node execution
4. Add tests for schema validation, state persistence, and mode policy checks
5. Add follow-up milestones for graph builder agent, scheduler, and cloud adapters

## Constraints
- No direct graph-state mutation from tool or AI nodes
- Validation and policy gates must run before execution
- Every run must be auditable and replay-debuggable via checkpoints

## Current Status (Implemented)
- Graph runtime scaffold exists (`schema`, `state_store`, `executor`).
- `/graph` CLI supports `list`, `save`, `validate`, `show`, `run`, `runs`, and `state` inspection.
- Execution guarantee modes are wired: `strict`, `bounded`, `flex`.
- State-aware nodes are wired: `read_state`, `write_state`, `read_prior_runs`.
- Daily memory baseline is implemented:
  - append-only `.md` files per day
  - token chunking with overlap
  - CLI retrieval (`/memory daily days`, `/memory daily search`)

## Next Steps (Immediate)
1. Replace `mock_output` execution for `tool` and `ai_template` nodes with real runtime adapters.
2. Integrate Graph Builder Anton to generate schema-valid graph JSON from intent.
3. Add run replay and checkpoint resume controls in CLI (`/graph replay`, `/graph resume`).
4. Add stronger policy enforcement for strict mode (idempotency, timeouts, retry caps per node).
5. Expand graph tests with failure injection and replay determinism checks.

## Deferred v0.2 Backlog (Implement Later)
### Agent Layer
- Graph Builder Anton:
  - constrained output to internal graph schema
  - template registry for AI nodes
  - iterative edit flow for user-directed node changes
- Error Handler Anton:
  - policy/runbook-driven remediation only
  - bounded retry/backoff and escalation decisions
  - no unrestricted autonomous actions

### Graph UX
- Graph renderer in UI for node/edge visualization.
- Node-level editing from UI and prompt-based graph patching.
- Graph diff/history view before execution.

### Scheduler and Automation
- Full in-app scheduler (cron-like) persisted in SQLite.
- External trigger endpoint/command contracts for hybrid orchestration.
- Scheduled run audit logs and failure notifications.

### Memory v0.2 Expansion
- Long-term fact memory lifecycle:
  - extraction rules
  - update/merge
  - expiry/pruning of stale facts
- Semantic retrieval upgrade path:
  - keep local baseline now
  - add pluggable embedding index backends later
- Retrieval planner across hot state, daily archive, and long-term facts.

### Hooks
- Lifecycle hooks implementation:
  - before_run
  - after_run
  - before_node
  - after_node
  - on_error
- Hook safety model and side-effect boundaries.

### Cloud Version (Cloud-Ready -> Full Cloud)
- Hosted MCP execution path with user JWT identity propagation.
- Encrypted OAuth token storage/retrieval on server side.
- Cloud run workers so scheduled graphs execute while user machine is offline.
- Local/cloud execution backend abstraction and compatibility tests.
