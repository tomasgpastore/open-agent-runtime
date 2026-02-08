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
