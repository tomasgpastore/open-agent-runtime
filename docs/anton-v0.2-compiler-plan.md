# Anton v0.2 Compiler Plan (AI-Native Reliable Graph Orchestration)

_Date: 2026-02-19_

## Goals

1. Make graph execution reliable through explicit contracts, not implicit templating.
2. Keep Anton AI-native by supporting automatic LLM normalization between node boundaries.
3. Fail early at compile time when contracts or policy are invalid.
4. Enforce the same contracts at runtime so execution and compile assumptions stay aligned.

## Compiler Architecture

### 1) Node I/O contracts (mandatory)

- Every executable node must define:
  - `input_schema`
  - `output_schema`
- `start` can omit `input_schema`, `end` can omit `output_schema`.
- Contracts use a constrained JSON-schema-like structure (`type`, `properties`, `required`, `items`, `enum`, `anyOf`, `oneOf`).

### 2) Edge policy for AI normalization

- New graph execution default:
  - `execution_defaults.ai_edge_policy`
  - values: `always | auto | never`
- Behavior:
  - `always`: compiler inserts `ai_template` normalizer nodes on every non-terminal edge.
  - `auto`: compiler inserts normalizers only when source `output_schema` is incompatible with target `input_schema`.
  - `never`: incompatible edges are compile errors.

### 3) Pass pipeline

1. Canonicalize aliases
2. Inject defaults
3. Base graph schema validation
4. Node I/O contract validation
5. Edge contract analysis + optional AI normalizer insertion
6. CFG validation
7. Template reference validation
8. Tool argument contract checks
9. Guarantee-mode policy checks
10. Finalize compiled artifact (`compile_hash`)

### 4) Runtime contract enforcement

- Before each node runs: validate resolved node input against `input_schema`.
- After each node returns: validate output against `output_schema`.
- Contract violations fail the run with node-scoped error messages.

## Builder Anton Changes

- Builder prompt now explicitly requires node contracts.
- Builder fallback graphs now include contracts on all nodes.
- Builder fallback defaults now include:
  - `execution_defaults.guarantee_mode = bounded`
  - `execution_defaults.ai_edge_policy = always`

## Implemented in this change

- Added graph contract module:
  - `assistant_cli/graph/contracts.py`
- Added compiler passes:
  - `assistant_cli/graph/compiler/passes/io_contract_pass.py`
  - `assistant_cli/graph/compiler/passes/edge_contract_pass.py`
- Updated compile pipeline integration:
  - `assistant_cli/graph/compiler/compiler.py`
  - `assistant_cli/graph/compiler/passes/__init__.py`
  - `assistant_cli/graph/compiler/models.py`
  - `assistant_cli/graph/compiler/passes/defaults_pass.py`
- Updated graph validation:
  - `assistant_cli/graph/schema.py`
  - `assistant_cli/graph/schema/graph.schema.json`
- Updated runtime enforcement:
  - `assistant_cli/graph/executor.py`
- Updated builder outputs and prompt constraints:
  - `assistant_cli/graph/builder.py`
- Updated tests for mandatory contracts and compiler behavior:
  - `tests/graph_compiler_test.py`
  - `tests/graph_runtime_test.py`
  - `tests/graph_v2_test.py`

## Acceptance criteria

- Graphs without required node contracts fail compilation.
- Graphs with incompatible edges either:
  - auto-insert normalizers (`always/auto`) or
  - fail (`never`).
- Runtime rejects node payloads that violate declared contracts.
- Builder-generated fallback graphs pass validation under new contract rules.
