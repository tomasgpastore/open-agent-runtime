from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


CompileTargetMode = Literal["strict", "bounded", "flex"]
AiEdgePolicy = Literal["always", "auto", "never"]
DiagnosticSeverity = Literal["error", "warning", "info"]


@dataclass(slots=True)
class CompileDiagnostic:
    code: str
    severity: DiagnosticSeverity
    message: str
    node_id: str | None = None
    path: str | None = None
    hint: str | None = None


@dataclass(slots=True)
class CompileOptions:
    mode: CompileTargetMode = "bounded"
    inject_defaults: bool = True
    allow_auto_rewrites: bool = True
    ai_edge_policy: AiEdgePolicy | None = None


@dataclass(slots=True)
class CompiledNode:
    node_id: str
    node_type: str
    raw: dict[str, Any]
    next_nodes: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    execution_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompiledGraph:
    graph_id: str
    name: str
    start: str
    mode: CompileTargetMode
    graph: dict[str, Any]
    nodes: dict[str, CompiledNode]
    topology: list[str]
    compile_hash: str
    warnings: list[CompileDiagnostic] = field(default_factory=list)
    compiler_version: str = "0.2.0"


@dataclass(slots=True)
class CompileResult:
    ok: bool
    diagnostics: list[CompileDiagnostic]
    rewritten_graph: dict[str, Any] | None = None
    compiled: CompiledGraph | None = None

    @property
    def errors(self) -> list[CompileDiagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]

    @property
    def warnings(self) -> list[CompileDiagnostic]:
        return [item for item in self.diagnostics if item.severity in {"warning", "info"}]
