from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from assistant_cli.graph.compiler.models import (
    CompileDiagnostic,
    CompileOptions,
    CompileResult,
    CompiledGraph,
)
from assistant_cli.graph.compiler.passes import (
    run_canonicalize_pass,
    run_cfg_pass,
    run_defaults_pass,
    run_edge_contract_pass,
    run_finalize_pass,
    run_io_contract_pass,
    run_policy_pass,
    run_schema_pass,
    run_template_pass,
    run_tool_contract_pass,
)
from assistant_cli.graph.schema import ALLOWED_GUARANTEE_MODES


class GraphCompiler:
    """Deterministic graph compiler that normalizes and validates executable graphs."""

    VERSION = "0.2.0"

    def compile(
        self,
        *,
        graph: dict[str, Any],
        tool_map: dict[str, BaseTool] | None = None,
        options: CompileOptions | None = None,
    ) -> CompileResult:
        opts = options or CompileOptions()
        tool_map = tool_map or {}
        diagnostics: list[CompileDiagnostic] = []

        if opts.mode not in ALLOWED_GUARANTEE_MODES:
            diagnostics.append(
                CompileDiagnostic(
                    code="COMPILE_MODE_INVALID",
                    severity="error",
                    message=f"Unsupported compile mode '{opts.mode}'.",
                    path="options.mode",
                    hint="Use strict, bounded, or flex.",
                )
            )
            return CompileResult(ok=False, diagnostics=diagnostics, rewritten_graph=graph, compiled=None)

        rewritten, canonical_diags = run_canonicalize_pass(graph)
        diagnostics.extend(canonical_diags)

        if opts.inject_defaults:
            rewritten, default_diags = run_defaults_pass(
                rewritten,
                opts.mode,
                opts.ai_edge_policy,
            )
            diagnostics.extend(default_diags)

        diagnostics.extend(run_schema_pass(rewritten))
        diagnostics.extend(run_io_contract_pass(rewritten))

        rewritten, edge_diags = run_edge_contract_pass(rewritten)
        diagnostics.extend(edge_diags)

        cfg_analysis, cfg_diags = run_cfg_pass(rewritten)
        diagnostics.extend(cfg_diags)

        template_analysis, template_diags = run_template_pass(rewritten)
        diagnostics.extend(template_diags)

        diagnostics.extend(run_tool_contract_pass(rewritten, tool_map))
        diagnostics.extend(run_policy_pass(rewritten, opts.mode))

        has_errors = any(item.severity == "error" for item in diagnostics)
        if has_errors:
            return CompileResult(
                ok=False,
                diagnostics=diagnostics,
                rewritten_graph=rewritten,
                compiled=None,
            )

        compiled: CompiledGraph = run_finalize_pass(
            graph=rewritten,
            mode=opts.mode,
            cfg=cfg_analysis,
            templates=template_analysis,
            compiler_version=self.VERSION,
        )
        compiled.warnings = [item for item in diagnostics if item.severity != "error"]

        return CompileResult(
            ok=True,
            diagnostics=diagnostics,
            rewritten_graph=rewritten,
            compiled=compiled,
        )
