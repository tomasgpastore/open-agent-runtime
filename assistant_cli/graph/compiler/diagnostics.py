from __future__ import annotations

from assistant_cli.graph.compiler.models import CompileDiagnostic


def render_diagnostic(diagnostic: CompileDiagnostic) -> str:
    location_bits: list[str] = []
    if diagnostic.node_id:
        location_bits.append(f"node={diagnostic.node_id}")
    if diagnostic.path:
        location_bits.append(f"path={diagnostic.path}")

    location = f" ({', '.join(location_bits)})" if location_bits else ""
    hint = f" Hint: {diagnostic.hint}" if diagnostic.hint else ""
    return f"[{diagnostic.severity.upper()}] {diagnostic.code}: {diagnostic.message}{location}.{hint}".rstrip()


def render_diagnostics(diagnostics: list[CompileDiagnostic]) -> str:
    if not diagnostics:
        return ""

    order = {"error": 0, "warning": 1, "info": 2}
    sorted_items = sorted(
        diagnostics,
        key=lambda item: (order.get(item.severity, 9), item.code, item.node_id or ""),
    )
    return "\n".join(f"- {render_diagnostic(item)}" for item in sorted_items)
