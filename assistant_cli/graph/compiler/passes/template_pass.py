from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic


TEMPLATE_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
BUILTIN_ROOTS = {
    "input",
    "last",
    "date",
    "now",
    "datetime",
    "run_id",
    "graph_id",
}


@dataclass(slots=True)
class TemplateAnalysis:
    node_input_refs: dict[str, list[str]]


def run_template_pass(graph: dict[str, Any]) -> tuple[TemplateAnalysis, list[CompileDiagnostic]]:
    diagnostics: list[CompileDiagnostic] = []
    refs_map: dict[str, list[str]] = {}

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return TemplateAnalysis(node_input_refs={}), diagnostics

    node_order: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if isinstance(node, dict):
            node_id = node.get("node_id")
            if isinstance(node_id, str) and node_id:
                node_order[node_id] = index

    known_nodes = set(node_order.keys())

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue

        refs = sorted(_extract_template_refs(node))
        refs_map[node_id] = refs

        for ref in refs:
            root = _root_from_ref(ref)
            if root in BUILTIN_ROOTS:
                continue

            if root not in known_nodes:
                diagnostics.append(
                    CompileDiagnostic(
                        code="TEMPLATE_UNKNOWN_REFERENCE",
                        severity="error",
                        message=f"Template reference '{{{{{ref}}}}}' points to unknown source '{root}'.",
                        node_id=node_id,
                        hint="Use input.*, built-in context values, or an existing node id.",
                    )
                )
                continue

            current_index = node_order.get(node_id, -1)
            source_index = node_order.get(root, -1)
            if source_index >= current_index:
                diagnostics.append(
                    CompileDiagnostic(
                        code="TEMPLATE_FORWARD_REFERENCE",
                        severity="warning",
                        message=f"Template reference '{{{{{ref}}}}}' points to same/future node '{root}'.",
                        node_id=node_id,
                        hint="Verify this is intentional and safe across branches/loops.",
                    )
                )

    return TemplateAnalysis(node_input_refs=refs_map), diagnostics


def _extract_template_refs(value: object) -> set[str]:
    refs: set[str] = set()

    if isinstance(value, str):
        for match in TEMPLATE_RE.finditer(value):
            refs.add(match.group(1).strip())
        return refs

    if isinstance(value, dict):
        for item in value.values():
            refs.update(_extract_template_refs(item))
        return refs

    if isinstance(value, list):
        for item in value:
            refs.update(_extract_template_refs(item))
        return refs

    return refs


def _root_from_ref(ref: str) -> str:
    root = ref.split(".", 1)[0].strip()
    if root == "":
        return root
    if root == "output":
        return "last"
    return root
