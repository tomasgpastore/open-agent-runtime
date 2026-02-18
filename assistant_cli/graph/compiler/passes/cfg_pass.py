from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic


@dataclass(slots=True)
class CFGAnalysis:
    adjacency: dict[str, list[str]]
    reachable: set[str]
    has_cycle: bool
    has_reachable_end: bool


def run_cfg_pass(graph: dict[str, Any]) -> tuple[CFGAnalysis, list[CompileDiagnostic]]:
    diagnostics: list[CompileDiagnostic] = []

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return CFGAnalysis(adjacency={}, reachable=set(), has_cycle=False, has_reachable_end=False), diagnostics

    node_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if isinstance(node_id, str) and node_id:
            node_map[node_id] = node

    adjacency: dict[str, list[str]] = {}
    for node_id, node in node_map.items():
        adjacency[node_id] = _node_targets(node)

    start = graph.get("start")
    reachable: set[str] = set()
    if isinstance(start, str) and start in node_map:
        _dfs_reachable(start, adjacency, reachable)

    for node_id in node_map:
        if node_id not in reachable:
            diagnostics.append(
                CompileDiagnostic(
                    code="CFG_UNREACHABLE_NODE",
                    severity="warning",
                    message=f"Node '{node_id}' is not reachable from start node.",
                    node_id=node_id,
                    hint="Remove it or connect it with a valid edge.",
                )
            )

    end_nodes = {node_id for node_id, node in node_map.items() if node.get("type") == "end"}
    has_reachable_end = any(node_id in reachable for node_id in end_nodes)
    if not end_nodes:
        diagnostics.append(
            CompileDiagnostic(
                code="CFG_NO_END_NODE",
                severity="error",
                message="Graph does not define any end node.",
                hint="Add at least one node with type='end'.",
            )
        )
    elif not has_reachable_end:
        diagnostics.append(
            CompileDiagnostic(
                code="CFG_END_UNREACHABLE",
                severity="error",
                message="No end node is reachable from the start node.",
                hint="Ensure at least one path from start reaches an end node.",
            )
        )

    has_cycle = _detect_cycle(start, adjacency) if isinstance(start, str) else False
    if has_cycle:
        diagnostics.append(
            CompileDiagnostic(
                code="CFG_LOOP_DETECTED",
                severity="warning",
                message="Graph contains at least one cycle.",
                hint="Verify max_steps and loop exit conditions are safe.",
            )
        )

    return (
        CFGAnalysis(
            adjacency=adjacency,
            reachable=reachable,
            has_cycle=has_cycle,
            has_reachable_end=has_reachable_end,
        ),
        diagnostics,
    )


def _node_targets(node: dict[str, Any]) -> list[str]:
    node_type = node.get("type")
    if node_type == "condition":
        targets: list[str] = []
        if isinstance(node.get("if_true"), str):
            targets.append(str(node["if_true"]))
        if isinstance(node.get("if_false"), str):
            targets.append(str(node["if_false"]))

        branch_targets = node.get("branch_targets")
        if isinstance(branch_targets, dict):
            for value in branch_targets.values():
                if isinstance(value, str):
                    targets.append(value)
        return _dedupe(targets)

    next_field = node.get("next")
    if isinstance(next_field, str):
        return [next_field]
    if isinstance(next_field, list):
        return [item for item in next_field if isinstance(item, str)]
    return []


def _dfs_reachable(start: str, adjacency: dict[str, list[str]], reachable: set[str]) -> None:
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        for target in adjacency.get(node_id, []):
            if target not in reachable:
                stack.append(target)


def _detect_cycle(start: str, adjacency: dict[str, list[str]]) -> bool:
    visited: set[str] = set()
    active: set[str] = set()

    def _visit(node_id: str) -> bool:
        if node_id in active:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        active.add(node_id)
        for target in adjacency.get(node_id, []):
            if _visit(target):
                return True
        active.remove(node_id)
        return False

    return _visit(start)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
