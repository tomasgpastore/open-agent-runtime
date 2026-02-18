from __future__ import annotations

import hashlib
import json
from typing import Any

from assistant_cli.graph.compiler.models import CompiledGraph, CompiledNode, CompileTargetMode
from assistant_cli.graph.compiler.passes.cfg_pass import CFGAnalysis
from assistant_cli.graph.compiler.passes.template_pass import TemplateAnalysis


def run_finalize_pass(
    *,
    graph: dict[str, Any],
    mode: CompileTargetMode,
    cfg: CFGAnalysis,
    templates: TemplateAnalysis,
    compiler_version: str,
) -> CompiledGraph:
    nodes_raw = graph.get("nodes")
    if not isinstance(nodes_raw, list):
        nodes_raw = []

    node_map: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    for node in nodes_raw:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        node_map[node_id] = node
        node_order.append(node_id)

    topology = _topological_order(node_order=node_order, adjacency=cfg.adjacency)

    compiled_nodes: dict[str, CompiledNode] = {}
    for node_id in node_order:
        node = node_map[node_id]
        compiled_nodes[node_id] = CompiledNode(
            node_id=node_id,
            node_type=str(node.get("type") or ""),
            raw=node,
            next_nodes=_node_targets(node),
            input_refs=list(templates.node_input_refs.get(node_id, [])),
            execution_contract={
                "timeout_seconds": node.get("timeout_seconds"),
                "max_retries": node.get("max_retries"),
                "idempotency_key": node.get("idempotency_key"),
            },
        )

    payload = {
        "compiler_version": compiler_version,
        "mode": mode,
        "graph": graph,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    compile_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    warnings = []

    return CompiledGraph(
        graph_id=str(graph.get("id") or ""),
        name=str(graph.get("name") or ""),
        start=str(graph.get("start") or ""),
        mode=mode,
        graph=graph,
        nodes=compiled_nodes,
        topology=topology,
        compile_hash=compile_hash,
        warnings=warnings,
        compiler_version=compiler_version,
    )


def _node_targets(node: dict[str, Any]) -> list[str]:
    if node.get("type") == "condition":
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


def _topological_order(*, node_order: list[str], adjacency: dict[str, list[str]]) -> list[str]:
    indegree: dict[str, int] = {node_id: 0 for node_id in node_order}
    for source in node_order:
        for target in adjacency.get(source, []):
            if target in indegree:
                indegree[target] += 1

    queue = [node_id for node_id in node_order if indegree[node_id] == 0]
    order: list[str] = []

    while queue:
        current = queue.pop(0)
        order.append(current)
        for target in adjacency.get(current, []):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(order) == len(node_order):
        return order

    # Cycles or disconnected structures: keep deterministic fallback order.
    remaining = [node_id for node_id in node_order if node_id not in order]
    return order + remaining


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
