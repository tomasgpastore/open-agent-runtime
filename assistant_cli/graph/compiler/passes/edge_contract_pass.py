from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from assistant_cli.graph.compiler.models import CompileDiagnostic
from assistant_cli.graph.contracts import schemas_are_compatible


ALLOWED_AI_EDGE_POLICIES = {"always", "auto", "never"}


@dataclass(slots=True)
class _EdgeRef:
    source_id: str
    target_id: str
    kind: str
    index: int | None = None
    branch_key: str | None = None


def run_edge_contract_pass(graph: dict[str, Any]) -> tuple[dict[str, Any], list[CompileDiagnostic]]:
    diagnostics: list[CompileDiagnostic] = []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return graph, diagnostics

    node_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if isinstance(node_id, str) and node_id:
            node_map[node_id] = node

    execution_defaults = graph.get("execution_defaults")
    policy = "auto"
    if isinstance(execution_defaults, dict):
        raw_policy = execution_defaults.get("ai_edge_policy")
        if isinstance(raw_policy, str) and raw_policy in ALLOWED_AI_EDGE_POLICIES:
            policy = raw_policy

    edges = _collect_edges(nodes)
    graph_id = str(graph.get("id") or "graph")

    for edge in edges:
        source_node = node_map.get(edge.source_id)
        target_node = node_map.get(edge.target_id)
        if source_node is None or target_node is None:
            continue

        if source_node.get("compiler_generated") == "ai_normalizer":
            continue
        if target_node.get("compiler_generated") == "ai_normalizer":
            continue
        if str(target_node.get("type") or "") == "end":
            continue

        source_schema = source_node.get("output_schema")
        target_schema = target_node.get("input_schema")
        if not isinstance(source_schema, dict):
            source_schema = {"type": "any"}
        if not isinstance(target_schema, dict):
            target_schema = {"type": "any"}

        compatible = schemas_are_compatible(source_schema, target_schema)
        if policy == "never":
            if not compatible:
                diagnostics.append(
                    CompileDiagnostic(
                        code="EDGE_SCHEMA_MISMATCH",
                        severity="error",
                        message=(
                            f"Edge '{edge.source_id}' -> '{edge.target_id}' has incompatible contracts "
                            "and ai_edge_policy='never'."
                        ),
                        node_id=edge.source_id,
                        hint="Align output_schema/input_schema or use ai_edge_policy=auto|always.",
                    )
                )
            continue

        should_insert = policy == "always" or (policy == "auto" and not compatible)
        if not should_insert:
            continue

        normalizer_id = _unique_node_id(
            base=f"normalize__{edge.source_id}__to__{edge.target_id}",
            node_map=node_map,
        )
        normalizer_node = _build_normalizer_node(
            graph_id=graph_id,
            normalizer_id=normalizer_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            source_schema=source_schema,
            target_schema=target_schema,
            policy=policy,
        )

        nodes.append(normalizer_node)
        node_map[normalizer_id] = normalizer_node
        _rewrite_edge(edge=edge, node_map=node_map, new_target=normalizer_id)

        severity = "warning" if policy == "auto" and not compatible else "info"
        diagnostics.append(
            CompileDiagnostic(
                code="EDGE_NORMALIZER_INSERTED",
                severity=severity,
                message=(
                    f"Inserted AI normalizer '{normalizer_id}' for edge "
                    f"'{edge.source_id}' -> '{edge.target_id}' (policy={policy})."
                ),
                node_id=edge.source_id,
            )
        )

    return graph, diagnostics


def _build_normalizer_node(
    *,
    graph_id: str,
    normalizer_id: str,
    source_id: str,
    target_id: str,
    source_schema: dict[str, Any],
    target_schema: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    source_schema_json = json.dumps(source_schema, sort_keys=True, ensure_ascii=False)
    target_schema_json = json.dumps(target_schema, sort_keys=True, ensure_ascii=False)

    return {
        "node_id": normalizer_id,
        "type": "ai_template",
        "compiler_generated": "ai_normalizer",
        "normalizes_edge": {
            "from": source_id,
            "to": target_id,
            "policy": policy,
        },
        "input": f"{{{{{source_id}}}}}",
        "input_schema": source_schema,
        "output_schema": target_schema,
        "system_prompt": (
            "You are an adapter that returns only valid JSON matching the target schema exactly. "
            "Do not include markdown or commentary."
        ),
        "prompt_template": (
            f"Source node: {source_id}\n"
            f"Target node: {target_id}\n"
            f"Source schema: {source_schema_json}\n"
            f"Target schema: {target_schema_json}\n"
            f"Payload: {{{{{source_id}}}}}\n"
            "Return only JSON that conforms to target schema."
        ),
        "output_format": "json",
        "timeout_seconds": 45,
        "max_retries": 1,
        "idempotency_key": f"{graph_id}:{normalizer_id}",
        "next": target_id,
    }


def _collect_edges(nodes: list[dict[str, Any] | Any]) -> list[_EdgeRef]:
    refs: list[_EdgeRef] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        source_id = node.get("node_id")
        if not isinstance(source_id, str) or not source_id:
            continue

        node_type = node.get("type")
        if node_type == "condition":
            if isinstance(node.get("if_true"), str):
                refs.append(_EdgeRef(source_id=source_id, target_id=str(node["if_true"]), kind="if_true"))
            if isinstance(node.get("if_false"), str):
                refs.append(_EdgeRef(source_id=source_id, target_id=str(node["if_false"]), kind="if_false"))
            branch_targets = node.get("branch_targets")
            if isinstance(branch_targets, dict):
                for key, value in branch_targets.items():
                    if isinstance(value, str):
                        refs.append(
                            _EdgeRef(
                                source_id=source_id,
                                target_id=value,
                                kind="branch_targets",
                                branch_key=str(key),
                            )
                        )
            continue

        next_field = node.get("next")
        if isinstance(next_field, str):
            refs.append(_EdgeRef(source_id=source_id, target_id=next_field, kind="next"))
            continue
        if isinstance(next_field, list):
            for index, value in enumerate(next_field):
                if isinstance(value, str):
                    refs.append(
                        _EdgeRef(
                            source_id=source_id,
                            target_id=value,
                            kind="next_list",
                            index=index,
                        )
                    )
    return refs


def _rewrite_edge(*, edge: _EdgeRef, node_map: dict[str, dict[str, Any]], new_target: str) -> None:
    source_node = node_map.get(edge.source_id)
    if source_node is None:
        return

    if edge.kind == "next":
        source_node["next"] = new_target
        return
    if edge.kind == "next_list":
        next_field = source_node.get("next")
        if isinstance(next_field, list) and edge.index is not None and edge.index < len(next_field):
            next_field[edge.index] = new_target
        return
    if edge.kind == "if_true":
        source_node["if_true"] = new_target
        return
    if edge.kind == "if_false":
        source_node["if_false"] = new_target
        return
    if edge.kind == "branch_targets":
        branch_targets = source_node.get("branch_targets")
        if isinstance(branch_targets, dict) and edge.branch_key is not None:
            branch_targets[edge.branch_key] = new_target


def _unique_node_id(*, base: str, node_map: dict[str, dict[str, Any]]) -> str:
    if base not in node_map:
        return base
    counter = 1
    while True:
        candidate = f"{base}__{counter}"
        if candidate not in node_map:
            return candidate
        counter += 1
