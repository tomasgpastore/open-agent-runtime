from __future__ import annotations

import json
from typing import Iterable

from langchain_core.tools import BaseTool


def _find_tool_by_suffix(
    tools: dict[str, BaseTool], suffixes: Iterable[str]
) -> tuple[str, BaseTool] | None:
    for suffix in suffixes:
        for name, tool in tools.items():
            if name.endswith(suffix):
                return name, tool
    return None


async def wipe_memory_graph(tools: dict[str, BaseTool]) -> tuple[bool, str]:
    """Safely wipes MCP memory entities and relations when matching tools exist."""

    read_graph_pair = _find_tool_by_suffix(tools, ["read_graph", "get_graph"])
    delete_entities_pair = _find_tool_by_suffix(
        tools,
        ["delete_entities", "delete_nodes", "remove_entities"],
    )
    delete_relations_pair = _find_tool_by_suffix(
        tools,
        ["delete_relations", "delete_edges", "remove_relations"],
    )

    if not read_graph_pair:
        return (
            False,
            "No memory graph reader tool available. Could not safely inspect graph before wipe.",
        )
    if not delete_entities_pair:
        return (
            False,
            "No memory entity deletion tool available. Could not safely wipe long-term memory.",
        )

    _, read_graph_tool = read_graph_pair
    _, delete_entities_tool = delete_entities_pair

    graph_raw = await read_graph_tool.ainvoke({})
    graph = _coerce_to_dict(graph_raw)

    entities = _extract_entities(graph)
    relations = _extract_relations(graph)

    if delete_relations_pair and relations:
        _, delete_relations_tool = delete_relations_pair
        await _delete_relations(delete_relations_tool, relations)

    if entities:
        await _delete_entities(delete_entities_tool, entities)

    return (
        True,
        f"Deleted {len(entities)} entities and {len(relations)} relations from long-term memory.",
    )


async def _delete_entities(tool: BaseTool, entities: list[str]) -> None:
    payload_options = [
        {"entity_names": entities},
        {"entities": entities},
        {"names": entities},
    ]

    for payload in payload_options:
        try:
            await tool.ainvoke(payload)
            return
        except Exception:  # noqa: BLE001
            continue

    raise RuntimeError("Failed to delete memory entities with supported payload shapes.")


async def _delete_relations(tool: BaseTool, relations: list[dict]) -> None:
    payload_options = [
        {"relations": relations},
        {"edges": relations},
    ]

    for payload in payload_options:
        try:
            await tool.ainvoke(payload)
            return
        except Exception:  # noqa: BLE001
            continue

    raise RuntimeError("Failed to delete memory relations with supported payload shapes.")



def _coerce_to_dict(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}



def _extract_entities(graph: dict) -> list[str]:
    raw_entities = graph.get("entities") or graph.get("nodes") or []
    names: list[str] = []

    for entity in raw_entities:
        if isinstance(entity, str):
            names.append(entity)
            continue
        if isinstance(entity, dict):
            name = entity.get("name") or entity.get("id")
            if isinstance(name, str) and name:
                names.append(name)

    return sorted(set(names))



def _extract_relations(graph: dict) -> list[dict]:
    raw_relations = graph.get("relations") or graph.get("edges") or []
    normalized: list[dict] = []

    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        source = relation.get("source")
        relation_type = relation.get("relation") or relation.get("type")
        target = relation.get("target")
        if not source or not relation_type or not target:
            continue
        normalized.append(
            {
                "source": source,
                "relation": relation_type,
                "target": target,
            }
        )

    return normalized
