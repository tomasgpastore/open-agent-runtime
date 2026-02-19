from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ALLOWED_CONTRACT_TYPES = {
    "any",
    "object",
    "array",
    "string",
    "number",
    "integer",
    "boolean",
    "null",
}


def validate_contract_schema(schema: object, *, path: str) -> list[str]:
    if not isinstance(schema, Mapping):
        return [f"{path} must be an object schema."]

    errors: list[str] = []
    _validate_contract_schema_recursive(schema=dict(schema), path=path, errors=errors)
    return errors


def validate_payload_against_schema(
    payload: object,
    schema: object,
    *,
    path: str = "$",
) -> list[str]:
    if not isinstance(schema, Mapping):
        return [f"{path}: schema is not an object."]
    errors: list[str] = []
    _validate_payload_recursive(payload=payload, schema=dict(schema), path=path, errors=errors)
    return errors


def schemas_are_compatible(source: object, target: object) -> bool:
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return False

    source_schema = dict(source)
    target_schema = dict(target)

    # Target accepts anything.
    target_type = _schema_type(target_schema)
    if target_type == "any":
        return True

    source_any_of = source_schema.get("anyOf")
    if isinstance(source_any_of, list) and source_any_of:
        return all(schemas_are_compatible(variant, target_schema) for variant in source_any_of)

    source_one_of = source_schema.get("oneOf")
    if isinstance(source_one_of, list) and source_one_of:
        return all(schemas_are_compatible(variant, target_schema) for variant in source_one_of)

    target_any_of = target_schema.get("anyOf")
    if isinstance(target_any_of, list) and target_any_of:
        return any(schemas_are_compatible(source_schema, variant) for variant in target_any_of)

    target_one_of = target_schema.get("oneOf")
    if isinstance(target_one_of, list) and target_one_of:
        return any(schemas_are_compatible(source_schema, variant) for variant in target_one_of)

    source_type = _schema_type(source_schema)
    if source_type == "any":
        return False

    if source_type == "integer" and target_type == "number":
        return True
    if source_type != target_type:
        return False

    if target_type == "object":
        source_props = source_schema.get("properties") if isinstance(source_schema.get("properties"), Mapping) else {}
        target_props = target_schema.get("properties") if isinstance(target_schema.get("properties"), Mapping) else {}
        source_required = _required_fields(source_schema)
        target_required = _required_fields(target_schema)

        if not target_required.issubset(set(target_props.keys()) | set(source_props.keys())):
            return False
        if not target_required.issubset(set(source_props.keys()) | source_required):
            return False

        for key in target_required:
            source_property_schema = source_props.get(key)
            target_property_schema = target_props.get(key)
            if not isinstance(source_property_schema, Mapping) or not isinstance(target_property_schema, Mapping):
                return False
            if not schemas_are_compatible(source_property_schema, target_property_schema):
                return False
        return True

    if target_type == "array":
        source_items = source_schema.get("items")
        target_items = target_schema.get("items")
        if isinstance(target_items, Mapping):
            if not isinstance(source_items, Mapping):
                return False
            return schemas_are_compatible(source_items, target_items)
        return True

    target_enum = target_schema.get("enum")
    source_enum = source_schema.get("enum")
    if isinstance(target_enum, list):
        if not isinstance(source_enum, list):
            return False
        return set(source_enum).issubset(set(target_enum))

    return True


def _validate_contract_schema_recursive(
    schema: dict[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    schema_type = schema.get("type")
    any_of = schema.get("anyOf")
    one_of = schema.get("oneOf")

    if schema_type is None and not isinstance(any_of, list) and not isinstance(one_of, list):
        errors.append(f"{path}.type is required.")
        return

    if isinstance(schema_type, str) and schema_type not in ALLOWED_CONTRACT_TYPES:
        errors.append(
            f"{path}.type must be one of: {', '.join(sorted(ALLOWED_CONTRACT_TYPES))}."
        )
        return

    if isinstance(any_of, list):
        if not any_of:
            errors.append(f"{path}.anyOf must be a non-empty array when provided.")
        for index, variant in enumerate(any_of):
            if not isinstance(variant, Mapping):
                errors.append(f"{path}.anyOf[{index}] must be an object schema.")
                continue
            _validate_contract_schema_recursive(
                schema=dict(variant),
                path=f"{path}.anyOf[{index}]",
                errors=errors,
            )

    if isinstance(one_of, list):
        if not one_of:
            errors.append(f"{path}.oneOf must be a non-empty array when provided.")
        for index, variant in enumerate(one_of):
            if not isinstance(variant, Mapping):
                errors.append(f"{path}.oneOf[{index}] must be an object schema.")
                continue
            _validate_contract_schema_recursive(
                schema=dict(variant),
                path=f"{path}.oneOf[{index}]",
                errors=errors,
            )

    if schema_type == "object":
        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, Mapping):
            errors.append(f"{path}.properties must be an object when provided.")
            return

        required = schema.get("required")
        if required is not None:
            if not isinstance(required, list) or not all(isinstance(item, str) and item for item in required):
                errors.append(f"{path}.required must be an array of non-empty strings.")
            elif isinstance(properties, Mapping):
                missing = [item for item in required if item not in properties]
                if missing:
                    errors.append(
                        f"{path}.required contains fields not declared in properties: {missing}."
                    )

        if isinstance(properties, Mapping):
            for key, property_schema in properties.items():
                if not isinstance(property_schema, Mapping):
                    errors.append(f"{path}.properties.{key} must be an object schema.")
                    continue
                _validate_contract_schema_recursive(
                    schema=dict(property_schema),
                    path=f"{path}.properties.{key}",
                    errors=errors,
                )
        return

    if schema_type == "array":
        items = schema.get("items")
        if items is None:
            return
        if not isinstance(items, Mapping):
            errors.append(f"{path}.items must be an object schema.")
            return
        _validate_contract_schema_recursive(schema=dict(items), path=f"{path}.items", errors=errors)
        return

    enum_values = schema.get("enum")
    if enum_values is not None and (
        not isinstance(enum_values, list) or len(enum_values) == 0
    ):
        errors.append(f"{path}.enum must be a non-empty array when provided.")


def _validate_payload_recursive(
    *,
    payload: object,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        for variant in any_of:
            if not isinstance(variant, Mapping):
                continue
            branch_errors: list[str] = []
            _validate_payload_recursive(
                payload=payload,
                schema=dict(variant),
                path=path,
                errors=branch_errors,
            )
            if not branch_errors:
                return
        errors.append(f"{path}: payload does not satisfy anyOf schema variants.")
        return

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        passed = 0
        for variant in one_of:
            if not isinstance(variant, Mapping):
                continue
            branch_errors: list[str] = []
            _validate_payload_recursive(
                payload=payload,
                schema=dict(variant),
                path=path,
                errors=branch_errors,
            )
            if not branch_errors:
                passed += 1
        if passed != 1:
            errors.append(f"{path}: payload must satisfy exactly one oneOf schema variant.")
        return

    schema_type = _schema_type(schema)
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and payload not in enum_values:
        errors.append(f"{path}: value {payload!r} is not in enum {enum_values!r}.")
        return

    if schema_type == "any":
        return

    if schema_type == "object":
        if not isinstance(payload, Mapping):
            errors.append(f"{path}: expected object, received {type(payload).__name__}.")
            return

        properties = schema.get("properties")
        properties_map = dict(properties) if isinstance(properties, Mapping) else {}
        required = _required_fields(schema)

        for key in sorted(required):
            if key not in payload:
                errors.append(f"{path}: missing required field '{key}'.")

        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            unknown_keys = [key for key in payload.keys() if key not in properties_map]
            for key in unknown_keys:
                errors.append(f"{path}: field '{key}' is not allowed by schema.")

        for key, value in payload.items():
            property_schema = properties_map.get(key)
            if isinstance(property_schema, Mapping):
                _validate_payload_recursive(
                    payload=value,
                    schema=dict(property_schema),
                    path=f"{path}.{key}",
                    errors=errors,
                )
        return

    if schema_type == "array":
        if not isinstance(payload, list):
            errors.append(f"{path}: expected array, received {type(payload).__name__}.")
            return
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(payload):
                _validate_payload_recursive(
                    payload=item,
                    schema=dict(item_schema),
                    path=f"{path}[{index}]",
                    errors=errors,
                )
        return

    if schema_type == "string":
        if not isinstance(payload, str):
            errors.append(f"{path}: expected string, received {type(payload).__name__}.")
        return

    if schema_type == "boolean":
        if not isinstance(payload, bool):
            errors.append(f"{path}: expected boolean, received {type(payload).__name__}.")
        return

    if schema_type == "integer":
        if not isinstance(payload, int) or isinstance(payload, bool):
            errors.append(f"{path}: expected integer, received {type(payload).__name__}.")
        return

    if schema_type == "number":
        if not isinstance(payload, (int, float)) or isinstance(payload, bool):
            errors.append(f"{path}: expected number, received {type(payload).__name__}.")
        return

    if schema_type == "null":
        if payload is not None:
            errors.append(f"{path}: expected null, received {type(payload).__name__}.")
        return


def _schema_type(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and schema_type in ALLOWED_CONTRACT_TYPES:
        return schema_type
    return "any"


def _required_fields(schema: dict[str, Any]) -> set[str]:
    required = schema.get("required")
    if not isinstance(required, list):
        return set()
    return {item for item in required if isinstance(item, str) and item}
