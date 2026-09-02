"""V2 source gates and tool-schema validation, independent of production."""
from __future__ import annotations

import json

from training.trajectory_dataset.adapters.common import AdapterError
from training.trajectory_dataset.validate import validate_trajectory

SOURCES = frozenset({"hermes_function_calling", "uit_viquad2_grounded"})


def validate_v2_trajectory(row):
    from jsonschema import Draft202012Validator, SchemaError
    from referencing.exceptions import Unresolvable
    errors = validate_trajectory(row)
    if errors:
        return errors
    if row.get("source_dataset") not in SOURCES:
        errors.append("Central V2 permits only Hermes function calling and UIT-ViQuAD2.0")
    definitions = {}
    for tool in row.get("tools", []):
        function = tool.get("function", {})
        name, schema = function.get("name"), function.get("parameters")
        if not name or name in definitions:
            errors.append("tool names must be present and unique")
            continue
        if not isinstance(schema, dict):
            errors.append(f"{name}: parameters must be a JSON schema object")
            continue
        # External schema resolution must never download data during validation.
        def remote_ref(value):
            if isinstance(value, dict):
                return any(k in {"$ref", "$dynamicRef"} and isinstance(v, str) and not v.startswith("#")
                           or remote_ref(v) for k, v in value.items())
            return isinstance(value, list) and any(remote_ref(v) for v in value)
        if remote_ref(schema):
            errors.append(f"{name}: external schema references are unsupported")
            continue
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            errors.append(f"{name}: invalid tool schema: {exc.message}")
            continue
        definitions[name] = Draft202012Validator(schema)
    for message in row.get("messages", []):
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            name, arguments = function.get("name"), function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    continue  # The base conversation validator reports this.
            if name in definitions:
                try:
                    errors.extend(f"{name}: {error.message}" for error in definitions[name].iter_errors(arguments))
                except Unresolvable as exc:
                    errors.append(f"{name}: unresolved local schema reference: {exc.ref}")
    return errors


def require_v2_trajectory(row):
    errors = validate_v2_trajectory(row)
    if errors:
        raise AdapterError("; ".join(errors))
    return row
