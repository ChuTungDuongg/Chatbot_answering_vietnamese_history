"""YAML declarations for explicit built-ins only; MCP retains discovered schemas."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel
import yaml


DEFINITIONS_PATH = Path(__file__).resolve().parent / "definitions" / "tools.yaml"


class ToolDeclarationError(ValueError):
    """Invalid declarations or a mismatch with an executable Pydantic contract."""


def _check_duplicate_keys(node) -> None:
    if isinstance(node, yaml.MappingNode):
        keys = set()
        for key, value in node.value:
            if not isinstance(key, yaml.ScalarNode) or key.value in keys:
                raise ToolDeclarationError("Duplicate or non-scalar YAML mapping key")
            keys.add(key.value)
            _check_duplicate_keys(value)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _check_duplicate_keys(child)


def parse_declarations(text: str) -> dict[str, dict[str, Any]]:
    """Parse trusted local YAML safely; never import executables from declarations."""
    try:
        if any(isinstance(token, yaml.AliasToken) for token in yaml.scan(text)):
            raise ToolDeclarationError("YAML aliases are not supported in built-in declarations")
        document = yaml.safe_load(text)
        _check_duplicate_keys(yaml.compose(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as exc:
        raise ToolDeclarationError("Malformed built-in tools YAML") from exc
    if (not isinstance(document, dict) or set(document) != {"version", "tools"}
            or type(document["version"]) is not int or document["version"] != 1
            or not isinstance(document["tools"], list) or not document["tools"]):
        raise ToolDeclarationError("Expected version: 1 and a nonempty tools list")
    declarations = {}
    for item in document["tools"]:
        if not isinstance(item, dict) or set(item) != {"name", "description", "parameters"}:
            raise ToolDeclarationError("Each tool requires name, description, and parameters")
        name, description, parameters = item["name"], item["description"], item["parameters"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise ToolDeclarationError("Invalid built-in tool name")
        if name in declarations:
            raise ToolDeclarationError(f"Duplicate built-in tool: {name}")
        if not isinstance(description, str) or not description.strip():
            raise ToolDeclarationError(f"Missing description for built-in tool: {name}")
        if not isinstance(parameters, dict):
            raise ToolDeclarationError(f"Expected object parameters for built-in tool: {name}")
        try:
            Draft202012Validator.check_schema(parameters)
        except SchemaError as exc:
            raise ToolDeclarationError(f"Invalid JSON Schema for built-in tool: {name}") from exc
        properties = parameters.get("properties")
        if (parameters.get("type") != "object" or not isinstance(properties, dict)
                or not set(parameters.get("required", [])).issubset(properties)):
            raise ToolDeclarationError(f"Invalid object properties/required fields for built-in tool: {name}")
        declarations[name] = item
    return declarations


@lru_cache(maxsize=1)
def _load_declarations() -> dict[str, dict[str, Any]]:
    try:
        text = DEFINITIONS_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ToolDeclarationError("Cannot load built-in tool declarations: definitions/tools.yaml") from exc
    return parse_declarations(text)


def builtin_declarations() -> dict[str, dict[str, Any]]:
    # Callers may decorate schemas for planning without mutating cached declarations.
    return deepcopy(_load_declarations())


def _declaration(name: str) -> dict[str, Any]:
    try:
        return _load_declarations()[name]
    except KeyError as exc:
        raise ToolDeclarationError(f"Missing built-in tool declaration: {name}") from exc


def builtin_description(name: str) -> str:
    return _declaration(name)["description"]


def _argument_contract(schema: dict[str, Any] | bool) -> dict[str, Any] | bool:
    """Ignore documentation annotations, never types/defaults/validation keywords.

    Property names are data, including a property literally named 'title'.
    """
    if isinstance(schema, bool):
        return schema
    output = {}
    for key, value in schema.items():
        if key in {"title", "description", "$comment", "examples"}:
            continue
        if key in {"properties", "$defs", "patternProperties", "dependentSchemas"} and isinstance(value, dict):
            output[key] = {name: _argument_contract(child) for name, child in value.items()}
        elif key in {"items", "additionalProperties", "unevaluatedProperties", "contains", "not", "if", "then", "else", "propertyNames"} and isinstance(value, dict):
            output[key] = _argument_contract(value)
        elif key in {"allOf", "anyOf", "oneOf", "prefixItems"} and isinstance(value, list):
            output[key] = [_argument_contract(child) if isinstance(child, dict) else child for child in value]
        else:
            output[key] = deepcopy(value)
    return output


def validate_argument_contract(name: str, model: type[BaseModel]) -> None:
    if _argument_contract(_declaration(name)["parameters"]) != _argument_contract(model.model_json_schema()):
        raise ToolDeclarationError(f"YAML/Pydantic argument contract drift for built-in tool: {name}")


class BuiltinTool:
    """Declaration adapter only; the registry retains validation and execution."""

    @property
    def descriptor(self) -> dict[str, Any]:
        validate_argument_contract(self.name, self.input_schema)
        declaration = deepcopy(_declaration(self.name))
        return {"name": declaration["name"], "description": declaration["description"],
                "input_schema": declaration["parameters"], "provider": "builtin"}


def validate_builtin_contracts() -> None:
    from app.tools.builtin import BUILTIN_TOOL_TYPES

    names = [tool.name for tool in BUILTIN_TOOL_TYPES]
    if len(names) != len(set(names)) or set(names) != set(_load_declarations()):
        raise ToolDeclarationError("Built-in implementation inventory does not match YAML declarations")
    for tool in BUILTIN_TOOL_TYPES:
        validate_argument_contract(tool.name, tool.input_schema)
