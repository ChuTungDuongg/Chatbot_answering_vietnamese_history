import asyncio
from copy import deepcopy
import importlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import BaseModel
import pytest
import yaml

from app.tools import declarations as declarations
from app.tools.builtin import BUILTIN_TOOL_TYPES
from app.tools.policy import tool_view
from app.tools.registry import ToolExecutionContext, ToolRegistry


BASELINE = json.loads((Path(__file__).parent / "fixtures/builtin_tool_contracts.json").read_text(encoding="utf-8"))


def test_inventory_pydantic_contracts_and_human_docs_are_complete():
    declarations.validate_builtin_contracts()
    manifest = declarations.builtin_declarations()
    assert set(manifest) == set(BASELINE) == {tool.name for tool in BUILTIN_TOOL_TYPES}
    for tool in BUILTIN_TOOL_TYPES:
        assert tool.input_schema.model_json_schema() == BASELINE[tool.name]
        item = manifest[tool.name]
        assert item["description"].strip() and "Do not" in item["description"]
        assert item["description"] == tool.description
        Draft202012Validator.check_schema(item["parameters"])
        doc = Path(importlib.import_module(tool.__module__).__file__).with_name("TOOL.md").read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(doc.split("---", 2)[1])
        assert frontmatter["name"] == tool.name
        assert {"kind", "provider", "requires_env", "inputs", "outputs", "side_effect", "requires_confirmation"} <= frontmatter.keys()
        assert set(frontmatter["inputs"]) == set(item["parameters"]["properties"])


def test_legacy_imports_are_the_same_implementations():
    from app.tools.local_search import SearchHistoryTool
    from app.tools.attachment_search import SearchUploadedDocumentsTool
    from app.tools.wikipedia import SearchWikipediaTool, FetchWikipediaPageTool
    from app.tools.web_search import SearchWebTool
    from app.tools.page_fetcher import FetchPageTool
    assert BUILTIN_TOOL_TYPES == (SearchHistoryTool, SearchUploadedDocumentsTool, SearchWikipediaTool,
                                  FetchWikipediaPageTool, SearchWebTool, FetchPageTool)


def test_model_visible_schemas_come_from_yaml_and_remain_provider_compatible():
    from app.central.planning import planner_tool_schemas
    from app.models.openai_responses import _tools
    registry = ToolRegistry()
    # Construction is deliberately bypassed: descriptors must not initialize dependencies.
    for tool in BUILTIN_TOOL_TYPES:
        registry.register(object.__new__(tool))
    view = tool_view(registry)
    manifest = declarations.builtin_declarations()
    for schema in view.schemas:
        function = schema["function"]
        assert function == manifest[function["name"]]
    planned, controlled = planner_tool_schemas(view.schemas)
    assert controlled == frozenset(manifest)
    converted = _tools(planned)
    assert len(converted) == len(manifest)
    assert all(item["type"] == "function" and item["strict"] is False for item in converted)
    assert declarations.builtin_declarations() == manifest


def test_declarations_cached_and_copy_isolated(monkeypatch):
    manifest = declarations.builtin_declarations()
    original = deepcopy(manifest)
    manifest["search_history"]["parameters"]["properties"]["query"]["type"] = "integer"
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("Unexpected per-request YAML read"))
    assert declarations.builtin_declarations() == original


@pytest.mark.parametrize("field,value", [("default", 9), ("maximum", 21), ("minimum", 0), ("type", "string")])
def test_argument_drift_detected(monkeypatch, field, value):
    manifest = declarations.builtin_declarations()
    manifest["search_history"]["parameters"]["properties"]["top_k"][field] = value
    monkeypatch.setattr(declarations, "_load_declarations", lambda: manifest)
    with pytest.raises(declarations.ToolDeclarationError, match="contract drift.*search_history"):
        declarations.validate_builtin_contracts()


def test_invalid_registration_is_atomic(monkeypatch):
    from app.tools.local_search import SearchHistoryTool
    class WrongInput(BaseModel):
        query: str
    tool = SearchHistoryTool(None)
    monkeypatch.setattr(tool, "input_schema", WrongInput)
    registry = ToolRegistry()
    with pytest.raises(declarations.ToolDeclarationError, match="contract drift"):
        registry.register(tool)
    assert registry.names() == registry.describe() == []


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_inventory_drift_fails_startup(monkeypatch, mutation):
    manifest = declarations.builtin_declarations()
    if mutation == "missing":
        del manifest["search_history"]
    elif mutation == "extra":
        manifest["unknown"] = deepcopy(manifest["search_history"])
    else:
        monkeypatch.setattr(importlib.import_module("app.tools.builtin"), "BUILTIN_TOOL_TYPES", BUILTIN_TOOL_TYPES * 2)
    monkeypatch.setattr(declarations, "_load_declarations", lambda: manifest)
    with pytest.raises(declarations.ToolDeclarationError, match="inventory"):
        declarations.validate_builtin_contracts()


@pytest.mark.parametrize("payload", ["", "[]", "tools: []", "!!python/object:os.system {}",
    "version: 1\nversion: 1\ntools: []", "version: 1\ntools: &tools [*tools]"])
def test_malformed_yaml_fails_clearly(payload):
    with pytest.raises(declarations.ToolDeclarationError):
        declarations.parse_declarations(payload)


@pytest.mark.parametrize("mutation", ["duplicate", "description", "required", "schema", "name"])
def test_bad_declarations_rejected(mutation):
    items = list(declarations.builtin_declarations().values())
    if mutation == "duplicate":
        items.append(deepcopy(items[0]))
    elif mutation == "description":
        items[0]["description"] = "  "
    elif mutation == "required":
        items[0]["parameters"]["required"].append("missing")
    elif mutation == "schema":
        items[0]["parameters"]["properties"]["query"]["type"] = "invalid"
    else:
        items[0]["name"] = "mcp__unauthorized!"
    with pytest.raises(declarations.ToolDeclarationError):
        declarations.parse_declarations(yaml.safe_dump({"version": 1, "tools": items}))


def test_missing_yaml_file_fails_clearly(tmp_path, monkeypatch):
    declarations._load_declarations.cache_clear()
    with monkeypatch.context() as context:
        context.setattr(declarations, "DEFINITIONS_PATH", tmp_path / "missing.yaml")
        with pytest.raises(declarations.ToolDeclarationError, match="Cannot load"):
            declarations.builtin_declarations()
    declarations._load_declarations.cache_clear()


def test_yaml_does_not_register_or_authorize_tools():
    from app.tools.local_search import SearchHistoryTool
    registry = ToolRegistry()
    assert registry.names() == []
    registry.register(SearchHistoryTool(None))
    assert tool_view(registry).allowed_names == frozenset({"search_history"})
    _, record = asyncio.run(registry.call("search_history", {"query": "Bạch Đằng"},
        context=ToolExecutionContext(allowed_tools=frozenset())))
    assert record.error == "tool_not_allowed"
    _, record = asyncio.run(registry.call("search_history", {"query": "", "top_k": 21}))
    assert record.error == "tool_failed"


def test_dynamic_mcp_descriptors_never_consult_builtin_declarations(monkeypatch):
    class Remote:
        name = "mcp__research__dynamic"
        provider = "mcp"
        server = "research"
        descriptor = {"name": name, "description": "Discovered remotely", "provider": provider,
                      "server": server, "input_schema": {"type": "object", "properties": {
                          "remote_argument": {"type": "integer", "minimum": 42}},
                          "required": ["remote_argument"]}}
    monkeypatch.setattr(declarations, "_load_declarations", lambda: pytest.fail("MCP consulted static YAML"))
    registry = ToolRegistry()
    registry.register(Remote())
    assert registry.describe() == [Remote.descriptor]
    assert tool_view(registry).allowed_names == frozenset()


def test_contract_normalization_preserves_property_names_and_default_data():
    schema = {"title": "Annotations", "type": "object", "properties": {
        "title": {"type": "string", "description": "Documentation", "default": "Kept"},
        "object": {"type": "object", "default": {"title": "Data", "description": "Also data"}}}}
    normalized = declarations._argument_contract(schema)
    assert normalized["properties"]["title"] == {"type": "string", "default": "Kept"}
    assert normalized["properties"]["object"]["default"] == {"title": "Data", "description": "Also data"}
    assert declarations._argument_contract({"properties": {"allowed": True, "forbidden": False}}) == {
        "properties": {"allowed": True, "forbidden": False}}
