import json

import pytest

from training.research_agent.converters.xlam import convert_xlam
from training.research_agent.prepare_dataset import convert_rows


def test_xlam_parses_official_string_schema_and_preserves_multiple_calls():
    row = {
        "query": json.dumps("Compute two things"),
        "tools": json.dumps([
            {"name": "sum", "description": "sum", "parameters": {"x": {"type": "int", "required": True}}},
            {"name": "product", "description": "multiply", "parameters": {"n": {"type": "int", "required": True}}},
        ]),
        "answers": json.dumps([
            {"name": "sum", "arguments": {"x": 2}},
            {"name": "product", "arguments": {"n": 3}},
        ]),
    }
    converted = convert_xlam(row)
    assert converted["training_target"]["action"] == "tool_batch"
    assert [call["tool_name"] for call in converted["training_target"]["tool_calls"]] == ["sum", "product"]
    assert len(converted["training_prompt"]["tools"]) == 2


def test_xlam_rejects_undefined_tool():
    with pytest.raises(LookupError):
        convert_xlam({"query": "q", "tools": '[{"name":"ok","parameters":{}}]', "answers": '[{"name":"bad","arguments":{}}]'})


def test_conversion_report_counts_parse_and_invalid_tool_rows():
    rows, stats = convert_rows("xlam", [
        {"query": "q", "tools": "not-json", "answers": "[]"},
        {"query": "q", "tools": '[{"name":"ok","parameters":{}}]', "answers": '[{"name":"bad","arguments":{}}]'},
    ])
    assert rows == []
    assert stats == {"converted_count": 0, "skipped_count": 2, "parse_error_count": 1, "invalid_tool_count": 1}
