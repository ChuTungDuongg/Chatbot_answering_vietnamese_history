"""Bounded routing hints and explicit planner completion, without hidden reasoning."""

import json
from copy import deepcopy
import re
import unicodedata
from typing import Any

from app.prompts.loader import load_prompt


COMPLETION_DESCRIPTION = load_prompt("central/completion_parameter")

COMPLETION_FIELD = "__central_final_after_success"


PLANNER_INSTRUCTION = load_prompt("central/planner")


def question_domain(question: str) -> str:
    """Only explicit domain cues are constrained; ambiguous queries stay with the planner."""
    value = unicodedata.normalize("NFD", question.casefold())
    value = "".join(c for c in value if not unicodedata.combining(c)).replace("đ", "d")
    vietnam = re.search(r"\b(viet nam|vietnam|vietnamese|dai viet|su viet|van lang|au lac|"
                        r"bach dang|ngo quyen|tran hung dao|nha tran|nha ly|nha le|"
                        r"hai ba trung|quang trung|dien bien phu)\b", value)
    external = re.search(
        r"\b(the gioi|world history|global history|chien tranh lanh|cold war|"
        r"chau au|chau my|chau phi|europe|roman empire|la ma|hy lap|ai cap co dai|"
        r"co dai ai cap|napoleon|napoleon bonaparte|chien tranh the gioi|"
        r"world war|the chien|cach mang phap|french revolution|"
        r"lich su (?:trung quoc|nhat ban|han quoc|anh|phap|my|nga|an do))\b", value,
    )
    if vietnam:
        return "mixed" if external else "vietnamese"
    return "external" if external else "unspecified"


def final_after_tools(text: str) -> bool:
    matches = re.findall(r"<plan_status>(.{1,500}?)</plan_status>", text, flags=re.S)
    if len(matches) != 1:
        return False
    try:
        status = json.loads(matches[0])
    except (ValueError, TypeError):
        return False
    return isinstance(status, dict) and status.get("final_after_tools") is True and status.get("pending_tools") == []


def planner_tool_schemas(schemas, *, mcp_schema_budget: int = 16384) -> tuple[list[dict], frozenset[str]]:
    """Add an optional planner-only flag without mutating capabilities or remote schemas."""
    output = deepcopy(list(schemas))
    controlled = set()
    for schema in output:
        function = schema["function"]
        parameters = function["parameters"]
        # Leave complex/non-object and colliding remote schemas untouched.
        if parameters.get("type") != "object" or any(key in parameters for key in ("allOf", "anyOf", "oneOf", "$ref")):
            continue
        properties = parameters.setdefault("properties", {})
        if COMPLETION_FIELD in properties:
            continue
        properties[COMPLETION_FIELD] = {"type": "boolean", "description": COMPLETION_DESCRIPTION}
        controlled.add(function["name"])
    mcp_schemas = [item for item in output if item["function"]["name"].startswith("mcp__")]
    if sum(len(json.dumps(item, ensure_ascii=False).encode()) for item in mcp_schemas) > mcp_schema_budget:
        # Planner controls must not consume the existing MCP schema budget.
        originals = {item["function"]["name"]: item for item in schemas}
        for index, schema in enumerate(output):
            name = schema["function"]["name"]
            if name.startswith("mcp__") and name in controlled:
                output[index] = deepcopy(originals[name])
                controlled.remove(name)
    return output, frozenset(controlled)


def evidence_chunks(output: Any) -> list[dict[str, Any]]:
    rows = [output] if isinstance(output, dict) else output if isinstance(output, list) else []
    return [row for row in rows if isinstance(row, dict) and row.get("chunk_id") and str(row.get("text") or "").strip()]


def terminal_evidence(name: str, output: Any, allowed: frozenset[str]) -> bool:
    # Search snippets retain their follow-up planner when a reader is available.
    if name == "search_wikipedia" and "fetch_wikipedia_page" in allowed:
        return False
    if name == "search_web" and "fetch_page" in allowed:
        return False
    rows = evidence_chunks(output)
    return bool(rows) and not any((row.get("metadata") or {}).get("truncated") for row in rows)
