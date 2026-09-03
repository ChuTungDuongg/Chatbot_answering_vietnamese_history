from __future__ import annotations

import json
from typing import Any, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


RESEARCH_AGENT_SYSTEM = (
    "You are a Vietnamese-history research tool policy. Do not answer the history question. "
    "Use the provided tool definitions and only information already present in observations. "
    "Return JSON only: either action=tool with tool_name and arguments, or action=finish "
    "with sufficient and missing_information. Never reveal hidden reasoning."
)

GENERIC_TOOL_USE_SYSTEM = (
    "You are a function-calling policy. Use only the provided tool definitions. Return JSON only. "
    "A parallel request may use action=tool_batch with every required call; do not answer the task."
)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PolicyLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=6, ge=1)
    web_searches_left: int = Field(default=0, ge=0)
    page_fetches_left: int = Field(default=0, ge=0)


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    result_count: int | None = Field(default=None, ge=0)
    evidence_ids: list[str] = Field(default_factory=list)
    result: Any | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_result_or_error(self):
        if self.result_count is None and self.result is None and not self.error:
            raise ValueError("observation requires an actual result, result_count, or error")
        if self.error and self.evidence_ids:
            raise ValueError("failed observation cannot contain evidence IDs")
        if any(not str(item).strip() for item in self.evidence_ids):
            raise ValueError("evidence IDs must be non-empty")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence IDs must be unique within an observation")
        return self


class ResearchPolicyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    retrieval_question: str | None = None
    step: int = Field(default=1, ge=1)
    limits: PolicyLimits = Field(default_factory=PolicyLimits)
    tools: list[ToolDefinition]
    observations: list[ToolObservation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    trajectory_class: str | None = None

    @model_validator(mode="after")
    def evidence_must_have_been_observed(self):
        observed = {item for obs in self.observations for item in obs.evidence_ids}
        if not set(self.evidence_ids).issubset(observed):
            raise ValueError("state evidence_ids must come from prior observations")
        return self


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolDecision(ToolCall):
    action: Literal["tool"]


class ToolBatchDecision(BaseModel):
    """Generic xLAM-only target for parallel calls; runtime ResearchAgent rejects it."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool_batch"]
    tool_calls: list[ToolCall] = Field(min_length=1)


class FinishDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["finish"]
    sufficient: bool
    missing_information: list[str]


RuntimeDecision = Annotated[Union[ToolDecision, FinishDecision], Field(discriminator="action")]
TrainingDecision = Annotated[
    Union[ToolDecision, ToolBatchDecision, FinishDecision], Field(discriminator="action")
]
_RUNTIME_DECISION_ADAPTER = TypeAdapter(RuntimeDecision)
_TRAINING_DECISION_ADAPTER = TypeAdapter(TrainingDecision)


def validate_runtime_decision(value: Any, *, tool_names: set[str] | None = None) -> ToolDecision | FinishDecision:
    decision = _RUNTIME_DECISION_ADAPTER.validate_python(value)
    if isinstance(decision, ToolDecision) and tool_names is not None and decision.tool_name not in tool_names:
        raise ValueError(f"unknown tool: {decision.tool_name}")
    return decision


def validate_training_decision(value: Any, *, tool_names: set[str] | None = None):
    decision = _TRAINING_DECISION_ADAPTER.validate_python(value)
    calls = decision.tool_calls if isinstance(decision, ToolBatchDecision) else [decision]
    if tool_names is not None:
        unknown = [call.tool_name for call in calls if isinstance(call, (ToolCall, ToolDecision)) and call.tool_name not in tool_names]
        if unknown:
            raise ValueError(f"unknown tool(s): {', '.join(unknown)}")
    return decision


def serialize_policy_state(state: ResearchPolicyState | dict[str, Any]) -> str:
    parsed = state if isinstance(state, ResearchPolicyState) else ResearchPolicyState.model_validate(state)
    return json.dumps(parsed.model_dump(exclude_none=True), ensure_ascii=False, sort_keys=True)


def policy_messages(
    state: ResearchPolicyState | dict[str, Any],
    decision: dict[str, Any],
    *,
    generic: bool = False,
) -> list[dict[str, str]]:
    parsed_state = state if isinstance(state, ResearchPolicyState) else ResearchPolicyState.model_validate(state)
    tool_names = {tool.name for tool in parsed_state.tools}
    parsed_decision = validate_training_decision(decision, tool_names=tool_names)
    system = GENERIC_TOOL_USE_SYSTEM if generic else RESEARCH_AGENT_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": serialize_policy_state(parsed_state)},
        {
            "role": "assistant",
            "content": json.dumps(parsed_decision.model_dump(), ensure_ascii=False, sort_keys=True),
        },
    ]


def default_research_tool_definitions(*, include_attachment: bool = False) -> list[dict[str, Any]]:
    """Describe training tools from the same classes registered at runtime."""
    from app.tools.evidence_tools import InspectEvidenceTool, RetrieveEvidenceTool
    from app.tools.local_search import SearchHistoryTool
    from app.tools.page_fetcher import FetchPageTool
    from app.tools.web_search import SearchWebTool
    from app.tools.wikipedia import FetchWikipediaPageTool, SearchWikipediaTool

    classes = [
        SearchHistoryTool,
        SearchWikipediaTool,
        FetchWikipediaPageTool,
        SearchWebTool,
        FetchPageTool,
        RetrieveEvidenceTool,
        InspectEvidenceTool,
    ]
    if include_attachment:
        from app.tools.attachment_search import SearchUploadedDocumentsTool

        classes.append(SearchUploadedDocumentsTool)
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema.model_json_schema(),
        }
        for tool in classes
    ]
