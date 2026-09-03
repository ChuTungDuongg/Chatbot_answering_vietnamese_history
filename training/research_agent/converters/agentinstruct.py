from __future__ import annotations

import re
from typing import Any

from app.agents.research.policy import PolicyLimits, ResearchPolicyState, ToolObservation, policy_messages


ACTION_RE = re.compile(r"(?im)^\s*Act\s*:\s*([a-zA-Z_][\w.-]*)")
CODE_RE = re.compile(r"```(?:bash|sh)?\s*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)


def _is_observation(text: str) -> bool:
    lowered = text.lstrip().lower()
    return lowered.startswith(("the output of the os:", "observation:", "the environment returns:"))


def convert_agentinstruct(row: dict[str, Any], *, source_split: str = "os") -> list[dict[str, Any]]:
    """Convert the reliably mapped AgentInstruct OS/bash subset.

    Other environments have environment-specific action languages without formal
    tool definitions in each row, so callers must skip/report them.
    """
    if source_split != "os":
        raise NotImplementedError(f"AgentInstruct split {source_split!r} has no safe policy-schema mapping")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("AgentInstruct row requires a non-empty conversations list")
    tool = {
        "name": "bash",
        "description": "Execute one non-interactive bash command in the task environment.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "minLength": 1}},
            "required": ["command"],
        },
    }
    row_id = str(row.get("id") or "agentinstruct-unknown")
    question = ""
    observations: list[ToolObservation] = []
    pending_tool = ""
    output: list[dict[str, Any]] = []
    has_loss_flags = any(turn.get("loss") is not None for turn in conversations if isinstance(turn, dict))

    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError("AgentInstruct conversation turn must be an object")
        role = str(turn.get("from") or "").lower()
        value = str(turn.get("value") or "").strip()
        if role in {"human", "user"}:
            if _is_observation(value):
                if not pending_tool:
                    raise ValueError("AgentInstruct observation has no preceding action")
                observations.append(ToolObservation(tool=pending_tool, result=value, result_count=1))
                pending_tool = ""
            else:
                question = value
                observations = []
                pending_tool = ""
            continue
        if role not in {"gpt", "assistant"}:
            raise ValueError(f"unsupported AgentInstruct role: {role}")
        action_match = ACTION_RE.search(value)
        if not action_match or not question:
            raise ValueError("AgentInstruct assistant turn is missing a parseable Act or active question")
        action = action_match.group(1).lower()
        if action == "bash":
            code_match = CODE_RE.search(value)
            if not code_match or not code_match.group("code").strip():
                raise ValueError("AgentInstruct bash action has no fenced command")
            target = {"action": "tool", "tool_name": "bash", "arguments": {"command": code_match.group("code").strip()}}
            pending_tool = "bash"
        elif action in {"answer", "finish"}:
            target = {"action": "finish", "sufficient": True, "missing_information": []}
            pending_tool = ""
        else:
            raise NotImplementedError(f"unsupported AgentInstruct OS action: {action}")

        should_train = bool(turn.get("loss")) if has_loss_flags else True
        if should_train:
            step = len(output) + 1
            state = ResearchPolicyState(
                question=question,
                step=step,
                limits=PolicyLimits(max_steps=32),
                tools=[tool],
                observations=list(observations),
                evidence_ids=[],
                trajectory_class="multi_step_agent",
            )
            output.append({
                "id": f"{row_id}-step-{step:03d}",
                "source_dataset": "agentinstruct",
                "source": "multi_step_agent",
                "stage": "multi_step_agent",
                "original_sample_id": row_id,
                "group_id": row_id,
                "trajectory_id": row_id,
                "trajectory_class": "multi_step_agent",
                "step": step,
                "grounded": True,
                "synthetic": False,
                "messages": policy_messages(state, target, generic=True),
                "training_prompt": state.model_dump(exclude_none=True),
                "training_target": target,
            })
    if not output:
        raise ValueError("AgentInstruct row produced no supervised turns")
    return output
