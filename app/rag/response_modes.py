"""Frozen SFT V1/Citation V2 and offline evaluation compatibility.

Keep instruction strings stable for research provenance. Live application
requests and Hybrid/Central runtime do not use these response controls.
"""

from __future__ import annotations

from typing import Literal

from app.prompts.loader import load_prompt

ResponseMode = Literal["concise", "standard", "detailed"]

MODE_INSTRUCTIONS: dict[str, str] = {
    mode: load_prompt(f"offline/modes/{mode}") for mode in ("concise", "standard", "detailed")
}

SFT_GROUNDING_INSTRUCTION = load_prompt("offline/sft_grounding")


def mode_instruction(mode: ResponseMode) -> str:
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"Unknown response mode: {mode}")
    return MODE_INSTRUCTIONS[mode]


def sft_system_instruction(mode: ResponseMode) -> str:
    return f"{SFT_GROUNDING_INSTRUCTION} {mode_instruction(mode)}"
