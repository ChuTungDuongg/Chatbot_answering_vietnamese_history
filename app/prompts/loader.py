"""Deterministic, cached prompt loading with a closed set of package resources."""

from functools import lru_cache
from pathlib import Path


PROMPT_ROOT = Path(__file__).resolve().parent
PROMPT_NAMES = (
    "shared/grounded_system", "hybrid/live_style",
    "central/planner", "central/completion_parameter", "central/external_domain", "central/grounded_final",
    "central/agentic_loop", "central/independent_parameter", "central/execution_limit",
    "offline/no_rag", "offline/sft_grounding",
    "offline/modes/concise", "offline/modes/standard", "offline/modes/detailed",
    "offline/semantic/generate", "offline/semantic/relevance", "offline/semantic/verify",
    "offline/semantic/contradiction", "offline/semantic/deep_answer", "offline/semantic/deep_verify",
    "offline/semantic/json_suffix",
)


class PromptResourceError(RuntimeError):
    """A checked-in prompt is missing, unknown, or malformed."""


@lru_cache(maxsize=len(PROMPT_NAMES))
def load_prompt(name: str) -> str:
    if name not in PROMPT_NAMES:
        raise PromptResourceError(f"Unknown prompt resource: {name!r}")
    try:
        # Never strip whitespace: spaces at composition boundaries are contractual.
        value = (PROMPT_ROOT / f"{name}.md").read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PromptResourceError(f"Cannot load prompt resource: {name}") from exc
    if not value.strip() or any(char in value for char in ("\x00", "\ufeff", "\r")):
        raise PromptResourceError(f"Malformed prompt resource (expected nonempty UTF-8/LF): {name}")
    return value


def validate_prompt_resources() -> None:
    """Warm all prompts at startup, including offline resources, before loading models."""
    for name in PROMPT_NAMES:
        load_prompt(name)
