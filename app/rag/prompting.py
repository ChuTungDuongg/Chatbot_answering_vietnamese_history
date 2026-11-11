"""Fixed grounded prompt for live generators, with explicit offline compatibility.

Live Hybrid/Central callers omit the optional mode and use LIVE_SYSTEM_PROMPT.
Explicit mode arguments preserve frozen six-way evaluation prompt bytes.
"""

from typing import Any

from app.rag.response_modes import ResponseMode, mode_instruction
from app.prompts.loader import load_prompt


SYSTEM_PROMPT = load_prompt("shared/grounded_system")


# Live-only instructions: evaluation modes continue to use the frozen SYSTEM_PROMPT.
LIVE_STYLE_INSTRUCTION = load_prompt("hybrid/live_style")
NO_RAG_SYSTEM_PROMPT = load_prompt("offline/no_rag")
LIVE_SYSTEM_PROMPT = f"{SYSTEM_PROMPT} {LIVE_STYLE_INSTRUCTION}"


def build_messages(question: str, contexts: list[dict[str, Any]],
                   history: list[dict[str, str]] | None = None,
                   *, response_mode: ResponseMode | None = None) -> list[dict[str, str]]:
    source_sections = []
    for item in contexts:
        chunk_id = str(item.get("chunk_id") or "")
        title = str(item.get("title") or "")[:250]
        text = str(item.get("text") or "")[:2000]
        source_sections.append(f"[{chunk_id}] {title}\n{text}")
    context_text = "\n\n".join(source_sections) if source_sections else "Không có nguồn phù hợp."
    system = LIVE_SYSTEM_PROMPT if response_mode is None else f"{SYSTEM_PROMPT} {mode_instruction(response_mode)}"
    messages = [{"role": "system", "content": system}]
    for item in (history or [])[-6:]:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content") or "")[:1800]})
    messages.append({"role": "user", "content": f"Câu hỏi: {question}\n\nNguồn được truy xuất:\n{context_text}"})
    return messages


def build_no_rag_messages(question: str, *, response_mode: ResponseMode = "standard") -> list[dict[str, str]]:
    """Offline knowledge-only baseline; retain historical mode instructions."""
    return [
        {"role": "system", "content": (
            NO_RAG_SYSTEM_PROMPT + mode_instruction(response_mode))},
        {"role": "user", "content": question},
    ]
