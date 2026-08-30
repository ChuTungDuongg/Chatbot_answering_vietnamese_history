from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TeacherRequest:
    task_type: str
    question: str
    evidence: str
    allowed_evidence_ids: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class TeacherResponse:
    answer: str
    # Accepted for compatibility with old mocks; V4 never uses teacher-authored questions.
    question: str | None = None


class Teacher(Protocol):
    def generate(self, requests: list[TeacherRequest]) -> list[TeacherResponse]:
        ...


class NoTeacher:
    def generate(self, requests: list[TeacherRequest]) -> list[TeacherResponse]:
        raise RuntimeError(
            "teacher-backend=none leaves deterministic evidence-grounded answers unchanged; "
            "explicitly configure local_hf only for post-retrieval answer enhancement"
        )
