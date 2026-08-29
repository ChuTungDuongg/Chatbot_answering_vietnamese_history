from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TeacherRequest:
    task_type: str
    title: str
    evidence: str
    seed: int


@dataclass(frozen=True)
class TeacherResponse:
    question: str
    answer: str


class Teacher(Protocol):
    def generate(self, requests: list[TeacherRequest]) -> list[TeacherResponse]:
        ...


class NoTeacher:
    def generate(self, requests: list[TeacherRequest]) -> list[TeacherResponse]:
        raise RuntimeError(
            "teacher-backend=none cannot synthesize free-form targets; use deterministic corpus templates, "
            "precomputed fixtures, or explicitly configure local_hf"
        )
