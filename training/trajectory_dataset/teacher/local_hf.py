from __future__ import annotations

import json
from typing import Any

from .base import TeacherRequest, TeacherResponse


class LocalHFTeacher:
    """Optional local teacher. Construction is the only point that loads a model."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        if batch_size < 1:
            raise ValueError("teacher batch size must be at least 1")
        from transformers import pipeline

        device_map: Any = "auto" if device == "auto" else device
        self._pipeline = pipeline("text-generation", model=model_id, device_map=device_map)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, requests: list[TeacherRequest]) -> list[TeacherResponse]:
        prompts = [
            (
                "Tạo đúng một JSON object có hai khóa question và answer. "
                f"Loại câu hỏi: {request.task_type}. Tiêu đề: {request.title}. "
                "Chỉ dùng bằng chứng sau, không thêm kiến thức ngoài:\n"
                f"{request.evidence}"
            )
            for request in requests
        ]
        generated = self._pipeline(
            prompts,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
        responses: list[TeacherResponse] = []
        for prompt, item in zip(prompts, generated):
            candidate = item[0] if isinstance(item, list) else item
            text = str(candidate.get("generated_text") or "")
            if text.startswith(prompt):
                text = text[len(prompt) :]
            try:
                value = json.loads(text.strip())
            except json.JSONDecodeError as exc:
                raise ValueError("local teacher did not return valid JSON") from exc
            question = str(value.get("question") or "").strip()
            answer = str(value.get("answer") or "").strip()
            if not question or not answer:
                raise ValueError("local teacher response is missing question or answer")
            responses.append(TeacherResponse(question=question, answer=answer))
        return responses
