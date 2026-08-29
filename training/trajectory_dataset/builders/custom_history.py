from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..io_utils import iter_jsonl
from ..schema import (
    DEFAULT_SYSTEM_PROMPT,
    SEARCH_HISTORY_TOOL,
    SEARCH_WEB_TOOL,
    SEARCH_WIKIPEDIA_TOOL,
    canonical_id,
    make_trajectory,
    tool_call,
)
from ..teacher.base import Teacher, TeacherRequest


TASK_TYPES = (
    "factual",
    "cause",
    "significance",
    "compare",
    "summary",
    "multihop",
    "verification",
    "hard_negative",
    "insufficient_evidence",
)


@dataclass(frozen=True)
class CustomBuildConfig:
    task_counts: dict[str, int] = field(default_factory=lambda: {
        "factual": 20,
        "cause": 10,
        "significance": 10,
        "compare": 8,
        "summary": 10,
        "multihop": 10,
        "verification": 6,
        "hard_negative": 6,
        "insufficient_evidence": 6,
    })
    top_k: int = 6
    seed: int = 42
    max_corpus_records: int = 10_000

    def __post_init__(self) -> None:
        unknown = set(self.task_counts) - set(TASK_TYPES)
        if unknown:
            raise ValueError(f"unknown custom task types: {sorted(unknown)}")
        if any(value < 0 for value in self.task_counts.values()):
            raise ValueError("custom task counts must be non-negative")
        if self.top_k < 1 or self.max_corpus_records < 1:
            raise ValueError("top_k and max_corpus_records must be positive")


def inspect_corpus(path: str | Path, *, max_records: int = 1000) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    facets: Counter[str] = Counter()
    titles: set[str] = set()
    rows = 0
    malformed = 0
    for record in iter_jsonl(path):
        rows += 1
        if not record.get("chunk_id") or not record.get("text"):
            malformed += 1
        sources[str(record.get("source_type") or record.get("source") or "unknown")] += 1
        titles.add(str(record.get("title") or ""))
        for facet in (record.get("metadata") or {}).get("content_facets", record.get("content_facets", [])) or []:
            facets[str(facet)] += 1
        if rows >= max_records:
            break
    return {
        "path": str(Path(path).resolve()),
        "sampled_records": rows,
        "malformed_records": malformed,
        "unique_titles": len(titles),
        "sources": dict(sources),
        "content_facets": dict(facets),
        "read_only": True,
    }


def load_seed_records(path: str | Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(iter_jsonl(path)):
        if index >= limit:
            break
        if not str(row.get("chunk_id") or "").strip() or not str(row.get("text") or "").strip():
            continue
        score = hashlib.sha256(f"{seed}:{row['chunk_id']}".encode("utf-8")).hexdigest()
        candidates.append((score, row))
    return [row for _, row in sorted(candidates)]


def _supports(row: dict[str, Any], task_type: str) -> bool:
    text = str(row.get("text") or "").casefold()
    metadata = row.get("metadata") or {}
    facets = " ".join(str(value).casefold() for value in metadata.get("content_facets", []) or [])
    if task_type == "cause":
        return "nguyên nhân" in facets or bool(re.search(r"\b(do|vì|nguyên nhân|dẫn đến)\b", text))
    if task_type == "significance":
        return "ý nghĩa" in facets or bool(re.search(r"ý nghĩa|đánh dấu|mở ra|góp phần", text))
    if task_type == "verification":
        return bool(row.get("url") or metadata.get("quality_flags") or row.get("quality_flags"))
    return True


def _question(task_type: str, primary: dict[str, Any], secondary: dict[str, Any] | None = None) -> str:
    title = str(primary.get("title") or "sự kiện này").strip()
    other = str((secondary or {}).get("title") or "đối tượng còn lại").strip()
    templates = {
        "factual": f"Theo tư liệu lịch sử, những thông tin chính về {title} là gì?",
        "cause": f"Những nguyên nhân hoặc điều kiện nào dẫn đến {title}?",
        "significance": f"{title} có ý nghĩa và tác động lịch sử như thế nào?",
        "compare": f"So sánh {title} và {other} theo bối cảnh, diễn biến, kết quả và ý nghĩa.",
        "summary": f"Hãy tóm tắt có cấu trúc về {title}, gồm bối cảnh, mốc chính, kết quả và ý nghĩa.",
        "multihop": f"Từ bối cảnh đến kết quả, các dữ kiện liên quan giúp giải thích {title} như thế nào?",
        "verification": f"Hãy kiểm chứng thận trọng nhận định về {title} bằng các nguồn có thể truy xuất.",
        "hard_negative": f"Nguyên nhân suy yếu hoặc thất bại liên quan đến {title} là gì?",
        "insufficient_evidence": f"Có đủ bằng chứng để khẳng định một nhận định còn tranh luận về {title} hay không?",
    }
    return templates[task_type]


def _observation(results: list[dict[str, Any]]) -> str:
    return json.dumps(results, ensure_ascii=False, sort_keys=True)


def _answer(task_type: str, evidence: list[dict[str, Any]]) -> str:
    usable = [item for item in evidence if item.get("chunk_id") and item.get("text")]
    if not usable:
        return "Các kết quả hiện có chưa cung cấp đủ bằng chứng để kết luận; cần tìm kiếm lại với truy vấn cụ thể hơn."
    paragraphs: list[str] = []
    for item in usable[:3]:
        text = " ".join(str(item["text"]).split())
        excerpt = text[:700].rsplit(" ", 1)[0] if len(text) > 700 else text
        paragraphs.append(f"- {excerpt} [{item['chunk_id']}]")
    prefix = {
        "verification": "Các nguồn truy xuất cho phép kết luận thận trọng như sau:",
        "hard_negative": "Kết quả gần nghĩa nhưng sai khía cạnh không đủ làm bằng chứng. Phần phù hợp là:",
        "compare": "Đối chiếu các nguồn theo từng đối tượng:",
    }.get(task_type, "Từ bằng chứng truy xuất được:")
    return prefix + "\n" + "\n".join(paragraphs)


def _trajectory(
    *,
    task_type: str,
    question: str,
    queries_and_results: list[tuple[str, str, list[dict[str, Any]]]],
    primary: dict[str, Any],
    ordinal: int,
    answer_override: str | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_results: list[dict[str, Any]] = []
    for index, (tool_name, query, results) in enumerate(queries_and_results, 1):
        call_id = f"call_{tool_name}_{index:04d}"
        arguments: dict[str, Any] = {"query": query, "top_k": len(results) or 5}
        if tool_name == "search_wikipedia":
            arguments["language"] = "vi"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, tool_name, arguments)]},
            {"role": "tool", "name": tool_name, "tool_call_id": call_id, "content": _observation(results)},
        ])
        all_results.extend(results)
    evidence_ids = list(dict.fromkeys(str(item.get("chunk_id")) for item in all_results if item.get("chunk_id")))
    answer_evidence = queries_and_results[-1][2] if task_type == "hard_negative" else all_results
    if task_type == "insufficient_evidence":
        answer_evidence = []
        evidence_ids = []
        final_answer = (
            "Các kết quả truy xuất hiện tại chưa trực tiếp chứng minh nhận định được hỏi. "
            "Vì vậy chưa đủ bằng chứng để kết luận; cần một truy vấn cụ thể hơn hoặc nguồn kiểm chứng bổ sung."
        )
    else:
        final_answer = _answer(task_type, answer_evidence)
    if answer_override:
        citations = " ".join(
            f"[{item['chunk_id']}]" for item in answer_evidence[:3] if item.get("chunk_id")
        )
        final_answer = f"{answer_override.strip()}\n\nBằng chứng: {citations}".strip()
    messages.append({"role": "assistant", "content": final_answer})
    group = str(primary.get("url") or primary.get("title") or primary.get("chunk_id"))
    tool_map = {
        "search_history": SEARCH_HISTORY_TOOL,
        "search_wikipedia": SEARCH_WIKIPEDIA_TOOL,
        "search_web": SEARCH_WEB_TOOL,
    }
    used_tools = list(dict.fromkeys(tool for tool, _, _ in queries_and_results))
    return make_trajectory(
        trajectory_id=canonical_id("custom_history", {"task": task_type, "chunk": primary["chunk_id"], "ordinal": ordinal}),
        source_dataset="custom_history",
        task_type=task_type,
        difficulty="hard" if task_type in {"compare", "multihop", "verification", "hard_negative"} else "medium",
        tools=[tool_map[name] for name in used_tools],
        messages=messages,
        provenance={
            "dataset_id": "local_enriched_vietnamese_history_corpus",
            "corpus_chunk_id": primary["chunk_id"],
            "source_document_id": group,
            "source_group": group,
            "grounded": True,
            "evidence_ids": evidence_ids,
            "retrieval_queries": [
                {"tool": tool, "query": query} for tool, query, _ in queries_and_results
            ],
            "external_verification": any(tool != "search_history" for tool in used_tools),
            "requires_final_answer": True,
            "corpus_read_only": True,
        },
    )


def build_custom_trajectories(
    corpus_path: str | Path,
    retriever: Any,
    *,
    config: CustomBuildConfig,
    completed_ids: set[str] | None = None,
    teacher: Teacher | None = None,
    external_retriever: Any | None = None,
) -> Iterable[dict[str, Any]]:
    completed = completed_ids if completed_ids is not None else set()
    records = load_seed_records(corpus_path, limit=config.max_corpus_records, seed=config.seed)
    if not records:
        raise ValueError("corpus contains no usable records")
    cursor = 0
    for task_type in TASK_TYPES:
        wanted = config.task_counts.get(task_type, 0)
        created = 0
        attempts = 0
        pending: list[tuple[int, dict[str, Any], dict[str, Any], str, int]] = []
        while created < wanted and attempts < len(records) * 2:
            primary = records[cursor % len(records)]
            secondary = records[(cursor + 1) % len(records)]
            cursor += 1
            attempts += 1
            if not _supports(primary, task_type):
                continue
            question = _question(task_type, primary, secondary)
            ordinal = created
            candidate_id = canonical_id(
                "custom_history",
                {"task": task_type, "chunk": primary["chunk_id"], "ordinal": ordinal},
            )
            created += 1
            if candidate_id in completed:
                continue
            pending.append((ordinal, primary, secondary, question, config.seed + cursor))
        if created < wanted:
            raise ValueError(f"corpus records could produce only {created}/{wanted} valid {task_type} trajectories")

        teacher_outputs: list[Any]
        if teacher is None or not pending:
            teacher_outputs = [None] * len(pending)
        else:
            teacher_outputs = teacher.generate([
                TeacherRequest(
                    task_type=task_type,
                    title=str(primary.get("title") or ""),
                    evidence=str(primary.get("text") or ""),
                    seed=teacher_seed,
                )
                for _, primary, _, _, teacher_seed in pending
            ])
            if len(teacher_outputs) != len(pending):
                raise ValueError("teacher must return exactly one response per request")

        for selection, teacher_output in zip(pending, teacher_outputs):
            ordinal, primary, secondary, question, _ = selection
            teacher_answer: str | None = None
            if teacher_output is not None:
                question = teacher_output.question
                teacher_answer = teacher_output.answer
            if task_type == "compare":
                first_query = f"{primary.get('title')} bối cảnh diễn biến kết quả ý nghĩa"
                second_query = f"{secondary.get('title')} bối cảnh diễn biến kết quả ý nghĩa"
                searches = [
                    ("search_history", first_query, retriever.search(first_query, top_k=config.top_k)),
                    ("search_history", second_query, retriever.search(second_query, top_k=config.top_k)),
                ]
            elif task_type in {"multihop", "verification"}:
                first_query = question
                second_query = f"{primary.get('title')} nguồn đối chiếu mốc thời gian kết quả"
                searches = [
                    ("search_history", first_query, retriever.search(first_query, top_k=config.top_k)),
                    ("search_history", second_query, retriever.search(second_query, top_k=config.top_k)),
                ]
                if task_type == "verification" and external_retriever is not None:
                    external_query = f"{primary.get('title')} kiểm chứng lịch sử"
                    searches.append((
                        "search_wikipedia",
                        external_query,
                        external_retriever.search("search_wikipedia", external_query, top_k=config.top_k),
                    ))
            elif task_type == "hard_negative":
                negative_query = f"{primary.get('title')} thành công thắng lợi"
                corrective_query = question
                searches = [
                    ("search_history", negative_query, retriever.search(negative_query, top_k=config.top_k)),
                    ("search_history", corrective_query, retriever.search(corrective_query, top_k=config.top_k)),
                ]
            else:
                searches = [("search_history", question, retriever.search(question, top_k=config.top_k))]
            yield _trajectory(
                task_type=task_type,
                question=question,
                queries_and_results=searches,
                primary=primary,
                ordinal=ordinal,
                answer_override=teacher_answer,
            )


def build_no_tool_trajectories() -> list[dict[str, Any]]:
    pairs = [
        ("Xin chào!", "Xin chào! Tôi có thể hỗ trợ bạn tìm hiểu lịch sử Việt Nam."),
        ("Bạn có thể làm gì?", "Tôi có thể hỗ trợ tra cứu, đối chiếu và giải thích các vấn đề lịch sử Việt Nam."),
    ]
    rows: list[dict[str, Any]] = []
    for index, (question, answer) in enumerate(pairs):
        rows.append(make_trajectory(
            trajectory_id=canonical_id("custom_history_no_tool", {"index": index, "question": question}),
            source_dataset="custom_history",
            task_type="no_tool",
            difficulty="easy",
            tools=[],
            messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            provenance={"synthetic": True, "source_group": "no-tool", "requires_final_answer": True},
        ))
    return rows
