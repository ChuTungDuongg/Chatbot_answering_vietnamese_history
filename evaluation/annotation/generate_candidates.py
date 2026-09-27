"""Stream Corpus V1 into balanced draft candidates, then retrieve review evidence."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.policy import CATEGORIES, fold, similar
from evaluation.annotation.workspace import Workspace, dumps


TEMPLATE_VERSION = "history_draft_templates_v1"
ROOT = Path(__file__).resolve().parents[2]
FAMOUS = re.compile(r"dien bien phu|ho chi minh|cach mang thang tam")
OUT_OF_DOMAIN_PROMPTS = (
    "Thời tiết Hà Nội ngày mai như thế nào?",
    "Làm thế nào để sửa lỗi Python trong chương trình này?",
    "Giá cổ phiếu hôm nay biến động ra sao?",
    "Công thức nấu phở bò tại nhà gồm những bước nào?",
    "Đội bóng nào đang dẫn đầu giải ngoại hạng?",
    "Điện thoại nào có pin bền nhất năm nay?",
    "Tôi nên tập bài thể dục nào vào buổi sáng?",
    "Làm sao giải phương trình bậc hai này?",
    "Có thể dự báo mưa ở Đà Nẵng cuối tuần không?",
    "Mẫu email xin nghỉ phép ngắn gọn nên viết thế nào?",
    "Làm sao cấu hình mạng Wi-Fi cho máy tính?",
    "Tỷ giá ngoại tệ hôm nay là bao nhiêu?",
    "Món ăn chay nào phù hợp cho bữa tối?",
    "Cách chăm sóc cây cảnh trong căn hộ là gì?",
    "Bài hát nào đang phổ biến trên bảng xếp hạng?",
    "Lịch chiếu phim ở rạp gần tôi ra sao?",
    "Làm thế nào nén một tệp PDF trên máy tính?",
    "Làm sao đặt vé tàu cho chuyến đi ngày mai?",
    "Tôi nên mua loại máy ảnh nào cho du lịch?",
    "Cách sửa vòi nước bị rò rỉ trong bếp?",
)
PATTERNS = {
    "dynasties": r"\bnha (ly|tran|le|nguyen|dinh|tien le|tay son)\b|trieu dai|vuong trieu",
    "rulers": r"\bvua\b|hoang de|quan vuong|de vuong",
    "historical_figures": r"nhan vat|danh nhan|tuong linh|anh hung|lanh tu",
    "battles": r"\btran\b|chien thang|chien dich|bach dang",
    "wars": r"chien tranh|khang chien|xam luoc",
    "revolutions": r"cach mang|khoi nghia|noi day",
    "treaties": r"hiep dinh|hiep uoc|hoa uoc|cong uoc",
    "colonial_period": r"thuoc dia|thuc dan|phap thuoc|dong duong",
    "independence": r"doc lap|giai phong|thong nhat",
    "political_events": r"chinh tri|dai hoi|dang|nha nuoc|chinh phu",
    "cultural_history": r"van hoa|van hoc|nghe thuat|giao duc|chu nom",
    "religious_history": r"ton giao|phat giao|cong giao|dao giao|tin nguong",
    "economic_history": r"kinh te|thuong mai|nong nghiep|thue|tien te",
    "social_history": r"xa hoi|dan cu|phong tuc|giai cap|phu nu",
    "historical_geography": r"dia ly|dia danh|kinh do|thanh co|song|bien gioi",
    "archaeology_heritage": r"khao co|di san|di tich|lang mo|hien vat",
    "cause_consequence": r"nguyen nhan|hau qua|ket qua|y nghia",
}
TEMPLATES = {
    "dynasties": "Triều đại {title} hình thành và để lại dấu ấn gì trong lịch sử Việt Nam?",
    "rulers": "Vai trò của {title} trong bối cảnh lịch sử đương thời là gì?",
    "historical_figures": "{title} đã có những đóng góp nào trong lịch sử Việt Nam?",
    "battles": "Diễn biến và kết quả chính của {title} là gì?",
    "wars": "Nguyên nhân và kết quả của {title} là gì?",
    "revolutions": "Bối cảnh và kết quả của {title} là gì?",
    "treaties": "Nội dung và tác động lịch sử của {title} là gì?",
    "colonial_period": "{title} phản ánh điều gì về thời kỳ thuộc địa?",
    "independence": "{title} có vai trò gì đối với tiến trình độc lập của Việt Nam?",
    "political_events": "Bối cảnh và hệ quả chính trị của {title} là gì?",
    "cultural_history": "{title} có ý nghĩa gì trong lịch sử văn hóa Việt Nam?",
    "religious_history": "{title} có vai trò gì trong lịch sử tôn giáo Việt Nam?",
    "economic_history": "{title} cho thấy những biến đổi kinh tế nào trong lịch sử?",
    "social_history": "{title} phản ánh những thay đổi xã hội nào?",
    "historical_geography": "{title} có ý nghĩa địa lý và lịch sử như thế nào?",
    "archaeology_heritage": "{title} giúp hiểu thêm điều gì về lịch sử Việt Nam?",
    "chronology": "{title} gắn với những mốc thời gian lịch sử nào?",
    "cause_consequence": "Những nguyên nhân và hệ quả chính của {title} là gì?",
}


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    draft_question: str = Field(min_length=1)
    category: str
    difficulty: str | None = None
    question_type: str | None = None
    origin_chunk_id: str | None = None
    origin_source_id: str | None = None
    origin_title: str | None = None
    draft_answer: str | None = None
    candidate_required_facts: list[str] = Field(default_factory=list)
    generation_metadata: dict[str, Any]
    review_status: str = "pending"


def _score(seed: int, category: str, chunk_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{category}:{chunk_id}".encode()).digest()[:8], "big")


def scaled_targets(config: dict[str, Any], limit: int) -> dict[str, int]:
    raw = {**config["category_targets"], **config["edge_targets"]}
    if any(category not in CATEGORIES or value < 0 for category, value in raw.items()):
        raise ValueError("Invalid category target")
    if sum(raw.values()) != int(config["base_target"]):
        raise ValueError("Category and edge targets must sum to base_target")
    weighted = {name: count * limit / config["base_target"] for name, count in raw.items()}
    result = {name: int(value) for name, value in weighted.items()}
    remaining = limit - sum(result.values())
    for name in sorted(raw, key=lambda key: (-(weighted[key] - result[key]), key))[:remaining]:
        result[name] += 1
    return result


def categories_for(row: dict[str, Any]) -> set[str]:
    title = fold(str(row.get("title") or ""))
    section = fold(str(row.get("section") or ""))
    source_meta = (row.get("metadata") or {}).get("source_metadata") or {}
    main_category = fold(str(source_meta.get("main_category") or ""))
    context = f"{title} {section} {main_category}"
    categories = {name for name, pattern in PATTERNS.items() if re.search(pattern, context)}
    if row.get("years"):
        categories.add("chronology")
    return categories


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("chunk_id", "source_id", "document_id", "title",
                                          "url", "section", "years", "chunk_index")} | {
        "excerpt": str(row.get("text") or "")[:450]}


def candidate_pools(corpus: Path, targets: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[tuple[int, str, dict[str, Any]]]] = {name: [] for name in targets}
    pools["_general"] = []
    capacities = {name: max(30, count * 12) for name, count in targets.items()}
    capacities["_general"] = max(100, sum(targets.values()) * 4)
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("source_id") or not row.get("chunk_id") or not row.get("title"):
                continue
            eligible = categories_for(row) & set(targets)
            eligible.add("_general")
            compact = _compact(row)
            for category in eligible:
                heap = pools[category]
                score = _score(seed, category, row["chunk_id"])
                entry = (-score, row["chunk_id"], compact)
                if len(heap) < capacities[category]:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    return {name: [entry[2] for entry in sorted(heap, reverse=True)] for name, heap in pools.items()}


def _candidate(row: dict[str, Any] | None, category: str, seed: int, *,
               paired: dict[str, Any] | None = None, serial: int = 0,
               category_match: bool = True) -> dict[str, Any]:
    title = str(row["title"]).strip() if row else ""
    excerpt = re.split(r"(?<=[.!?])\s+", row["excerpt"], maxsplit=1)[0].strip()[:300] if row else ""
    if category == "comparison":
        question = f"Điểm giống và khác giữa {title} và {paired['title']} là gì?"
        excerpt = (excerpt + " " + paired["excerpt"][:200]).strip()
    elif category == "multi_hop":
        question = f"{title} có liên hệ như thế nào với {paired['title']} trong lịch sử Việt Nam?"
        excerpt = (excerpt + " " + paired["excerpt"][:200]).strip()
    elif category == "out_of_domain":
        question = OUT_OF_DOMAIN_PROMPTS[serial % len(OUT_OF_DOMAIN_PROMPTS)]
    elif category == "ambiguous":
        question = f"Sự kiện liên quan đến {title} xảy ra khi nào?"
    elif category == "insufficient_evidence":
        question = f"Có thể xác định đầy đủ mọi người tham dự {title} từ các nguồn hiện có không?"
    elif category == "false_premise":
        years = row.get("years") or []
        year = int(years[0]) + 17 if years and str(years[0]).isdigit() else 1800 + serial
        question = f"Có đúng {title} diễn ra vào năm {year} không?"
    else:
        question = TEMPLATES[category].format(title=title)
    difficulty = "hard" if category in ("comparison", "multi_hop", "false_premise") else (
        "easy" if category == "chronology" else "medium")
    identifier = "cand_" + hashlib.sha256(
        f"{seed}|{category}|{row['chunk_id'] if row else 'synthetic'}|{serial}|{question}".encode()).hexdigest()[:24]
    metadata = {"provider": "deterministic-template", "model_id": None,
                "revision": TEMPLATE_VERSION, "seed": seed, "category_match": category_match,
                "draft_difficulty_reason": "template evidence scope, pending human review",
                "edge_case": category if category in ("out_of_domain", "false_premise",
                                                       "ambiguous", "insufficient_evidence") else None,
                "paired_chunk_id": paired["chunk_id"] if paired else None}
    result = {"candidate_id": identifier, "question": question, "draft_question": question,
              "category": category, "difficulty": difficulty,
              "question_type": category, "origin_chunk_id": row["chunk_id"] if row else None,
              "origin_source_id": row["source_id"] if row else None,
              "origin_title": title or None, "draft_answer": excerpt or None,
              "candidate_required_facts": [], "generation_metadata": metadata,
              "review_status": "pending"}
    return Candidate.model_validate(result).model_dump()


def draft_candidates(corpus: Path, config: dict[str, Any], *, limit: int = 50,
                     existing: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    existing = existing or []
    targets = scaled_targets(config, limit)
    seed = int(config["seed"])
    pools = candidate_pools(corpus, targets, seed)
    counts = Counter(item["category"] for item in existing)
    used_sources = Counter(item.get("origin_source_id") for item in existing if item.get("origin_source_id"))
    used_titles = Counter(fold(item.get("origin_title") or "") for item in existing if item.get("origin_title"))
    used_questions = [item["question"] for item in existing]
    used_ids = {item["candidate_id"] for item in existing}
    famous_count = sum(bool(FAMOUS.search(fold(item.get("origin_title") or ""))) for item in existing)
    output: list[dict[str, Any]] = []
    normal = [name for name in config["category_targets"] if targets.get(name, 0)]
    normal.sort(key=lambda name: (len(pools.get(name, [])), name))
    for category in [*normal, *config["edge_targets"]]:
        needed = max(0, targets.get(category, 0) - counts[category])
        if not needed:
            continue
        # Try unused synthetic prompts as well when expanding an existing pilot.
        choices = [None] * len(OUT_OF_DOMAIN_PROMPTS) if category == "out_of_domain" else [
            *pools.get(category, []), *pools["_general"]]
        serial = 0
        for row in choices:
            if needed == 0:
                break
            if row:
                source, title = row["source_id"], fold(row["title"])
                if used_sources[source] >= int(config["max_candidates_per_source"]):
                    continue
                if used_titles[title] >= int(config["max_candidates_per_title"]):
                    continue
                is_famous = bool(FAMOUS.search(title))
                if is_famous and famous_count >= int(config["max_famous_topics"]):
                    continue
            paired = None
            if category in ("comparison", "multi_hop") and row:
                paired = next((item for item in pools["_general"]
                               if item["source_id"] != row["source_id"] and
                               fold(item["title"]) != fold(row["title"])), None)
                if paired is None:
                    continue
            candidate = _candidate(row, category, seed, paired=paired, serial=serial,
                                   category_match=bool(row in pools.get(category, [])))
            serial += 1
            if candidate["candidate_id"] in used_ids or any(
                    similar(candidate["question"], question) for question in used_questions):
                continue
            output.append(candidate)
            used_ids.add(candidate["candidate_id"])
            used_questions.append(candidate["question"])
            if row:
                used_sources[row["source_id"]] += 1
                used_titles[fold(row["title"])] += 1
                famous_count += int(bool(FAMOUS.search(fold(row["title"]))))
            needed -= 1
        if needed:
            raise RuntimeError(f"Could not fill {category} target; refine category targets or source sampling")
    return output


def evidence_from_result(result: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for stage, rows in (result.get("annotation_stages") or {}).items():
        for item in rows:
            row_index = int(item["row_id"])
            if not 0 <= row_index < len(chunks):
                raise RuntimeError("Retrieval row outside corpus")
            chunk = chunks[row_index]
            evidence.append({"stage": stage, "rank": int(item["rank"]),
                             "chunk_id": chunk["chunk_id"], "source_id": chunk.get("source_id"),
                             "document_id": chunk.get("document_id"),
                             "score": float(item["score"]),
                             "details": {key: value for key, value in item.items()
                                         if key not in ("row_id", "rank", "score")}})
    return evidence


def hydrate_workspace(workspace: Workspace, retriever, chunks: list[dict[str, Any]]) -> int:
    pending = [row for row in workspace.all_candidates() if not row["evidence_complete"]]
    completed = 0
    for index, candidate in enumerate(pending, 1):
        try:
            result = retriever.retrieve(candidate["question"], include_stage_diagnostics=True)
            workspace.set_evidence(candidate["candidate_id"], evidence_from_result(result, chunks))
            completed += 1
            print(f"[annotation] evidence {index}/{len(pending)} id={candidate['candidate_id']}", flush=True)
        except Exception as exc:
            # Avoid printing provider/network exception details or any secret value.
            print(f"[annotation] evidence failed id={candidate['candidate_id']} type={type(exc).__name__}",
                  file=sys.stderr, flush=True)
    return completed


def write_candidate_file(workspace: Workspace) -> Path:
    """Refresh the generated JSONL view from transactional drafts after a crash."""
    path = workspace.directory / "candidates.jsonl"
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for (draft_json,) in workspace.connection.execute(
                "SELECT draft_json FROM candidates ORDER BY candidate_id"):
            handle.write(draft_json + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("artifacts/corpus_v1/retrieval"))
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--config", type=Path, default=Path("configs/annotation_pilot.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--dense-backend", choices=("faiss", "qdrant"), default="faiss")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    corpus, retrieval = args.corpus.resolve(), args.retrieval_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    workspace = Workspace(args.workspace)
    try:
        workspace.set_metadata("corpus_path", str(corpus))
        if not (args.workspace / "corpus_lookup.sqlite3").is_file():
            runtime_manifest = corpus.parent / "runtime" / "manifest.json"
            identity = json.loads(runtime_manifest.read_text(encoding="utf-8"))["corpus"] if runtime_manifest.is_file() else {}
            lookup = CorpusLookup.build(corpus, args.workspace / "corpus_lookup.sqlite3",
                                        expected_sha=identity.get("corpus_sha256"),
                                        expected_count=identity.get("count"))
        else:
            lookup = CorpusLookup(corpus, args.workspace / "corpus_lookup.sqlite3")
        workspace.set_metadata("corpus_sha256", lookup.corpus_sha256)
        lookup.close()
        existing = workspace.all_candidates()
        config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
        config_key = f"expansion_config_sha256_{args.limit}" if args.expand or args.limit > 50 else "generator_config_sha256"
        workspace.set_metadata(config_key, config_hash)
        if existing and not (args.resume or args.expand):
            raise RuntimeError("Workspace already has candidates; use --resume or --expand")
        if args.expand and args.limit > 50 and existing and any(
                row["review_status"] not in ("accepted", "rejected") for row in existing):
            raise RuntimeError("Review all pilot candidates before expanding")
        if args.limit > 50 and not (args.expand or args.resume):
            raise RuntimeError("Expansion beyond the 50-question pilot requires --expand")
        if not existing and args.expand:
            raise RuntimeError("Create and review the 50-question pilot first")
        planned_target = workspace.get_metadata("target_count")
        if args.limit > 50 and workspace.get_metadata(f"expansion_target_{args.limit}") == args.limit:
            planned_target = args.limit
        if args.resume and existing and args.limit != planned_target:
            raise RuntimeError("Resume target count differs from workspace")
        if not existing:
            workspace.set_metadata("target_count", args.limit)
        if args.expand:
            # A new target is recorded as an event-like metadata key, preserving prior plans.
            workspace.set_metadata(f"expansion_target_{args.limit}", args.limit)
        new = [] if args.resume and existing else draft_candidates(corpus, config, limit=args.limit, existing=existing)
        if new:
            workspace.insert_candidates(new)
            print(f"[annotation] drafted={len(new)} total={workspace.count()}", flush=True)
        write_candidate_file(workspace)
        pending = [row for row in workspace.all_candidates() if not row["evidence_complete"]]
        if pending:
            # Set before import so annotation always loads retrieval, never Qwen.
            os.environ["APP_MODE"] = "retrieval-only"
            os.environ["CORPUS_PATH"] = str(corpus)
            os.environ["RETRIEVAL_ROOT"] = str(retrieval)
            os.environ["RETRIEVAL_DENSE_BACKEND"] = args.dense_backend
            from app.config import Settings
            import app.services.rag_service as service_module
            from app.rag.retrieval import HybridRetriever
            service_module.settings = Settings()
            service = service_module.RAGService()
            try:
                service.load()
                hydrated = hydrate_workspace(workspace, HybridRetriever(service), service.chunks)
            finally:
                service.shutdown()
            if hydrated != len(pending):
                return 1
        print(f"[annotation] ready={workspace.count()} workspace={args.workspace}")
        return 0
    finally:
        workspace.close()


if __name__ == "__main__":
    raise SystemExit(main())
