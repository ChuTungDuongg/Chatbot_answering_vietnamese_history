"""Derive citation-supervised SILVER V2 from frozen V1 SFT and batch audits.

No model is called. V1 inputs are opened read-only and are never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.rag.prompting import SYSTEM_PROMPT
from app.rag.response_modes import MODE_INSTRUCTIONS, mode_instruction
from evaluation.metrics.citations import BRACKET, parse_citations
from evaluation.schema import Question
from evaluation.silver_v1 import validate_sft
from training.train_qwen3 import read_sft

CANONICAL_SHA256 = "97acf491e7409e54ed27f38f1e28a16b9e85107c9c2ed296bffaf48814d539d8"
COUNTS = {"train": 2400, "validation": 300, "test": 300, "train_sft": 3000, "validation_sft": 300}
CHUNK_ID_BYTES = re.compile(rb'"chunk_id"\s*:\s*"([^"\\]+)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                    for row in rows)


def _unique(rows: list[dict], name: str) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs in {name}")
    return result


def load_audits(directory: Path, canonical: list[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    """Pair original batch records and audits by their immutable row order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Audit batch directory missing: {directory}")
    expected = {f"batch_{number:02d}.audit.jsonl" for number in range(1, 61)}
    unexpected = {path.name for path in directory.glob("batch_*.audit.jsonl")} - expected
    if unexpected:
        raise ValueError(f"Unexpected audit batches beyond frozen V1: {sorted(unexpected)}")
    by_id = _unique(canonical, "canonical")
    audits: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for number in range(1, 61):
        batch = directory / f"batch_{number:02d}.jsonl"
        audit = directory / f"batch_{number:02d}.audit.jsonl"
        if not batch.exists() and not audit.exists():
            continue  # Explicitly reported as unresolved below.
        if not batch.is_file() or not audit.is_file():
            raise ValueError(f"Incomplete batch/audit pair: {number:02d}")
        records, evidence = jsonl(batch), jsonl(audit)
        if len(records) != 50 or len(evidence) != 50:
            raise ValueError(f"Batch {number:02d} must have 50 paired rows")
        for offset, (record, item) in enumerate(zip(records, evidence, strict=True), 1):
            expected = f"vn_hist_silver_{(number - 1) * 50 + offset:04d}"
            if record.get("id") != expected or record != by_id.get(expected):
                raise ValueError(f"Batch {number:02d} differs from frozen canonical: {expected}")
            audits[expected] = item
        hashes[batch.name] = sha256(batch)
        hashes[audit.name] = sha256(audit)
    if not audits:
        raise ValueError(f"No frozen V1 batch audits found: {directory}")
    return audits, hashes


def load_corpus_chunks(path: Path, wanted: set[str]) -> tuple[dict[str, dict], str, int]:
    """Scan the actual corpus once; retain only canonical evidence chunks."""
    chunks: dict[str, dict] = {}
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            count += 1
            match = CHUNK_ID_BYTES.search(raw)
            if match is None:
                raise ValueError(f"Corpus row {count} has no chunk_id")
            chunk_id = match.group(1).decode("utf-8")
            if chunk_id in wanted:
                if chunk_id in chunks:
                    raise ValueError(f"Duplicate corpus chunk ID: {chunk_id}")
                row = json.loads(raw)
                if row.get("chunk_id") != chunk_id:
                    raise ValueError(f"Corpus chunk ID mismatch: {chunk_id}")
                chunks[chunk_id] = row
    missing = wanted - chunks.keys()
    if missing:
        raise ValueError(f"Canonical chunks absent from corpus: {sorted(missing)[:5]}")
    return chunks, digest.hexdigest(), count


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    v1_directory = args.train_sft_v1.resolve().parent
    output_directory = args.output_dir.resolve()
    if output_directory == v1_directory or output_directory.is_relative_to(v1_directory):
        raise ValueError("V2 output must be outside the frozen V1 SFT directory")
    canonical = jsonl(args.canonical)
    for row in canonical:
        Question.model_validate(row)
    if len(canonical) != 3000 or sha256(args.canonical) != CANONICAL_SHA256:
        raise ValueError("Canonical SILVER V1 count/SHA-256 mismatch")
    canonical_by_id = _unique(canonical, "canonical")
    split_dir = args.train_split.parent
    manifest_path = split_dir / "split_manifest.json"
    test_path = split_dir / "test.jsonl"
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {"train": jsonl(args.train_split), "validation": jsonl(args.validation_split), "test": jsonl(test_path)}
    split_paths = {"train": args.train_split, "validation": args.validation_split, "test": test_path}
    if split_manifest.get("seed") != 42 or split_manifest.get("canonical_dataset_sha256") != CANONICAL_SHA256:
        raise ValueError("Frozen split identity mismatch")
    for name, rows in splits.items():
        if (len(rows) != COUNTS[name] or [row["id"] for row in rows] != split_manifest["ids"][name]
                or sha256(split_paths[name]) != split_manifest["split_sha256"][name]
                or any(row != canonical_by_id.get(row["id"]) for row in rows)):
            raise ValueError(f"Frozen {name} split mismatch")
    if set().union(*(set(row["id"] for row in rows) for rows in splits.values())) != set(canonical_by_id):
        raise ValueError("Frozen splits do not cover canonical IDs")
    if any(set(row["id"] for row in splits[left]) & set(row["id"] for row in splits[right])
           for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Frozen splits overlap")
    v1 = {"train": read_sft(args.train_sft_v1), "validation": read_sft(args.validation_sft_v1)}
    v1_paths = {"train": args.train_sft_v1, "validation": args.validation_sft_v1}
    v1_manifest_path = args.train_sft_v1.parent / "manifest.json"
    if args.train_sft_v1.parent != args.validation_sft_v1.parent or not v1_manifest_path.is_file():
        raise ValueError("V1 SFT inputs need their common frozen manifest")
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    if (v1_manifest.get("canonical_dataset_sha256") != CANONICAL_SHA256 or
        v1_manifest.get("split_manifest_sha256") != sha256(manifest_path)):
        raise ValueError("V1 SFT manifest provenance mismatch")
    for name, rows in v1.items():
        if len(rows) != COUNTS[f"{name}_sft"] or v1_manifest.get(f"{name}_sft_sha256") != sha256(v1_paths[name]):
            raise ValueError(f"Frozen V1 {name} SFT count/SHA-256 mismatch")
        owners = set(row["id"] for row in splits[name])
        if any(row["canonical_id"] not in owners or row["id"] != f"{row['canonical_id']}:{row['response_mode']}"
               for row in rows):
            raise ValueError(f"V1 {name} SFT split/variant mismatch")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate V1 {name} SFT variant")
        if name == "train" and set(row["canonical_id"] for row in rows) != owners:
            raise ValueError("V1 train SFT canonical coverage mismatch")
        if name == "validation" and set(row["canonical_id"] for row in rows) != owners:
            raise ValueError("V1 validation SFT canonical coverage mismatch")
    audits, audit_hashes = load_audits(args.audit_dir, canonical)
    wanted = {chunk for row in canonical for chunk in row.get("relevant_chunk_ids", [])}
    chunks, corpus_sha, corpus_count = load_corpus_chunks(args.corpus, wanted)
    if corpus_sha != v1_manifest.get("corpus_sha256") or corpus_count != v1_manifest.get("corpus_chunk_count"):
        raise ValueError("Corpus differs from frozen V1 SFT manifest")
    class CorpusView:
        def chunk(self, chunk_id: str) -> dict | None:
            return chunks.get(chunk_id)
    # Reconstruct the original messages with the untouched V1 implementation.
    validate_sft(splits, v1["train"], v1["validation"], CorpusView())
    return {"canonical": canonical_by_id, "splits": splits, "v1": v1, "audits": audits,
            "chunks": chunks, "corpus_sha": corpus_sha, "corpus_count": corpus_count,
            "provenance": {"canonical_sha256": sha256(args.canonical),
                "split_manifest_sha256": sha256(manifest_path),
                "train_split_sha256": sha256(args.train_split),
                "validation_split_sha256": sha256(args.validation_split),
                "test_split_sha256": sha256(test_path),
                "train_sft_v1_sha256": sha256(args.train_sft_v1),
                "validation_sft_v1_sha256": sha256(args.validation_sft_v1),
                "v1_manifest_sha256": sha256(v1_manifest_path),
                "corpus_sha256": corpus_sha, "corpus_chunk_count": corpus_count,
                "audit_file_sha256": audit_hashes, "audited_canonical_ids": len(audits)}}


def fact_support(record: dict, audit: dict | None, chunks: dict[str, dict]) -> dict[int, list[str]]:
    """Validate evidence labels and return only explicitly audited fact support."""
    support: dict[int, list[str]] = defaultdict(list)
    if audit is None:
        return support
    if not isinstance(audit, dict):
        raise ValueError(f"Invalid audit object: {record['id']}")
    evidence = audit.get("evidence")
    if not isinstance(evidence, list):
        return support
    relevant = set(record.get("relevant_chunk_ids", []))
    source_ids = set(record.get("relevant_source_ids", []))
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid audit evidence item: {record['id']}")
        chunk_id = item.get("chunk_id")
        chunk = chunks.get(chunk_id) if isinstance(chunk_id, str) else None
        indices = item.get("supports_fact_indices", [])
        if (chunk is None or not isinstance(indices, list) or
            any(type(i) is not int or i < 0 or i >= len(record["required_facts"]) for i in indices) or
            not isinstance(item.get("quote"), str) or not item["quote"].strip() or
            item["quote"] not in chunk.get("text", "") or
            item.get("decision") not in {"relevant", "not_relevant", "uncertain"}):
            raise ValueError(f"Invalid audit evidence: {record['id']}/{chunk_id}")
        if item["decision"] != "relevant":
            if indices:
                raise ValueError(f"Non-relevant evidence supports fact: {record['id']}/{chunk_id}")
            continue
        if chunk_id not in relevant or chunk.get("source_id") not in source_ids:
            raise ValueError(f"Audit/canonical/corpus mapping mismatch: {record['id']}/{chunk_id}")
        for index in indices:
            if chunk_id not in support[index]:
                support[index].append(chunk_id)
    return support


def _chosen_chunk(record: dict, support: dict[int, list[str]], fact_index: int,
                  chunks: dict[str, dict]) -> str | None:
    gold = set(record.get("gold_citation_source_ids", []))
    candidates = set(support.get(fact_index, []))
    for chunk_id in record.get("relevant_chunk_ids", []):
        chunk = chunks[chunk_id]
        if (chunk_id in candidates and chunk["source_id"] in record.get("relevant_source_ids", [])
                and (not gold or chunk["source_id"] in gold)):
            return chunk_id
    return None


def cite_answer(answer: str, record: dict, audit: dict | None,
                chunks: dict[str, dict]) -> tuple[str, dict]:
    """Insert citations at exact fact spans, or one conservative paragraph end."""
    if BRACKET.search(answer):
        raise ValueError(f"V1 assistant already has bracket text: {record['id']}")
    if not record.get("answerable"):
        return answer, {"status": "unanswerable_unchanged", "cited_chunks": [], "unresolved_facts": []}
    support = fact_support(record, audit, chunks)
    facts = record.get("required_facts") or []
    positions: list[tuple[int, int, str, int]] = []
    unresolved: list[int] = []
    for index, fact in enumerate(facts):
        chunk_id = _chosen_chunk(record, support, index, chunks)
        at = answer.find(fact)
        if chunk_id is None or at < 0:
            unresolved.append(index)
        else:
            positions.append((at, at + len(fact), chunk_id, index))
    positions.sort()
    if any(left[1] > right[0] for left, right in zip(positions, positions[1:])):
        raise ValueError(f"Overlapping required-fact spans: {record['id']}")
    if positions:
        cited = answer
        for _, end, chunk_id, _ in reversed(positions):
            cited = cited[:end] + f" [{chunk_id}]" + cited[end:]
        cited_chunks = [item[2] for item in positions]
        # A cited fact in one paragraph does not justify a different factual
        # paragraph whose claim-level wording has no exact audited span.
        paragraph_spans = []
        beginning = 0
        for separator in re.finditer(r"\n\s*\n", answer):
            paragraph_spans.append((beginning, separator.start()))
            beginning = separator.end()
        paragraph_spans.append((beginning, len(answer)))
        uncited_paragraphs = [index for index in record.get("factual_paragraph_indices", [])
                             if 0 <= index < len(paragraph_spans) and
                             not any(paragraph_spans[index][0] <= start < paragraph_spans[index][1]
                                     for start, *_ in positions)]
        return cited, {"status": "partial" if unresolved or uncited_paragraphs else "cited",
                       "cited_chunks": cited_chunks, "unresolved_facts": unresolved,
                       "uncited_factual_paragraph_indices": uncited_paragraphs}
    # Paragraph-level attribution is allowed only when a single relevant
    # audited chunk supports every required fact in this one factual paragraph.
    relevant = record.get("relevant_chunk_ids", [])
    if (audit is not None and len(relevant) == 1 and "\n\n" not in answer and
        record.get("factual_paragraph_indices") == [0] and facts and
        all(_chosen_chunk(record, support, i, chunks) == relevant[0] for i in range(len(facts)))):
        chunk_id = relevant[0]
        return answer + f" [{chunk_id}]", {"status": "paragraph_cited",
                                          "cited_chunks": [chunk_id], "unresolved_facts": []}
    return answer, {"status": "unresolved", "cited_chunks": [],
                    "unresolved_facts": list(range(len(facts)))}


def strip_citations(answer: str, chunk_ids: set[str]) -> str:
    pattern = re.compile(r" \[(?:" + "|".join(re.escape(value) for value in sorted(chunk_ids, key=len, reverse=True)) + r")\]") if chunk_ids else None
    return pattern.sub("", answer) if pattern else answer


def transform_row(v1: dict, record: dict, audit: dict | None,
                  chunks: dict[str, dict]) -> tuple[dict, dict]:
    if v1["canonical_id"] != record["id"]:
        raise ValueError("V1/canonical ID mismatch")
    if (v1["relevant_chunk_ids"] != record["relevant_chunk_ids"] or
        v1["relevant_source_ids"] != record["relevant_source_ids"] or
        v1["messages"][1]["content"] != _context_message(record, chunks)):
        raise ValueError(f"V1 context/canonical/corpus mismatch: {v1['id']}")
    answer, info = cite_answer(v1["messages"][2]["content"], record, audit, chunks)
    row = deepcopy(v1)
    row["messages"][0]["content"] = f"{SYSTEM_PROMPT} {mode_instruction(v1['response_mode'])}"
    row["messages"][2]["content"] = answer
    validate_row(v1, row, record, chunks)
    return row, info


def _context_message(record: dict, chunks: dict[str, dict]) -> str:
    sections = [f"[{chunk_id}] {chunks[chunk_id]['title']}\n{chunks[chunk_id]['text']}"
                for chunk_id in record.get("relevant_chunk_ids", [])]
    context = "\n\n".join(sections) if sections else "Không có nguồn phù hợp."
    return f"Câu hỏi: {record['question']}\n\nNguồn được truy xuất:\n{context}"


def validate_row(v1: dict, v2: dict, record: dict, chunks: dict[str, dict]) -> None:
    original = v1["messages"][2]["content"]
    answer = v2["messages"][2]["content"]
    if (set(v1) != set(v2) or any(v1[key] != v2[key] for key in v1 if key != "messages") or
        [item["role"] for item in v2["messages"]] != ["system", "user", "assistant"] or
        v1["messages"][1] != v2["messages"][1] or
        v2["messages"][0]["content"] != f"{SYSTEM_PROMPT} {mode_instruction(v1['response_mode'])}" or
        strip_citations(answer, set(record["relevant_chunk_ids"])) != original):
        raise ValueError(f"V2 changed frozen V1 content or metadata: {v1['id']}")
    aliases = parse_citations(answer, set(record["relevant_chunk_ids"]))
    bracketed = [match.group(1) for match in BRACKET.finditer(answer)]
    if len(aliases) != len(bracketed) or aliases != bracketed:
        raise ValueError(f"Invalid citation alias: {v1['id']}")
    gold = set(record.get("gold_citation_source_ids", []))
    for chunk_id in aliases:
        chunk = chunks.get(chunk_id)
        if (chunk_id not in record["relevant_chunk_ids"] or chunk is None or
            chunk["source_id"] not in record["relevant_source_ids"] or
            (gold and chunk["source_id"] not in gold) or
            f"[{chunk_id}] " not in v2["messages"][1]["content"]):
            raise ValueError(f"Citation outside context/corpus/gold sources: {v1['id']}/{chunk_id}")
    if any(a == b for a, b in zip(aliases, aliases[1:]) if f"[{a}] [{b}]" in answer):
        raise ValueError(f"Adjacent duplicate citations: {v1['id']}")


def build(args: argparse.Namespace) -> tuple[dict[str, list[dict]], dict, dict, list[dict]]:
    inputs = load_inputs(args)
    output: dict[str, list[dict]] = {}
    details: list[dict] = []
    for part in ("train", "validation"):
        output[part] = []
        for v1 in inputs["v1"][part]:
            identifier = v1["canonical_id"]
            record = inputs["canonical"][identifier]
            v2, info = transform_row(v1, record, inputs["audits"].get(identifier), inputs["chunks"])
            output[part].append(v2)
            details.append({"id": v1["id"], "part": part, "mode": v1["response_mode"],
                            "difficulty": v1["difficulty"], "category": v1["category"],
                            "before": v1["messages"][2]["content"], "after": v2["messages"][2]["content"],
                            **info})
    citations = [len(item["cited_chunks"]) for item in details]
    cited = [count for count in citations if count]
    unresolved = [{"id": item["id"], "status": item["status"],
                   "unresolved_fact_indices": item["unresolved_facts"],
                   "uncited_factual_paragraph_indices": item.get("uncited_factual_paragraph_indices", [])}
                  for item in details if item["status"] in {"partial", "unresolved"}]
    def grouping(field: str) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for item in details:
            groups[str(item[field])].append(item)
        return {key: {"rows": len(group), "cited_answers": sum(bool(x["cited_chunks"]) for x in group),
                      "citations": sum(len(x["cited_chunks"]) for x in group)}
                for key, group in sorted(groups.items())}
    hashes = {"train_sft_v2_sha256": hashlib.sha256(jsonl_bytes(output["train"])).hexdigest(),
              "validation_sft_v2_sha256": hashlib.sha256(jsonl_bytes(output["validation"])).hexdigest()}
    stats = {"train_rows": len(output["train"]), "validation_rows": len(output["validation"]),
             "unique_train_canonical_ids": len({r["canonical_id"] for r in output["train"]}),
             "unique_validation_canonical_ids": len({r["canonical_id"] for r in output["validation"]}),
             "answers_with_citation": len(cited),
             "citation_answer_coverage_pct": round(100 * len(cited) / len(details), 2),
             "total_citations": sum(citations),
             "mean_citations_per_cited_answer": round(statistics.mean(cited), 3) if cited else 0,
             "median_citations_per_cited_answer": statistics.median(cited) if cited else 0,
             "unresolved_samples": len(unresolved), "unresolved_rows": unresolved,
             "status_counts": dict(sorted(Counter(item["status"] for item in details).items())),
             "invalid_citations": 0, "citations_outside_context": 0,
             "citations_outside_gold_sources": 0,
             "by_response_mode": grouping("mode"), "by_difficulty": grouping("difficulty"),
             "by_category": grouping("category"),
             "before_after_sha256": {"train_sft_v1": inputs["provenance"]["train_sft_v1_sha256"],
                                     "validation_sft_v1": inputs["provenance"]["validation_sft_v1_sha256"], **hashes}}
    manifest = {"dataset_name": "VN History RAG-SFT citation V2", "schema_version": 2,
                "source_tier": "SILVER", "derivation": "deterministic audit fact support; no model call",
                "citation_policy": "exact required-fact spans; single audited chunk for one factual paragraph; otherwise unresolved",
                "provenance": inputs["provenance"], **hashes,
                # The training loader expects these top-level compatibility keys.
                "split_manifest_sha256": inputs["provenance"]["split_manifest_sha256"],
                "train_sft_sha256": hashes["train_sft_v2_sha256"],
                "validation_sft_sha256": hashes["validation_sft_v2_sha256"],
                "corpus_sha256": inputs["provenance"]["corpus_sha256"]}
    return output, stats, manifest, details


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> dict:
    if args.action == "derive" and not args.dry_run and args.output_dir.exists():
        raise FileExistsError(f"V2 output already exists: {args.output_dir}")
    output, stats, manifest, details = build(args)
    if args.action == "validate":
        split_copy = args.output_dir / "split_manifest.json"
        if sha256(split_copy) != manifest["split_manifest_sha256"]:
            raise ValueError("V2 split manifest copy differs from frozen V1")
        for part in ("train", "validation"):
            path = args.output_dir / f"{part}_sft.jsonl"
            if jsonl(path) != output[part] or len(read_sft(path)) != len(output[part]):
                raise ValueError(f"V2 {part} differs from deterministic reconstruction")
            if sha256(path) != manifest[f"{part}_sft_v2_sha256"]:
                raise ValueError(f"V2 {part} hash mismatch")
        if (json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8")) != manifest or
            json.loads((args.output_dir / "stats.json").read_text(encoding="utf-8")) != stats):
            raise ValueError("V2 manifest/stats provenance mismatch")
    elif not args.dry_run:
        if args.output_dir.exists():
            raise FileExistsError(f"V2 output already exists: {args.output_dir}")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".citation-v2-", dir=args.output_dir.parent) as temporary:
            staging = Path(temporary)
            for part in ("train", "validation"):
                _write(staging / f"{part}_sft.jsonl", jsonl_bytes(output[part]))
                if len(read_sft(staging / f"{part}_sft.jsonl")) != len(output[part]):
                    raise ValueError("Training loader rejected V2 rows")
            _write(staging / "split_manifest.json", (args.train_split.parent / "split_manifest.json").read_bytes())
            _write(staging / "manifest.json", (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            _write(staging / "stats.json", (json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            staging.rename(args.output_dir)
    examples = []
    for status in ("cited", "paragraph_cited", "partial", "unresolved", "unanswerable_unchanged"):
        example = next((item for item in details if item["status"] == status), None)
        if example:
            examples.append({key: example[key] for key in ("id", "status", "before", "after")})
    report = {key: value for key, value in stats.items() if key != "unresolved_rows"}
    report.update(action=args.action, dry_run=bool(args.dry_run), output_dir=str(args.output_dir),
                  provenance=inputs_provenance_summary(manifest["provenance"]), examples=examples)
    return report


def inputs_provenance_summary(provenance: dict) -> dict:
    return {key: value for key, value in provenance.items() if key != "audit_file_sha256"} | {
        "audit_files": len(provenance["audit_file_sha256"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("derive", "validate"))
    for name in ("train-sft-v1", "validation-sft-v1", "canonical", "train-split",
                 "validation-split", "corpus", "audit-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing V2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "validate" and args.dry_run:
        raise SystemExit("--dry-run applies only to derive")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(f"[citation-v2] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
