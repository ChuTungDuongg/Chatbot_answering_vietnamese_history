"""Offline evaluation of saved SSE benchmark predictions; no model or app imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.metrics.answer import score_answer
from evaluation.metrics.citations import score_citations
from evaluation.metrics.grounding import score_grounding
from evaluation.metrics.retrieval import score_retrieval
from evaluation.report import render_markdown, summarize
from evaluation.schema import Question, load_questions


ROOT = Path(__file__).resolve().parents[1]
RANK_FIELDS = ("dense_rank", "bm25_rank", "rrf_rank", "reranker_rank", "final_rank",
               "final_retrieval_score", "retrieval_score", "reranker_score", "rrf_score")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("question_id"), str):
                    raise ValueError(f"invalid prediction at line {line_number}")
                records.append(value)
    return records


def _load_source_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    content = path.read_text(encoding="utf-8")
    if content.lstrip().startswith("["):
        values = json.loads(content)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("source IDs JSON must be a string array")
        return set(values)
    return {line.strip() for line in content.splitlines() if line.strip()}


def evaluate_one(question: Question, prediction: dict[str, Any], *,
                 known_source_ids: set[str] | None = None) -> dict[str, Any]:
    sources = prediction.get("sources") or prediction.get("retrieved") or []
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        raise ValueError(f"invalid sources for {question.id}")
    answer = prediction.get("answer") or ""
    if not isinstance(answer, str):
        raise ValueError(f"invalid answer for {question.id}")
    success = prediction.get("success") is True
    diagnostics = [{"chunk_id": source.get("chunk_id"), "source_id": source.get("source_id"),
                    **{name: source.get(name) for name in RANK_FIELDS if name in source},
                    "final_rank": source.get("final_rank", index)}
                   for index, source in enumerate(sources, 1)]
    if success:
        retrieval = score_retrieval(sources, relevant_chunk_ids=question.relevant_chunk_ids,
                                    relevant_source_ids=question.relevant_source_ids)
        answer_metrics = score_answer(answer, question.gold_answer)
        grounding = score_grounding(answer, question.question, sources,
                                    required_facts=question.required_facts,
                                    answerable=question.answerable, in_domain=question.in_domain)
        citations = score_citations(answer, sources, gold_source_ids=question.gold_citation_source_ids,
                                    known_source_ids=known_source_ids,
                                    factual_paragraph_indices=question.factual_paragraph_indices)
    else:
        # Failed requests are counted in success rate, not scored as incorrect
        # historical answers or zero-recall retrieval.
        retrieval = score_retrieval([], relevant_chunk_ids=None, relevant_source_ids=None)
        answer_metrics = score_answer("", None)
        grounding = score_grounding("", question.question, [], required_facts=None)
        citations = score_citations("", [], gold_source_ids=None)
    return {"schema_version": 1, "question_id": question.id, "category": question.category,
            "request_id": prediction.get("request_id"), "mode": prediction.get("mode"),
            "phase": prediction.get("phase"), "run_index": prediction.get("run_index"),
            "success": success, "error": prediction.get("error"), "answer": answer if success else None,
            "retrieval_diagnostics": diagnostics, "retrieval": retrieval,
            "answer_metrics": answer_metrics, "grounding": grounding, "citations": citations}


def run(dataset: Path, predictions: Path, output: Path, *, phase: str = "warm",
        source_ids: Path | None = None) -> Path:
    questions = load_questions(dataset)
    by_id = {question.id: question for question in questions}
    rows = _load_predictions(predictions)
    if phase != "all":
        rows = [row for row in rows if row.get("phase") == phase]
    unknown = sorted({row["question_id"] for row in rows} - set(by_id))
    if unknown:
        raise ValueError(f"predictions contain unknown question IDs: {unknown}")
    if not rows:
        raise ValueError("no predictions selected")
    known = _load_source_ids(source_ids)
    records = [evaluate_one(by_id[row["question_id"]], row, known_source_ids=known) for row in rows]
    summary = summarize(records)
    summary["dataset_question_count"] = len(questions)
    summary["predicted_question_count"] = len({row["question_id"] for row in rows})
    summary["missing_question_ids"] = sorted(set(by_id) - {row["question_id"] for row in rows})
    summary["inputs"] = {"dataset": str(dataset.resolve()), "dataset_sha256": _sha256(dataset),
                         "predictions": str(predictions.resolve()),
                         "predictions_sha256": _sha256(predictions), "phase": phase,
                         "source_ids_sha256": _sha256(source_ids) if source_ids else None}
    benchmark_metadata_path = predictions.parent / "run_metadata.json"
    benchmark_metadata = json.loads(benchmark_metadata_path.read_text(encoding="utf-8")) if benchmark_metadata_path.exists() else {}
    if not isinstance(benchmark_metadata, dict):
        raise ValueError("run_metadata.json must be an object")
    model_ids = sorted({row["model_id"] for row in rows if isinstance(row.get("model_id"), str)})
    model_revisions = sorted({row["model_revision"] for row in rows if isinstance(row.get("model_revision"), str)})
    summary["provenance"] = {
        "benchmark_run_id": benchmark_metadata.get("run_id"),
        "git_commit": benchmark_metadata.get("git_commit"),
        "model_id": benchmark_metadata.get("model_id") or (model_ids[0] if len(model_ids) == 1 else None),
        "model_revision": benchmark_metadata.get("model_revision") or (model_revisions[0] if len(model_revisions) == 1 else None),
        "corpus_hash": benchmark_metadata.get("corpus_hash"),
        "retrieval_index_hash": benchmark_metadata.get("retrieval_index_hash"),
        "generation_settings": benchmark_metadata.get("generation_settings"),
        "retrieval_settings": benchmark_metadata.get("retrieval_settings"),
        "server_hardware": benchmark_metadata.get("server_hardware"),
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_question.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                              allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_markdown(summary, dataset=str(dataset),
                                                          predictions=str(predictions)), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/datasets/fixtures/questions.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("warm", "cold", "warmup", "all"), default="warm")
    parser.add_argument("--source-ids", type=Path, help="Optional corpus source ID file, one ID per line or JSON array")
    args = parser.parse_args(argv)
    print(run(args.dataset, args.predictions, args.output, phase=args.phase,
              source_ids=args.source_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
