"""Capture synthetic live SSE answers and flag a few high-risk Waterloo errors.

Advisory lexical checks, not an entailment judge or a replacement for human review.
The captured output can contain retrieved public source text; it is not telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from uuid import uuid4

from benchmarks.latency.sse_client import measure_request
from evaluation.metrics.citations import score_citations

WATERLOO_QUESTION = ("Vì sao Napoleon thua trận Waterloo? Nếu Grouchy dẫn quân tới tiếp viện "
                     "trong khi đi truy kích quân Phổ thì cục diện có đảo chiều không?")


def review_waterloo(answer: str, sources: list[dict]) -> list[str]:
    flags = []
    text = answer.casefold()
    if re.match(r"\s*(theo thông tin được cung cấp|dựa trên các nguồn|tài liệu cho biết)", text):
        flags.append("formulaic_opening")
    if re.search(r"theo thông tin được cung cấp|dựa trên các nguồn|theo tài liệu", text):
        flags.append("formulaic_phrase")
    if "phối hợp không hiệu quả giữa các lực lượng liên minh" in text:
        flags.append("coalition_coordination_claim_requires_review")
    if re.search(r"quân phổ[^.!?\n]{0,120}không thể tham gia[^.!?\n]{0,40}waterloo", text):
        flags.append("prussian_participation_claim_requires_review")
    # Examine sentence fragments and recognize common explicit negations.
    for sentence in re.split(r"[;!?\n]|(?<=[a-zà-ỹ])\.\s", text):
        if (re.search(r"33[., ]?000|33 nghìn", sentence)
                and re.search(r"thiệt mạng|thương vong|tử trận|bị giết", sentence)
                and not re.search(r"không phải|không có nghĩa|không đồng nghĩa|không thể coi", sentence)):
            flags.append("strength_as_casualties")
            break
    if re.search(r"chắc chắn (?:sẽ )?(?:thắng|chiến thắng|đảo chiều)|tất yếu (?:sẽ )?thắng", text):
        flags.append("certain_counterfactual")
    citations = score_citations(answer, sources)
    if citations["invalid_citation_ids"]:
        flags.append("invalid_citations")
    if sources and not citations["cited_aliases"]:
        flags.append("missing_citations")
    return flags


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="CPU frontend URL for end-to-end checks")
    parser.add_argument("--mode", choices=["central", "hybrid"], default="central")
    parser.add_argument("--retrieval-backend", choices=["faiss", "qdrant"], default="faiss")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    record = measure_request(base_url=args.base_url,
        question={"id": "waterloo-live-regression", "question": WATERLOO_QUESTION},
        mode=args.mode, client_id="live-smoke-" + uuid4().hex,
        timeout=args.timeout, phase="live_smoke", run_index=0,
        request_options={"retrieval_backend": args.retrieval_backend}, strict_contract=True)
    flags = review_waterloo(record["answer"], record["sources"])
    result = {"record": record, "review_flags": flags, "human_review_required": True,
              "notice": "No flags means lexical checks passed, not that factual correctness was proved."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": record["success"], "review_flags": flags,
                      "answer_ttft_ms": record["answer_ttft_ms"],
                      "e2e_ms": record["e2e_ms"], "human_review_required": True}))
    return 0 if record["success"] and not flags else 1


if __name__ == "__main__":
    raise SystemExit(main())
