from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


DEFAULT_QUESTION = "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?"


def _post(base_url: str, path: str, payload: dict[str, Any], client_id: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Client-ID": client_id},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the production Central-mode API smoke and print isolation telemetry.")
    parser.add_argument("--base-url", required=True, help="Deployed API origin, for example https://example.modal.run")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--client-id", default="central-smoke-client")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conversation = _post(args.base_url, "/api/v1/conversations", {"title": "Central smoke"}, args.client_id)
    response = _post(
        args.base_url,
        "/api/v1/chat",
        {
            "conversation_id": conversation["id"],
            "question": args.question,
            "mode": "central",
            "final_k": 6,
            "debug": True,
        },
        args.client_id,
    )
    debug = response.get("debug") or {}
    performance = debug.get("performance") or {}
    provenance = debug.get("answer_provenance") or {}
    central = debug.get("central") or {}
    report = {
        "answer": response.get("answer"),
        "source_ids": [item.get("chunk_id") for item in response.get("sources") or []],
        "tools_called": [item.get("name") for item in central.get("tools") or []],
        "central_model_calls": performance.get("central_model_calls") or provenance.get("central_model_calls"),
        "timeout_stage": performance.get("timeout_stage") or provenance.get("timeout_stage"),
        "central_model_ready": performance.get("central_model_ready") or provenance.get("central_model_ready"),
        "central_model_load_ms": performance.get("central_model_load_ms") or provenance.get("central_model_load_ms"),
        "question_type": performance.get("question_type") or provenance.get("question_type"),
        "comparison_targets": performance.get("comparison_targets") or provenance.get("comparison_targets"),
        "central_tool_schema_count": performance.get("central_tool_schema_count") or provenance.get("central_tool_schema_count"),
        "central_tools_exposed_to_model": provenance.get("central_tools_exposed_to_model"),
        "central_tool_parse_failures": performance.get("central_tool_parse_failures") or provenance.get("central_tool_parse_failures"),
        "central_malformed_tool_calls": performance.get("central_malformed_tool_calls") or provenance.get("central_malformed_tool_calls"),
        "central_input_tokens": performance.get("central_input_tokens"),
        "central_output_tokens": performance.get("central_output_tokens"),
        "central_generation_ms": performance.get("central_generation_ms"),
        "central_tool_ms": performance.get("central_tool_ms"),
        "central_total_latency_ms": performance.get("central_total_latency_ms"),
        "wikipedia_used": any("wikipedia" in str(name) for name in [item.get("name") for item in central.get("tools") or []]),
        "web_used": any("web" in str(name) for name in [item.get("name") for item in central.get("tools") or []]),
        "gpu": (provenance.get("model_placement") or {}).get("gpu_name"),
        "model_placement": provenance.get("model_placement"),
        "central_cache_root": provenance.get("central_cache_root"),
        "central_model_snapshot_resolved": provenance.get("central_model_snapshot_resolved"),
        "central_cache_hit": provenance.get("central_cache_hit"),
        "central_cache_miss": provenance.get("central_cache_miss"),
        "central_model_resolve_ms": provenance.get("central_model_resolve_ms"),
        "central_model_load_ms": provenance.get("central_model_load_ms"),
        "central_adapter_load_ms": provenance.get("central_adapter_load_ms"),
        "research_generation_calls": performance.get("research_generation_calls"),
        "evidence_generation_calls": performance.get("evidence_generation_calls"),
        "history_generation_calls": performance.get("history_generation_calls"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    leaked = any(int(report.get(key) or 0) != 0 for key in (
        "research_generation_calls", "evidence_generation_calls", "history_generation_calls",
    ))
    return 2 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
