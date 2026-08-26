from __future__ import annotations

import argparse
import time

from training.common.jsonl import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple retrieval/chat benchmark harness.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/v1/retrieve")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import requests

    rows = read_jsonl(args.questions)
    latencies = []
    for row in rows:
        started = time.perf_counter()
        response = requests.post(args.endpoint, json={"question": row.get("question") or row.get("user"), "final_k": 6}, timeout=60)
        response.raise_for_status()
        latencies.append((time.perf_counter() - started) * 1000)
    print({"count": len(latencies), "avg_ms": sum(latencies) / max(len(latencies), 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



