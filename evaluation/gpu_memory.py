"""Low-overhead CUDA memory samples for long, resumable evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MIB = 1024 * 1024


class CudaMemory:
    def __init__(self, device: str, torch_module=None):
        if device != "cuda":
            self.cuda = None
            self.enabled = False
            self.index = self.name = None
            return
        if torch_module is None:
            import torch as torch_module
        self.cuda = torch_module.cuda
        self.enabled = self.cuda.is_available()
        self.index = self.cuda.current_device() if self.enabled else None
        self.name = self.cuda.get_device_name(self.index) if self.enabled else None

    def sample(self) -> dict | None:
        if not self.enabled:
            return None
        index = self.index
        return {"device": index, "device_name": self.name,
                "allocated_mib": round(self.cuda.memory_allocated(index) / MIB, 2),
                "reserved_mib": round(self.cuda.memory_reserved(index) / MIB, 2),
                "peak_allocated_mib": round(self.cuda.max_memory_allocated(index) / MIB, 2),
                "peak_reserved_mib": round(self.cuda.max_memory_reserved(index) / MIB, 2)}

    def reset_peaks(self) -> None:
        if self.enabled:
            self.cuda.reset_peak_memory_stats(self.index)

    def empty_cache(self) -> None:
        if self.enabled:
            self.cuda.empty_cache()


def append_sample(path: Path, record: dict) -> None:
    """Append one durable line, discarding only an interrupted final line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                while end:
                    end -= 1
                    stream.seek(end)
                    if stream.read(1) == b"\n":
                        end += 1
                        break
                stream.truncate(end)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
    return [json.loads(line) for line in data.splitlines() if line]


def summarize(path: Path, system: str | None = None) -> dict:
    rows = [row for row in read_samples(path)
            if row.get("event") == "question" and (system is None or row.get("system") == system)]
    samples = [row["after_cleanup"] for row in rows if row.get("after_cleanup")]
    if not samples:
        return {"questions": len(rows), "cuda_samples": 0}
    result = {"questions": len(rows), "cuda_samples": len(samples)}
    for metric in ("allocated_mib", "reserved_mib"):
        values = [sample[metric] for sample in samples]
        result[metric] = {"start": values[0], "max": max(values), "end": values[-1],
                          "growth_mib": round(values[-1] - values[0], 2),
                          "slope_mib_per_question": round((values[-1] - values[0]) / max(len(values) - 1, 1), 2)}
    result["max_peak_allocated_mib"] = max(sample["peak_allocated_mib"] for sample in samples)
    result["max_peak_reserved_mib"] = max(sample["peak_reserved_mib"] for sample in samples)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize six-way GPU memory samples")
    parser.add_argument("file", type=Path, help="Path to gpu_memory.jsonl")
    parser.add_argument("--system")
    args = parser.parse_args()
    print(json.dumps(summarize(args.file, args.system), indent=2))


if __name__ == "__main__":
    main()
