# 🧱 Shared Training Library

[⬅️ Training overview](../README.md)

- `cli.py`: thêm/validate các cờ training dùng chung; giữ alias CLI cũ.
- `qlora.py`: NF4 4-bit, double quantization và LoRA target modules.
- `trainer.py`: `TrainingArguments`, reproducible seed, checkpoint policy và `training_log.jsonl` với GPU telemetry.
- `datasets.py`: load chat rows, lấy user/assistant pair và deterministic train/eval/test split.
- `jsonl.py`: UTF-8 JSONL reader/writer dùng chung.

Không import Transformers/Torch nặng ở module load nếu CLI chỉ chạy `--help` hoặc `--dry-run`.
