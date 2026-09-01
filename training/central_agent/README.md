# Central Qwen3-8B agent training

Package này huấn luyện **một** central history/tool agent từ canonical trajectories. Source of truth cho loss là `training.trajectory_dataset.preprocess.build_canonical_sft_example`: mọi assistant tool call và final answer được supervise; system, user và tool observation được mask. Không packing sequence.

## Modules

```text
central_agent/
├── config.py      # argparse, JSON config, validation, QLoRA/LoRA settings
├── data.py        # Drive/dataset paths, canonical validation, SHA256, leakage, tokenizer audit
├── runtime.py     # GPU/disk inspection, checkpoint discovery, resume guard, manifest
├── engine.py      # tokenizer/model/Trainer wiring, metrics, best/final adapters
├── cli.py         # orchestration and fail-fast order
└── configs/       # explicit informational starting profiles
```

`python -m training.train_qwen3_8b_agent` remains the stable compatibility command. `python -m training.central_agent.cli` is equivalent.

## Colab and Drive

Mount Drive outside the trainer:

```python
from google.colab import drive
drive.mount("/content/drive")
```

The core package never imports `google.colab`. With `--drive-root`, relative dataset/output paths resolve under that root, and checkpoints are written directly to persistent Drive storage.

## Validation-only commands

Dry-run validates paths, canonical JSONL rows, distributions, hashes, leakage, and output writability. It loads neither tokenizer nor model:

```bash
python -m training.train_qwen3_8b_agent \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --dry-run
```

Tokenizer preflight runs the exact training preprocessing path but never loads model weights:

```bash
python -m training.train_qwen3_8b_agent \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --max-seq-length 4096 \
  --preflight-only
```

## L4 starting command

These values are conservative starting points, not claimed optimal:

```bash
python -m training.train_qwen3_8b_agent \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --model-id Qwen/Qwen3-8B \
  --max-seq-length 4096 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --num-train-epochs 3 \
  --optim paged_adamw_8bit \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --eval-steps 50 \
  --save-steps 50 \
  --logging-steps 5 \
  --save-total-limit 3 \
  --evaluate-test-after-train \
  --test-diagnostics
```

`--test-diagnostics` is optional and disabled by default. It requires `--evaluate-test-after-train` and writes `test_diagnostics.json` after training, using the same best model that normal held-out test evaluation sees. The report contains streaming, teacher-forced token NLL/perplexity/accuracy and row-level exact-match diagnostics for tool-call and final-answer assistant spans, plus task/source breakdowns. These are not free-generation accuracy metrics. Use `--test-diagnostics-max-samples N` for a deterministic prefix of the already-selected test split; normal `test_metrics.json` still evaluates the complete selected split.

Resume after disconnect using the same hyperparameters:

```bash
python -m training.train_qwen3_8b_agent \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --auto-resume \
  --max-seq-length 4096 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --num-train-epochs 3
```

Auto-resume only inspects `checkpoint-<step>` inside that run directory. It refuses model, LoRA, QLoRA, sequence-length, critical optimizer-setting, or train/validation SHA256 mismatch. Dataset mismatch requires the explicit `--allow-resume-data-mismatch` override.

## JSON config and hardware profiles

Precedence is hardcoded defaults, then JSON config, then explicit CLI flags:

```bash
python -m training.train_qwen3_8b_agent \
  --config training/central_agent/configs/l4_24gb.json \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --learning-rate 8e-5
```

Available starting profiles are `l4_24gb.json`, `a100_40gb.json`, and `a100_80gb.json`. They never auto-apply. Inspect hardware-based informational output with:

```bash
python -m training.train_qwen3_8b_agent --print-recommended-config
python -m training.train_qwen3_8b_agent --config training/central_agent/configs/l4_24gb.json --dump-config
```

Measure real memory before increasing microbatch. `--attn-implementation auto` resolves to SDPA and does not require flash-attn; an explicit unavailable `flash_attention_2` request fails instead of silently falling back. Full determinism can reduce performance.

## Output contract

```text
training_runs/<run-name>/
├── run_manifest.json
├── training_log.jsonl
├── checkpoint-*/
├── best_adapter/
├── final_adapter/
├── tokenizer/
├── train_metrics.json
├── validation_metrics.json
├── test_metrics.json          # only when requested and a test split exists
└── test_diagnostics.json      # optional teacher-forced held-out diagnostics
```

Only PEFT adapters and tokenizer/config metadata are saved; base Qwen weights are not duplicated. `final_adapter` is captured from the actual last optimized in-memory state, immediately before Transformers reloads the best checkpoint for `load_best_model_at_end`. `best_adapter` is copied from `state.best_model_checkpoint`; therefore the two adapters may intentionally differ. `run_manifest.json` records both global steps and artifact sources.
