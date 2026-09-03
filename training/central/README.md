# Central V2 training preparation

This is the canonical Central V2 preparation and trainer package. Production never imports it. The earlier `training/trajectory_dataset` package retains reusable schema, assistant masking, deduplication and group-splitting utilities, plus explicitly legacy workflows; those legacy mixes are not V2 defaults. Training and evaluation do not import each other.

```text
training/central/
  normalization/  hermes.py, viquad.py, validation.py
  mixing/         mix.py
  data/           prepare.py; generated/ is ignored
  configs/        mix.json, train.json, optional hardware profiles
  train/          cli.py, config.py, data.py, engine.py, runtime.py,
                  diagnostics.py, constants.py, README.md
```

## Intended sources and normalization

Only [Hermes function-calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/blob/main/README.md) and [UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0/blob/main/README.md) are accepted by the V2 entry point. Obtain and review local exports separately, retaining the original source filename and official split. This CLI never downloads datasets.

Hermes preserves tool schemas, structured calls and observed results, canonicalizes roles and stable call IDs, and rejects JSON-mode, hidden reasoning, undefined/malformed calls, mismatched results and invalid ordering. `validation.py` also validates arguments against the declared JSON Schema and rejects external schema references. The single-turn function-calling subset may end with an assistant action; multi-turn rows must resolve pending calls before final answers. Missing observations are never fabricated.

ViQuAD uses a conservative historical-context filter. Accepted rows become a `search_history` call, supplied-context tool observation and assistant answer. Answerable rows use the sentence containing a verified source answer and cite the context's stable source ID. Impossible rows explicitly report insufficient evidence; plausible answers never become generated facts. The filter is a preparation heuristic, not a historical truth label.

`configs/mix.json` is the single ratio source: initially Hermes 65%, grounded ViQuAD 35%. Mixing is seeded, capacity-limited without oversampling/duplicating rows, followed by source-group/question-aware splitting. Exact proportions can vary with capacity and indivisible groups; inspect the preparation manifest and splits. Direct history SFT, Agent-FLAN and V1 mixtures are not accepted as V2 sources.

Future commands (not executed as part of this refactor):

```bash
python -m training.central.data.prepare normalize --source hermes --input local-hermes.jsonl --source-file func-calling.json --split train --output training/central/data/generated/hermes.jsonl
python -m training.central.data.prepare normalize --source viquad --input local-viquad.jsonl --split train --output training/central/data/generated/viquad.jsonl
python -m training.central.data.prepare mix --hermes training/central/data/generated/hermes.jsonl --viquad training/central/data/generated/viquad.jsonl --output-dir training/central/data/generated
python -m training.central.train.cli --config training/central/configs/train.json --dry-run
```

Use `--split validation` / `test` for corresponding official exports; retain all normalized exports in the supplied source pools. Do not relabel official held-out examples as train. Normalization writes explicit rejection reasons; empty/invalid trainer splits are rejected. The real source distributions and curation quality still need review when preparing the actual data.

## Template and training contract

The trainer uses the tokenizer's native `apply_chat_template` with structured `tools` and `enable_thinking=False`, matching Central runtime. It does not manually inject Qwen special tokens or ReAct text. System, user and tool observations are masked; assistant calls and final answers are supervised. Existing preprocessing checks truncation and supervision invariants before training.

`training.central.mixing.mix.assistant_token_share(rows, tokenizer)` audits **actual supervised token counts** by source using that same preprocessing path. Sample ratios are not token ratios. Supply an explicitly loaded/cached tokenizer later; no tokenizer or model was downloaded during this refactor.

`python -m training.central.train.cli` is the canonical training entry point. `python -m training.train_qwen3_8b_agent` is a compatibility re-export/CLI, not another trainer. `--dump-config` and `--dry-run` do not load a tokenizer/model. `--preflight-only` loads the tokenizer but no model weights. Omitting those flags executes training, so use that only in the later authorized training phase.

Config precedence is defaults < JSON config < explicit CLI. Configure seed, learning rate, epochs/max steps, batch/accumulation, sequence length, LoRA rank/alpha/dropout/targets, precision, checkpointing, logging/evaluation/save intervals, output and resume. `--no-load-in-4bit --optim adamw_torch` selects ordinary LoRA; NF4 QLoRA is the starting profile. Hardware presets are explicit suggestions, not measured guarantees. See [trainer details](train/README.md) for resume checks and best/final adapter handling.

## Future artifact handoff

The trainer saves PEFT `final_adapter/`, optionally `best_adapter/`, tokenizer metadata and a run manifest; it does not copy base weights or deploy artifacts. Select and validate the intended adapter later before placing its PEFT files at `/artifacts/adapters/central-v2` using the existing artifact workflow.

- Empty/unset `CENTRAL_AGENT_ADAPTER_PATH`: BASE `Qwen/Qwen3-8B`.
- Explicit `/artifacts/adapters/central-v2`: the future validated V2 adapter.
- Central V1 is not a default or fallback.

No training, full data processing, model download, adapter upload or Modal command was run for this preparation.
