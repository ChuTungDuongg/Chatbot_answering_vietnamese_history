# History Answerer — fresh Qwen3 grounded QLoRA

Active Phase-6 trains a new, role-specific History adapter directly from the frozen shared base `Qwen/Qwen3-4B-Instruct-2507`. The Qwen2.5 Phase-1/Phase-6 artifacts are legacy benchmark baselines only: they must never be mounted on Qwen3 or used as an init adapter.

The role consumes the question plus upstream-selected textual evidence and writes the final Vietnamese answer with existing source IDs. It does not retrieve, call tools, manage an evidence pool, or receive embedding vectors.

## Prepare and validate

```bash
python -m training.history_answerer.prepare_dataset \
  --input Dataset/merged_jsonl/all_messages.jsonl \
  --output datasets/history_answerer/train.jsonl

python -m training.history_answerer.validate_dataset \
  --dataset datasets/history_answerer/train.jsonl

python -m training.history_answerer.preflight \
  --dataset datasets/history_answerer/train.jsonl \
  --tokenizer-id Qwen/Qwen3-4B-Instruct-2507 \
  --max-length 4096
```

Preparation drops source rows whose gold citation is absent from their own input rather than remapping or fabricating an ID. Preflight loads tokenizer metadata only. It checks every split for empty inputs, invented citation IDs, embedding leakage, assistant truncation, and zero-supervised labels.

## Safe dry-run

```bash
python -m training.history_answerer.train \
  --dataset datasets/history_answerer/train.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir outputs/history-answerer-full \
  --max-length 4096 \
  --dry-run
```

Dry-run validates data, builds group-disjoint splits, runs the exact target-preserving tokenization path, and writes a preflight manifest. It never loads Qwen weights or runs optimization.

## Recommended A100 starting command

```bash
python -m training.history_answerer.train \
  --dataset datasets/history_answerer/train.jsonl \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir outputs/history-answerer-full \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --max-length 4096 \
  --eval-steps 50 \
  --save-steps 50 \
  --bf16 --no-fp16 --gradient-checkpointing
```

Assistant targets are preserved in full. Prompt/context tokens are removed from the left when necessary, an assistant target longer than `max_length` fails, and an all-`-100` sample is impossible. Source-line tokens retain weight 1.6 and answer tokens weight 1.0.

The canonical split keeps groups disjoint and uses three whole groups for each holdout (34/3/3 groups) so `grounded_qa`, `noisy_context`, `insufficient_context`, and `false_premise` are all represented in both eval and test.

`train_instruction_sft.py` and `merge_phase1.py` are legacy Qwen2.5 compatibility utilities. They are not part of active Phase-6. `merge_adapter.py` may merge a trained adapter only into the exact base declared by that adapter; shared multi-LoRA deployment normally exports adapters without bundling or triplicating base weights.
