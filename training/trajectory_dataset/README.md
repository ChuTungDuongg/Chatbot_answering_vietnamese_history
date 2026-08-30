# Central Qwen3-8B trajectory dataset pipeline (custom builder V4)

This additive pipeline prepares behavioral training examples for one possible future central Qwen3-8B Vietnamese-history agent. It does not replace the current Research Agent, Evidence Critic, or History Answerer. It does not rebuild the retrieval indexes and it never edits the enriched corpus.

All executable code is Python. There are no notebooks.

## Corpus and trajectory are different things

The enriched RAG corpus is the local knowledge collection at `artifacts/vn_history_deployment/corpus/vn_history_rag_chunks_enriched.jsonl`. Each record is a source chunk enriched with identifiers and historical metadata. FAISS, BM25, E5, RRF, and the reranker use that corpus at runtime to find evidence.

A training trajectory is a conversation that teaches behavior. It can show the model how to choose a tool, write a useful query, inspect noisy observations, search again, reject a hard negative, stop when evidence is insufficient, synthesize an answer, and cite evidence. A no-tool trajectory can teach the model to answer a greeting directly.

The corpus is knowledge; the trajectory dataset is behavior. Blindly putting all corpus chunks into QLoRA would encourage memorization and would not teach the runtime decision process.

## Sources and normalization

V1 supports three bounded public sources plus custom corpus-grounded rows:

| Canonical source | Public dataset | Contribution |
|---|---|---|
| `agent_flan` | `internlm/Agent-FLAN` | Generic agent actions, tool selection, negative and no-tool behavior. Reasoning targets are removed by default. Original tools are retained when semantic tool definitions are available. |
| `multi_hop_function_calling` | `khaimaitien/multi-hop-qa-function-calling-format-V1.0` | `retrieve → observation → retrieve again → answer` behavior. Its documented generic `retrieve` function is never renamed to `search_history`. |
| `vietnam_history_200k` | `minhxthanh/Vietnam-History-200K-Vi` | Vietnamese historical answer style in bounded `style_only` mode, or actual/precomputed retrieval observations in `rag_grounded` mode. |
| `custom_history` | Local enriched corpus | Project-specific `search_history` trajectories with real chunk IDs, observations, citations, and source-group metadata. |

These datasets do not share one raw schema. Each has a separate adapter in `adapters/`; incompatible rows become rejected records with a reason. Original split, source metadata, row identity, transformations, and known license are preserved under `provenance`. Missing licenses stay `null`; the pipeline does not invent them.

Agent-FLAN has heterogeneous subsets. The adapter accepts `messages`, `conversations`, and `conversation`; those raw conversation fields are excluded from `source_metadata` to avoid duplicating the transcript. Its default compatible split is `agent_instruct_react` (override with `--split`). Rows that contain semantic calls but omit matching tool definitions are rejected. Text-only action examples remain text-only instead of receiving invented tools. Use `--include-reasoning` only when deliberately evaluating a subset whose reasoning text is suitable as a public training target; the default is `--no-include-reasoning`.

Vietnam-History filtering utilities favor causes, context, significance, comparison, summary, and multi-part explanation and can reject malformed or extreme-length answers. Do not normalize all 200K rows by default. `rag_grounded` is preferred when a compatible retriever or precomputed observation file is available; `style_only` is deliberately bounded in the final mix.

## Canonical format

Stored JSONL remains semantic and does not contain manually baked Qwen tokens:

```json
{
  "id": "traj-custom-history-...",
  "schema_version": "trajectory-v1",
  "source_dataset": "custom_history",
  "task_type": "multihop",
  "uses_tools": true,
  "difficulty": "hard",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_history",
        "description": "...",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {"id": "call_0001", "type": "function", "function": {"name": "search_history", "arguments": {"query": "..."}}}
      ]
    },
    {"role": "tool", "name": "search_history", "tool_call_id": "call_0001", "content": "[...]"},
    {"role": "assistant", "content": "... [chunk-id]"}
  ],
  "provenance": {"source_group": "...", "evidence_ids": ["chunk-id"]}
}
```

The trainer calls `tokenizer.apply_chat_template(messages, tools=tools, ...)`. It masks system, user, and tool-observation spans from loss. Every assistant action is supervised: tool calls and final answers both receive labels.

## Custom builder V4 contract

V4 classifies each selected corpus subject deterministically as `person`, `event`, `organization`, `state`, `dynasty`, `document`, `location`, `date`, or `topic`. Enriched metadata is checked before title cues, and the result is stored in provenance. Task eligibility then excludes semantically invalid combinations: for example, plain people, locations, dates, and general topics cannot seed cause trajectories; analytical summaries avoid location/date/topic pages; compare pairs must be distinct subjects of the same type.

Seed selection streams the entire corpus read-only and retains a bounded deterministic top-hash sample. It is not a sample of only the first `--max-corpus-records` lines. Within a build, normalized questions, repeated article titles, and canonical unordered compare pairs are deduplicated before requested task counts are credited.

Query plans are task-specific. Summary and multihop use separate context/cause and result/significance calls; compare searches each subject; verification searches a proposition derived from seed evidence plus a corroboration facet; hard negatives issue a wrong-facet call followed by a corrective call. Assistant tool calls preserve the requested `top_k` even when retrieval returns fewer rows.

Tool observations use a compact training-only evidence contract. Each result retains its ID, title, task/query-relevant sentences, source identity, optional ranking score, and a small metadata allow-list. Configure the serialized per-observation and per-result limits with:

```text
--observation-char-budget 12000
--max-result-text-chars 1600
```

Deterministic answers select task-relevant sentences only from those compact observations and cite only observed IDs. Verification uses a conservative normalized exact-substring support check. Insufficient-evidence examples contain a concrete deliberately unsupported proposition marked `Z-1901`, never treat merely related chunks as proof, and do not cite unrelated results.

## Installation and help

From the repository root:

```powershell
python -m pip install -r requirements-training.txt
python -m training.trajectory_dataset.cli --help
python -m training.trajectory_dataset.cli build-custom --help
python training/train_qwen3_8b_agent.py --help
```

Importing the package does not load Hugging Face datasets, a retriever, a teacher, or Qwen. Public loading is streaming and starts only in a non-dry-run normalization command.

## Local Windows workflow

Inspect a bounded read-only sample:

```powershell
python -m training.trajectory_dataset.cli inspect `
  --corpus-path artifacts/vn_history_deployment/corpus `
  --max-samples 1000
```

First verify a custom build without loading the production retriever:

```powershell
python -m training.trajectory_dataset.cli build-custom `
  --corpus-path artifacts/vn_history_deployment/corpus `
  --output-dir D:/vn-history/trajectory_dataset_v1 `
  --retrieval-backend project `
  --rerank-batch-size 4 `
  --observation-char-budget 12000 `
  --max-result-text-chars 1600 `
  --num-factual 100 `
  --num-cause 50 `
  --num-significance 50 `
  --num-compare 40 `
  --num-summary 50 `
  --num-multihop 50 `
  --num-verification 30 `
  --num-hard-negative 30 `
  --num-insufficient-evidence 30 `
  --checkpoint-every 25 `
  --dry-run
```

Remove `--dry-run` only when ready to load the existing retrieval artifacts. `--retrieval-backend project` wraps the actual `RAGService`, `HybridRetriever`, and `SearchHistoryTool`; it does not duplicate or rebuild RAG. It expects the corpus to keep the normal sibling artifact layout (`corpus/`, `retrieval/`, `config/`, and manifests).

For cheap tests or a deliberately small offline fixture, use `--retrieval-backend fixture`. That backend is lexical and is not production-parity data. For captured production observations, use:

```powershell
python -m training.trajectory_dataset.cli build-custom `
  --corpus-path artifacts/vn_history_deployment/corpus `
  --output-dir D:/vn-history/trajectory_dataset_v1 `
  --retrieval-backend precomputed `
  --retrieval-results D:/vn-history/retrieval_results.jsonl `
  --resume
```

Each precomputed row is `{"query": "...", "results": [...]}` and must cover every generated query exactly.

Disputed-claim trajectories can optionally include a project-named external step without making a live request during generation. Pass `--external-results external.jsonl`; each row must be `{"tool": "search_wikipedia", "query": "...", "results": [...]}` (or `search_web`). The builder records the real tool name and observation. Without this file, verification examples remain cautious local cross-checks.

## Build public sources separately

Commands are bounded by `--max-samples`; they never request all rows by default. `--max-samples N` means up to N successfully written canonical rows, while `--max-attempts` bounds raw rows scanned (default `max(N*10, N+100)`). Reports distinguish attempted, written, rejected, resume-skipped, and whether the attempt limit was hit.

```powershell
python -m training.trajectory_dataset.cli normalize-public `
  --source agent_flan `
  --split agent_instruct_react `
  --max-samples 2000 `
  --max-attempts 20000 `
  --cache-dir D:/hf-cache `
  --output D:/vn-history/trajectory_dataset_v1/intermediate/agent_flan.jsonl `
  --no-include-reasoning `
  --resume

python -m training.trajectory_dataset.cli normalize-public `
  --source multihop `
  --max-samples 3000 `
  --output D:/vn-history/trajectory_dataset_v1/intermediate/multihop.jsonl `
  --resume

python -m training.trajectory_dataset.cli normalize-public `
  --source vietnam_history `
  --history-mode style_only `
  --max-samples 2000 `
  --output D:/vn-history/trajectory_dataset_v1/intermediate/vietnam_history.jsonl `
  --resume
```

Add `--input-jsonl path/to/raw_fixture.jsonl` to exercise adapters offline without a network call. In `rag_grounded` mode, also select `project`, `precomputed`, or fixture retrieval and provide the corpus where required. `--rerank-batch-size` overrides the loaded project service config in memory only; it never edits `inference_config.json` or any deployment artifact.

## Mix, validate, and split

The checked-in `configs/mix_v1.json` is only a V1 starting point: 55% custom, 17% multi-hop, 12% Agent-FLAN, and 16% Vietnam-History. Ratios are explicit and overrideable.

```powershell
python -m training.trajectory_dataset.cli mix `
  --input custom_history=D:/vn-history/trajectory_dataset_v1/custom_history.jsonl `
  --input multi_hop_function_calling=D:/vn-history/trajectory_dataset_v1/intermediate/multihop.jsonl `
  --input agent_flan=D:/vn-history/trajectory_dataset_v1/intermediate/agent_flan.jsonl `
  --input vietnam_history_200k=D:/vn-history/trajectory_dataset_v1/intermediate/vietnam_history.jsonl `
  --ratio custom_history=0.55 `
  --ratio multi_hop_function_calling=0.17 `
  --ratio agent_flan=0.12 `
  --ratio vietnam_history_200k=0.16 `
  --output D:/vn-history/trajectory_dataset_v1/mixed.jsonl `
  --seed 42

python -m training.trajectory_dataset.cli validate `
  --input D:/vn-history/trajectory_dataset_v1/mixed.jsonl `
  --output D:/vn-history/trajectory_dataset_v1/mixed.validated.jsonl `
  --rejected-output D:/vn-history/trajectory_dataset_v1/rejected.jsonl

python -m training.trajectory_dataset.cli split `
  --input D:/vn-history/trajectory_dataset_v1/mixed.validated.jsonl `
  --output-dir D:/vn-history/trajectory_dataset_v1/final `
  --train-ratio 0.90 `
  --val-ratio 0.05 `
  --test-ratio 0.05 `
  --seed 42
```

Mixing uses deterministic sampling without duplicating rows. Exact normalized user questions and trajectory IDs are deduplicated. Custom provenance retains legacy `source_group` and also records every contributing stable group in `source_groups`. Splitting connects rows that share any group, then assigns the whole connected component to one split. This prevents a compare row using A+B from leaking apart from another row based on B, while preserving legacy single-group behavior and unambiguous official public validation/test splits.

Validation checks roles, final answers, tool definitions, call/result ID consistency, non-empty search queries, unique IDs, provenance, and internal evidence/citation IDs. Every rejected row contains a reason.

## Resume and checkpoints

Public normalization and custom generation append incrementally. `--checkpoint-every N` flushes and fsyncs progress; `--resume` reads completed deterministic IDs and skips them. A stopped process therefore keeps all checkpointed rows and does not regenerate completed records.

Do not combine outputs from different generation configurations under one resumed file. Use a new output directory when changing task counts, seed, source revision, teacher, or retrieval backend.

## Audit before training

The tokenizer-free audit loads no model and checks semantic eligibility, concrete claims, pair consistency, multihop call counts, duplicate questions, grounding, empty observations, observation sizes, and tool-call counts:

```powershell
python -m training.trajectory_dataset.cli audit `
  --input D:/vn-history/trajectory_dataset_v1/custom_history.jsonl `
  --strict-custom
```

An optional tokenizer-only pass renders the canonical chat template and reports token p50/p95/max, overlength rows, lost initial questions, lost tool-call targets, and lost final/all assistant targets. It does not load model weights:

```powershell
python -m training.trajectory_dataset.cli audit `
  --input D:/vn-history/trajectory_dataset_v1/custom_history.jsonl `
  --strict-custom `
  --tokenizer-model-id YOUR_TOKENIZER_ID_OR_PATH `
  --max-seq-length 4096
```

Preprocessing no longer silently left-truncates a row when doing so would damage the initial user question or any assistant action target; compact observations or increase the sequence length instead.

## Optional post-retrieval local teacher

The default is `--teacher-backend none`; no large model is loaded unexpectedly. This mode is the deterministic controller/tool-behavior validation baseline. Deterministic generation, public normalization, fixture/precomputed retrieval, validation, mixing, and splitting all work without a teacher.

Teacher enhancement is answer-only and happens after retrieval. The builder owns the task, question, tool calls, and compact observations. The teacher receives those immutable fields plus allowed evidence IDs and may change only the final assistant answer. Unknown citations, empty output, and invalid rows follow an explicit `fallback` or `reject` policy. A teacher-enhanced answer is not assumed to be better; audit it.

For a standalone pilot that does not regenerate retrieval:

```powershell
python -m training.trajectory_dataset.cli enhance-teacher `
  --input D:/vn-history/trajectory_dataset_v1/custom_history.jsonl `
  --output D:/vn-history/trajectory_dataset_v1/custom_history.teacher.jsonl `
  --teacher-backend local_hf `
  --teacher-model YOUR_LOCAL_TEACHER_ID_OR_PATH `
  --task-type cause --task-type significance --task-type compare `
  --task-type summary --task-type multihop --task-type verification `
  --teacher-device auto --teacher-batch-size 1 `
  --max-new-tokens 512 --temperature 0
```

Compatibility mode on `build-custom` uses the same two-stage process:

```powershell
python -m training.trajectory_dataset.cli build-custom `
  --corpus-path artifacts/vn_history_deployment/corpus `
  --output-dir D:/vn-history/trajectory_dataset_teacher `
  --retrieval-backend project `
  --teacher-backend local_hf `
  --teacher-model YOUR_LOCAL_TEACHER_ID_OR_PATH `
  --teacher-device auto `
  --teacher-batch-size 1 `
  --teacher-failure-policy fallback `
  --max-new-tokens 512 `
  --temperature 0 `
  --dry-run
```

When this mode is enabled, deterministic trajectories are written first, the project retriever is closed, and only then is the local teacher loaded. The intermediate file is `custom_history.deterministic.jsonl`; the final output remains `custom_history.jsonl`. Factual, hard-negative, insufficient-evidence, and no-tool rows remain deterministic by default. Remove `--dry-run` manually when you intend to load the teacher. No paid API is required.

## Colab and Google Drive

The corpus is intentionally not in Git, so cloning the repository in Colab is not enough. Put the deployment artifacts or at least the corpus in your own Drive. Drive mounting is opt-in; `google.colab` is imported only with `--mount-drive`.

```bash
git clone YOUR_REPOSITORY_URL
cd Chatbot_answering_vietnamese_history
python -m pip install -r requirements-training.txt

python -m training.trajectory_dataset.cli build-all \
  --mount-drive \
  --drive-mount-point /content/drive \
  --corpus-path /content/drive/MyDrive/vn-history/artifacts/vn_history_deployment/corpus \
  --output-dir /content/drive/MyDrive/vn-history/trajectory_dataset_v1 \
  --cache-dir /content/drive/MyDrive/vn-history/hf_cache \
  --max-samples-per-source 2000 \
  --retrieval-backend project \
  --teacher-backend none \
  --checkpoint-every 25 \
  --resume \
  --dry-run
```

Paths above are examples; the command does not assume they exist. Remove `--dry-run` only after checking them. Requesting `--mount-drive` outside Colab raises an actionable error rather than hiding the failure.

If the corpus is already mounted externally, omit `--mount-drive` and pass its absolute path. Generated datasets can be written directly to Drive with `--output-dir`.

## Train one Qwen3-8B adapter

Training is a separate manual action. First run a CPU-only configuration/data dry run, which does not load a tokenizer or model:

```bash
python training/train_qwen3_8b_agent.py \
  --model-id Qwen/Qwen3-8B \
  --train-file /content/drive/MyDrive/vn-history/trajectory_dataset_v1/final/train.jsonl \
  --validation-file /content/drive/MyDrive/vn-history/trajectory_dataset_v1/final/validation.jsonl \
  --output-dir /content/drive/MyDrive/vn-history/qwen3_8b_central_qlora \
  --dry-run
```

Then remove `--dry-run` and choose hardware-appropriate settings:

```bash
python training/train_qwen3_8b_agent.py \
  --model-id Qwen/Qwen3-8B \
  --train-file /content/drive/MyDrive/vn-history/trajectory_dataset_v1/final/train.jsonl \
  --validation-file /content/drive/MyDrive/vn-history/trajectory_dataset_v1/final/validation.jsonl \
  --output-dir /content/drive/MyDrive/vn-history/qwen3_8b_central_qlora \
  --learning-rate 1e-4 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-seq-length 4096 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --bf16 \
  --gradient-checkpointing \
  --resume-from-checkpoint /path/to/checkpoint
```

The model ID is configurable and appears in one training CLI, not scattered through dataset files. QLoRA uses PEFT and bitsandbytes. This repository task does not run training.

## Expected output

```text
trajectory_dataset_v1/
├── intermediate/
│   ├── agent_flan.jsonl
│   ├── agent_flan.rejected.jsonl
│   ├── multihop.jsonl
│   ├── multihop.rejected.jsonl
│   ├── vietnam_history.jsonl
│   └── vietnam_history.rejected.jsonl
├── custom_history.jsonl
├── custom_history.rejected.jsonl
├── mixed.validated.jsonl
├── rejected.jsonl
└── final/
    ├── train.jsonl
    ├── validation.jsonl
    ├── test.jsonl
    ├── dataset_stats.json
    └── manifest.json
```

Statistics include rows per source and task, tool/no-tool ratio, single/multi-tool ratio, answer word lengths, trajectory turn counts, observation characters, and observation result counts.

## Troubleshooting

- **Corpus not found:** pass the directory containing `vn_history_rag_chunks_enriched.jsonl` or the file itself. Relative paths resolve from the repository root.
- **Project retriever cannot find FAISS/BM25/config:** point at the corpus inside the complete `vn_history_deployment` layout. The pipeline never rebuilds missing indexes automatically.
- **Outside Colab mount error:** remove `--mount-drive`, or run in Colab and verify `google.colab` is available.
- **Adapter rejected rows:** inspect the emitted `*.rejected.jsonl`; schema mismatches are intentionally visible.
- **Precomputed query missing:** capture every deterministic generated query or select another retrieval backend.
- **Resume seems to skip rows:** deterministic IDs already present in the output are treated as completed. Use a new output directory for a changed configuration.
- **Empty split:** increase source groups or adjust ratios. Tiny sets with fewer than three groups cannot provide meaningful train/validation/test isolation.
- **Chat-template preprocessing error:** use a tokenizer revision that supports the stored semantic `messages`, `tools`, and `tool_calls` format.
- **Question or assistant target would be truncated:** reduce observation budgets or increase `--max-seq-length`; preprocessing refuses silent loss of the initial question or any assistant target.

## Large-file safety

Never commit the enriched corpus, generated trajectory JSONL, caches, model checkpoints, or adapters. The root `.gitignore` protects the normal repository-local heavy paths, but an output path outside those conventions is still the operator's responsibility. The builder opens the canonical corpus only for reading and never enriches, rewrites, deletes, moves, or indexes it.
