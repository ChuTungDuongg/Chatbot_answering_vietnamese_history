# VN History SILVER V1: frozen benchmark and RAG-SFT

The Codex-generated **SILVER** collection ends at 60 batches and 3,000 canonical questions. It is not human-reviewed GOLD. The canonical file preserves the completed batch records byte-for-byte, including their `gold_answer` field (an evaluation-schema name, not a human-review claim). No Batch 61 is permitted after the local `frozen_v1.json` marker is created.

## Local outputs

Generated JSONL and manifests follow the existing Git policy and stay local:

- `evaluation/datasets/v1_silver/questions_3000.jsonl`: unchanged canonical records.
- `evaluation/datasets/v1_silver/splits/v1_seed42/{train,validation,test}.jsonl`: canonical subsets.
- `evaluation/datasets/v1_silver/splits/v1_seed42/split_manifest.json`: exact IDs, SHA-256 hashes, distributions, seed, grouping rule.
- `evaluation/datasets/v1_silver/splits/v1_seed42/leakage_report.json`: cross-split source/document and question overlap.
- `training/datasets/vn_history_rag_sft_v1/{train_sft,validation_sft}.jsonl`: derived Qwen-style chat messages.
- `training/datasets/vn_history_rag_sft_v1/{manifest,stats}.json`: SFT provenance and counts.

Run `python -m evaluation.silver_v1 all` once after all 60 completed batches are present. Run `python -m evaluation.silver_v1 validate` to recheck frozen data. Individual `freeze`, `split`, and `derive` actions support recovery if a process stops between stages. Existing frozen files are verified, never silently overwritten. A changed input requires an explicit new dataset/split version.

## Split policy

Seed 42 assigns whole groups keyed by each question's **first relevant source ID**. An evidence-free question would use its own ID as a stable fallback group. The greedy assignment balances count, difficulty, category, question type, answerability and domain while keeping primary-source groups intact. The frozen manifest, rather than the seed alone, is authoritative. No source is primary in two splits. Other sources cited by cross-source questions can occur across splits; their overlap and document overlap are measured in the leakage report. Strict grouping on every cited source is unsuitable here because the largest resulting connected component contains 1,186 questions.

Train/validation/test are **benchmark and model-development splits**, independent of any UVW source partitions. Test is held out. Do not derive paraphrases, variants or other training examples from validation/test answers. Validation SFT has one row per canonical validation ID for checkpoint selection only; it never updates model weights.

## RAG-SFT representation

The SFT tool reads actual Corpus V1 chunk text by stable ID through the read-only corpus lookup. It places all selected chunks in canonical order, untruncated, into the user message. The assistant message is derived only from the canonical SILVER answer and its audited `required_facts`. It does not query a model, add historical facts from memory, or modify canonical records. A detailed answer includes at most two additional audited facts only if they add information absent from the canonical answer. A simple lookup stays short even when the requested mode is detailed. These deterministic variants are presentation variants, **not independent historical questions**.

Exactly one base response mode is assigned to each train ID, with one additional mode for a deterministic subset. No train ID has more than two variants. Mode counts and assignments are frozen in the SFT files and manifest. Validation has exactly one deterministic mode per ID. `response_mode` appears as metadata **and** as a stable Vietnamese instruction in the system message. The three values are `concise`, `standard`, `detailed`; their shared wording is in `app/rag/response_modes.py`. Future inference can pass the same value to `app.rag.prompting.build_messages(..., response_mode=...)`. The UI need not be changed to produce these data.

Mechanical validation checks schema, completed batch audits, corpus IDs and exact source text, disjoint split IDs, primary-source grouping, frozen hashes, reproducible assignments, SFT chat messages, train/validation ID membership, and at most two variants. It cannot prove that every SILVER historical claim is true; human GOLD review remains necessary for trustworthy final evaluation. The small number of unanswerable cases in the canonical data also limits robustness training.
