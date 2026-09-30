# Legacy data cleanup

At the time of this V2 cleanup, the runtime had two vanilla Qwen models and loaded no role adapters. Hybrid now also supports a separately configured PEFT adapter; the legacy role adapters listed here remain unused. These tracked files were removed from the active tree after checking that `app/`, `scripts/`, `evaluation/`, and `benchmarks/` did not import them. Git commit history can still recover every removed tracked file; history was not rewritten.

| Removed path | Former purpose and architecture | Why removed |
| --- | --- | --- |
| `datasets/research_agent/history_trajectories.jsonl` | Research Agent trajectory SFT, former 3-LLM system | No active agent role training or runtime consumer. |
| `datasets/evidence_agent/train.jsonl` | Evidence Agent / critic SFT, former 3-LLM system | No active critic adapter or consumer. |
| `datasets/evidence_agent/train_v2.jsonl` | Revised Evidence Agent / critic SFT, former 3-LLM system | No active critic adapter or consumer. |
| `datasets/history_answerer/train.jsonl` | History Answerer SFT, former 3-LLM system | Hybrid switched to Qwen3-4B; current Hybrid can optionally attach a separate PEFT adapter. |
| `Dataset/merged_jsonl/all_messages.jsonl` | 1,000 RAG-SFT messages | Not the 58,603-chunk V0 runtime corpus; no current consumer. |
| `Dataset/README.md` | Guide for the removed RAG-SFT export | Its active-looking instructions became misleading. |
| `artifacts/reports/evidence_v23_validation.json` | Evidence Agent V2.3 validation | Describes an obsolete training run. |
| `artifacts/reports/research_v23_validation.json` | Research Agent V2.3 validation | Describes an obsolete training run. |
| `artifacts/reports/history_qwen3_preflight.json` | History Answerer Qwen3 preflight | Describes an obsolete training run. |
| `artifacts/training/evidence_agent/baseline_validation.json` | Evidence Agent baseline validation | No active role-training consumer. |
| `artifacts/training/evidence_agent/qwen3_critic_v2/split_manifest.json` | Evidence Agent Qwen3 critic split | No active role-training consumer. |
| `artifacts/training/evidence_agent/v23_dry_run/split_manifest.json` | Evidence Agent V2.3 dry-run split | No active role-training consumer. |
| `artifacts/training/evidence_agent/v23_validation.json` | Evidence Agent V2.3 validation | No active role-training consumer. |
| `artifacts/training/history_answerer/qwen3_dry_run/preflight_manifest.json` | History Answerer Qwen3 preflight | No active role-training consumer. |
| `artifacts/training/history_answerer/qwen3_validation.json` | History Answerer Qwen3 validation | No active role-training consumer. |
| `artifacts/training/research_agent/qwen3_tool_agent/split_manifest.json` | Research Agent Qwen3 split | No active role-training consumer. |
| `artifacts/training/research_agent/v23_dry_run/split_manifest.json` | Research Agent V2.3 dry-run split | No active role-training consumer. |
| `artifacts/training/research_agent/v23_validation.json` | Research Agent V2.3 validation | No active role-training consumer. |
| `tests/trajectory_dataset/fixtures/fake_corpus.jsonl` | Orphaned Research Agent trajectory fixture | No active test imports it. |
| `artifacts/agentic_smoke_result.json` | Old `agentic_rag` smoke output | Reports an inactive public mode. |
| `artifacts/bounded_acceptance_smoke_result.json` | Old `agentic_rag` acceptance smoke output | Reports an inactive public mode. |

The original preservation snapshot still records the removed five SFT JSONL paths and obsolete README with their hashes, now with `retired_legacy` tier. The read-only default audit excludes retired entries while checking every remaining protected entry. The original snapshot counts remain historical; the current protected count is reported by the audit.

## Intentionally retained

`training/Dataset/` and `training/InvestigatingDataset.zip` are protected historical source data. Ignored `Dataset/Samples/` files and protected `artifacts/training/**/*.jsonl` are also retained until a separate source-data review. Neither deployment's `corpus/`, FAISS, or BM25S files changed. The preservation manifest retains their original hashes. V0 is still the active runtime source.
