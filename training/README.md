Historical data retained from the former training workspace. The current
baseline does not train or load adapters. `training/Dataset/` and
`training/InvestigatingDataset.zip` are intentionally preserved byte-for-byte.
Corpus and index utilities now live in `scripts/corpus/` and
`scripts/retrieval/`.

The frozen Codex SILVER V1 RAG-SFT preparation is documented in
`docs/SILVER_V1_FREEZE_SFT.md`. Its generated training JSONL remains local
under `training/datasets/vn_history_rag_sft_v1/` and is not human GOLD.
