# 🛠️ Corpus, Index and Export Scripts

[⬅️ Training overview](../README.md) · [📦 Artifact contract](../../artifacts/README.md)

| CLI | Vai trò |
|---|---|
| `training.scripts.build_corpus` | Gộp chunk packs và dedup theo `chunk_id`. |
| `training.scripts.enrich_corpus` | Thêm year metadata, char length và history score mặc định. |
| `training.scripts.build_index` | Build normalized E5 FAISS IP index và BM25S index. |
| `training.scripts.merge_model` | Merge LoRA adapter vào đúng base model. |
| `training.scripts.export_artifacts` | Xuất ba role adapters 4B + optional future Central V2 adapter 8B + retrieval. |
| `training.scripts.benchmark` | Gọi API retrieval trên bộ question JSONL và đo latency. |

## 🔨 Build tuần tự

```bash
python -m training.scripts.build_corpus \
  --input-dir training/Dataset/Chunk_id \
  --output artifacts/corpus/vn_history_rag_chunks.jsonl

python -m training.scripts.enrich_corpus \
  --input artifacts/corpus/vn_history_rag_chunks.jsonl \
  --output artifacts/corpus/vn_history_rag_chunks_enriched.jsonl

python -m training.scripts.build_index \
  --corpus artifacts/corpus/vn_history_rag_chunks_enriched.jsonl \
  --output-dir artifacts/retrieval
```

Corpus và index count phải khớp. Không sửa manifest để né mismatch.

## 📦 Export

```bash
python -m training.scripts.export_artifacts \
  --model-dir outputs/history_answerer/merged \
  --research-agent outputs/research_agent \
  --evidence-agent outputs/evidence_agent \
  --history-agent outputs/history-answerer-full/adapter \
  --corpus artifacts/corpus/vn_history_rag_chunks_enriched.jsonl \
  --retrieval-dir artifacts/retrieval \
  --output-root artifacts/vn_history_deployment
```

Central V2 chạy base-only mặc định. Sau khi train adapter mới, thêm `--central-agent outputs/central-v2/final_adapter --central-adapter-relative-path adapters/central-v2`; không dùng lại `adapters/central` V1.

Sau export, kiểm tra snapshot cục bộ trước khi upload Modal:

```bash
python scripts/validate_artifact_bundle.py artifacts/vn_history_deployment
```

Không upload riêng adapter/config vào production Volume; `artifact_lock.json` chỉ hợp lệ cho đúng bytes của toàn snapshot.
