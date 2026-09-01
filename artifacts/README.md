# 📦 Deployment Artifact Contract

[🏠 Project README](../README.md) · [☁️ Upload CLI](../scripts/README.md)

Git chỉ giữ tài liệu contract. Model weights, corpus và indexes thật bị `.gitignore` loại khỏi repository.

## 🗂️ Layout chuẩn

```text
artifacts/vn_history_deployment/       # local ARTIFACT_ROOT
├── adapters/
│   ├── research/                      # Qwen3 Research LoRA
│   ├── evidence/                      # Qwen3 Evidence LoRA
│   ├── history/                       # fresh Qwen3 History LoRA
│   ├── central/                       # optional retained V1 baseline; never referenced by default
│   └── central-v2/                    # optional future Qwen3-8B Central V2 LoRA
├── retrieval/
│   ├── faiss/chunks.index
│   ├── faiss/manifest.json
│   └── bm25s_index/
├── corpus/vn_history_rag_chunks_enriched.jsonl
├── config/inference_config.json
├── config/model_registry.json
├── legacy/qwen25_history/model/       # optional benchmark-only baseline
├── manifest.json
├── artifact_lock.json
└── EXPORT_SUCCESS.txt
```

Trên Modal, Volume được mount trực tiếp ở `/artifacts`. Qwen3-4B role base và Qwen3-8B Central base tải/cache ở `/hf-cache`; bundle chỉ chứa adapters.

## 🧱 Artifact theo mode

| Mode | Bắt buộc |
|---|---|
| `api-only` | Không artifact. |
| `retrieval-only` | corpus, config, manifest, FAISS, BM25S. |
| `full` + `legacy-merged` | retrieval artifacts + legacy History model (baseline only). |
| `full` + `transformers`, mọi mode bật | retrieval artifacts + ba role adapters 4B; Central chạy Qwen3-8B base nếu chưa cấu hình adapter V2. |
| `full` + `transformers`, chỉ `central` | retrieval artifacts; Central chạy Qwen3-8B base, không cần adapter role 4B hay Central adapter. |
| `full` + `transformers`, chỉ `hybrid` | retrieval artifacts + History adapter 4B. |
| `full` + `transformers`, chỉ `three_llm` | retrieval artifacts + ba role adapters 4B. |
| `full` + `vllm` | retrieval artifacts + endpoint đã phục vụ ba role names. |

Cả Research, Evidence và History adapter phải khai báo cùng `Qwen/Qwen3-4B-Instruct-2507`; exporter/runtime fail sớm khi metadata lệch base.
Một Central V2 adapter tương lai là tùy chọn và phải khai báo riêng `Qwen/Qwen3-8B`; adapter 4B bị từ chối. `central_adapter_present=false`/`central=null` là bundle hợp lệ. Thư mục V1 `adapters/central` có thể được giữ làm baseline nhưng không được config/registry/manifest/lock tham chiếu.

## 🏗️ Tạo bundle

```bash
python -m training.scripts.export_artifacts \
  --research-agent outputs/research_agent \
  --evidence-agent outputs/evidence_agent \
  --history-agent outputs/history-answerer-full/adapter \
  --corpus artifacts/corpus/vn_history_rag_chunks_enriched.jsonl \
  --retrieval-dir artifacts/retrieval \
  --output-root artifacts/vn_history_deployment
```

## ☁️ Upload

Validate local trước:

```bash
python scripts/validate_artifact_bundle.py artifacts/vn_history_deployment
```

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --local-dir artifacts/vn_history_deployment \
  --remote-dir / \
  --dry-run
```

Bỏ `--dry-run` khi danh sách lệnh đúng. CLI từ chối component upload, validate toàn bộ canonical bundle trước khi ghi Volume và upload lock cuối cùng. Với production update, thêm `--exact-sync --allow-replace-adapter-weights` sau khi đã xem mutation plan.

## ✅ Sanity

```bash
modal volume ls vn-history-artifacts
modal run scripts/modal_artifact_sanity.py
modal run scripts/modal_runtime_sanity.py
```

Artifact sanity kiểm tra layout, base contracts, corpus/index, ba role adapters và Central adapter nếu được cấu hình. Runtime sanity kiểm tra retrieval; mọi lệnh Modal có thể phát sinh quota/chi phí.

## 🔒 Quy tắc

- Corpus count, FAISS `ntotal` và BM25 manifest phải khớp.
- Không sửa manifest để né validation.
- Không upload riêng adapter/config vào một bundle đã khóa; luôn export lại snapshot và upload cả bundle.
- Không force-add `.safetensors`, `.index`, corpus lớn hoặc SQLite vào Git.
- Upload config mới cần restart/redeploy container; runtime không hot-reload config.
