# 📦 Deployment Artifact Contract

[🏠 Project README](../README.md) · [☁️ Upload CLI](../scripts/README.md)

Git chỉ giữ tài liệu contract. Model weights, corpus và indexes thật bị `.gitignore` loại khỏi repository.

## 🗂️ Layout chuẩn

```text
artifacts/vn_history_deployment/       # local ARTIFACT_ROOT
├── history_answerer/
│   ├── model/                         # merged Qwen2.5 Phase 1 + Phase 6
│   └── adapter/                       # optional, không cần nếu model đã merge
├── research_agent/
│   └── adapter/                       # Qwen3 Research LoRA
├── evidence_agent/
│   └── adapter/                       # Qwen3 Evidence LoRA
├── retrieval/
│   ├── faiss/chunks.index
│   ├── faiss/manifest.json
│   └── bm25s_index/
├── corpus/vn_history_rag_chunks_enriched.jsonl
├── config/inference_config.json
├── manifest.json
└── EXPORT_SUCCESS.txt
```

Trên Modal, Volume được mount trực tiếp ở `/artifacts`; vì vậy `/artifacts/history_answerer/model` là model path, không lồng thêm thư mục `vn_history_deployment`.

## 🧱 Artifact theo mode

| Mode | Bắt buộc |
|---|---|
| `api-only` | Không artifact. |
| `retrieval-only` | corpus, config, manifest, FAISS, BM25S. |
| `full` + deterministic | retrieval artifacts + History model. |
| `full` + model | toàn bộ layout ba model. |

Research và Evidence adapter dùng cùng `Qwen/Qwen3-4B-Instruct-2507` base mặc định. Base tải vào `/hf-cache`; không duplicate nó trong artifact Volume.

## 🏗️ Tạo bundle

```bash
python -m training.scripts.export_artifacts \
  --model-dir outputs/history_answerer/merged \
  --research-agent outputs/research_agent \
  --evidence-agent outputs/evidence_agent \
  --corpus artifacts/corpus/vn_history_rag_chunks_enriched.jsonl \
  --retrieval-dir artifacts/retrieval \
  --output-root artifacts/vn_history_deployment
```

## ☁️ Upload

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --local-dir artifacts/vn_history_deployment \
  --remote-dir / \
  --dry-run
```

Bỏ `--dry-run` khi danh sách lệnh đúng. CLI validate local path trước khi ghi Volume.

## ✅ Sanity

```bash
modal volume ls vn-history-artifacts
modal run modal_artifact_sanity.py
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py
```

Artifact sanity kiểm tra layout, manifest/corpus count, index và ba model roles. Runtime sanity có thể tải model/GPU và phát sinh chi phí.

## 🔒 Quy tắc

- Corpus count, FAISS `ntotal` và BM25 manifest phải khớp.
- Không sửa manifest để né validation.
- Không force-add `.safetensors`, `.index`, corpus lớn hoặc SQLite vào Git.
- Upload config mới cần restart/redeploy container; runtime không hot-reload config.
- Sao lưu Volume trước khi dùng `modal_fix.py`; script đó có thể đổi shard files.
