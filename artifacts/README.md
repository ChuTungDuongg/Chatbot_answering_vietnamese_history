# Deployment artifacts

[Về README gốc](../README.md)

Thư mục `artifacts/` là vị trí runtime local tìm model, corpus, retrieval indexes và inference config đã export từ Phase 10. Git chỉ giữ tài liệu này và placeholder [`vn_history_deployment/manifest.json`](vn_history_deployment/manifest.json); model/index thật quá lớn nên không được commit.

## Trạng thái trong Git

`vn_history_deployment/manifest.json` hiện là file rỗng để giữ cấu trúc thư mục. Nó không đủ để chạy `retrieval-only` hoặc `full`.

| Mode | Cần deployment bundle |
|---|---:|
| `api-only` | Không |
| `retrieval-only` | Có corpus, config, FAISS và BM25S |
| `full` | Có toàn bộ retrieval artifacts và merged model |

## Cấu trúc runtime yêu cầu

```text
artifacts/vn_history_deployment/
├── EXPORT_SUCCESS.txt
├── manifest.json
├── config/
│   └── inference_config.json
├── corpus/
│   └── vn_history_rag_chunks_enriched.jsonl
├── retrieval/
│   ├── faiss/
│   │   ├── chunks.index
│   │   └── manifest.json
│   └── bm25s_index/
│       ├── phase9_manifest.json
│       └── ...
├── model/
│   └── qwen2_5_3b_vnhistory_stage12_merged/
└── evaluation/
    ├── benchmark_results_v3_unique_batched.jsonl
    └── benchmark_summary_v3_unique_batched.csv
```

Các đường dẫn này được định nghĩa tại [`../app/config.py`](../app/config.py) và được kiểm tra khi [`../app/services/rag_service.py`](../app/services/rag_service.py) khởi động.

## Nguồn tạo artifact

1. Phase 8 tạo enriched corpus.
2. Phase 9 tạo FAISS, BM25S, inference config và benchmark output.
3. Phase 10 merge model Stage 1 + Stage 2 và export deployment bundle.
4. Bundle được copy vào thư mục này khi chạy local hoặc upload lên Modal Volume `vn-history-artifacts`.

Trên Modal, Volume được mount tại `/artifacts`, nên `ARTIFACT_ROOT` là `/artifacts/vn_history_deployment`. Local mặc định dùng `./artifacts/vn_history_deployment`.

## Kiểm tra tính toàn vẹn

Chạy từ repository root:

```powershell
modal run modal_artifact_sanity.py
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py
```

Sanity check xác nhận corpus count, unique `chunk_id`, FAISS vector count, BM25S manifest, model shards và success marker khớp nhau.

## Quy tắc khi cập nhật

- Không force-add model weights, FAISS index, BM25S data hoặc corpus lớn vào Git.
- Không sửa thủ công manifest để vượt validation; hãy export lại từ notebook tạo artifact.
- Chỉ chạy `modal_fix.py` khi sanity check xác nhận model shard trên Volume bị sai tên.
- Khi đổi layout hoặc tên file, cập nhật đồng thời Phase 10 export, `app/config.py`, sanity scripts và README.
