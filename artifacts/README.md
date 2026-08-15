# Deployment Artifacts

[Về README gốc](../README.md)

Thư mục `artifacts/` mô tả deployment bundle được Phase 10 export cho backend. Git không chứa
model, corpus và retrieval indexes thật vì các file này lớn; file
[`vn_history_deployment/manifest.json`](vn_history_deployment/manifest.json) hiện chỉ là
placeholder rỗng để giữ cấu trúc thư mục.

## Mode và artifact cần thiết

| `APP_MODE` | Artifact cần có |
|---|---|
| `api-only` | Không cần deployment bundle |
| `retrieval-only` | Corpus, inference config, FAISS và BM25S |
| `full` | Toàn bộ retrieval artifacts và merged model |

Placeholder trong Git không đủ để chạy `retrieval-only` hoặc `full`.

## Contract thư mục

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

Các đường dẫn được định nghĩa trong [`../app/config.py`](../app/config.py) và được load/validate
bởi [`../app/services/rag_service.py`](../app/services/rag_service.py).

## Nguồn tạo bundle

1. Phase 8 tạo enriched corpus.
2. Phase 9 tạo FAISS, BM25S, inference config và benchmark output.
3. Phase 10 merge model Stage 1 + Stage 2 và export bundle hoàn chỉnh.
4. Bundle được đặt tại `artifacts/vn_history_deployment/` khi chạy local, hoặc upload vào
   Modal Volume `vn-history-artifacts`.

Trên Modal, Volume được mount tại `/artifacts`, nên runtime root là
`/artifacts/vn_history_deployment`. Hugging Face cache nằm trên Volume riêng
`vn-history-hf-cache`.

## Cập nhật inference config trên Modal

Runtime dùng file:

```text
/artifacts/vn_history_deployment/config/inference_config.json
```

Nếu đã chuẩn bị một config local mới, ví dụ `inference_config_long.json`, chạy từ repository
root:

```powershell
modal volume get --force vn-history-artifacts vn_history_deployment/config/inference_config.json inference_config.backup.json
modal volume put --force vn-history-artifacts inference_config_long.json vn_history_deployment/config/inference_config.json
modal volume ls vn-history-artifacts vn_history_deployment/config
```

Nếu `modal` chưa có trong `PATH` nhưng repository đang dùng environment `.conda`:

```powershell
.\.conda\Scripts\modal.exe volume put --force vn-history-artifacts inference_config_long.json vn_history_deployment/config/inference_config.json
```

Sau khi upload, dừng và chạy lại `npm run dev`, hoặc redeploy `modal_app.py`, để container
mới load config:

```powershell
modal deploy modal_app.py
```

Không thể chỉ upload config rồi kỳ vọng container đang warm tự reload; `RAGService` đọc config
một lần trong application startup.

Để answer dài hơn và có bố cục Markdown, thường cần cập nhật đồng thời:

- `generation.max_new_tokens`;
- `prompt.max_new_tokens` nếu config có field này;
- `prompt.default_system` với yêu cầu rõ cho các phần answer, lý do/bằng chứng, góc nhìn khác
  và kết luận.

Giữ output contract về source IDs tương thích với parser/guards. Tăng token quá cao làm tăng
latency và VRAM usage; cần smoke test lại sau mỗi thay đổi.

## Manifest và config

Manifest Phase 10 có thể chứa checksum của config tại thời điểm export. Backend hiện kiểm tra
file cần thiết và parse config/manifest, nhưng không so sánh lại config SHA khi startup. Vì vậy,
thay riêng `inference_config.json` có thể làm checksum trong manifest cũ mà runtime vẫn load.

Đây là trade-off có chủ đích khi thử nghiệm config. Với release cần reproducibility, hãy export
lại bundle/manifest hoặc ghi rõ config revision thay vì để checksum lệch.

## Kiểm tra

Chạy từ repository root, từ nhẹ đến nặng:

```powershell
modal run modal_artifact_sanity.py
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py
```

- `modal_artifact_sanity.py`: kiểm tra corpus count, unique `chunk_id`, FAISS count, BM25S
  manifest, model shards và success marker.
- `modal_runtime_sanity.py`: load retrieval runtime và chạy truy vấn mẫu.
- `full_modal_runtime_sanity.py`: load merged model trên GPU và chạy full answer pipeline.

Các script này không thay thế benchmark, load test hoặc test upload/OCR/conversation memory.

## Quy tắc cập nhật

- Không force-add model weights, corpus, FAISS hoặc BM25S data vào Git.
- Không sửa manifest để né validation; export lại khi contract artifact thay đổi.
- Chỉ chạy `modal_fix.py` khi sanity check xác nhận model shard trên Volume sai tên.
- Khi đổi layout, cập nhật đồng thời Phase 10 export, `app/config.py`, sanity scripts và docs.
- Sao lưu config đang chạy trước khi ghi đè trên Volume.
