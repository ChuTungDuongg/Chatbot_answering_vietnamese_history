# Corpus, model artifacts và khôi phục

Git chứa mã nguồn, cấu hình mẫu, metadata model và một số kết quả evaluation đã lưu. Corpus/index lớn, adapter weights, model cache, dữ liệu annotation/training sinh ra và backup nằm ngoài Git. Clone repository không cung cấp đủ artifacts cho full inference. Startup không xây lại hoặc sửa artifacts.

## Layout runtime

```text
artifacts/
  corpus_v1/                         # ARTIFACT_ROOT mặc định local
    documents.jsonl                  # bài viết và provenance
    chunks.jsonl                     # retrieval theo thứ tự cố định
    retrieval/
      faiss/{chunks.index,manifest.json}
      bm25s_index/{*.npy,params.index.json,vocab.index.json,phase9_manifest.json}
      qdrant/manifest.json            # metadata cho collection bên ngoài
      index_manifest.json
    runtime/{inference_config.json,manifest.json}
  vn_history_deployment/             # V0 khi chọn rõ ràng
    corpus/vn_history_rag_chunks_enriched.jsonl
    retrieval/{faiss,bm25s_index}/...
    config/inference_config.json
    manifest.json
  models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/
    adapter/{adapter_config.json,adapter_model.safetensors}
    manifest.json
```

`api-only` không cần retrieval artifacts. `retrieval-only` cần corpus, dense index/manifest, BM25S và retrieval config; `full` cần thêm Qwen weights/tokenizer và adapter nếu dùng SFT. `documents.jsonl` phục vụ provenance/build; `chunks.jsonl` là đầu vào runtime retrieval.

## Danh tính bản Corpus V1 đã lưu

Các số dưới đây mô tả release V1 trong runtime/index manifests, không phải kết quả chất lượng hay xác nhận trạng thái server bên ngoài.

| Thuộc tính | Giá trị |
| --- | --- |
| Dataset nguồn | `undertheseanlp/UVW-2026`, config `default` |
| Dataset revision | `a0a79294e4568137e25828bb3f2a4cde8546e1fb` |
| Số dòng nguồn | 1.118.224: train 894.579 / validation 111.822 / test 111.823 |
| Documents giữ lại | 145.648; 926.690.133 bytes |
| Retrieval chunks | 624.288; 1.331.391.190 bytes |
| Chunking | E5 tokenizer, tối đa 384 tokens, overlap cấu hình 48 tokens |
| History filter | `broad_history_category_v3` |
| Embedding | `intfloat/multilingual-e5-base`, 768 dimensions, normalized float32 |
| E5 revision | `d128750597153bb5987e10b1c3493a34e5a4502a` |
| FAISS | `IndexFlatIP`, 624.288 vectors |
| BM25S | 624.288 records; Lucene, k1=1.5, b=0.75, delta=0.5 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Reranker revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Qdrant | `vn_history_v1_e5`, named vector `dense_e5`, 768 dimensions, COSINE, full precision |

Corpus SHA-256: `4d1700c20c25e8f2c77b801e6ef6242224a8f59a37469ab13fdbe1dc62266255`.

Ordered chunk-ID SHA-256: `b42f792bc8b59795861a5329df5fc0835abdfd3449028dcf2eef8e86f3b55d8f`.

Các partition UVW là provenance dữ liệu nguồn; chúng không phải TRAIN/VALIDATION/TEST của benchmark/SFT. Xem [xây corpus](CORPUS_BUILD.md), [Retrieval](RETRIEVAL.md) và [training](TRAINING_AND_EVALUATION.md).

Runtime kiểm tra count, corpus hash, ordered-ID hash và fingerprints của manifests. Thứ tự dòng corpus là hợp đồng chung của FAISS, BM25S và Qdrant point IDs. Không sắp xếp hoặc viết lại JSONL của bundle đã cố định.

## Adapter và kết quả six-way

Hybrid dùng base tokenizer/model `Qwen/Qwen3-4B-Instruct-2507`; Central giữ vanilla Qwen3-8B. Adapter weights phải được cung cấp local hoặc qua Modal Volume. Không tự merge/train hay tải adapter khác khi thiếu file.

Backup adapter ghi fingerprint `742c5ffdfb6f0bc0ff8be65558b1836ad7c008033ad3828473064ebb67fe3ece`, còn benchmark six-way đã lưu ghi `ab11d1371a3045c4be75bbb1612df6785ab13ad6c3b508ca5c56cc30a9f91ebc`. Không coi adapter backup là tái lập chính xác run đó hoặc resume vào output directory có fingerprint khác. Manifest của run và fingerprint runtime thực tế quyết định tính tương thích.

Kết quả đã lưu tại [`six_way_best_b4_ga4_e2`](../artifacts/evaluation/reports/six_way_best_b4_ga4_e2/), gồm summary, metrics và predictions. Đây là kết quả với nhãn SILVER của thí nghiệm đã ghi, không phải điểm chất lượng của mọi deployment hiện tại.

## Khôi phục và đồng bộ Modal

Đặt ba ZIP release vào `portable_backup/` hoặc root repository:

- `vn_history_v1_research_runtime_2026-09-30_release.zip`
- `vn_history_v0_legacy_runtime_2026-09-30_release.zip`
- `vn_history_peft_adapter_2026-09-30_release.zip`

Từ root repository:

```powershell
python -m scripts.restore_portable_backup --restore-line-endings --dry-run
python -m scripts.restore_portable_backup --restore-line-endings
python -m scripts.restore_portable_backup --verify-only
python -m scripts.retrieval.validate_indexes --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --components local
```

Restore kiểm tra manifest/hash và đường dẫn payload, từ chối file khác nội dung. Tùy chọn line endings xử lý metadata JSON qua CRLF checkout khi khớp nội dung chuẩn hóa. Backup không chứa upstream Qwen/E5/BGE model cache hoặc Qdrant snapshot; phải cung cấp riêng.

Để chuyển artifacts đã kiểm tra sang Volume:

```powershell
python -m scripts.upload_modal_volume --include-v0 --dry-run
python -m scripts.upload_modal_volume --include-v0
python -m scripts.upload_modal_volume --include-v0 --verify-only
```

Uploader chỉ gửi file còn thiếu, kiểm tra hash local/remote và từ chối xung đột. `--ipv4` là workaround tùy chọn cho lỗi kết nối upload. Trên Volume, V1 ở `/corpus_v1/`, V0 release ở `/v0/`, adapter ở `/models/...`; container mount thêm tiền tố `/artifacts`. Xem [Modal](DEPLOYMENT.md).

Khôi phục Qdrant là bước riêng: manifest local không chứa vector database. Cần collection cùng danh tính; `build_index --component qdrant` encode corpus lại nên không phải lệnh restore thông thường. Kiểm tra collection bằng smoke read-only trước khi dùng.

## Bảo toàn dữ liệu cũ

[`corpus_preservation_manifest.json`](corpus_preservation_manifest.json) giữ inventory/hash lịch sử V0, source packs và dữ liệu cũ. Snapshot này khác runtime manifest V1. Các mục `retired_legacy` ghi nhận dữ liệu đã ngừng dùng; không đổi hash để làm audit pass.

```powershell
python -m scripts.corpus.audit_corpus
python -m scripts.corpus.audit_corpus --corpus artifacts/corpus_v1/chunks.jsonl --output reports/corpus/v1-audit.json
```

Lệnh đầu kiểm tra bytes tại đường dẫn lịch sử và trả nonzero khi thiếu/khác; V1 chạy được không có nghĩa audit V0 phải pass. `--all-files` thêm các mục inventory, kể cả file mutable; `--strict-new` kiểm tra file mới dưới root đã ghi. Lệnh thứ hai tạo báo cáo mô tả riêng, không sửa corpus.

Giữ nguyên `training/Dataset/`, `training/InvestigatingDataset.zip`, backup ZIP, model/index/corpus và archive local `artifacts/old_corpus/`. V0 có 58.603 chunks; chọn local bằng `ARTIFACT_ROOT=./artifacts/vn_history_deployment` và bỏ override V1. Các role adapters/datasets của kiến trúc agent cũ không thuộc runtime hiện tại; lịch sử Git giữ phần đã loại bỏ.
