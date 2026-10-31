# Retrieval và tạo index

Hybrid và Central dùng chung [`HybridRetriever`](../app/rag/retrieval.py). [`RAGService`](../app/services/rag_service.py) tải corpus, dense backends, BM25S, E5 và BGE reranker một lần; server không xây index khi startup.

## Luồng truy vấn

```text
question → domain/query analysis và query variants
         → E5 dense search (FAISS hoặc Qdrant) + BM25S
         → weighted reciprocal-rank fusion
         → cross-encoder rerank
         → metadata signals, deduplication, context diversity
         → selected chunks → grounded generation prompt
```

| Lane | Dense search | Các bước dùng chung |
| --- | --- | --- |
| `faiss` | Normalized E5 → exact `IndexFlatIP` | BM25S, weighted RRF, BGE reranker, context selection |
| `qdrant` | Cùng E5 → Qdrant HNSW (`exact=false`) | Cùng pipeline phía sau |

Qdrant-native sparse/fusion không thuộc hai lane này. Query dùng tiền tố `query: `; passage embedding dùng `passage: {title}\n{text}`. Qdrant lưu vector `dense_e5`, COSINE, float32 không quantization; payload gọn, text đọc từ corpus local. Point ID là số dòng zero-based, tương ứng FAISS/BM25 row. Runtime kiểm tra `chunk_id` của hit với row map.

## Thông số release V1

Thông số của bundle lấy từ `runtime/inference_config.json`, không phải một bộ mặc định dùng cho mọi artifact. [Artifacts](ARTIFACTS.md) ghi model revisions và danh tính corpus.

| Tham số | Giá trị V1 |
| --- | ---: |
| E5 dimensions | 768 |
| `dense_fetch_k` / `bm25_fetch_k` | 80 / 80 |
| `rrf_k` / `rrf_top_k` | 60 / 20 |
| `final_context_k` | 6 |
| `rerank_batch_size` | 32 |
| `max_query_variants` / `query_expansion_weight` | 3 / 0.82 |
| `max_chunks_per_title` / `enable_context_diversity` | 2 / true |
| `metadata_max_bonus` / `intent_facet_bonus` | 0.18 / 0.025 |

API `final_k` mặc định 6, cho phép 1..10. Dense backend được chọn từng request trong số backend đã validate; tham khảo [Cấu hình](CONFIGURATION.md). Không có fallback ngầm giữa Qdrant và FAISS. Các query variants có thể chạy nhiều dense/sparse calls; stage timing cộng active work, không phải toàn bộ E2E.

## Tạo index cho corpus mới

Các lệnh dưới dành cho output mới chưa có index, không phải quy trình startup hay restore bundle đã có. Không ghi đè artifact V1 đã cố định. Corpus xây mới theo [pipeline corpus](CORPUS_BUILD.md).

Preflight read-only, không tải model hoặc tạo output directory:

```powershell
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --device cpu --preflight
```

Preflight kiểm tra schema/IDs/text, duplicate IDs, corpus SHA-256 và ordered-ID SHA-256. RAM chứa tập IDs, không giữ toàn bộ text. FAISS float32 cần `chunk_count × dimension × 4` bytes cho vectors; bản V1 cần 1.917.812.736 bytes trước overhead.

Build FAISS và BM25S theo hai phase:

```powershell
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --model-revision main --embedding-batch-size 64 --device cuda --component faiss
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --embedding-batch-size 64 --device cpu --component bm25
```

`--component all` tạo FAISS + BM25S. `dense-all` encode mỗi batch một lần rồi gửi cùng float32 stream tới FAISS và Qdrant; `all-backends` thêm BM25S. `qdrant` tạo riêng collection. Kết nối và key lấy từ môi trường; collection/output phải mới. Builder từ chối overwrite hoặc tự xóa collection.

Dense build resolve model revision về SHA bất biến, chuẩn hóa embeddings và ghi sidecar với corpus/order hashes, revision, dimension, normalization, prefixes, batch/device và embedding stream hash. Dùng SHA cụ thể qua `--model-revision` để tái lập. Hai dense manifests từ paired build có cùng stream hash. Qdrant manifest ghi cả cấu hình HNSW server trả về; không dùng in-process local Qdrant cho corpus 624k points.

FAISS build theo batch nhưng giữ toàn bộ flat index trong RAM. BM25S giữ token-ID lists/sparse arrays trong khi index, có thể cần high-RAM runtime riêng. Local component ghi vào `.partial` rồi rename khi hoàn tất. Index build không có resume/checkpoint; Qdrant upload bị ngắt có thể để lại collection và `qdrant.partial`. Cần kiểm tra/xử lý rõ ràng trước lần thử mới. Resume của corpus builder không áp dụng cho index build.

## Validate và runtime metadata

Sau khi có local components:

```powershell
python -m scripts.retrieval.validate_indexes --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --components local
python -m scripts.retrieval.prepare_v1_runtime
```

`--components all-backends` kiểm tra thêm Qdrant thật khi đã cấu hình kết nối. Validator kiểm tra count, corpus/order identity và component compatibility; không viết lại sidecars. `prepare_v1_runtime` đọc corpus/index có sẵn, chỉ tạo metadata nhỏ dưới `runtime/`, không copy corpus hoặc build index. Qdrant chỉ được đưa vào runtime metadata khi manifest hoàn chỉnh và hợp lệ; partial directory không bật backend.

Startup V1 kiểm tra manifest fingerprints, FAISS vector count/dimension, BM25S record count và corpus identity. Qdrant kiểm tra collection/count, named vector/distance/dimension, quantization, HNSW settings và hit identity. Đây không phải phép tính checksum lại toàn bộ vectors từ server. `QDRANT_HNSW_EF` là override tùy chọn, cần ghi lại trong thí nghiệm.

Runtime hiện giữ parsed chunks và ID-to-row map trong RAM, bên cạnh FAISS và memory-mapped BM25S; dung lượng JSONL không đại diện lượng RAM cần dùng.

## Đánh giá hai lane

Dense diagnostic so sánh FAISS exact, Qdrant exact và HNSW trên cùng E5 query vectors:

```powershell
python -m scripts.retrieval.compare_dense --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --queries evaluation/datasets/fixtures/questions.jsonl --hnsw-ef 128
```

ANN recall/overlap so với FAISS không phải historical relevance. Với end-to-end evaluation, giữ dataset, corpus/index identities, E5/BGE revisions, BM25/RRF, candidate depths và context policy; chỉ đổi dense backend. Dùng nhãn đã kiểm duyệt để so chất lượng, ghi local/remote Qdrant vì network ảnh hưởng latency. Lệnh headless retrieval và cách diễn giải metric nằm trong [Evaluation](EVALUATION.md); timing nằm trong [Benchmark](BENCHMARKS.md).
