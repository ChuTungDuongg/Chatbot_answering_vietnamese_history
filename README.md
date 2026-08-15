<div align="center">

# 🇻🇳✨ Chatbot Hỏi Đáp Lịch Sử Việt Nam ✨🇻🇳

### FastAPI · Qwen2.5 · Hybrid RAG · FAISS · BM25S · RRF · Cross-Encoder

<p>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white">
  <img alt="Qwen2.5 3B" src="https://img.shields.io/badge/Qwen2.5-3B--Instruct-7C3AED">
  <img alt="Hybrid RAG" src="https://img.shields.io/badge/RAG-Hybrid-FFB000">
  <img alt="React 19" src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=0B1220">
  <img alt="Vite 8" src="https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-100%20questions-EC4899">
  <img alt="API" src="https://img.shields.io/badge/API-REST%20%2B%20SSE-0EA5E9">
</p>

Chatbot hỏi đáp lịch sử Việt Nam với giao diện React hiện đại, truy xuất bằng chứng, kiểm tra nguồn và niên đại, từ chối câu hỏi ngoài phạm vi, và hỗ trợ phản hồi streaming qua SSE.

[🚀 Chạy ứng dụng](#-chạy-nhanh) · [🧠 Kiến trúc](#-kiến-trúc-hệ-thống) · [📊 Benchmark Phase-9](#-benchmark-phase-9--100-câu--4-cấu-hình) · [🧪 Metrics Phase-6](#-toàn-bộ-metrics-phase-6) · [🔌 API](#-api-reference)

</div>

> [!IMPORTANT]
> Git hiện chỉ chứa khung <code>artifacts/vn_history_deployment/</code>; model, corpus 58.603 dòng, FAISS và BM25S đầy đủ không được commit. Chế độ <code>api-only</code> chạy ngay; <code>retrieval-only</code> và <code>full</code> cần bộ artifact do Phase 10 export.

## 🌟 Điểm nổi bật

- 🧠 Mô hình sinh: <code>Qwen/Qwen2.5-3B-Instruct</code>, merge tuần tự LoRA Phase 1 và Phase 6.
- 🔎 Hybrid retrieval: multilingual E5 + FAISS và BM25S, hợp nhất bằng weighted Reciprocal Rank Fusion.
- 🎯 Reranking: <code>BAAI/bge-reranker-v2-m3</code>, metadata soft boost và chọn context đa dạng.
- 🛡️ Guardrails: phát hiện off-topic, kiểm tra <code>chunk_id</code>, chặn niên đại không có trong evidence, từ chối khi thiếu bằng chứng.
- 🩹 Grounded repair: tự sửa tối đa một lần nếu câu trả lời thiếu ý hoặc vi phạm guard.
- ⚡ FastAPI: REST cho retrieval/chat và SSE streaming chỉ phát câu trả lời sau khi validation hoàn tất.
- 💻 Giao diện React responsive: hội thoại, trạng thái xử lý và bằng chứng truy xuất được trình bày rõ ràng trên desktop lẫn thiết bị di động.
- 🌓 Dark/light mode: chuyển theme ngay trên thanh tiêu đề, tự nhận theme hệ thống ở lần đầu và ghi nhớ lựa chọn trong trình duyệt.
- 🇻🇳 Nhận diện Sử Việt AI: logo SVG riêng đặt cạnh tên chatbot, hiển thị sắc nét ở mọi kích thước.
- 📊 Đánh giá đầy đủ: 100 câu × 4 cấu hình = 400 lượt benchmark Phase 9.
- 🧪 RAG-SFT có theo dõi loss, throughput, source metrics, ROUGE-L, format, abstention và composite score ở Phase 6.

## 🧠 Kiến trúc hệ thống

~~~mermaid
flowchart TD
    U["👤 Người dùng"] --> API["⚡ FastAPI"]
    API --> SYS["GET /health · GET /ready"]
    API --> RET["POST /api/v1/retrieve"]
    API --> CHAT["POST /api/v1/chat · /chat/stream"]

    RET --> HR
    CHAT --> HR

    subgraph HR["🔎 Hybrid Retriever"]
        QA["Phân tích câu hỏi · facets · năm · OOD anchors"]
        MQ["Mở rộng tối đa 3 query"]
        D["E5 + FAISS Top 80"]
        B["BM25S Top 80"]
        RRF["Weighted RRF Top 20"]
        CE["Cross-Encoder rerank"]
        META["Metadata boost + context diversity"]
        CTX["Tối đa 6 evidence chunks"]

        QA --> MQ
        MQ --> D
        MQ --> B
        D --> RRF
        B --> RRF
        RRF --> CE
        CE --> META
        META --> CTX
    end

    CTX --> GEN["🤖 Qwen2.5 Stage 1 + Stage 2"]
    GEN --> GUARD["🛡️ Source/year/format/quality guards"]
    GUARD -->|có lỗi có thể sửa| REPAIR["🩹 Evidence-only repair"]
    REPAIR --> GUARD
    GUARD --> OUT["✅ Answer + sources + status"]
    OUT --> API
~~~

Luồng retrieval giữ đúng cấu hình Phase 9:

~~~text
question
  → intent/OOD guard
  → multi-query
  → FAISS top 80 + BM25S top 80
  → weighted RRF top 20
  → BGE cross-encoder
  → 0,72 × reranker + 0,28 × RRF + metadata bonus
  → context diversity
  → tối đa 6 chunks
~~~

Luồng generation:

~~~text
retrieval evidence
  → prompt động theo loại câu hỏi
  → cắt prompt theo token budget
  → Qwen2.5 sinh answer + source IDs
  → kiểm tra source / year / format / completeness
  → evidence-only repair tối đa 1 lần
  → answer được chấp nhận hoặc safe refusal
~~~

## 🗂️ Sơ đồ thư mục

~~~text
Chatbot_answering_vietnamese_history/
├── app/
│   ├── main.py                         # FastAPI app, lifespan, health/readiness
│   ├── config.py                       # APP_MODE, DEVICE và đường dẫn artifact
│   ├── schemas.py                      # Pydantic request/response schemas
│   ├── api/
│   │   └── routes.py                   # retrieve, chat và validated SSE stream
│   ├── rag/
│   │   ├── retrieval.py                # E5 + FAISS + BM25S + RRF + reranker
│   │   ├── prompting.py                # prompt động, token budget, parse output
│   │   ├── generation.py               # generation, guard và repair
│   │   └── guards.py                   # source/year/OOD/quality checks
│   └── services/
│       └── rag_service.py              # load, validate và shutdown runtime
├── artifacts/
│   └── vn_history_deployment/
│       └── manifest.json               # placeholder trong Git hiện tại
├── frontend/
│   ├── src/
│   │   ├── components/                 # message, composer, status và evidence UI
│   │   ├── services/api.js             # client SSE cho endpoint chat stream
│   │   ├── App.jsx                     # giao diện chính, logo và theme state
│   │   ├── App.css                     # layout responsive và component styles
│   │   └── index.css                   # design tokens cho dark/light mode
│   ├── .env.example                    # mẫu VITE_API_BASE_URL
│   └── package.json                    # React 19, Vite 8 và scripts frontend
├── modal_test/
│   ├── modal_hello.py
│   ├── modal_gpu_test.py
│   └── modal_volume_test.py
├── modal_artifact_sanity.py             # kiểm tra bộ artifact trên Modal Volume
├── modal_runtime_sanity.py              # smoke test retrieval runtime trên Modal
├── full_modal_runtime_sanity.py         # smoke test full RAG trên GPU Modal
├── modal_fix.py                         # sửa tên model shards trên Modal Volume
├── modal_app.py                         # triển khai FastAPI lên Modal với GPU L4
├── Dockerfile                           # image Python 3.11 cho API/Modal
├── Training/
│   ├── Dataset/
│   │   ├── Chunk_id/                    # 31 pack JSONL, tổng 520 dòng
│   │   └── merged_jsonl/
│   │       └── all_chunk_id.jsonl       # 520 dòng, 511 chunk_id duy nhất
│   ├── InvestigatingDataset.ipynb
│   ├── ReadingJSONCorpus.ipynb
│   ├── requirement.txt                  # dependency cho notebook/training
│   └── Training/
│       ├── Phase1_SFT.ipynb
│       ├── Phase2_CreatingCorpus.ipynb
│       ├── Phase3_ChunkExporter.ipynb
│       ├── Phase4_ChunkExporter_extra_topics.ipynb
│       ├── Phase5.ipynb
│       ├── Phase6_RAG_SFT_Qwen2_5_LoRA.ipynb
│       ├── Phase7_InferneceTesting.ipynb
│       ├── Phase8_VN_History_Chunk_Metadata_Enrichment_v4.ipynb
│       ├── Phase9_VN_History_Hybrid_RAG_ToolUse_v2_Grounded_Direct.ipynb
│       └── Phase10_VN_History_FastAPI_Export_From_Phase9 (1).ipynb
├── package.json                         # launcher chung cho frontend + Modal backend
├── package-lock.json                    # dependency lock của launcher root
├── requirements.txt                     # dependency runtime FastAPI/RAG
├── .dockerignore                        # loại artifact/cache khỏi Docker context
└── README.md
~~~

> [!NOTE]
> Corpus mẫu dùng ở Phase 6 có 520 dòng. Corpus deployment của Phase 8–10 là bộ khác, gồm 58.603 chunks và phải khớp 58.603 vectors FAISS cùng 58.603 records BM25S.

## 🧭 Hành trình từ Phase 1 đến Phase 10

| Phase | Vai trò |
|---|---|
| 1 | SFT Qwen2.5-3B bằng QLoRA và assistant-only weighted cross entropy. |
| 2 | Làm sạch, lọc phạm vi và xây corpus lịch sử Việt Nam. |
| 3 | Tìm kiếm, chọn và export các chunk theo chủ đề. |
| 4 | Mở rộng thêm chủ đề và pack dữ liệu. |
| 5 | Gộp các JSONL chunk/messages thành dữ liệu thống nhất. |
| 6 | Merge adapter Phase 1 vào base, tạo LoRA mới và RAG-SFT chọn nguồn + trả lời. |
| 7 | Kiểm thử inference Phase 1 + Phase 2 với FAISS. |
| 8 | Làm giàu metadata precision-first theo <code>chunk_id</code>. |
| 9 | Hybrid RAG hoàn chỉnh, deterministic tool use, guards, grounded repair và benchmark. |
| 10 | Merge model cuối, copy corpus/index/config/evaluation và export artifact cho FastAPI. |

Hai notebook tiện ích dùng để khảo sát dataset và đọc/audit JSON corpus nằm trực tiếp trong <code>Training/</code>.

## 🚀 Chạy nhanh

Ứng dụng được khởi động từ **thư mục gốc** bằng một lệnh duy nhất. Script root dùng <code>concurrently</code> để chạy song song:

- <code>FRONTEND</code>: Vite development server cho giao diện React;
- <code>BACKEND</code>: <code>modal serve modal_app.py</code>, phục vụ FastAPI full RAG trên GPU L4 của Modal.

Máy local không cần GPU để chạy theo luồng này. Model, corpus và các retrieval index được nạp từ Modal Volume.

### 1. Yêu cầu

- Node.js 20.19+ hoặc 22.12+ và npm.
- Python 3.10+ và pip để cài Modal CLI.
- Tài khoản Modal đã đăng nhập bằng <code>modal setup</code>.
- Hai Modal Volume <code>vn-history-artifacts</code> và <code>vn-history-hf-cache</code> đã tồn tại.
- Volume artifact đã chứa đầy đủ deployment bundle của Phase 10.

### 2. Chuẩn bị lần đầu

Chạy các lệnh sau tại thư mục gốc của repository.

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install modal
modal setup

npm install
npm --prefix frontend install
~~~

Linux/macOS:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install modal
modal setup

npm install
npm --prefix frontend install
~~~

Hai lệnh npm cài hai nhóm dependency khác nhau: package root chứa <code>concurrently</code>/<code>cross-env</code>, còn <code>frontend/</code> chứa React, Vite và ESLint.

Có thể kiểm tra bộ artifact trên Modal trước khi chạy ứng dụng:

~~~powershell
modal run modal_artifact_sanity.py
~~~

### 3. Cấu hình URL backend cho frontend

Tạo file <code>frontend/.env</code> từ file mẫu. Vẫn chạy lệnh ở thư mục gốc:

Windows PowerShell:

~~~powershell
Copy-Item frontend/.env.example frontend/.env
~~~

Linux/macOS:

~~~bash
cp frontend/.env.example frontend/.env
~~~

Đặt <code>VITE_API_BASE_URL</code> thành URL development do <code>modal serve</code> cấp, không thêm dấu <code>/</code> ở cuối:

~~~dotenv
VITE_API_BASE_URL=https://your-modal-api.modal.run
~~~

Nếu chưa biết URL ở lần chạy đầu tiên, chạy <code>npm run dev</code> và tìm URL <code>https://...modal.run</code> trong log có nhãn <code>[BACKEND]</code>. Cập nhật <code>frontend/.env</code>, sau đó khởi động lại lệnh để Vite nạp biến môi trường mới. Những lần chạy sau không cần lặp lại bước này nếu URL development không thay đổi.

> [!NOTE]
> Luồng development tích hợp lấy cấu hình backend từ <code>modal_app.py</code>: <code>APP_MODE=full</code>, <code>DEVICE=cuda</code> và <code>ARTIFACT_ROOT=/artifacts/vn_history_deployment</code>. File <code>.env</code> ở root chỉ dành cho trường hợp chạy FastAPI trực tiếp trên máy local, không điều khiển backend Modal của <code>npm run dev</code>.

### 4. Khởi động toàn bộ ứng dụng

Mỗi lần phát triển, mở terminal tại thư mục gốc, bảo đảm lệnh <code>modal</code> khả dụng trong môi trường hiện tại, rồi chạy:

~~~powershell
npm run dev
~~~

Lệnh này khởi động cả frontend và backend. Không cần chuyển vào thư mục <code>frontend/</code> hoặc mở thêm terminal để chạy backend.

Sau khi hai tiến trình sẵn sàng:

- Giao diện chatbot: <code>http://localhost:5173</code>;
- API backend: URL <code>https://...modal.run</code> hiển thị trong log <code>[BACKEND]</code>;
- Swagger UI: <code>&lt;MODAL_API_URL&gt;/docs</code>;
- Health: <code>&lt;MODAL_API_URL&gt;/health</code>;
- Readiness: <code>&lt;MODAL_API_URL&gt;/ready</code>.

Lần khởi động backend đầu tiên có thể lâu hơn vì Modal cần build image hoặc khởi tạo container và nạp model/index. Chỉ bắt đầu chat sau khi endpoint <code>/ready</code> trả <code>ready=true</code>.

Nhấn <code>Ctrl+C</code> trong terminal để dừng. Tùy chọn <code>-k</code> của <code>concurrently</code> bảo đảm tiến trình còn lại cũng được tắt khi một tiến trình kết thúc.

### 5. Các script hữu ích

| Lệnh tại root | Tác dụng |
|---|---|
| <code>npm run dev</code> | Chạy đồng thời frontend và backend Modal. |
| <code>npm run frontend</code> | Chỉ chạy Vite, hữu ích khi backend Modal đã chạy sẵn. |
| <code>npm run backend</code> | Chỉ chạy <code>modal serve modal_app.py</code>. |
| <code>npm run frontend:lint</code> | Kiểm tra mã nguồn frontend bằng ESLint. |
| <code>npm run frontend:build</code> | Build frontend production vào <code>frontend/dist/</code>. |

Biến <code>VITE_API_BASE_URL</code> được Vite nhúng tại thời điểm build, vì vậy cần kiểm tra URL API trước khi chạy <code>npm run frontend:build</code>.

### 6. Chạy API bằng Docker (tùy chọn)

Docker không thuộc luồng <code>npm run dev</code>. Cách này chủ yếu dùng để kiểm tra image API độc lập ở chế độ <code>api-only</code>:

~~~powershell
docker build -t vn-history-rag-api .
docker run --rm -p 8000:8000 vn-history-rag-api
~~~

Image mặc định dùng <code>APP_MODE=api-only</code>, <code>DEVICE=cpu</code> và <code>ARTIFACT_ROOT=/artifacts/vn_history_deployment</code>. Endpoint kiểm tra container là <code>http://localhost:8000/health</code> và <code>http://localhost:8000/ready</code>.

## 📦 Chuẩn bị artifact

Cấu trúc Phase 10 mà runtime mong đợi:

~~~text
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
│       └── ... các file BM25S ...
├── model/
│   └── qwen2_5_3b_vnhistory_stage12_merged/
│       ├── config.json
│       ├── tokenizer_config.json
│       └── ... model shards ...
└── evaluation/
    ├── benchmark_results_v3_unique_batched.jsonl
    └── benchmark_summary_v3_unique_batched.csv
~~~

Để tạo bộ này:

1. Chạy Phase 8 để có enriched corpus.
2. Chạy Phase 9 để có FAISS, BM25S và benchmark.
3. Chạy Phase 10 để merge Stage 1 + Stage 2 và export deployment bundle.
4. Đưa bundle lên Modal Volume <code>vn-history-artifacts</code> tại <code>/vn_history_deployment/</code> để dùng với <code>npm run dev</code>.
5. Giữ Hugging Face cache trong Modal Volume <code>vn-history-hf-cache</code> để giảm thời gian khởi động lại.
6. Chạy <code>modal run modal_artifact_sanity.py</code> để xác nhận artifact hợp lệ.
7. Khởi động bằng <code>npm run dev</code> và chỉ sử dụng chatbot khi <code>GET /ready</code> trả <code>ready=true</code>.

Nếu chạy FastAPI trực tiếp trên máy thay vì qua Modal, copy bundle vào <code>artifacts/vn_history_deployment/</code> ở repository và đặt <code>APP_MODE=retrieval-only</code> hoặc <code>APP_MODE=full</code> trong file <code>.env</code> root. Đây là luồng tùy chọn, không phải luồng development mặc định của launcher npm.

Runtime kiểm tra chặt:

- corpus count khớp manifest;
- <code>chunk_id</code> không trùng;
- số vectors FAISS khớp corpus;
- BM25 manifest count khớp corpus;
- model tồn tại khi chạy <code>full</code>.

## 🔌 API reference

### Endpoint hệ thống

| Method | Path | Mô tả |
|---|---|---|
| GET | <code>/</code> | Tên service, version, environment, mode và docs path. |
| GET | <code>/health</code> | Liveness của tiến trình API. |
| GET | <code>/ready</code> | Trạng thái corpus, FAISS, BM25S, E5, reranker, model và device. |

### Endpoint RAG

| Method | Path | Mode tối thiểu | Mô tả |
|---|---|---|---|
| POST | <code>/api/v1/retrieve</code> | retrieval-only | Trả candidates/evidence sau Hybrid RAG. |
| POST | <code>/api/v1/chat</code> | full | Trả answer, status, sources, latency và debug tùy chọn. |
| POST | <code>/api/v1/chat/stream</code> | full | SSE; chạy retrieval → generation → guards → repair trước khi stream answer đã duyệt. |

Request dùng chung:

- <code>question</code>: 2–1.000 ký tự.
- <code>final_k</code>: 1–10, mặc định 6.
- <code>debug</code>: bật candidates, tool trace và diagnostic fields.

### Retrieval

~~~bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Khởi nghĩa Lam Sơn diễn ra trong bối cảnh nào và kết quả ra sao?\",\"final_k\":6,\"debug\":true}"
~~~

Các trường chính trong response:

- <code>is_ood</code>, <code>ood_reason</code>;
- intent anchors và phân tích facets/năm;
- query variants;
- final context và retrieval scores;
- candidates/tool trace khi <code>debug=true</code>;
- latency phía API.

### Chat

~~~bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?\",\"final_k\":6,\"debug\":true}"
~~~

Response gồm <code>answer</code>, <code>status</code>, danh sách <code>sources</code>, <code>latency_ms</code>, <code>rewrite_used</code> và debug diagnostics.

### Validated SSE stream

~~~bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Vì sao chiến thắng Bạch Đằng năm 938 là một bước ngoặt?\",\"final_k\":6}"
~~~

Event thực tế:

1. <code>status: processing</code>
2. <code>ping</code> mỗi khoảng 8 giây nếu inference còn chạy
3. <code>status: validated</code>
4. một hoặc nhiều <code>answer_delta</code>
5. <code>sources</code>
6. <code>debug</code> nếu yêu cầu
7. <code>done</code> hoặc <code>error</code>

> [!TIP]
> Đây là validated streaming: hệ thống không phát token thô từ model. Chỉ answer cuối đã qua guards/repair mới được chia nhỏ để stream cho giao diện.

## 📊 Benchmark Phase 9 — 100 câu × 4 cấu hình

Nguồn số liệu: output cuối của [Phase 9](Training/Training/Phase9_VN_History_Hybrid_RAG_ToolUse_v2_Grounded_Direct.ipynb) và deployment export <code>benchmark_results_v3_unique_batched.jsonl</code> / <code>benchmark_summary_v3_unique_batched.csv</code>.

### Thiết kế benchmark

- 90 câu lịch sử lấy từ held-out split Phase 6 với seed 42.
- 10 câu ngoài phạm vi được thêm thủ công.
- Phân bố cuối: 60 <code>noisy_context</code>, 18 <code>grounded_qa</code>, 7 <code>insufficient_context</code>, 5 <code>false_premise</code>, 10 <code>off_topic</code>.
- 22/100 câu kỳ vọng hệ thống từ chối.
- Có 2 dòng chứa gold ID cũ không còn trong corpus Phase 8.
- 4 variants, mỗi variant đủ 100/100; tổng 400 result records, không trùng <code>benchmark_uid</code>.
- Hardware notebook: NVIDIA A100-SXM4-80GB, VRAM báo cáo 79,3 GB.
- Batch generation: 16; tối đa 180 new tokens cho benchmark.

Bốn cấu hình:

1. <code>vanilla</code>: Qwen2.5-3B-Instruct nguyên bản.
2. <code>stage1</code>: base + LoRA Phase 1 đã merge.
3. <code>stage12_weights</code>: merge Phase 1 + Phase 6 nhưng không có retrieval context.
4. <code>stage12_full_rag</code>: model đầy đủ + Hybrid RAG + reranker + metadata + guards + repair.

### Chất lượng, hành vi, format và latency

| Variant | N | Semantic ↑ | Token F1 ↑ | ROUGE-L F1 ↑ | Year F1 ↑ | Behavior acc. ↑ | OOD refusal ↑ | History behavior ↑ | Quality pass ↑ | Style spam ↓ | Format ↑ | Avg latency (s) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 100 | 0,8627 | 0,2999 | 0,2116 | 0,5041 | 0,7700 | 0,0000 | 0,8556 | 0,9667 | 0,0000 | N/A | 0,5792 |
| stage1 | 100 | 0,8301 | 0,2030 | 0,1559 | 0,1948 | 0,7800 | 0,0000 | 0,8667 | 0,9000 | 0,0000 | N/A | 0,5768 |
| stage12_weights | 100 | 0,8579 | 0,3429 | 0,2797 | 0,5533 | 0,8200 | 0,2000 | 0,8889 | 0,5667 | 0,4000 | 1,0000 | 0,3732 |
| stage12_full_rag | 100 | **0,8741** | **0,4332** | **0,3706** | **0,6294** | **0,8600** | **1,0000** | 0,8444 | 0,9000 | 0,0111 | 1,0000 | 1,8036 |

### Grounding và retrieval của Full RAG

Các metric dưới đây chỉ có ý nghĩa khi cấu hình thực sự nhận evidence, vì vậy ba baseline được ghi N/A trong CSV.

| Metric | Kết quả |
|---|---:|
| Source precision | 0,3770 |
| Source recall | 0,4321 |
| Source F1 | 0,3930 |
| Source validity | **1,0000** |
| Unsupported-year-free | 0,9300 |
| Retrieval Recall@20 | 0,7654 |
| Final-context recall | 0,6173 |
| MRR@20 | 0,4800 |
| Grounding support | 0,8727 |
| Context title diversity | 0,7333 |
| Rewrite rate | 0,0300 |
| Số query variants trung bình | 1,3800 |

Diagnostic notebook ghi nhận 14 lỗi behavior, 9 câu lịch sử không qua quality critic và 1/90 câu bị style spam. Full RAG đổi lại latency cao hơn baseline vì phải retrieval, rerank, validate và đôi khi repair.

> [!NOTE]
> <code>avg_latency_sec</code> là trường latency do benchmark ghi lại, không phải phép đo tải đồng thời hay SLA production. Benchmark chưa đo p50/p95/p99, cold start, throughput API hoặc mức dùng RAM/VRAM.

### Định nghĩa metric ngắn gọn

- Semantic similarity: cosine similarity giữa answer và reference qua embedder.
- Token F1 / ROUGE-L F1: độ phủ từ và chuỗi con chung dài nhất.
- Year F1: F1 trên tập niên đại 3–4 chữ số.
- Behavior accuracy: trả lời hay từ chối đúng theo <code>expected_refusal</code>.
- Source P/R/F1: source IDs dự đoán so với gold evidence.
- Source validity: không bịa source ID ngoài context.
- Unsupported-year-free: không sinh năm ngoài evidence được trích dẫn.
- Recall@20 / final-context recall / MRR@20: chất lượng retrieval trước và sau reranking.
- Grounding support: semantic support proxy giữa answer và chunks được cite.

<details>
<summary><strong>📚 Xem toàn bộ 100 câu benchmark</strong></summary>

1. <code>noisy_context</code> — Phùng Hưng đánh phủ đô hộ và Cao Chính Bình thất bại ra sao?
2. <code>grounded_qa</code> — Phan Châu Trinh sinh và mất vào thời gian nào?
3. <code>grounded_qa</code> — Tháng 4 âm lịch năm 1289, Trần Hưng Đạo được phong tước gì?
4. <code>noisy_context</code> — Khoa thi đầu tiên dưới thời Lý được mở khi nào và ai là Trạng nguyên đầu tiên?
5. <code>noisy_context</code> — Đoàn 559 được thành lập trong bối cảnh nào năm 1959?
6. <code>noisy_context</code> — Nhà Lý là triều đại nào trong lịch sử Việt Nam?
7. <code>noisy_context</code> — Trần Hưng Đạo nêu thượng sách giữ nước là gì trước khi qua đời?
8. <code>noisy_context</code> — Nguồn gốc và tên gọi ban đầu của Hồ Quý Ly được tài liệu mô tả như thế nào?
9. <code>grounded_qa</code> — Đề Nắm bị giết vào thời điểm nào?
10. <code>grounded_qa</code> — Thành Hoa Lư có diện tích hơn bao nhiêu hecta?
11. <code>noisy_context</code> — Cuộc biểu tình Hưng Nguyên ngày 12/9/1930 diễn ra như thế nào?
12. 🛡️ <code>insufficient_context</code> — Trong ngày đầu tiên SOG vượt biên vào Lào, toàn bộ tên các biệt kích tham gia là gì?
13. 🛡️ <code>insufficient_context</code> — Nguyễn Trãi đã viết chính xác bao nhiêu bức thư chiêu dụ Vương Thông?
14. <code>noisy_context</code> — Cuộc giảng hòa lần thứ nhất giữa nghĩa quân Yên Thế và Pháp diễn ra trong hoàn cảnh nào?
15. 🛡️ <code>off_topic</code> — Viết cho tôi một hàm Python để merge hai dictionary.
16. <code>grounded_qa</code> — Phan Đình Phùng có hiệu và tự là gì?
17. <code>grounded_qa</code> — Ngô Quyền trị vì trong khoảng thời gian nào?
18. <code>grounded_qa</code> — Những ai là con của Đặng Tất và Nguyễn Cảnh Chân trong phong trào Hậu Trần?
19. <code>noisy_context</code> — Giai đoạn đầu của khởi nghĩa Yên Thế có đặc điểm gì?
20. <code>noisy_context</code> — Vì sao Nguyễn Hoàng vào Thuận Hóa và điều này liên hệ thế nào với Đàng Trong?
21. <code>noisy_context</code> — Vì sao cuộc di tản cuối tháng 4 năm 1975 ở Sài Gòn trở nên hỗn loạn?
22. <code>grounded_qa</code> — Cuộc biểu tình Hưng Nguyên diễn ra ngày nào?
23. <code>grounded_qa</code> — Trong buổi lễ ngày 2 tháng 9 năm 1945, Võ Nguyên Giáp giữ chức vụ gì theo tài liệu?
24. <code>noisy_context</code> — Vì sao Võ Nguyên Giáp bị đuổi học ở Quốc học Huế?
25. <code>grounded_qa</code> — Bảo Đại đọc Tuyên ngôn Thoái vị ngày nào?
26. <code>noisy_context</code> — Hồ Quý Ly từng đối phó với Chế Bồng Nga và Chiêm Thành như thế nào?
27. <code>noisy_context</code> — Trong những năm cuối đời, Phan Bội Châu sống và hoạt động ra sao?
28. 🛡️ <code>off_topic</code> — Giá Bitcoin hôm nay là bao nhiêu?
29. <code>noisy_context</code> — Thời Lê Nhân Tông, Nguyễn Thái hậu giữ vai trò gì?
30. <code>noisy_context</code> — Sau khi thu được kế hoạch tiến công Việt Bắc, Bộ Tổng chỉ huy tổ chức các mặt trận như thế nào?
31. <code>noisy_context</code> — Hiệp ước Versailles năm 1787 quy định những gì theo tài liệu?
32. <code>noisy_context</code> — Sự sụp đổ ở Đà Nẵng cuối tháng 3 năm 1975 được mô tả như thế nào?
33. <code>noisy_context</code> — Hồ Chí Minh tìm cách tranh thủ Mỹ và Liên Xô sau Cách mạng tháng Tám như thế nào?
34. <code>noisy_context</code> — Chỉ thị “Nhật - Pháp bắn nhau và hành động của chúng ta” được ban hành khi nào và nhằm mục đích gì?
35. 🛡️ <code>off_topic</code> — Cách nấu bò kho ngon tại nhà như thế nào?
36. <code>noisy_context</code> — Kiều Công Tiễn có vai trò gì trong bối cảnh dẫn tới cuộc đối đầu giữa Ngô Quyền và Nam Hán?
37. 🛡️ <code>false_premise</code> — Chúa Nguyễn là người cai trị Đàng Ngoài và phục tùng trực tiếp Chúa Trịnh phải không?
38. <code>noisy_context</code> — Nghĩa quân Hương Khê tìm cách khắc phục khó khăn về vũ khí như thế nào?
39. <code>noisy_context</code> — Vì sao nhà Minh có ý định đánh Đại Ngu?
40. <code>noisy_context</code> — Khởi nghĩa Tây Sơn năm 1771 lấy danh nghĩa gì?
41. <code>noisy_context</code> — Nguyễn Trãi có quan điểm gì khi một số tướng muốn đánh thành Đông Quan để trả thù quân Minh?
42. <code>grounded_qa</code> — Trận Bô Cô diễn ra vào năm nào?
43. 🛡️ <code>false_premise</code> — Đường Trường Sơn là tuyến hậu cần do Mỹ xây dựng để đưa quân vào miền Bắc Việt Nam phải không?
44. 🛡️ <code>off_topic</code> — Đau đầu và sốt nhẹ thì nên uống thuốc gì?
45. <code>noisy_context</code> — Trần Hưng Đạo có tên thật và tước hiệu là gì?
46. <code>grounded_qa</code> — Khởi nghĩa Yên Thế diễn ra trong khoảng thời gian nào?
47. <code>noisy_context</code> — Đền Vua Lê Đại Hành thờ những nhân vật nào?
48. <code>grounded_qa</code> — Mạc Đăng Dung lập ra nhà Mạc vào năm nào?
49. 🛡️ <code>insufficient_context</code> — Ngô Quyền có chính sách thuế cụ thể nào sau khi lên ngôi?
50. <code>noisy_context</code> — Dương Đình Nghệ đánh đuổi quân Nam Hán năm 931 ra sao?
51. 🛡️ <code>false_premise</code> — Chiến tranh Đông Dương lần thứ nhất bắt đầu năm 1954 sau Hiệp định Genève phải không?
52. <code>noisy_context</code> — Võ Nguyên Giáp giữ vai trò gì trong các cuộc chiến tranh lớn của Việt Nam?
53. <code>noisy_context</code> — Sau Roosevelt, chính sách của Mỹ về Đông Dương chuyển biến như thế nào?
54. <code>noisy_context</code> — Nguyễn Tri Phương và con trai Nguyễn Lâm gặp chuyện gì trong trận Hà Nội năm 1873?
55. <code>noisy_context</code> — Kế hoạch tấn công biên giới của Việt Nam Quang phục Hội sau năm 1914 gặp thất bại gì?
56. <code>noisy_context</code> — Chiến dịch Biên giới Thu đông 1950 là chiến dịch gì và diễn ra khi nào?
57. 🛡️ <code>false_premise</code> — Nhà Lý chấm dứt vì Lý Công Uẩn nhường ngôi cho nhà Tiền Lê phải không?
58. <code>noisy_context</code> — Khởi nghĩa Hương Khê kết thúc trong hoàn cảnh nào?
59. <code>grounded_qa</code> — Phan Bội Châu sinh ở đâu?
60. <code>noisy_context</code> — Nhà Lê sơ được thành lập trong hoàn cảnh nào?
61. <code>noisy_context</code> — Những cải cách ban đầu của Việt Nam Dân chủ Cộng hòa sau ngày 2 tháng 9 năm 1945 là gì?
62. 🛡️ <code>off_topic</code> — So sánh iPhone và Samsung đời mới nhất.
63. 🛡️ <code>insufficient_context</code> — Nguyên văn đầy đủ toàn bộ Hiệp định Genève 1954 là gì?
64. <code>noisy_context</code> — Lực lượng Quân đội Nhân dân Việt Nam tham gia Chiến dịch Biên giới 1950 gồm những đơn vị nào?
65. <code>noisy_context</code> — Đoàn 559 lúc mới thành lập có lực lượng và nhiệm vụ gì?
66. <code>noisy_context</code> — Kết quả trận Như Nguyệt năm 1077 được mô tả như thế nào?
67. <code>noisy_context</code> — Sau khi chiếm Đa Bang, quân Minh tiến vào Đông Đô như thế nào?
68. <code>noisy_context</code> — Trước khi trở thành thủ lĩnh Yên Thế, Hoàng Hoa Thám từng tham gia những lực lượng nào?
69. <code>noisy_context</code> — Vì sao Hiệp ước Versailles 1787 được xem là di họa dù không thành hiện thực?
70. 🛡️ <code>off_topic</code> — Tôi nên làm gì khi người yêu không trả lời tin nhắn?
71. <code>grounded_qa</code> — Năm 866, ai được cho làm Tiết độ sứ Tĩnh Hải quân?
72. <code>noisy_context</code> — Nguyên nhân khởi nghĩa Mai Thúc Loan được tài liệu giải thích thế nào?
73. <code>noisy_context</code> — Nhà Lý bắt đầu suy vong dưới thời Lý Cao Tông như thế nào?
74. <code>grounded_qa</code> — Việt Minh do ai thành lập và vào năm nào?
75. 🛡️ <code>off_topic</code> — Dịch câu "machine learning is useful" sang tiếng Việt.
76. <code>noisy_context</code> — Việt Nam Dân chủ Cộng hòa bảo vệ lợi ích của Pathet Lào và Khmer Issarak như thế nào?
77. 🛡️ <code>insufficient_context</code> — Danh sách đầy đủ mọi trận đánh giữa quân Tây Sơn và Chúa Nguyễn từ 1771 đến 1777 là gì?
78. <code>noisy_context</code> — Cương vực Âu Lạc được tài liệu mô tả như thế nào?
79. 🛡️ <code>off_topic</code> — Giải phương trình x^2 - 5x + 6 = 0.
80. <code>noisy_context</code> — Thực dân Pháp đàn áp phong trào Xô Viết Nghệ Tĩnh như thế nào?
81. <code>grounded_qa</code> — Đại đồn Chí Hòa bị quân Pháp công phá vào ngày nào?
82. <code>grounded_qa</code> — Khúc Hạo mất năm nào?
83. 🛡️ <code>false_premise</code> — Gia Long là vua Tây Sơn đánh bại Nguyễn Ánh để lập nhà Nguyễn phải không?
84. <code>noisy_context</code> — Nguyễn Ánh trốn chạy ở Phú Quốc, Côn Lôn và các đảo ra sao?
85. <code>noisy_context</code> — Đời sau tưởng nhớ Mai Hắc Đế như thế nào?
86. <code>noisy_context</code> — Việt Nam Dân chủ Cộng hòa nhượng bộ thế nào về giới tuyến quân sự?
87. 🛡️ <code>insufficient_context</code> — Nguyên văn đầy đủ truyền thuyết Mỵ Châu - Trọng Thủy là gì?
88. <code>noisy_context</code> — Âu Lạc kế tục nhà nước nào?
89. <code>noisy_context</code> — Võ Nguyên Giáp là ai theo tài liệu?
90. <code>noisy_context</code> — Trương Phúc Loan đã thao túng chính quyền Chúa Nguyễn như thế nào?
91. 🛡️ <code>off_topic</code> — Thời tiết Thành phố Hồ Chí Minh hôm nay có mưa không?
92. <code>noisy_context</code> — Nguồn gốc gia đình ba anh em Tây Sơn được nêu như thế nào?
93. <code>noisy_context</code> — Hiệp định Genève 1954 được ký sau quá trình đàm phán như thế nào?
94. 🛡️ <code>insufficient_context</code> — Danh sách đầy đủ mọi đoàn di dân người Việt vào Nam Bộ từ thế kỷ XVII đến XVIII gồm những ai?
95. <code>noisy_context</code> — Những bảo vật quốc gia nào được lưu giữ ở Cố đô Hoa Lư?
96. <code>noisy_context</code> — Khu di tích Cố đô Hoa Lư nằm ở đâu?
97. <code>noisy_context</code> — Vì sao Hoa Lư được xem là nơi ghi dấu quá trình thống nhất giang sơn thế kỷ X?
98. <code>noisy_context</code> — Người nắm quyền đầu tiên của nhà Tây Sơn là ai và Nguyễn Huệ lên ngôi khi nào?
99. 🛡️ <code>off_topic</code> — Messi đã ghi bao nhiêu bàn ở mùa giải gần nhất?
100. <code>noisy_context</code> — Lãnh thổ Đại Việt thời đầu độc lập được mô tả như thế nào trước khi mở rộng về phương Nam?

🛡️ = <code>expected_refusal=true</code> trong benchmark.

</details>

## 🧪 Toàn bộ metrics Phase 6

Nguồn: output đã lưu trong [Phase 6 RAG-SFT](Training/Training/Phase6_RAG_SFT_Qwen2_5_LoRA.ipynb). Các số dưới đây là kết quả đo thật của notebook, không phải mục tiêu dự kiến.

### Môi trường và cấu hình chạy

| Thuộc tính | Giá trị |
|---|---|
| PyTorch | 2.11.0+cu128 |
| CUDA | Có |
| GPU | NVIDIA A100-SXM4-40GB |
| Compute capability | 8.0 |
| Compute dtype | bfloat16 |
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Max sequence length | 4.096 |
| Epochs | 5 |
| Train batch / device | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | 1,5e-4 |
| Weight decay | 0,01 |
| Warmup ratio | 0,05 |
| Source-line loss weight | 1,6 |
| Answer loss weight | 1,0 |
| LoRA r / alpha / dropout | 32 / 64 / 0,05 |
| Trainable params | 59.867.136 / 3.145.805.824 = 1,9031% |

### Corpus và dataset validation

| Metric | Giá trị |
|---|---:|
| File chunks | 2,00 MB |
| File messages | 9,30 MB |
| Phase 1 adapter | 239,35 MB |
| Số dòng chunks | 520 |
| Unique <code>chunk_id</code> | 511 |
| Message samples | 1.000 |
| Valid records | 1.000 |
| Bad rows | 0 |
| Unique context IDs | 469 |
| Missing context IDs | 0 |

| Split | Tổng | noisy_context | grounded_qa | insufficient_context | false_premise |
|---|---:|---:|---:|---:|---:|
| Toàn bộ | 1.000 | 650 | 200 | 100 | 50 |
| Train | 900 | 585 | 180 | 90 | 45 |
| Eval | 50 | 33 | 10 | 5 | 2 |
| Test | 50 | 32 | 10 | 5 | 3 |

Không có sample nào bị loại sau bước filter length/loss: 900/50/50 trước và sau filter.

### Độ dài token

| Split | N | Min | P50 | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Train | 900 | 391 | 2.553 | 3.127 | 3.415 |
| Eval | 50 | 773 | 2.354 | 3.110 | 3.271 |
| Test | 50 | 856 | 2.472 | 3.038 | 3.226 |

Sanity batch có shape <code>(2, 3128)</code> cho input/attention/labels/loss weights, 185 active label tokens và mean active loss weight 1,233513.

### Training loss và validation loss

| Step | Epoch | Training loss | Validation loss |
|---:|---:|---:|---:|
| 50 | 0,8889 | 2,669996 | 0,298431 |
| 100 | 1,7644 | 1,761843 | **0,288516** |
| 150 | 2,6400 | 0,934587 | 0,318169 |
| 200 | 3,5156 | 0,438666 | 0,391338 |
| 250 | 4,3911 | 0,246027 | 0,413053 |
| 285 | 5,0000 | 0,226130 | 0,423512 |

Best checkpoint theo eval loss: <code>checkpoint-100</code>, với <code>eval_loss=0,2885164320</code>.

### Tổng kết hiệu năng training

| Metric | Giá trị |
|---|---:|
| Global steps | 285 |
| Epoch | 5,0 |
| Aggregate training loss | 1,7191895012 |
| Train runtime | 9.595,5251 giây ≈ 2 giờ 39 phút 55,5 giây |
| Samples/second | 0,469 |
| Steps/second | 0,030 |
| Total FLOPs | 2,1314476738 × 10¹⁷ |

### Generation callback — source, format và abstention

Mỗi lần callback chạy trên 50 eval examples.

| Step | Epoch | Source exact | Precision | Recall | Source F1 | Format OK | Answer nonempty | IDs in context | IDs in corpus | Insufficient empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0,8889 | 0,90 | 0,90 | 0,90 | 0,90 | 1,00 | 1,00 | 1,00 | 1,00 | 0,40 |
| 100 | 1,7644 | 0,96 | 0,96 | 0,96 | 0,96 | 1,00 | 1,00 | 1,00 | 1,00 | 0,80 |
| 150 | 2,6400 | 0,96 | 0,96 | 0,96 | 0,96 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| 200 | 3,5156 | 0,96 | 0,96 | 0,96 | 0,96 | 1,00 | 1,00 | 1,00 | 1,00 | 0,80 |
| 250 | 4,3911 | 0,96 | 0,96 | 0,96 | 0,96 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| 285 | 5,0000 | 0,94 | 0,94 | 0,94 | 0,94 | 1,00 | 1,00 | 1,00 | 1,00 | 0,80 |

### Generation callback — ROUGE, composite và runtime

| Step | ROUGE-L F1 | Composite score | Runtime (s) | Trạng thái |
|---:|---:|---:|---:|---|
| 50 | 0,589027 | 0,893903 | 947,120 | best mới |
| 100 | 0,642660 | 0,938266 | 896,086 | best mới |
| 150 | **0,648870** | **0,938887** | 808,801 | best generation |
| 200 | 0,616756 | 0,935676 | 869,623 | không cải thiện |
| 250 | 0,628284 | 0,936828 | 888,415 | không cải thiện |
| 285 | 0,609518 | 0,921952 | 896,102 | không cải thiện |

Composite score của Phase 6:

~~~text
0,45 × source_f1
+ 0,20 × source_exact
+ 0,15 × format_ok
+ 0,10 × pred_ids_in_context
+ 0,10 × rouge_l_f1
~~~

Best adapter theo generation metric nằm ở step 150, khác best checkpoint theo eval loss ở step 100.

### Test loss cuối

| Metric | Giá trị |
|---|---:|
| Manual weighted test loss | 0,2983313243 |
| Active tokens | 5.787 |
| Weight sum | 7.060,7998 |
| Runtime | 13,7696 giây |
| Test samples | 50 |
| Batch size | 8 |

### Test generation cuối

| Metric | Giá trị |
|---|---:|
| Source exact | 0,96 |
| Source precision | 0,96 |
| Source recall | 0,96 |
| Source F1 | 0,96 |
| Format OK | 1,00 |
| Answer nonempty | 1,00 |
| Pred IDs in context | 1,00 |
| Pred IDs in corpus | 1,00 |
| Insufficient empty rate | 1,00 |
| ROUGE-L F1 | 0,6448342289 |
| Composite score | 0,9384834229 |
| Examples | 50 |
| Runtime | 223,3257 giây |
| Batch size | 8 |
| Max new tokens | 256 |

### Kích thước export và inference sanity

| Metric | Giá trị |
|---|---:|
| Final adapter folder | 239,3666 MB |
| Adapter safetensors | 228,44 MB |
| Tokenizer JSON | 10,89 MB |
| ZIP | 213,5793 MB |
| TF-IDF matrix | 520 × 110.637 |

Điểm đáng chú ý: training loss tiếp tục giảm nhưng validation loss tăng sau step 100; generation composite đạt đỉnh ở step 150. Vì vậy export ưu tiên adapter tốt nhất theo generation metric thay vì checkpoint cuối.

## ☁️ Kiểm tra với Modal

Các script Modal không nằm trong <code>requirements.txt</code>. Cài riêng:

~~~powershell
python -m pip install modal
modal setup
~~~

Lệnh hữu ích:

~~~powershell
modal run modal_test/modal_hello.py
modal run modal_test/modal_gpu_test.py
modal run modal_test/modal_volume_test.py
modal run modal_artifact_sanity.py
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py
~~~

<code>modal_artifact_sanity.py</code> kiểm tra 58.603 corpus rows, unique IDs, model shards, FAISS, BM25S, manifest và success marker. <code>modal_runtime_sanity.py</code> kiểm tra retrieval-only; <code>full_modal_runtime_sanity.py</code> nạp cả model trên GPU và chạy luồng chat đầy đủ.

Triển khai API bằng image từ <code>Dockerfile</code>:

~~~powershell
modal serve modal_app.py
modal deploy modal_app.py
~~~

<code>modal_app.py</code> yêu cầu hai Volume đã tồn tại: <code>vn-history-artifacts</code> và <code>vn-history-hf-cache</code>. Cấu hình mặc định dùng GPU L4, <code>APP_MODE=full</code> và mount artifact tại <code>/artifacts</code>. Script <code>modal_fix.py</code> có thay đổi file model trên Volume; chỉ chạy khi kiểm tra artifact báo sai tên shard.

## 🔐 Bảo mật và vận hành

- Không commit token, API key, credentials hoặc nội dung bí mật trong <code>.env</code>.
- Model đang dùng <code>trust_remote_code=True</code>; chỉ load artifact/model từ nguồn tin cậy.
- Trước khi public API, nên thêm authentication, rate limiting, request timeout, CORS policy, structured logging và monitoring.
- SSE nên đi qua reverse proxy đã tắt buffering cho endpoint stream.
- Chỉ nhận traffic khi <code>/ready</code> trả <code>ready=true</code>.
- Không xem benchmark notebook là SLA production; cần benchmark tải riêng trên hạ tầng deploy thật.

## ⚠️ Hạn chế hiện tại

- Artifact lớn không có trong Git; manifest trong repo chỉ là placeholder.
- Đã có Dockerfile và các smoke test Modal, nhưng chưa có CI hoặc test suite tự động chuẩn hóa.
- Chưa đo p95/p99 latency, cold start, requests/second, RAM và VRAM production.
- Source F1 Full RAG còn thấp hơn các metric answer similarity; retrieval/final-context recall vẫn là hướng tối ưu chính.
- Repository chưa có file <code>LICENSE</code>.

## 🛣️ Hướng phát triển

- 📈 Thêm benchmark retrieval độc lập và error analysis theo từng giai đoạn lịch sử.
- ⚙️ Đo p50/p95/p99, throughput, cold start, RAM/VRAM và concurrency.
- 🧪 Bổ sung unit/integration tests cho OOD, RRF, source/year guards và SSE.
- 🐳 Bổ sung CI, container healthcheck và health/readiness probes ở tầng triển khai production.
- 🚀 Chuyển generation sang vLLM hoặc engine batching khi cần tải đồng thời cao.
- 🔍 Tối ưu recall sau reranker và source selection.
- 📜 Bổ sung giấy phép sử dụng dữ liệu, model và mã nguồn.

---

<div align="center">

Made with 🇻🇳, 📚, ⚡ and a slightly unreasonable amount of RAG ✨

</div>
