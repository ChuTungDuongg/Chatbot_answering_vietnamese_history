# Chatbot hỏi đáp lịch sử Việt Nam

API hỏi đáp lịch sử Việt Nam xây dựng bằng **FastAPI** và kiến trúc **Hybrid RAG**. Hệ thống được thiết kế để kết hợp tìm kiếm ngữ nghĩa bằng FAISS, tìm kiếm từ khóa bằng BM25S, hợp nhất kết quả, xếp hạng lại bằng Cross-Encoder và sinh câu trả lời bằng mô hình Qwen2.5 đã tinh chỉnh.

> **Trạng thái hiện tại:** API đã cung cấp endpoint kiểm tra sức khỏe và trạng thái sẵn sàng. Schema cho chức năng chat đã có trong mã nguồn, nhưng endpoint hỏi đáp chưa được triển khai trong `app/main.py`.

## Công nghệ sử dụng

- FastAPI, Uvicorn và Pydantic
- PyTorch, Transformers và Accelerate
- Sentence Transformers và Cross-Encoder
- FAISS cho tìm kiếm vector
- BM25S cho tìm kiếm từ khóa
- Qwen2.5 cho sinh câu trả lời
- Server-Sent Events (SSE) cho khả năng streaming trong tương lai

## Cấu trúc dự án

```text
.
├── app/
│   ├── main.py                 # Khởi tạo FastAPI và các endpoint hệ thống
│   ├── config.py               # Cấu hình ứng dụng và đường dẫn artifact
│   ├── schemas.py              # Các schema request/response
│   └── services/
│       └── rag_service.py      # Nạp corpus, index, model và kiểm tra readiness
├── artifacts/
│   └── vn_history_deployment/  # Artifact phục vụ triển khai
├── Training/                   # Notebook, dữ liệu và quy trình huấn luyện/RAG
├── requirements.txt
└── README.md
```

## Yêu cầu

- Python 3.10 trở lên
- pip
- GPU hỗ trợ CUDA được khuyến nghị khi chạy chế độ `full`
- Đủ artifact triển khai nếu chạy `retrieval-only` hoặc `full`

Chế độ `api-only` không cần model, corpus hay index và phù hợp để phát triển API cục bộ.

## Cài đặt

### 1. Clone dự án

```bash
git clone <repository-url>
cd Chatbot_answering_vietnamese_history
```

### 2. Tạo môi trường ảo

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Trên Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU, hãy cài phiên bản PyTorch tương thích với CUDA của máy trước khi cài các thư viện còn lại.

### 4. Cấu hình môi trường

Tạo file `.env` tại thư mục gốc:

```dotenv
APP_NAME="Vietnamese History RAG API"
APP_VERSION="1.0.0"
APP_ENV="development"
APP_MODE="api-only"
ARTIFACT_ROOT="./artifacts/vn_history_deployment"
DEVICE="cpu"
```

Các chế độ được hỗ trợ:

| `APP_MODE` | Thành phần được nạp | Mục đích |
|---|---|---|
| `api-only` | Chỉ FastAPI | Phát triển endpoint, Swagger và schema |
| `retrieval-only` | Corpus, FAISS, BM25S, embedder, reranker | Phát triển và kiểm thử truy xuất |
| `full` | Toàn bộ retrieval và mô hình Qwen | Suy luận đầy đủ |

`DEVICE` nhận giá trị `cpu` hoặc `cuda`. Với chế độ `full`, cấu hình `cuda` sẽ báo lỗi nếu PyTorch không phát hiện GPU CUDA.

## Artifact triển khai

Để chạy `retrieval-only`, `ARTIFACT_ROOT` cần có cấu trúc:

```text
vn_history_deployment/
├── manifest.json
├── config/
│   └── inference_config.json
├── corpus/
│   └── vn_history_rag_chunks_enriched.jsonl
└── retrieval/
    ├── faiss/
    │   ├── chunks.index
    │   └── manifest.json
    └── bm25s_index/
        └── phase9_manifest.json
```

Chế độ `full` yêu cầu thêm model:

```text
vn_history_deployment/
└── model/
    └── qwen2_5_3b_vnhistory_stage12_merged/
```

Các thư mục artifact trong repository hiện chỉ là khung và chưa chứa đầy đủ dữ liệu/model cần thiết. Khi khởi động, dịch vụ sẽ kiểm tra các đường dẫn, số lượng corpus, vector FAISS và manifest BM25S; quá trình sẽ dừng nếu artifact bị thiếu hoặc không đồng nhất.

## Chạy ứng dụng

Từ thư mục gốc của dự án:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi server khởi động:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Health check: <http://localhost:8000/health>
- Readiness check: <http://localhost:8000/ready>

Không nên dùng `--reload` trong môi trường production vì mỗi lần reload có thể phải nạp lại các model và index lớn.

## API hiện có

### `GET /health`

Kiểm tra tiến trình API có đang hoạt động hay không.

Ví dụ:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "service": "Vietnamese History RAG API",
  "version": "1.0.0"
}
```

### `GET /ready`

Cho biết các thành phần RAG đã được nạp hay chưa. Ở chế độ `api-only`, API vẫn sẵn sàng nhưng các trường corpus, index và model sẽ là `false`.

```bash
curl http://localhost:8000/ready
```

Ví dụ phản hồi trong chế độ `api-only`:

```json
{
  "ready": true,
  "corpus_loaded": false,
  "faiss_loaded": false,
  "bm25_loaded": false,
  "embedder_loaded": false,
  "reranker_loaded": false,
  "model_loaded": false,
  "corpus_chunks": null,
  "faiss_vectors": null,
  "device": "api-only"
}
```

## Quy trình xử lý dự kiến

```text
Câu hỏi người dùng
        │
        ├── Tìm kiếm ngữ nghĩa (FAISS + embedding)
        └── Tìm kiếm từ khóa (BM25S)
                    │
             Hợp nhất kết quả
                    │
          Cross-Encoder reranking
                    │
          Qwen2.5 sinh câu trả lời
                    │
       Câu trả lời kèm nguồn tham chiếu
```

## Dữ liệu và notebook huấn luyện

Thư mục `Training/` lưu các notebook theo từng giai đoạn, bao gồm chuẩn bị corpus, xuất chunk, SFT với LoRA, đánh giá suy luận, làm giàu metadata, xây dựng Hybrid RAG và xuất artifact FastAPI. Các notebook phục vụ nghiên cứu/thực nghiệm và không bắt buộc khi chỉ chạy API bằng artifact đã chuẩn bị sẵn.

## Hạn chế hiện tại

- Chưa có endpoint chat trong ứng dụng FastAPI.
- Chưa có bộ kiểm thử tự động trong repository.
- Artifact model, corpus và retrieval index chưa được lưu đầy đủ trong repository.
- Một số chuỗi tiếng Việt trong mã nguồn/notebook có dấu hiệu sai encoding và nên được chuẩn hóa về UTF-8.

## Bảo mật

- Không commit file `.env`, token, khóa API hoặc dữ liệu nhạy cảm.
- Chỉ nạp model từ nguồn tin cậy vì cấu hình hiện dùng `trust_remote_code=True`.
- Nên bổ sung xác thực, giới hạn tần suất và giới hạn kích thước request trước khi mở API ra Internet.

## Hướng phát triển

- Triển khai endpoint hỏi đáp và streaming.
- Hoàn thiện pipeline Hybrid RAG và trích dẫn nguồn.
- Bổ sung kiểm thử unit/integration và CI.
- Đóng gói Docker và cấu hình triển khai production.
- Bổ sung logging, metrics và theo dõi lỗi.

## Giấy phép

Repository hiện chưa khai báo giấy phép sử dụng. Hãy bổ sung file `LICENSE` trước khi phân phối hoặc tái sử dụng dự án.
