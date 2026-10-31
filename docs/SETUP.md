# Cài đặt và chạy local

Chạy các lệnh từ root repository. Python 3.11 và Node.js 24 là các phiên bản CI dùng. Full inference cần đủ RAM/VRAM cho corpus, retrieval và model; `api-only` phù hợp để kiểm tra API trên máy chưa có artifacts hoặc GPU.

## Chuẩn bị môi trường

PowerShell, với môi trường mới:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu đã có `.conda` của repository, có thể dùng `.\.conda\python.exe` hoặc activate môi trường đó thay cho việc tạo thêm `.venv`. Cài dependencies vào đúng interpreter dùng để chạy backend.

Tạo cấu hình một lần nếu file chưa tồn tại:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
if (!(Test-Path frontend/.env)) { Copy-Item frontend/.env.example frontend/.env }
npm ci
npm --prefix frontend ci
```

`requirements.txt` là runtime Transformers. Các bundle `requirements-corpus.txt` và `requirements-training.txt` phục vụ công việc offline tương ứng. `requirements-vllm.txt` có Torch/FastAPI/NumPy khác, cần môi trường Linux CUDA riêng; xem [vLLM](VLLM_LATENCY_EXPERIMENTS.md). Các file `requirements*.txt` ở root là đầu vào cài đặt, không phải tài liệu để di chuyển.

## Backend local

Kiểm tra API nhẹ:

```powershell
$env:APP_MODE = 'api-only'
$env:DEVICE = 'cpu'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Để chạy retrieval, khôi phục [artifacts V1](ARTIFACTS.md), đặt `APP_MODE=retrieval-only` và khởi động lại. Chế độ này dùng được `/api/v1/retrieve`, chưa sinh câu trả lời chat.

Để chạy chat trên máy CUDA đủ tài nguyên, đặt trong `.env` rồi khởi động backend từ terminal không có override cũ:

```dotenv
APP_MODE=full
DEVICE=cuda
DTYPE=bfloat16
ARTIFACT_ROOT=./artifacts/corpus_v1
RETRIEVAL_DENSE_BACKEND=faiss
RETRIEVAL_AVAILABLE_BACKENDS=faiss
MODEL_VARIANT=vanilla
```

Hybrid tải Qwen3-4B khi startup; Central mặc định tải Qwen3-8B khi được gọi. Model có thể được tải vào cache từ Hub nếu chưa có. Chọn `MODEL_VARIANT=sft` yêu cầu adapter local hợp lệ. Chi tiết ở [cấu hình](CONFIGURATION.md).

## Frontend

Đặt `VITE_API_BASE_URL=http://127.0.0.1:8000` trong `frontend/.env`, rồi chạy terminal khác:

```powershell
npm run frontend
```

Vite nhúng API origin khi dev/build; cần restart hoặc build lại sau khi đổi URL. Có thể trỏ frontend local tới URL GPU Modal được CLI trả về. Khi dùng frontend CPU trên Modal, để URL Vite rỗng để gọi API cùng origin; không cần Vite local, xem [Deployment](DEPLOYMENT.md).

`npm run dev` ở root chạy đồng thời Vite và `modal serve modal_app.py`. Để dùng backend Uvicorn local, chạy riêng lệnh ở trên. Xem [Modal](DEPLOYMENT.md) trước khi dùng lệnh kết hợp.

## Kiểm tra API

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

OpenAPI tại `http://127.0.0.1:8000/docs`. `/health` kiểm tra API phản hồi. `/ready` có thể trả HTTP 200 cùng `ready=false`; cần đọc JSON. Trong `api-only`, `ready=true` không có nghĩa retrieval/model đã tải. Trong full mode, xem thêm `hybrid_loaded` và `central_loaded`; Central lazy có thể chưa tải. Kiểm tra một request thực tế để xác nhận đường chạy cần dùng.

| Triệu chứng | Cách kiểm tra |
| --- | --- |
| Chat trả 503 | `APP_MODE=full`, chat mode đã bật, artifacts và model tải thành công. |
| Sai hoặc thiếu corpus/index | Đường dẫn và manifest của cùng bundle V1; startup không tự xây lại index. |
| `.env` không có tác dụng | Biến môi trường terminal ưu tiên hơn file; bỏ override cũ rồi restart. |
| Frontend không gọi được API | `VITE_API_BASE_URL`, CORS, URL/port và restart Vite. |
| Qdrant không xuất hiện | Manifest, URL/key và `RETRIEVAL_AVAILABLE_BACKENDS`; chỉ backend đã validate mới xuất hiện. |
| SFT không tải | Model gốc đúng, `adapter_config.json`, `adapter_model.safetensors` và đường dẫn adapter. |

Các lệnh test và CI nằm trong [hướng dẫn phát triển](DEVELOPMENT.md).

## Docker Compose local

[`compose.day03.yaml`](../compose.day03.yaml) chạy API `api-only` trên `127.0.0.1:8000` và Qdrant `v1.17.1` trên `127.0.0.1:6333`, với named volumes `api_data` và `qdrant_data`:

```powershell
docker compose -f compose.day03.yaml up -d --build
```

Compose không tự bật full inference, cấu hình app dùng Qdrant hay copy Cloud collection. Quy trình copy/verify collection có sẵn nằm trong [Qdrant migration](QDRANT_MIGRATION.md). Giữ named volumes khi restart để bảo toàn dữ liệu.
