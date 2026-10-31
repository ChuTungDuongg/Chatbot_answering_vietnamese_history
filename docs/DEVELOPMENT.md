# Phát triển và kiểm tra

## Bố cục mã nguồn

| Thư mục | Trách nhiệm |
| --- | --- |
| `app/` | FastAPI, chat runtimes, model engines, retrieval, tools, MCP, lưu hội thoại. Xem [kiến trúc](ARCHITECTURE.md). |
| `frontend_server/` | CPU static UI và gateway streaming tới API GPU. |
| `frontend/` | React 19/Vite: hội thoại, streamed answer, nguồn trích dẫn, attachments, theme và lựa chọn mode/backend/tools. |
| `scripts/` | Corpus/index CLI, khôi phục/upload artifacts, triển khai benchmark và smoke tests. |
| `benchmarks/latency/` | HTTP/SSE latency runner, workload cố định, so sánh thí nghiệm. |
| `evaluation/` | Schema câu hỏi, annotation, metric, headless retrieval và six-way evaluation. |
| `training/` | QLoRA và merge adapter; `Dataset/` và ZIP lịch sử được bảo toàn. |
| `tools/drive_cli.py` | Chuyển/kiểm tra dataset, adapter, evaluation và benchmark qua filesystem Drive đã mount. |
| `tests/`, `frontend/tests/`, `frontend/e2e/` | Python, Node/Vitest và Playwright. |
| `docs/` | Tài liệu hệ thống; README root là điểm vào. |

## Frontend state và streaming

[`api.js`](../frontend/src/services/api.js) thực hiện REST/SSE; [`useChatStream.js`](../frontend/src/hooks/useChatStream.js) điều khiển lifecycle; [`chatSessionReducer.js`](../frontend/src/state/chatSessionReducer.js) cập nhật deltas/sources theo request và conversation. Hoàn tất chat, frontend đồng bộ ID và nội dung đã lưu từ server. Stop abort request và giữ phần text nhận được.

`localStorage` giữ anonymous client ID, theme, chat mode (`vn-history-chat-mode-v2`) và dense backend (`vn-history-retrieval-backend-v1`). Messages/attachments lưu ở backend. Quyền MCP được chọn theo request, không lưu thành quyền dài hạn trong `localStorage`. `X-Client-ID` dùng scope dữ liệu, không phải xác thực người dùng.

## Kiểm tra tự động

```powershell
python -m pip install pytest
python -m compileall -q app frontend_server scripts evaluation benchmarks tools training
python -m pytest -q
npm --prefix frontend test
npm run frontend:lint
npm run frontend:build
git diff --check
```

Python tests dùng model/token stream giả và corpus tạm để kiểm tra routing, SSE, cancellation, tool permissions, telemetry, danh tính artifacts, metric và quy trình training. Một số kiểm tra dữ liệu/preservation đọc artifacts local đã cố định. Tests không tải Qwen weights hay chạy inference GPU thật.

`npm --prefix frontend test` chạy Node tests rồi Vitest. Browser tests chạy riêng:

```powershell
npm --prefix frontend run test:ui
```

Playwright config quản lý frontend test server và dùng Chrome đã cài (`channel=chrome`), không tự tải browser hoặc model weights. Smoke giao diện cần bao phủ Hybrid/Central, FAISS/Qdrant theo capabilities, stream một phần, hoàn tất/lỗi/Stop, citations, attachments và desktop/mobile.

[CI](../.github/workflows/ci.yml) chạy trên push/pull request: Python 3.11 cài runtime/corpus dependencies cùng Modal 1.5.3, compile và pytest; Node.js 24 chạy frontend tests/lint/build. Job `frontend-container` kiểm tra/build Docker image CPU, static UI và proxy 503 khi chưa cấu hình upstream. Playwright và live Modal/GPU checks là các bước riêng.

## Thay đổi dữ liệu hoặc thêm tool

Corpus/index không được rebuild trong startup, audit hay test. Dùng output riêng cho mỗi lần xây dữ liệu, lưu manifest/hash và kiểm tra trước khi thay bundle phục vụ; xem [Artifacts](ARTIFACTS.md) và [Retrieval](RETRIEVAL.md).

Tool mới có Pydantic input model, `name`, `description`, `input_schema` và `run` hoặc `run_with_context`; đăng ký trong [`app/main.py`](../app/main.py). Request context mang owner/conversation/backend/permissions riêng. Kiểm tra validation, scope, kết quả và lỗi trước khi cho planner dùng. Xem [Tools và MCP](TOOLS_AND_MCP.md).

Đo chất lượng và latency trên đúng artifacts/model revisions/hardware bằng [evaluation](EVALUATION.md) và [benchmark](BENCHMARKS.md). Kết quả unit test không thay thế các phép đo này.
