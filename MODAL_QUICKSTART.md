# Modal quick start — PowerShell

Mở terminal trong repository. **DEFAULT MODAL CORPUS: V1** — `/artifacts/corpus_v1/chunks.jsonl`, Volume `vn-history-artifacts`.

## Chạy backend bằng một lệnh

Ở root repository, chạy:

```powershell
modal serve modal_app.py
```

`modal_app.py` tự đọc `.env` ở root repository trước khi tạo cấu hình Modal.
Máy hiện tại đã có `.env`; trên checkout mới, tạo một lần bằng
`Copy-Item .env.example .env`. Không cần nhập lại các dòng `$env` mỗi lần chạy.
PowerShell trên máy hiện tại đã được cấu hình tự thêm Modal vào PATH và dùng UTF-8;
mở terminal PowerShell mới một lần sau khi thay đổi cấu hình này.
File `.env` được Git ignore; credentials Qdrant tiếp tục lấy từ Modal Secret.
Dùng URL `https://...-dev.modal.run` mà CLI in ra. `Ctrl+C` dừng phiên development.

### Đổi Hybrid vanilla / SFT

Sửa **MODEL_VARIANT** trong `.env`:

```dotenv
# Base model
MODEL_VARIANT=vanilla
# Muốn dùng PEFT adapter, thay dòng trên bằng:
# MODEL_VARIANT=sft
```

Sau khi sửa, `Ctrl+C` rồi chạy lại `modal serve modal_app.py`.
SFT dùng adapter `/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter`
đã lưu trên Volume; không cần đặt `MODEL_ADAPTER_PATH` cho Modal.
Central vẫn dùng vanilla Qwen3-8B.

Biến môi trường đã đặt trong terminal được ưu tiên hơn `.env`. Nếu trước đó đã
đặt `$env:MODEL_VARIANT`, bỏ override một lần bằng
`Remove-Item Env:MODEL_VARIANT -ErrorAction SilentlyContinue` để dùng giá trị trong file.

## Activate environment (chỉ khi terminal chưa nhận lệnh modal)

```powershell
Set-Location (git rev-parse --show-toplevel)
$repo = (Get-Location).Path
$env:Path = "$repo\.conda;$repo\.conda\Scripts;$repo\.conda\Library\bin;$env:Path"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MODAL_SERVER_URL = 'https://api.modal2.com'
python --version
modal --version
```

## Check Modal login and data

```powershell
modal volume list --json
modal secret list --json
modal volume ls vn-history-artifacts /corpus_v1/runtime --json
```

Chỉ nếu chưa đăng nhập: `modal token new`. Dữ liệu đã upload; không cần restore/upload mỗi lần mở máy.

Trong `.env`, đặt `RETRIEVAL_DENSE_BACKEND=faiss` và
`RETRIEVAL_AVAILABLE_BACKENDS=faiss,qdrant` để load/validate cả hai lane một lần;
chọn FAISS/Qdrant tại composer theo từng request. File của máy hiện tại đã đặt cả hai.
Nếu Secret Qdrant chưa có, đổi `.env` thành `RETRIEVAL_AVAILABLE_BACKENDS=faiss`.

MCP mặc định tắt. Khi cần server ngoài, xem [MCP setup](MCP_INTEGRATION_REPORT.md):
đặt `MCP_ENABLED=true`, `MCP_CONFIG_PATH` và optional `MODAL_MCP_SECRET_NAME` trong `.env` trước `serve`/`deploy`.
Tools chỉ hiện trong Central, theo capabilities và quyền từng request.

## Start API for development

```powershell
modal serve modal_app.py
```

Dùng URL `https://...-dev.modal.run` mà CLI in ra. `Ctrl+C` dừng phiên development.

## Deploy

```powershell
modal deploy modal_app.py
modal app logs vn-history-rag-api --tail 100
```

API chạy `APP_MODE=full` trên A100, tự tải base models vào `vn-history-hf-cache` nếu thiếu. Central tải model khi được gọi lần đầu.

Hybrid vanilla/SFT cùng mặc định tối đa **1536 output tokens**; model có thể tự EOS sớm.
Modal dùng default từ `app/config.py`, không hardcode budget riêng. Override trong
`.env` trước `serve`/`deploy`, ví dụ `HYBRID_MAX_NEW_TOKENS=2048` (hoặc `1024`).
Central final vẫn giữ 1536. DeveloperTrace và generation metadata ghi budget thực tế.
Xem [runtime telemetry](docs/RUNTIME_TELEMETRY.md) cho planner rounds, finish reason và smoke/benchmark.

## Run once / Qdrant

```powershell
New-Item -ItemType Directory -Force reports/restore | Out-Null
modal run --detach --write-result reports/restore/runtime_smoke.json modal_app.py::runtime_smoke --generate --central
modal run --write-result reports/restore/qdrant_smoke.json scripts/modal_qdrant_smoke.py::qdrant_smoke
```

Để API mặc định dùng Qdrant, sửa `.env` trước `serve`/`deploy`:

```dotenv
RETRIEVAL_DENSE_BACKEND=qdrant
RETRIEVAL_AVAILABLE_BACKENDS=faiss,qdrant
MODAL_QDRANT_SECRET_NAME=vn-history-qdrant
```

Để dùng adapter đã phục hồi: sửa `MODEL_VARIANT=sft` trong `.env`, như mục
**Đổi Hybrid vanilla / SFT** ở trên, rồi khởi động lại Modal.

## Test endpoint

```powershell
$base = 'https://tungduong156gli--vn-history-rag-api-fastapi-app.modal.run'
Invoke-RestMethod "$base/ready" -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
$headers = @{'X-Client-ID'='terminal-smoke'}
$c = Invoke-RestMethod -Method Post -Uri "$base/api/v1/conversations" -Headers $headers -ContentType 'application/json' -Body '{}'
$body = @{conversation_id=$c.id; question='Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?'; mode='hybrid'; retrieval_backend='faiss'; final_k=3} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/api/v1/chat" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
```

## Verify / sync again only when needed

```powershell
python -m scripts.restore_portable_backup --verify-only
python -m scripts.upload_modal_volume --include-v0 --verify-only
# Upload missing files; refuses different local/remote bytes:
python -m scripts.upload_modal_volume --include-v0 --ipv4
```

Xem [report đầy đủ](RESTORE_MODAL_QDRANT_REPORT.md) để tạo Secret, restore ZIP, logs và dừng deployment.
Xem [dynamic retrieval](DYNAMIC_RETRIEVAL_REPORT.md) để chạy hai backend cùng lúc và đo latency.
