# Modal quick start — PowerShell

Mở terminal trong repository. **DEFAULT MODAL CORPUS: V1** — `/artifacts/corpus_v1/chunks.jsonl`, Volume `vn-history-artifacts`.

## Activate environment

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
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
$env:MODEL_VARIANT = 'vanilla'
```

Chỉ nếu chưa đăng nhập: `modal token new`. Dữ liệu đã upload; không cần restore/upload mỗi lần mở máy.

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

## Run once / Qdrant

```powershell
New-Item -ItemType Directory -Force reports/restore | Out-Null
modal run --detach --write-result reports/restore/runtime_smoke.json modal_app.py::runtime_smoke --generate --central
modal run --write-result reports/restore/qdrant_smoke.json scripts/modal_qdrant_smoke.py::qdrant_smoke
```

Để API dùng Qdrant, chạy trước `serve`/`deploy`:

```powershell
$env:RETRIEVAL_DENSE_BACKEND = 'qdrant'
$env:MODAL_QDRANT_SECRET_NAME = 'vn-history-qdrant'
modal deploy modal_app.py
```

Để dùng adapter đã phục hồi: `$env:MODEL_VARIANT = 'sft'`. Modal tự dùng `/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter`.

## Test endpoint

```powershell
$base = 'https://tungduong156gli--vn-history-rag-api-fastapi-app.modal.run'
Invoke-RestMethod "$base/ready" -ConnectionTimeoutSeconds 60 -OperationTimeoutSeconds 1200
$headers = @{'X-Client-ID'='terminal-smoke'}
$c = Invoke-RestMethod -Method Post -Uri "$base/api/v1/conversations" -Headers $headers -ContentType 'application/json' -Body '{}'
$body = @{conversation_id=$c.id; question='Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?'; mode='hybrid'; final_k=3} | ConvertTo-Json
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
