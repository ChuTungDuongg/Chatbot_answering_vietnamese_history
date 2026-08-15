# Modal Smoke Tests

[Về README gốc](../README.md)

`modal_test/` chứa các script nhỏ để xác nhận Modal account, remote function, GPU và Volume
hoạt động trước khi chạy runtime RAG. Đây là smoke/learning scripts độc lập, không phải
automated test suite của ứng dụng.

## Chuẩn bị

```powershell
python -m pip install modal
modal setup
```

Chạy lệnh từ repository root. GPU function và container đang chạy có thể phát sinh chi phí.

## Script

| File | Lệnh | Phạm vi | Kết quả mong đợi |
|---|---|---|---|
| [`modal_hello.py`](modal_hello.py) | `modal run modal_test/modal_hello.py` | Remote function và OS | In lời chào cùng platform |
| [`modal_gpu_test.py`](modal_gpu_test.py) | `modal run modal_test/modal_gpu_test.py` | NVIDIA L4, CUDA, PyTorch, `nvidia-smi` | `CUDA available: True` |
| [`modal_volume_test.py`](modal_volume_test.py) | `modal run modal_test/modal_volume_test.py` | Mount/read artifact Volume | Đọc `/artifacts/test_modal.txt` |
| [`test_modal.txt`](test_modal.txt) | Không chạy trực tiếp | Fixture nhỏ cho Volume | Nội dung được remote function in ra |

`modal_volume_test.py` dùng `create_if_missing=False`. Volume `vn-history-artifacts` phải
tồn tại và có file `test_modal.txt`. Chuẩn bị fixture:

```powershell
modal volume put --force vn-history-artifacts modal_test/test_modal.txt test_modal.txt
modal volume ls vn-history-artifacts
```

## Runtime checks ở root

Sau khi ba smoke tests cơ bản thành công:

| File | Tài nguyên | Kiểm tra |
|---|---|---|
| [`../modal_artifact_sanity.py`](../modal_artifact_sanity.py) | Artifact Volume | Layout, corpus count, index count, model shards, manifest |
| [`../modal_runtime_sanity.py`](../modal_runtime_sanity.py) | CPU/retrieval artifacts | Load retrieval-only và chạy history/OOD queries |
| [`../full_modal_runtime_sanity.py`](../full_modal_runtime_sanity.py) | L4/full bundle | Load model và chạy full generation pipeline |
| [`../modal_app.py`](../modal_app.py) | L4 + 3 Volumes | ASGI development/deployment thật |

Thứ tự đề xuất:

```powershell
modal run modal_test/modal_hello.py
modal run modal_test/modal_volume_test.py
modal run modal_test/modal_gpu_test.py
modal run modal_artifact_sanity.py
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py
```

Full runtime sanity kiểm tra generation cơ bản nhưng chưa xác nhận đầy đủ:

- conversation CRUD và persistence qua restart;
- upload PDF, fallback OCR và temporary chunk cleanup;
- multi-turn memory;
- toàn bộ thứ tự/payload SSE;
- concurrency, cold start hoặc p95/p99 latency.

Các phần này cần integration test hoặc manual smoke test riêng.

## Quy tắc

- Giữ mỗi script độc lập và có `@app.local_entrypoint()`.
- Mỗi smoke test chỉ nên xác nhận một capability rõ ràng và fail bằng exception/exit code.
- Không đưa Modal token, API key hoặc credential vào source/fixture.
- Logic production nằm trong `app/` hoặc root deployment scripts, không đặt tại đây.
- Không dùng `modal_fix.py` như smoke test; script đó có thể sửa model files trên Volume.
