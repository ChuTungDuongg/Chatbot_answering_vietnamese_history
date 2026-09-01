# ☁️ Modal Upload Scripts

[🏠 Project README](../README.md) · [📦 Artifact contract](../artifacts/README.md)

## 🚚 Upload bundle đầy đủ

Trước tiên validate hoàn toàn cục bộ:

```bash
python scripts/validate_artifact_bundle.py artifacts/vn_history_deployment
```

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --local-dir artifacts/vn_history_deployment \
  --remote-dir / \
  --dry-run
```

`--dry-run` vẫn gọi `validate_artifact_lock()` trước khi chỉ in các lệnh `modal volume put --force`. Bỏ flag để chạy upload thật.

## 🔒 Exact sync production

```bash
python scripts/upload_modal_volume.py \
  --volume vn-history-artifacts \
  --local-dir artifacts/vn_history_deployment \
  --remote-dir / \
  --exact-sync \
  --allow-replace-adapter-weights \
  --dry-run
```

Component mutation đã bị vô hiệu hóa. Hãy build một snapshot bằng `training.scripts.export_artifacts`; uploader quản lý cả `/manifest.json` và `/artifact_lock.json`, với lock luôn được ghi cuối cùng.
`--exact-sync --dry-run` vẫn đọc inventory từ Modal để lập mutation plan, trừ khi truyền `--remote-inventory-json`; đây không phải kiểm tra hoàn toàn offline.

## 🧊 Hugging Face cache

Central Qwen3-8B không nằm trong artifact Volume; base weights/tokenizer phải có trong HF cache Volume `/hf-cache`. Seed CPU-only, không cần A100:

```bash
modal run scripts/modal_seed_hf_cache.py
```

Validate cache:

```bash
modal run scripts/modal_seed_hf_cache.py --validate-only
```

Local/offline check cho cache dir bất kỳ:

```bash
python scripts/hf_cache.py --validate-only --cache-dir /hf-cache/hub
```

## ✅ Sau upload

```bash
modal volume ls vn-history-artifacts
modal run scripts/modal_artifact_sanity.py
modal serve modal_app.py
```

Script không đọc hoặc hard-code Modal token. Authentication do `modal setup` hoặc Modal environment quản lý.
