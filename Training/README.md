# Training and dataset pipeline

[Về README gốc](../README.md)

Thư mục `Training/` lưu các archive notebook, chunk packs cho RAG-SFT và pipeline Phase 1-10. Đây là workspace nghiên cứu/huấn luyện, tách khỏi dependencies và runtime production trong `app/`.

## Cấu trúc

```text
Training/
├── README.md
├── InvestigatingDataset.zip       # 2 notebook audit dataset/corpus
├── requirement.txt
├── Dataset/
│   ├── Chunk_id/                  # 31 JSONL packs, tổng 520 records
│   └── merged_jsonl/
│       └── all_chunk_id.jsonl     # 520 records
└── Training/
    └── Training.zip               # archive notebook Phase 1-10
```

## Notebook audit

`InvestigatingDataset.zip` chứa hai notebook tiện ích:

| Notebook trong archive | Mục đích |
|---|---|
| `InvestigatingDataset.ipynb` | Khảo sát phân bố, chất lượng và cấu trúc dataset |
| `ReadingJSONCorpus.ipynb` | Đọc, tìm kiếm và audit JSON/JSONL corpus |

## Pipeline Phase 1-10

`Training/Training/Training.zip` chứa các notebook dưới đây. Giữ đúng thứ tự vì output phase trước là input hoặc checkpoint cho phase sau.

| Phase | Notebook trong archive | Output chính |
|---|---|---|
| 1 | `Phase1_SFT.ipynb` | QLoRA adapter SFT nền |
| 2 | `Phase2_CreatingCorpus.ipynb` | Corpus lịch sử đã lọc/làm sạch |
| 3 | `Phase3_ChunkExporter.ipynb` | Chunk packs theo chủ đề |
| 4 | `Phase4_ChunkExporter_extra_topics.ipynb` | Packs cho chủ đề bổ sung |
| 5 | `Phase5.ipynb` | Dữ liệu chunk/messages đã hợp nhất |
| 6 | `Phase6_RAG_SFT_Qwen2_5_LoRA.ipynb` | Adapter RAG-SFT và evaluation |
| 7 | `Phase7_InferneceTesting.ipynb` | Inference/FAISS sanity checks |
| 8 | `Phase8_VN_History_Chunk_Metadata_Enrichment_v4.ipynb` | Enriched corpus metadata |
| 9 | `Phase9_VN_History_Hybrid_RAG_ToolUse_v2_Grounded_Direct.ipynb` | FAISS/BM25S/config và benchmark |
| 10 | `Phase10_VN_History_FastAPI_Export_From_Phase9 (1).ipynb` | Merged model và FastAPI deployment bundle |

Tên file trong bảng giữ nguyên theo archive, kể cả chính tả hiện tại, để developer tìm đúng notebook.

## Hai bộ dữ liệu dễ nhầm

| Bộ dữ liệu | Quy mô | Vai trò |
|---|---:|---|
| `Training/Dataset/merged_jsonl/all_chunk_id.jsonl` | 520 records, 511 unique `chunk_id` | Corpus mẫu/contexts cho RAG-SFT Phase 6 |
| Deployment corpus Phase 8-10 | 58.603 chunks | Corpus thật cho Hybrid RAG/FastAPI |

Không dùng bộ 520 records để thay thế deployment corpus và không build FAISS production từ nhầm file này.

## Môi trường notebook

Tạo environment riêng để không làm xáo trộn runtime API:

```powershell
python -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Training/requirement.txt
python -m ipykernel install --user --name vn-history-training
```

Giải nén `InvestigatingDataset.zip` hoặc `Training/Training.zip` vào workspace ngoài repository trước khi chạy notebook; không commit lại checkpoint/output lớn sinh ra sau khi giải nén.

`requirement.txt` gồm Jupyter, Hugging Face training stack, PEFT/TRL, bitsandbytes, sentence-transformers và FAISS. QLoRA/bitsandbytes cần môi trường GPU/CUDA tương thích; notebook có thể cần cài thêm package theo runtime Colab/Modal cụ thể.

## Dòng output

```text
Phase 1 adapter
  + Phase 2-5 corpus/chunk/message preparation
  -> Phase 6 RAG-SFT adapter + metrics
  -> Phase 7 inference sanity
  -> Phase 8 enriched metadata
  -> Phase 9 retrieval indexes + benchmark
  -> Phase 10 merged model + artifacts/vn_history_deployment
```

Kết quả Phase 10 được dùng bởi [`../app`](../app) và được mô tả tại [`../artifacts/README.md`](../artifacts/README.md).

## Reproducibility checklist

- Ghi seed, model revision, GPU, CUDA, package versions và dataset hash cho mỗi run.
- Không ghi đè adapter/checkpoint tốt nhất nếu chưa lưu metrics và manifest.
- Validate JSONL, unique IDs, split leakage và evidence IDs trước khi train.
- Lưu benchmark output cạnh config đã dùng để có thể đối chiếu.
- Không commit token, cache, checkpoints, model weights hoặc generated dataset lớn.
- Sau khi đổi artifact contract, cập nhật Phase 10 export, backend loader và sanity scripts cùng lúc.
