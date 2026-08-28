FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    APP_MODE=api-only \
    LLM_BACKEND=transformers \
    SHARED_BASE_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
    DEVICE=cpu \
    ARTIFACT_ROOT=/artifacts/vn_history_deployment \
    CHAT_DATABASE_PATH=/data/chat.sqlite3 \
    HF_HOME=/hf-cache \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /workspace/requirements.txt

COPY app /workspace/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
