# ══════════════════════════════════════════════════════════════
# ChatbotCSKHv2 Runtime Image
# Base: ctgroup/python-base:1.0 (shared, already has langchain/fastapi/etc)
# KHÔNG dùng GPU — kết nối core_vllm/core_embedding qua API
# ══════════════════════════════════════════════════════════════
FROM ctgroup/python-base:1.0

WORKDIR /app

# OCR system deps (chỉ cần cho document parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install app-specific requirements (KHÔNG có torch/unstructured)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Copy source code
COPY app/ ./app/

# Storage directory
RUN mkdir -p /app/storage/documents

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Workers đọc từ env var GUNICORN_WORKERS (default 2)
# Tránh OOM: LangGraph + OpenAI client mỗi worker ~500MB
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${GUNICORN_WORKERS:-2} --access-log"]
