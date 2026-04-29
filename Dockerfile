# ══════════════════════════════════════════════════════════════
# Stage 1: builder — cài dependencies
# ══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps cho OCR + document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ══════════════════════════════════════════════════════════════
# Stage 2: runtime — image nhỏ gọn
# ══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

WORKDIR /app

# Chỉ copy runtime system libs cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv từ builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY app/ ./app/

# Storage directory
RUN mkdir -p /app/storage/documents

# Non-root user — bảo mật
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser \
    && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn: 4 workers để xử lý lượng connection lớn hơn
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--access-log"]
