# Stage 1: Install Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Production image
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root user for security
RUN groupadd -r raggroup && useradd -r -g raggroup raguser \
    && mkdir -p /app/data /app/logs \
    && chown -R raguser:raggroup /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/raguser/.local
ENV PATH=/home/raguser/.local/bin:$PATH

# Copy application code
COPY --chown=raguser:raggroup backend/ ./backend/
COPY --chown=raguser:raggroup data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

USER raguser

EXPOSE 8000

# Use python for healthcheck since curl is not available in slim images
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
