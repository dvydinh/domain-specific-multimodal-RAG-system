# Stage 1: Build dependencies in a Python builder environment
FROM python:3.11-slim as builder

WORKDIR /build

# Build text-embedding-inference dependencies (some packages need build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Final lightweight image
FROM python:3.11-slim as runner

WORKDIR /app

# Security: Create a non-privileged user to run the application
RUN groupadd -r raggroup && useradd -r -g raggroup raguser \
    && mkdir -p /app/data /app/logs \
    && chown -r raguser:raggroup /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/raguser/.local
ENV PATH=/home/raguser/.local/bin:$PATH

# Copy application code and ensure permissions
COPY --chown=raguser:raggroup backend/ ./backend/
COPY --chown=raguser:raggroup data/ ./data/

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Production configuration: Run as non-root user
USER raguser

# Expose FastAPI port
EXPOSE 8000

# Health check ensures orchestrators (Docker Compose/K8s) know the service is live
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production server: Run with multiple workers or higher concurrency might be better for high scale,
# but uvicorn --host 0.0.0.0 is standard for the initial Docker deployment.
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
