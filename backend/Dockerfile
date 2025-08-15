FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying code (for optimal layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual code and data
COPY backend/ ./backend/
COPY data/ ./data/

# Environment settings
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn server targeting the python module
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
