FROM python:3.11-slim

# Create user with UID 1000 (standard for HF Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for proper caching and execution
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app \
    API_HOST=0.0.0.0 \
    API_PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

WORKDIR $HOME/app

# Copy requirements and install (as user to avoid root permission issues)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY --chown=user backend/ backend/
RUN mkdir -p data/raw

EXPOSE 7860

# Run using python module invocation to ensure it uses the user's installed uvicorn
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
