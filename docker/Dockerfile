# docker/Dockerfile

FROM python:3.10-slim-bookworm

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal, CPU-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src /app/src
COPY api /app/api

# Default environment variables (can be overridden at runtime)
ENV MODEL1_ONNX_PATH=/models/optimized.onnx
ENV MODEL1_TOKENIZER_NAME_OR_PATH=hf-internal-testing/tiny-random-distilbert
ENV MODEL1_MAX_LENGTH=16

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
