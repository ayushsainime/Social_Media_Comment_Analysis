# Dockerfile for HuggingFace Spaces
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by ML wheels/runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in one resolver pass for compatibility stability
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

EXPOSE 3000

# Extra safety for Vite host allowlist in HF domains.
ENV __VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS=.hf.space

# Run backend API and Reflex frontend in production mode for Spaces.
# PROD mode avoids Vite dev-server host allowlist blocks on hf.space domains.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & reflex run --env prod --frontend-only --frontend-port 3000"]
