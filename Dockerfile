FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV TORCH_HOME=/app/models

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repository
RUN git clone --depth 1 --branch main https://github.com/Lightricks/LTX-2.git /app/repo

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install /app/repo
RUN pip install transformers accelerate safetensors
RUN pip install runpod

# Download model weights at build time (faster cold starts)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video', local_dir='/app/models/LTX-Video')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]