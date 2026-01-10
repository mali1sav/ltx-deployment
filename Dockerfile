FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV TORCH_HOME=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 --branch main https://github.com/Lightricks/LTX-2.git /app/repo

RUN pip install --upgrade pip
RUN pip install /app/repo
RUN pip install transformers accelerate safetensors huggingface_hub
RUN pip install runpod

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
