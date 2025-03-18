FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 1) Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget ffmpeg libgl1 libglib2.0-0 vim \
    cmake build-essential ninja-build \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# (Optional) Upgrade pip to the latest version
RUN python3 -m pip install --no-cache-dir --upgrade pip

# 2) Create a directory for storing models/caches
# We'll mount a volume at /models
RUN mkdir -p /models
VOLUME ["/models"]

# 3) Set environment variables so Transformers/HF store models & caches here
ENV HF_HOME="/models/hf" \
    TRANSFORMERS_CACHE="/models/transformers" \
    HF_DATASETS_CACHE="/models/hf-datasets"

WORKDIR /workspace

# 4) Install PyTorch and other dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face Transformers and other LLM tooling
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    xformers \
    optimum \
    auto-gptq \
    peft \
    huggingface_hub

# Install llama-cpp-python (after ninja is installed)
RUN pip3 install --no-cache-dir llama-cpp-python

RUN pip3 install --no-cache-dir sentencepiece protobuf

RUN pip3 install --no-cache-dir protobuf

# Install FastAPI and Uvicorn (needed for server.py)
RUN pip3 install --no-cache-dir fastapi uvicorn

# 5) Switch to your working directory (where your server code is)
WORKDIR /workspace/api

COPY ./api /workspace/api

# 6) Expose API port
EXPOSE 8000

# 7) (Optional) Create a non-root user
RUN useradd -m llamauser
USER llamauser

# 8) Start API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

