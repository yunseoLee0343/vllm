FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git python3 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install ninja packaging wheel

COPY . /workspace/vllm
WORKDIR /workspace/vllm

RUN pip install -e .

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
