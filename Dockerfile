FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git python3 python3-pip build-essential wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install numpy packaging wheel ninja
RUN pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /workspace/vllm
WORKDIR /workspace/vllm

ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV MAX_JOBS=12
ENV NVCC_THREADS=1
ENV PYTHONUNBUFFERED=1

RUN rm -rf build/ .deps/ && pip install -e .

CMD ["bash"]
