FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for PyTorch, MuJoCo, mujoco-py, and plotting utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    curl \
    wget \
    unzip \
    python3 \
    python3-venv \
    python3-pip \
    libgl1-mesa-dev \
    libglu1-mesa \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxi6 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo 2.1.0
RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /workspace

# Install Python dependencies (use --no-build-isolation so mujoco-py picks up Cython<3)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -r /tmp/requirements.txt

# Copy the rest of the repository
COPY . /workspace

# Default command
CMD ["/bin/bash"]
