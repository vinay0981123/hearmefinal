# Base: CUDA 12.8 runtime with Ubuntu 24.04 compatible with NVIDIA driver 570.x
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# System dependencies: ffmpeg, Python, build tools, OpenSSH
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    ffmpeg git curl ca-certificates build-essential \
    openssh-server sudo \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user (app) for SSH and runtime
ARG USER=app
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USER} && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} \
    && usermod -aG sudo ${USER} \
    && echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# SSH server setup (Runpod-like convenience)
RUN mkdir /var/run/sshd && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin no" >> /etc/ssh/sshd_config

# Workdir and caches on persistent volume path
WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV PYTHONUNBUFFERED=1

# Copy project descriptors first for layer caching
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch with CUDA 12.8 wheels, then app deps
# Use the official CUDA 12.8 index for torch/torchaudio
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio && \
    pip install -r /workspace/requirements.txt

# Copy application code
COPY hearme/ /workspace/hearme/
COPY run.sh /workspace/run.sh
# Optional: copy .env if you bake secrets at build-time (not recommended)
# COPY .env /workspace/.env

# Permissions
RUN chown -R ${USER}:${USER} /workspace

# Switch to non-root
USER ${USER}

# Create venv (optional; the base Python works too, but venv isolates)
RUN python3 -m venv /workspace/.venv && \
    . /workspace/.venv/bin/activate && pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio && \
    pip install -r /workspace/requirements.txt

# Ensure venv is used in shell
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Expose FastAPI and SSH ports
EXPOSE 8000 22

# Entry script: starts SSH and the API (Uvicorn) together
# -d flag for sshd via service; Uvicorn in foreground
CMD bash -lc '\
  mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
  if [ -f /workspace/authorized_keys ]; then \
    cat /workspace/authorized_keys >> ~/.ssh/authorized_keys; \
  fi && \
  chmod 600 ~/.ssh/authorized_keys || true; \
  sudo service ssh start; \
  # Print CUDA info for sanity
  python -c "import torch; print(\"torch.cuda.is_available()=\", torch.cuda.is_available()); \
             print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no cuda\")"; \
  uvicorn hearme.app:app --host 0.0.0.0 --port 8000 --log-level info'
