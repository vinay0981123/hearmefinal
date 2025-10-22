#!/usr/bin/env bash
# set -euo pipefail
# export PYTHONUNBUFFERED=1
# uvicorn hearme.app:app --host 0.0.0.0 --port 8000 --reload --log-level info --proxy-headers --forwarded-allow-ips="*"





#!/usr/bin/env bash
set -euo pipefail

# ---- Configurable env (override via .env or shell) ----
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export PYTHONUNBUFFERED=1

# Optional: load .env if present
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '\n' -I {} echo {})
fi

# Device selection only via env, no code changes needed
export DEVICE="${DEVICE:-auto}"

# ---- Ensure build tools ----
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev ffmpeg
fi

# ---- Install pyenv under /workspace (user space) ----
export PYENV_ROOT="/workspace/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if ! command -v pyenv >/dev/null 2>&1; then
  curl https://pyenv.run | bash
fi
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Initialize pyenv
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
fi

# ---- Install Python 3.10.12 if needed and set local version ----
PY_VERSION="3.10.12"
if ! pyenv versions --bare | grep -qx "$PY_VERSION"; then
  CFLAGS="-O2" pyenv install -s "$PY_VERSION"
fi
pyenv local "$PY_VERSION"

# ---- Create venv in /workspace/.venv and install deps ----
python -m venv /workspace/.venv
# shellcheck disable=SC1091
source /workspace/.venv/bin/activate
pip install --upgrade pip

# If requirements.txt exists, install; otherwise install known deps
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install fastapi uvicorn[standard] python-dotenv requests pydantic \
              numpy soundfile librosa torch torchaudio pyannote.audio \
              faster-whisper tqdm aiofiles
fi

# ---- Verify CUDA availability (for info) ----
python - <<'PY'
import os, torch
print("DEVICE env:", os.getenv("DEVICE", "auto"))
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

# ---- Run the app ----
exec uvicorn hearme.app:app --host 0.0.0.0 --port 8000 --log-level info --proxy-headers --forwarded-allow-ips="*"
