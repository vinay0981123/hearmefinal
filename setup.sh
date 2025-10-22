#!/usr/bin/env bash
set -euo pipefail

# 1) System deps (Ubuntu 24.04 on Runpod image)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    wget llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev ffmpeg
fi

# 2) Install pyenv under /workspace (user-level)
export PYENV_ROOT="/workspace/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if ! command -v pyenv >/dev/null 2>&1; then
  curl https://pyenv.run | bash
fi
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Initialize pyenv
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)" || true
fi

# 3) Install Python 3.10.12 and set it for the project directory
PY_VERSION="3.10.12"
if ! pyenv versions --bare | grep -qx "$PY_VERSION"; then
  CFLAGS="-O2" pyenv install -s "$PY_VERSION"
fi
pyenv local "$PY_VERSION"

# 4) Create venv at the requested location
VENV_PATH="/workspace/hearmefinal/.venv"
mkdir -p "$(dirname "$VENV_PATH")"
python -m venv "$VENV_PATH"

# 5) Activate venv and install deps
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install fastapi uvicorn[standard] python-dotenv requests pydantic \
              numpy soundfile librosa torch torchaudio pyannote.audio \
              faster-whisper tqdm aiofiles
fi

# 6) Recommend caches on the persistent /workspace volume
echo "Recommended to cache models on /workspace (add to .env or export in shell):"
echo "  HF_HOME=/workspace/.cache/huggingface"
echo "  TRANSFORMERS_CACHE=/workspace/.cache/huggingface"
echo "  TORCH_HOME=/workspace/.cache/torch"

echo "Setup complete."
echo "Activate venv before running:"
echo "  source /workspace/hearmefinal/.venv/bin/activate"
