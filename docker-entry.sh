#!/usr/bin/env bash
set -euo pipefail

mkdir -p ~/.ssh
chmod 700 ~/.ssh
if [ -f /workspace/authorized_keys ]; then
  touch ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
  cat /workspace/authorized_keys >> ~/.ssh/authorized_keys
fi

sudo service ssh start

python - <<'PY'
import torch
print("torch.cuda.is_available()=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

exec uvicorn hearme.app:app --host 0.0.0.0 --port 8000 --log-level info
