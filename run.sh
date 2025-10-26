#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
uvicorn hearme.app:app --host 0.0.0.0 --port 8000 --reload --log-level info --proxy-headers --forwarded-allow-ips="*"