#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/hearmefinal/logs
export PYTHONUNBUFFERED=1

# Start uvicorn detached, capture PID, log to file
nohup uvicorn hearme.app:app \
  --host 0.0.0.0 --port 8000 \
  --log-level info --proxy-headers --forwarded-allow-ips="*" \
  > /workspace/hearmefinal/logs/uvicorn.log 2>&1 &

echo $! > /workspace/hearmefinal/logs/uvicorn.pid
echo "started: $(date -Is) pid=$(cat /workspace/hearmefinal/logs/uvicorn.pid)"