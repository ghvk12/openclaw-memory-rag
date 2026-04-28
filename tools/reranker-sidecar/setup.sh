#!/usr/bin/env bash
# Bootstrap the reranker sidecar venv.
# Idempotent: re-running upgrades pip and re-installs requirements only.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3.13}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON not found. Install with: brew install python@3.13" >&2
  exit 1
fi

if [ ! -d venv ]; then
  echo "==> creating venv with $PYTHON"
  "$PYTHON" -m venv venv
fi

source venv/bin/activate
echo "==> upgrading pip"
pip install --upgrade pip wheel >/dev/null

echo "==> installing requirements (this is ~1.5GB, takes a few minutes on first run)"
pip install -r requirements.txt

echo "==> done. To run manually:"
echo "    source venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8089"
