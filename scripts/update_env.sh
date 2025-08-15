#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
LOCK_FILE="requirements.lock.txt"

# shellcheck disable=SC1090
if [[ -d "$VENV_DIR" ]]; then
  source "$VENV_DIR/bin/activate"
else
  echo "[error] $VENV_DIR not found. Run scripts/setup_env.sh first."
  exit 1
fi

python -m pip install --upgrade pip

if [[ -f "$LOCK_FILE" ]]; then
  echo "==> Syncing exact versions from $LOCK_FILE"
  if ! python -c "import piptools" >/dev/null 2>&1; then
    pip install pip-tools
  fi
  pip-sync "$LOCK_FILE"
else
  echo "==> Installing from $REQ_FILE"
  pip install -r "$REQ_FILE"
fi

echo "[ok] Environment updated."
