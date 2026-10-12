#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
APP_ENTRY="app.py"
export FLASK_APP="$APP_ENTRY"
export FLASK_ENV="development"

# Data/cache locations are resolved in Python (map_core/paths.py); see .env.example.
# Load optional .env if present
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -I {} echo {})
fi

# shellcheck disable=SC1090
if [[ -d "$VENV_DIR" ]]; then
  source "$VENV_DIR/bin/activate"
else
  echo "[error] $VENV_DIR not found. Run scripts/setup_env.sh first."
  exit 1
fi

# Run Flask dev server
python "$APP_ENTRY" "$@"
