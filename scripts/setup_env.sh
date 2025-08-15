#!/usr/bin/env bash
set -euo pipefail

# Configuration
PYTHON_VERSION="3.11.9"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"

echo "==> Checking Homebrew and geospatial system deps (geos, proj, spatialindex, pkg-config)"
if command -v brew >/dev/null 2>&1; then
  brew update
  brew install geos proj spatialindex pkg-config || true
else
  echo "[warn] Homebrew not found. Ensure GEOS/PROJ/spatialindex are installed on your system."
fi

echo "==> Ensuring pyenv is installed"
if ! command -v pyenv >/dev/null 2>&1; then
  echo "[info] pyenv not found. Installing via Homebrew..."
  if command -v brew >/dev/null 2>&1; then
    brew install pyenv
  else
    echo "[error] pyenv is required but not installed. Install pyenv and re-run."
    exit 1
  fi
fi

# Initialize pyenv in this shell if necessary
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

echo "==> Installing Python ${PYTHON_VERSION} via pyenv (if needed)"
pyenv install -s "${PYTHON_VERSION}"
pyenv local "${PYTHON_VERSION}"

echo "==> Creating venv at ${VENV_DIR}"
python -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

echo "==> Installing Python dependencies from ${REQ_FILE}"
pip install -r "${REQ_FILE}"

# Create cache directory in parent (can be overridden by OSMNX_CACHE_FOLDER)
mkdir -p ../cache/osmnx || true

cat <<EOF

[ok] Environment setup complete.
- Python: $(python --version)
- Venv:   ${VENV_DIR}
- pyenv:  $(pyenv --version 2>/dev/null || echo 'n/a')

Next steps:
  1) cp .env.example .env   # optional
  2) make run                # or: scripts/run.sh
EOF
