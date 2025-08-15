.PHONY: setup update run lock clean

PYTHON_VERSION := 3.11.9
VENV_DIR := .venv
REQ := requirements.txt
LOCK := requirements.lock.txt

setup:
	@echo "==> Project setup (pyenv + venv)"
	@chmod +x scripts/*.sh || true
	@bash scripts/setup_env.sh

update:
	@echo "==> Updating environment"
	@bash scripts/update_env.sh

run:
	@echo "==> Running app"
	@bash scripts/run.sh

lock:
	@echo "==> Generating lock file with pip-tools"
	@$(VENV_DIR)/bin/python -m pip install --upgrade pip pip-tools
	@$(VENV_DIR)/bin/pip-compile --generate-hashes -o $(LOCK) $(REQ)
	@echo "[ok] Wrote $(LOCK)"

clean:
	@echo "==> Cleaning caches and build outputs"
	rm -rf __pycache__ **/__pycache__ .pytest_cache build dist *.egg-info
	@echo "(not removing $(VENV_DIR); delete manually if desired)"
