# Usage

## Prerequisites
- Python 3.10+
- `osmium-tool` for clipping large PBF files into regions
  - macOS: `brew install osmium-tool`
  - Debian/Raspberry Pi: `sudo apt install osmium-tool`
- Check everything with `make doctor`

## Local Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
make doctor
```

## Run the Web UI
```
FLASK_APP=app.py flask run
```
Open http://127.0.0.1:5000 and configure parameters.

## Outputs
- Originals under `output/<run>/`
- Optimized under `output/<run>/optimized/`
- PNGs adjacent to the selected source (original/optimized)

## CLI (if applicable)
You can call `main.py` programmatically for batch runs; see code in `main.py` for the function that orchestrates generation.
