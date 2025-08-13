# Usage

## Local Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
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
