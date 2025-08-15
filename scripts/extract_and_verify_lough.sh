#!/usr/bin/env bash
# Extract Cork from a larger Ireland PBF (preserving relations) and verify
# whether "The Lough" is present. Produces a GeoJSON for inspection.
#
# Usage:
#   bash scripts/extract_and_verify_lough.sh
#
# Requirements:
#   - osmium-tool (brew install osmium-tool)
#   - Optional: Python + pyrosm (for richer verification)
#
set -euo pipefail

# Paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
INPUT_PBF="${ROOT_DIR}/../osm-data/ireland.pbf"
POLY_FILE="${ROOT_DIR}/tools/cork.poly"
CORK_PBF="${ROOT_DIR}/../osm-data/ireland_cork.osm.pbf"
OUT_DIR="${ROOT_DIR}/output"
OUT_DIR_OSMDATA="${ROOT_DIR}/../osm-data"

# Verify prerequisites
if ! command -v osmium >/dev/null 2>&1; then
  echo "ERROR: osmium-tool is required. Install with: brew install osmium-tool" >&2
  exit 2
fi

if [[ ! -f "$INPUT_PBF" ]]; then
  echo "ERROR: Input PBF not found: $INPUT_PBF" >&2
  exit 2
fi

if [[ ! -f "$POLY_FILE" ]]; then
  echo "ERROR: Polygon file not found: $POLY_FILE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$OUT_DIR_OSMDATA"

# 1) Extract Cork area from the full Ireland PBF, preserving relations
echo "[1/3] Extracting Cork from ${INPUT_PBF} using ${POLY_FILE} ..."
osmium extract --polygon "$POLY_FILE" \
  --set-bounds --strategy=smart \
  -o "$CORK_PBF" \
  "$INPUT_PBF"

# 2) Verify The Lough by name using Python+pyrosm if available; fall back to osmium
echo "[2/3] Verifying presence of 'The Lough' in the Cork extract ..."
if command -v python3 >/dev/null 2>&1 && python3 - <<'PY'
import sys
try:
    import pyrosm  # noqa: F401
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
  echo " - Using Python + pyrosm verifier"
  python3 "$ROOT_DIR/scripts/find_lough_in_pbf.py" "$CORK_PBF" \
    --name "The Lough" --contains \
    --center 51.894333 -8.480534 --dist 3000 \
    --save "$OUT_DIR/the_lough.geojson" || true
else
  echo " - pyrosm not available; using osmium + jq for contains-match"
  GEOJSON_SEQ="${OUT_DIR}/the_lough_contains.geojsonseq"
  GEOJSON_FIRST="${OUT_DIR}/the_lough_first.geojson"
  # Export as GeoJSONSeq and filter case-insensitive on name containing 'The Lough'
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required for contains search. Install with: brew install jq" >&2
  else
    osmium export "$CORK_PBF" -f geojsonseq \
      | jq -c 'select(.properties.name? | test("The Lough"; "i"))' \
      > "$GEOJSON_SEQ" || true
    COUNT=$(wc -l < "$GEOJSON_SEQ" | tr -d ' \t\n')
    echo " - Contains-match count in Cork extract: ${COUNT}"
    if [[ "${COUNT}" != "0" ]]; then
      head -n 1 "$GEOJSON_SEQ" | jq '.' > "$GEOJSON_FIRST" || true
      echo " - Wrote first match to: $GEOJSON_FIRST"
    else
      echo " - No matches in Cork extract; searching full Ireland PBF (this may take longer) ..."
      GEOJSON_SEQ_FULL="${OUT_DIR}/the_lough_contains_full.geojsonseq"
      GEOJSON_FIRST_FULL="${OUT_DIR}/the_lough_first_full.geojson"
      osmium export "$INPUT_PBF" -f geojsonseq \
        | jq -c 'select(.properties.name? | test("The Lough"; "i"))' \
        > "$GEOJSON_SEQ_FULL" || true
      COUNT_FULL=$(wc -l < "$GEOJSON_SEQ_FULL" | tr -d ' \t\n')
      echo " - Contains-match count in full Ireland: ${COUNT_FULL}"
      if [[ "${COUNT_FULL}" != "0" ]]; then
        head -n 1 "$GEOJSON_SEQ_FULL" | jq '.' > "$GEOJSON_FIRST_FULL" || true
        echo " - Wrote first full-file match to: $GEOJSON_FIRST_FULL"
      fi
    fi
  fi
fi

# 3) Summarize result locations
echo "[3/3] Done. Outputs:"
echo " - Cork extract: $CORK_PBF"
echo " - Verifier output (if pyrosm): $OUT_DIR/the_lough.geojson"
echo " - Verifier output (if osmium): $OUT_DIR/the_lough_osmium.geojson"
