#!/usr/bin/env bash
# Check external and Python prerequisites for the map generator.
#
# Usage: bash scripts/doctor.sh   (or: make doctor)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY="${ROOT_DIR}/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

status=0

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; status=1; }

echo "External tools"
if command -v osmium >/dev/null 2>&1; then
  if ver=$(osmium --version 2>/dev/null | head -n1); then
    ok "osmium-tool: $ver"
  else
    fail "osmium found but fails to run (library mismatch?). Try: brew reinstall osmium-tool"
  fi
else
  fail "osmium-tool not found. macOS: brew install osmium-tool | Debian/Pi: sudo apt install osmium-tool"
fi

echo "Python ($PY)"
"$PY" - <<'PY' || status=1
import importlib, sys
from importlib.metadata import version, PackageNotFoundError

def check(mod, hint=""):
    try:
        importlib.import_module(mod)
        try:
            v = version(mod)
        except PackageNotFoundError:
            v = ""
        print(f"  [ok]   {mod} {v}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {mod}: {e}{(' - ' + hint) if hint else ''}")
        return False

good = True
good &= check("osmnx")
good &= check("geopandas")
good &= check("matplotlib")
good &= check("pyrosm", "required for local PBF reading")
good &= check("scour", "SVG optimisation")
good &= check("cairosvg", "PNG/PDF export; needs libcairo")
good &= check("flask")
sys.exit(0 if good else 1)
PY

echo "Data"
if [[ -d "${ROOT_DIR}/../data" ]]; then
  n=$(find "${ROOT_DIR}/../data" -maxdepth 2 -name '*.pbf' | wc -l | tr -d ' ')
  ok "../data exists (${n} .pbf file(s))"
else
  warn "../data not found; place source PBFs there (see docs/regions.md)"
fi

exit $status
