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

echo "Data layout (resolved by map_core/paths.py)"
PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY' || status=1
from map_core import paths

def h(n):
    return f"{n/1e9:.1f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"

info = paths.describe()
src = f"  (MAPGEN_DATA_DIR={info['env_override']})" if info["env_override"] else ""
print(f"  data root:  {info['data_root']}{src}")
print(f"  free space: {h(info['free_space_bytes'])}")
for name, st in info["dirs"].items():
    mark = "[ok]  " if st["files"] else "[--]  "
    print(f"  {mark}{name:<9} {st['files']:>4} file(s)  {h(st['size_bytes']):>9}  {st['path']}")
if info["dirs"]["sources"]["files"] == 0:
    print("  [warn] no source PBFs in sources/; download a Geofabrik extract (see docs/regions.md)")
if not info["osmium_available"]:
    print("  [warn] osmium-tool missing: clipping regions will not work")
PY

exit $status
