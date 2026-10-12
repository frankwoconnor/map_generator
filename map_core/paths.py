"""Single source of truth for every filesystem location the project uses.

Nothing in the codebase should build a data path by hand. Call the getters
here instead: they return **absolute** paths, honour the ``MAPGEN_DATA_DIR``
environment variable, and create directories on demand.

Default layout (data root defaults to the directory *containing* the repo, so
the existing sibling workspace keeps working)::

    Maps/                    <- MAPGEN_DATA_DIR (default: repo parent)
    ├── map_generator/       <- the git repo (code only)
    ├── data/
    │   ├── sources/         raw PBF extracts (big, rarely read)
    │   └── regions/         clipped areas + regions.json (small, read often)
    ├── cache/
    │   ├── clips/           auto-clip results
    │   ├── features/        parquet feature cache
    │   └── osmnx/           OSMnx HTTP cache
    └── output/              run artifacts

Set ``MAPGEN_DATA_DIR`` to relocate everything (e.g. a big disk, or the Pi).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

# .../map_generator/map_core/paths.py -> parents[1] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

_ENV_VAR = "MAPGEN_DATA_DIR"


def _load_dotenv() -> None:
    """Best-effort .env load (optional dependency); never fatal."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001 - dotenv is optional
        pass


_load_dotenv()


def data_root() -> Path:
    """Root holding all runtime data.

    ``MAPGEN_DATA_DIR`` if set, otherwise the directory containing the repo
    (so ``../data``, ``../cache`` and ``../output`` keep working).
    """
    raw = os.environ.get(_ENV_VAR)
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT.parent


def sources_dir() -> Path:
    """Raw source PBF extracts, as downloaded (Geofabrik etc.)."""
    return data_root() / "data" / "sources"


def regions_dir() -> Path:
    """Clipped, named region PBFs and their catalog."""
    return data_root() / "data" / "regions"


def region_catalog_path() -> Path:
    return regions_dir() / "regions.json"


def cache_dir() -> Path:
    return data_root() / "cache"


def clips_dir() -> Path:
    """Auto-clip results (bbox slices extracted on demand)."""
    return cache_dir() / "clips"


def features_dir() -> Path:
    """Parquet feature cache keyed by query+tags."""
    return cache_dir() / "features"


def osmnx_cache_dir() -> Path:
    """OSMnx HTTP response cache."""
    return cache_dir() / "osmnx"


def outputs_dir() -> Path:
    """Base directory for run artifacts."""
    return data_root() / "output"


# Directories created by ensure_dirs() and reported by describe().
_DIR_GETTERS = {
    "sources": sources_dir,
    "regions": regions_dir,
    "clips": clips_dir,
    "features": features_dir,
    "osmnx": osmnx_cache_dir,
    "output": outputs_dir,
}


def ensure_dirs() -> None:
    """Create every managed directory if missing (idempotent)."""
    for getter in _DIR_GETTERS.values():
        getter().mkdir(parents=True, exist_ok=True)


# --- resolution helpers -----------------------------------------------------


def resolve_data_path(path: PathLike) -> Path:
    """Resolve a possibly-relative data path to an absolute one.

    Tries, in order: absolute as-is; relative to CWD; relative to the repo
    root; relative to ``sources_dir``; relative to ``regions_dir``. Returns the
    first that exists, else the CWD-relative candidate (so error messages show
    something sensible).
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p)
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    for base in (REPO_ROOT, sources_dir(), regions_dir(), data_root()):
        candidate = base / p
        if candidate.exists():
            return candidate.resolve()
    return cwd_candidate


def resolve_output_dir(path: PathLike) -> Path:
    """Resolve an output directory; relative paths are anchored to the repo root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# --- introspection ----------------------------------------------------------


def _dir_stats(path: Path) -> Dict[str, Any]:
    size = 0
    files = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    size += f.stat().st_size
                    files += 1
                except OSError:
                    pass
    return {"path": str(path), "files": files, "size_bytes": size}


def free_space_bytes(path: Optional[PathLike] = None) -> int:
    target = Path(path) if path else data_root()
    probe = target if target.exists() else target.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def osmium_available() -> bool:
    from map_core.core.regions import osmium_available as _avail

    return _avail()


def describe() -> Dict[str, Any]:
    """Everything the UI needs to explain where data lives and how much there is."""
    dirs = {name: _dir_stats(getter()) for name, getter in _DIR_GETTERS.items()}
    return {
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root()),
        "env_override": os.environ.get(_ENV_VAR),
        "free_space_bytes": free_space_bytes(),
        "dirs": dirs,
        "osmium_available": osmium_available(),
    }
