"""Region extraction and PBF inspection built on osmium-tool.

Large country-scale PBF files (e.g. the 370 MB Ireland extract) are far too
slow to read per map run. This module clips them into small, fast,
area-specific PBFs ("regions") and reports what a PBF contains.

All heavy lifting is done by the ``osmium`` CLI (https://osmcode.org/osmium-tool/),
which streams the input and finishes a city-sized extract in a minute or two.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from map_core.core.util import log_progress

BBox = Tuple[float, float, float, float]  # (west, south, east, north)

# Rough metres per degree at the equator; longitude is scaled by cos(lat).
_M_PER_DEG = 111_320.0

# auto_clip only helps for a *small* map area; a bbox this large means
# something is wrong (e.g. lat/lon scrambled) or clipping won't pay off.
_MAX_AUTO_CLIP_SPAN_DEG = 5.0


class OsmiumNotAvailable(RuntimeError):
    """Raised when the osmium CLI is missing or cannot run."""


# --- osmium discovery -------------------------------------------------------


def osmium_path() -> Optional[str]:
    return shutil.which("osmium")


def osmium_available() -> bool:
    """True if ``osmium --version`` runs successfully."""
    exe = osmium_path()
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "--version"], check=True, capture_output=True, timeout=10
        )
        return True
    except Exception:
        return False


def _require_osmium() -> str:
    exe = osmium_path()
    if not exe:
        raise OsmiumNotAvailable(
            "osmium-tool is required for region extraction. "
            "Install with 'brew install osmium-tool' (macOS) or "
            "'sudo apt install osmium-tool' (Debian/Raspberry Pi), then run 'make doctor'."
        )
    try:
        subprocess.run([exe, "--version"], check=True, capture_output=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        raise OsmiumNotAvailable(
            f"osmium is installed but fails to run ({e}). "
            "On macOS try 'brew reinstall osmium-tool'."
        ) from e
    return exe


# --- bbox helpers -----------------------------------------------------------


def bbox_from_center(lat: float, lon: float, dist_m: float, margin: float = 0.0) -> BBox:
    """Square bbox of half-width ``dist_m`` around a point, optionally grown by ``margin`` (fraction)."""
    import math

    d = float(dist_m) * (1.0 + float(margin))
    dlat = d / _M_PER_DEG
    dlon = d / (_M_PER_DEG * max(0.01, math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def normalize_bbox(bbox: Iterable[float]) -> BBox:
    """Coerce any 4-sequence to (west, south, east, north) with ordering fixed."""
    w, s, e, n = (float(v) for v in bbox)
    if w > e:
        w, e = e, w
    if s > n:
        s, n = n, s
    return (w, s, e, n)


def bbox_to_osmium(bbox: BBox) -> str:
    w, s, e, n = normalize_bbox(bbox)
    return f"{w:.7f},{s:.7f},{e:.7f},{n:.7f}"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "region"


# --- fileinfo ---------------------------------------------------------------


@dataclass
class PbfInfo:
    path: str
    size_bytes: int
    bbox: Optional[BBox]
    nodes: int
    ways: int
    relations: int
    timestamp: Optional[str]
    generator: Optional[str]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["size_mb"] = round(self.size_mb, 1)
        return d


def pbf_info(path: str | Path, extended: bool = True) -> PbfInfo:
    """Inspect a PBF with ``osmium fileinfo``.

    ``extended=True`` reads the whole file to compute the data bbox and object
    counts (about 1 s for a 370 MB file). With ``extended=False`` only the
    header is read, which is instant but may lack a bbox.
    """
    exe = _require_osmium()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PBF not found: {p}")

    cmd = [exe, "fileinfo", "-j"]
    if extended:
        cmd.append("-e")
    cmd.append(str(p))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    doc = json.loads(res.stdout)

    data = doc.get("data", {}) or {}
    header = doc.get("header", {}) or {}
    file_ = doc.get("file", {}) or {}

    # Prefer the header bbox (the requested extent written by --set-bounds).
    # The data bbox can be much larger for "smart" extracts because complete
    # relations pull in members from far outside the clip area.
    bbox: Optional[BBox] = None
    boxes = header.get("boxes") or []
    raw_bbox = boxes[0] if boxes else None
    if not raw_bbox:
        raw_bbox = data.get("bbox")
    if raw_bbox and len(raw_bbox) == 4:
        bbox = normalize_bbox(raw_bbox)

    counts = data.get("count", {}) or {}
    ts = (data.get("timestamp") or {}).get("last") or None
    if ts and ts.startswith("1970-"):
        ts = None
    opt = header.get("option", {}) or {}
    ts = opt.get("osmosis_replication_timestamp") or opt.get("timestamp") or ts

    return PbfInfo(
        path=str(p),
        size_bytes=int(file_.get("size") or p.stat().st_size),
        bbox=bbox,
        nodes=int(counts.get("nodes", 0)),
        ways=int(counts.get("ways", 0)),
        relations=int(counts.get("relations", 0)),
        timestamp=ts,
        generator=opt.get("generator"),
    )


# --- extraction -------------------------------------------------------------


def _run_osmium_extract(
    exe: str,
    src: Path,
    dst: Path,
    selector: List[str],
    strategy: str,
    overwrite: bool,
    progress_cb: Optional[Callable[[str], None]],
) -> Path:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst} (use overwrite=True)")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "extract",
        *selector,
        "--set-bounds",
        f"--strategy={strategy}",
        "-O",
        "-o",
        str(dst),
        str(src),
    ]
    log = progress_cb or log_progress
    log(f"[regions] osmium extract -> {dst.name} ({' '.join(selector)}, strategy={strategy})")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"osmium extract failed ({res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
        )
    size_mb = dst.stat().st_size / (1024 * 1024)
    log(f"[regions] wrote {dst} ({size_mb:.1f} MB)")
    return dst


def extract_bbox(
    src: str | Path,
    bbox: BBox,
    dst: str | Path,
    strategy: str = "smart",
    overwrite: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Clip ``src`` to ``bbox`` (west, south, east, north) and write ``dst``.

    ``strategy="smart"`` keeps multipolygon relations (lakes, rivers, parks)
    complete even when they cross the bbox edge; ``"simple"`` is faster but
    may drop them; ``"complete_ways"`` is the middle ground.
    """
    exe = _require_osmium()
    src_p, dst_p = Path(src), Path(dst)
    if not src_p.is_file():
        raise FileNotFoundError(f"Source PBF not found: {src_p}")
    return _run_osmium_extract(
        exe,
        src_p,
        dst_p,
        ["--bbox", bbox_to_osmium(bbox)],
        strategy,
        overwrite,
        progress_cb,
    )


def grow_bbox(bbox: BBox, margin: float) -> BBox:
    """Grow a bbox by ``margin`` (fraction of its width/height) on every side."""
    w, s, e, n = normalize_bbox(bbox)
    dx = (e - w) * float(margin)
    dy = (n - s) * float(margin)
    return (w - dx, s - dy, e + dx, n + dy)


def auto_clip(
    src: str | Path,
    bbox: Optional[BBox],
    cache_dir: str | Path = Path("..") / "cache" / "clips",
    threshold_mb: float = 20.0,
    margin: float = 0.2,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Return a small PBF covering ``bbox``, clipping ``src`` on first use.

    If ``src`` is below ``threshold_mb``, ``bbox`` is unknown, or osmium is not
    available, ``src`` is returned unchanged. Otherwise the bbox is grown by
    ``margin`` and extracted into ``cache_dir`` under a name derived from the
    source file identity and the bbox, so repeated runs over the same area
    reuse the clip.
    """
    log = progress_cb or log_progress
    src_p = Path(src)
    if bbox is None or not src_p.is_file():
        return src_p

    size_mb = src_p.stat().st_size / (1024 * 1024)
    if size_mb < float(threshold_mb):
        return src_p

    # Defense in depth: a scrambled bbox (e.g. lat/lon swapped upstream) would
    # make osmium rewrite the whole file. Reject anything implausible.
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError):
        log(f"[regions] auto-clip skipped: unreadable bbox {bbox!r}")
        return src_p
    if not (-90.0 <= s <= n <= 90.0 and -180.0 <= w <= e <= 180.0):
        log(f"[regions] auto-clip skipped: implausible bbox {(w, s, e, n)}")
        return src_p
    if (n - s) > _MAX_AUTO_CLIP_SPAN_DEG or (e - w) > _MAX_AUTO_CLIP_SPAN_DEG:
        log(
            f"[regions] auto-clip skipped: bbox spans {(n - s):.1f}x{(e - w):.1f} deg "
            "which is too large for a clipped read to help."
        )
        return src_p

    if not osmium_available():
        log(
            f"[regions] auto-clip skipped: osmium not available; reading full {size_mb:.0f} MB PBF (slow). "
            "Install osmium-tool to enable clipping."
        )
        return src_p

    clip_bbox = grow_bbox(bbox, margin)
    st = src_p.stat()
    key_src = json.dumps(
        {
            "src": str(src_p.resolve()),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "bbox": [round(v, 5) for v in clip_bbox],
        },
        sort_keys=True,
    )
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:12]
    dst = Path(cache_dir) / f"{src_p.name.split('.')[0]}_{key}.osm.pbf"

    if dst.is_file() and dst.stat().st_size > 0:
        log(f"[regions] auto-clip CACHE HIT -> {dst} ({dst.stat().st_size / (1024*1024):.1f} MB)")
        return dst

    log(
        f"[regions] auto-clip: {src_p.name} is {size_mb:.0f} MB (> {threshold_mb:.0f} MB); "
        f"clipping to bbox {tuple(round(v, 4) for v in clip_bbox)} once (cached for reuse)"
    )
    try:
        return extract_bbox(src_p, clip_bbox, dst, strategy="smart", overwrite=True, progress_cb=log)
    except Exception as e:  # noqa: BLE001
        log(f"[regions] auto-clip failed ({e}); falling back to full PBF")
        return src_p


def extract_polygon(
    src: str | Path,
    polygon_file: str | Path,
    dst: str | Path,
    strategy: str = "smart",
    overwrite: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Clip ``src`` to a ``.poly`` or GeoJSON polygon file and write ``dst``."""
    exe = _require_osmium()
    src_p, poly_p, dst_p = Path(src), Path(polygon_file), Path(dst)
    if not src_p.is_file():
        raise FileNotFoundError(f"Source PBF not found: {src_p}")
    if not poly_p.is_file():
        raise FileNotFoundError(f"Polygon file not found: {poly_p}")
    return _run_osmium_extract(
        exe,
        src_p,
        dst_p,
        ["--polygon", str(poly_p)],
        strategy,
        overwrite,
        progress_cb,
    )


# --- region catalog ---------------------------------------------------------

DEFAULT_CATALOG = Path("..") / "data" / "regions" / "regions.json"


@dataclass
class Region:
    """A clipped, named area PBF registered in the catalog."""

    slug: str
    name: str
    path: str  # relative to the catalog's directory, or absolute
    bbox: Optional[BBox] = None
    source: Optional[str] = None
    size_bytes: int = 0
    created: Optional[str] = None
    counts: Dict[str, Any] = field(default_factory=dict)
    thumbnail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Region":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        region = cls(**{k: v for k, v in data.items() if k in known})
        # JSON round-trips tuples as lists; keep bbox a consistent tuple.
        if isinstance(region.bbox, (list, tuple)) and len(region.bbox) == 4:
            region.bbox = tuple(float(v) for v in region.bbox)  # type: ignore[assignment]
        return region

    def resolved_path(self, catalog_dir: Path) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else (Path(catalog_dir) / p).resolve()

    def resolved_thumbnail(self, catalog_dir: Path) -> Optional[Path]:
        if not self.thumbnail:
            return None
        p = Path(self.thumbnail)
        return p if p.is_absolute() else (Path(catalog_dir) / p).resolve()

    def status(self, catalog_dir: Path) -> str:
        return "ok" if self.resolved_path(catalog_dir).is_file() else "missing"


class RegionCatalog:
    """JSON-backed index of clipped region PBFs.

    Keeps the catalog file and the region PBFs in the same directory by
    default so paths stay relative and the whole directory can be synced to
    another machine (e.g. the Pi) in one copy.
    """

    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG) -> None:
        self.path = Path(catalog_path)
        self.dir = self.path.parent
        self._regions: Dict[str, Region] = {}
        self.load()

    def load(self) -> "RegionCatalog":
        self._regions = {}
        if self.path.is_file():
            try:
                doc = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                doc = {}
            for item in doc.get("regions", []):
                region = Region.from_dict(item)
                self._regions[region.slug] = region
        return self

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": 1,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "regions": [r.to_dict() for r in self.list()],
        }
        self.path.write_text(json.dumps(doc, indent=2) + "\n")

    def list(self) -> List[Region]:
        return sorted(self._regions.values(), key=lambda r: r.name.lower())

    def get(self, slug: str) -> Optional[Region]:
        return self._regions.get(slugify(slug))

    def add(self, region: Region, overwrite: bool = False) -> Region:
        if region.slug in self._regions and not overwrite:
            raise FileExistsError(
                f"Region '{region.slug}' already exists (use overwrite=True)"
            )
        self._regions[region.slug] = region
        self.save()
        return region

    def remove(self, slug: str) -> bool:
        slug = slugify(slug)
        if slug in self._regions:
            del self._regions[slug]
            self.save()
            return True
        return False

    def refresh(self, slug: str) -> Region:
        """Recompute size/bbox/counts for a region from its file on disk."""
        region = self.get(slug)
        if region is None:
            raise KeyError(f"No such region: {slug}")
        p = region.resolved_path(self.dir)
        if not p.is_file():
            raise FileNotFoundError(f"Region file missing: {p}")
        info = pbf_info(p)
        region.size_bytes = info.size_bytes
        region.bbox = info.bbox or region.bbox
        region.counts = {
            "nodes": info.nodes,
            "ways": info.ways,
            "relations": info.relations,
        }
        self.save()
        return region

    def add_from_file(
        self,
        name: str,
        pbf_path: str | Path,
        source: Optional[str] = None,
        bbox: Optional[BBox] = None,
        overwrite: bool = False,
        inspect: bool = True,
    ) -> Region:
        """Register an existing region PBF, optionally inspecting it first."""
        p = Path(pbf_path)
        if not p.is_file():
            raise FileNotFoundError(f"Region file not found: {p}")
        slug = slugify(name)
        try:
            rel = str(p.resolve().relative_to(self.dir.resolve()))
        except ValueError:
            rel = str(p.resolve())

        size = p.stat().st_size
        used_bbox = bbox
        counts: Dict[str, Any] = {}
        if inspect:
            try:
                info = pbf_info(p)
                size = info.size_bytes
                used_bbox = used_bbox or info.bbox
                counts = {
                    "nodes": info.nodes,
                    "ways": info.ways,
                    "relations": info.relations,
                }
            except Exception as e:  # noqa: BLE001 - inspection is best-effort
                log_progress(f"[regions] could not inspect {p}: {e}")

        region = Region(
            slug=slug,
            name=name,
            path=rel,
            bbox=used_bbox,
            source=source,
            size_bytes=size,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            counts=counts,
        )
        return self.add(region, overwrite=overwrite)


# --- city presets / bootstrap ----------------------------------------------

DEFAULT_CITY_TABLE = Path("config") / "regions" / "ireland_cities.json"


def load_city_table(path: str | Path = DEFAULT_CITY_TABLE) -> Dict[str, Any]:
    """Load a city-centre table (name/center/radius_m) used for bootstrap."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"City table not found: {p}")
    return json.loads(p.read_text())


def _select_cities(
    table: Dict[str, Any], only: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    cities = list(table.get("cities", []))
    if not only:
        return cities
    wanted = {slugify(n) for n in only if str(n).strip()}
    if "all" in {str(n).strip().lower() for n in only}:
        return cities
    selected = [c for c in cities if slugify(c.get("name", "")) in wanted]
    missing = wanted - {slugify(c.get("name", "")) for c in selected}
    if missing:
        raise KeyError(
            f"Unknown city/cities: {sorted(missing)}. "
            f"Known: {sorted(slugify(c['name']) for c in cities)}"
        )
    return selected


def bootstrap_cities(
    src_pbf: str | Path,
    catalog: Optional[RegionCatalog] = None,
    table_path: str | Path = DEFAULT_CITY_TABLE,
    only: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    strategy: str = "smart",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Region]:
    """Clip one PBF per city from ``src_pbf`` and register them in ``catalog``.

    ``only`` is an optional list of city names/slugs to limit the run
    (e.g. ``["cork", "dublin"]``); omitting it (or passing ``"all"``) does
    every city in the table.
    """
    log = progress_cb or log_progress
    src_p = Path(src_pbf)
    if not src_p.is_file():
        raise FileNotFoundError(f"Source PBF not found: {src_p}")

    catalog = catalog or RegionCatalog()
    catalog.dir.mkdir(parents=True, exist_ok=True)
    table = load_city_table(table_path)
    default_radius = float(table.get("default_radius_m", 6000))
    cities = _select_cities(table, only)

    regions: List[Region] = []
    for i, city in enumerate(cities, 1):
        name = str(city["name"])
        slug = slugify(name)
        lat, lon = (float(v) for v in city["center"])
        radius = float(city.get("radius_m", default_radius))
        bbox = bbox_from_center(lat, lon, radius, margin=0.1)
        dst = catalog.dir / f"{slug}.osm.pbf"
        log(f"[bootstrap] ({i}/{len(cities)}) {name} -> {dst.name} ({radius / 1000:.0f} km box)")
        if dst.is_file() and not overwrite:
            log(f"[bootstrap] {name}: {dst.name} exists, registering as-is")
        else:
            extract_bbox(src_p, bbox, dst, strategy=strategy, overwrite=True, progress_cb=log)
        region = catalog.add_from_file(
            name, dst, source=str(src_p), bbox=bbox, overwrite=True
        )
        regions.append(region)
    return regions
