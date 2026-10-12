"""``mapgen`` - command line interface for the Map Art Generator.

Examples::

    mapgen info ../data/ireland-and-northern-ireland-latest.osm.pbf
    mapgen regions bootstrap --from ../data/ireland-and-northern-ireland-latest.osm.pbf --city cork
    mapgen regions list
    mapgen generate --region cork --out ../output --prefix cork_blue
    mapgen generate --pbf ../data/regions/cork.osm.pbf --center 51.8986 -8.4756 \\
        --dist 1500 --set layers.buildings.facecolor=#ff0066
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow running the installed console script
    sys.path.insert(0, str(ROOT))

# Base style applied by `generate`; overridable for tests.
STYLE_PATH = ROOT / "style.json"

from map_core.core import regions as R  # noqa: E402
from mapgen.overrides import apply_overrides  # noqa: E402


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _add_catalog_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--catalog",
        default=str(R.DEFAULT_CATALOG),
        help=f"Region catalog JSON (default: {R.DEFAULT_CATALOG})",
    )


# --- info -------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    info = R.pbf_info(args.pbf)
    if args.json:
        _print_json(info.to_dict())
        return 0
    print(info.path)
    print(f"  size:      {info.size_mb:.1f} MB ({info.size_bytes:,} bytes)")
    print(f"  bbox:      {info.bbox}")
    print(f"  nodes:     {info.nodes:,}")
    print(f"  ways:      {info.ways:,}")
    print(f"  relations: {info.relations:,}")
    print(f"  source:    {info.generator or '?'}")
    print(f"  updated:   {info.timestamp or '?'}")
    return 0


# --- regions ----------------------------------------------------------------


def cmd_regions_list(args: argparse.Namespace) -> int:
    cat = R.RegionCatalog(args.catalog)
    rows = [{**r.to_dict(), "status": r.status(cat.dir)} for r in cat.list()]
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("No regions yet.")
        print(
            "Create some with, e.g.:\n"
            "  mapgen regions bootstrap --from ../data/ireland-and-northern-ireland-latest.osm.pbf --city cork"
        )
        return 0
    print(f"{'slug':<14}{'name':<18}{'size':>9}  {'status':<8}bbox")
    for r in rows:
        size = f"{r['size_bytes'] / (1024 * 1024):.1f} MB"
        bbox = (
            ",".join(f"{v:.3f}" for v in r["bbox"]) if r["bbox"] else "-"
        )
        print(f"{r['slug']:<14}{r['name']:<18}{size:>9}  {r['status']:<8}{bbox}")
    return 0


def cmd_regions_info(args: argparse.Namespace) -> int:
    cat = R.RegionCatalog(args.catalog)
    region = cat.get(args.slug)
    if region is None:
        print(f"No such region: {args.slug}", file=sys.stderr)
        return 1
    doc = {**region.to_dict(), "status": region.status(cat.dir)}
    doc["resolved_path"] = str(region.resolved_path(cat.dir))
    if args.json:
        _print_json(doc)
        return 0
    for k in ("slug", "name", "status", "size_bytes", "bbox", "source", "created"):
        print(f"  {k:<13} {doc[k]}")
    print(f"  {'path':<13} {doc['resolved_path']}")
    return 0


def cmd_regions_extract(args: argparse.Namespace) -> int:
    cat = R.RegionCatalog(args.catalog)
    cat.dir.mkdir(parents=True, exist_ok=True)
    bbox = R.bbox_from_center(args.center[0], args.center[1], args.dist, margin=args.margin)
    dst = cat.dir / f"{R.slugify(args.name)}.osm.pbf"
    if dst.is_file() and not args.overwrite:
        print(f"{dst} exists (use --overwrite)", file=sys.stderr)
        return 1
    R.extract_bbox(args.source, bbox, dst, strategy=args.strategy, overwrite=True)
    region = cat.add_from_file(
        args.name, dst, source=str(Path(args.source).resolve()), bbox=bbox, overwrite=True
    )
    print(f"Registered region '{region.name}' ({region.slug}) -> {dst}")
    return 0


def cmd_regions_bootstrap(args: argparse.Namespace) -> int:
    cat = R.RegionCatalog(args.catalog)
    only: Optional[List[str]] = None
    if args.city:
        only = [c.strip() for c in args.city.split(",") if c.strip()]
    elif args.all:
        only = ["all"]
    regions = R.bootstrap_cities(
        args.source, catalog=cat, only=only, overwrite=args.overwrite
    )
    print(f"Bootstrapped {len(regions)} region(s) into {cat.dir}")
    return 0


def cmd_regions_remove(args: argparse.Namespace) -> int:
    cat = R.RegionCatalog(args.catalog)
    region = cat.get(args.slug)
    if region is None:
        print(f"No such region: {args.slug}", file=sys.stderr)
        return 1
    cat.remove(args.slug)
    print(f"Removed '{args.slug}' from the catalog")
    if args.delete_file:
        p = region.resolved_path(cat.dir)
        if p.is_file():
            p.unlink()
            print(f"Deleted {p}")
    return 0


# --- generate ---------------------------------------------------------------


def _region_geometry(cat: R.RegionCatalog, region: R.Region):
    """Return (center [lat, lon], half-width metres) from a region bbox."""
    if not region.bbox:
        return None, None
    w, s, e, n = region.bbox
    center = [(s + n) / 2.0, (w + e) / 2.0]
    dist = (n - s) / 2.0 * R._M_PER_DEG
    return center, dist


def cmd_generate(args: argparse.Namespace) -> int:
    pbf = args.pbf
    center = args.center
    dist = args.dist

    if args.region:
        cat = R.RegionCatalog(args.catalog)
        region = cat.get(args.region)
        if region is None:
            print(f"No such region: {args.region}", file=sys.stderr)
            return 1
        pbf = str(region.resolved_path(cat.dir))
        c, d = _region_geometry(cat, region)
        center = center or c
        dist = dist if dist is not None else d

    if not pbf:
        print("Provide --region <slug> or --pbf <file>", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(ROOT / "main.py")]
    tmp_config: Optional[str] = None
    if args.set:
        style = json.loads(STYLE_PATH.read_text())
        apply_overrides(style, args.set)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", prefix="mapgen_", delete=False
        ) as fh:
            json.dump(style, fh, indent=2)
            tmp_config = fh.name
        cmd += ["--config", tmp_config]

    cmd += ["--pbf", pbf]
    if center:
        cmd += ["--center", str(center[0]), str(center[1])]
    if dist is not None:
        cmd += ["--dist", str(dist)]
    if args.out:
        cmd += ["--out", args.out]
    if args.prefix:
        cmd += ["--prefix", args.prefix]
    if args.layers:
        cmd += ["--layers", args.layers]

    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    finally:
        if tmp_config:
            Path(tmp_config).unlink(missing_ok=True)


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mapgen", description="Generate stylised map artwork from OSM data."
    )
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Inspect a PBF (bbox, counts, size)")
    info.add_argument("pbf")
    info.add_argument("--json", action="store_true")
    info.set_defaults(func=cmd_info)

    regions = sub.add_parser("regions", help="Manage clipped region PBFs")
    rsub = regions.add_subparsers(dest="regions_command", required=True)

    rlist = rsub.add_parser("list", help="List catalogued regions")
    _add_catalog_arg(rlist)
    rlist.add_argument("--json", action="store_true")
    rlist.set_defaults(func=cmd_regions_list)

    rinfo = rsub.add_parser("info", help="Show one region")
    _add_catalog_arg(rinfo)
    rinfo.add_argument("slug")
    rinfo.add_argument("--json", action="store_true")
    rinfo.set_defaults(func=cmd_regions_info)

    rext = rsub.add_parser("extract", help="Clip a new region from a large PBF")
    _add_catalog_arg(rext)
    rext.add_argument("--name", required=True)
    rext.add_argument("--center", type=float, nargs=2, metavar=("LAT", "LON"), required=True)
    rext.add_argument("--dist", type=float, required=True, metavar="METRES",
                      help="Half-width of the region in metres")
    rext.add_argument("--from", dest="source", required=True, help="Source PBF to clip")
    rext.add_argument("--margin", type=float, default=0.1)
    rext.add_argument("--strategy", default="smart")
    rext.add_argument("--overwrite", action="store_true")
    rext.set_defaults(func=cmd_regions_extract)

    rboot = rsub.add_parser("bootstrap", help="Clip a set of preset cities from one PBF")
    _add_catalog_arg(rboot)
    rboot.add_argument("--from", dest="source", required=True, help="Source PBF to clip")
    rboot.add_argument("--city", help="Comma-separated city names (default: all in the table)")
    rboot.add_argument("--all", action="store_true", help="Explicitly select every city")
    rboot.add_argument("--overwrite", action="store_true")
    rboot.set_defaults(func=cmd_regions_bootstrap)

    rrm = rsub.add_parser("remove", help="Remove a region from the catalog")
    _add_catalog_arg(rrm)
    rrm.add_argument("slug")
    rrm.add_argument("--delete-file", action="store_true")
    rrm.set_defaults(func=cmd_regions_remove)

    gen = sub.add_parser("generate", help="Generate map artwork")
    _add_catalog_arg(gen)
    gen.add_argument("--region", help="Region slug from the catalog")
    gen.add_argument("--pbf", help="Explicit PBF path")
    gen.add_argument("--center", type=float, nargs=2, metavar=("LAT", "LON"))
    gen.add_argument("--dist", type=float, metavar="METRES")
    gen.add_argument("--out", help="Output directory")
    gen.add_argument("--prefix", help="Output filename prefix")
    gen.add_argument("--layers", help="Comma-separated layers to enable")
    gen.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="Override any style value (repeatable)")
    gen.set_defaults(func=cmd_generate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError, R.OsmiumNotAvailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
