"""Topology-safe water geometry: dissolve water-area polygons into one body.

Mirrors the approach that reads as a clean, connected water feature on the map
(instead of many loose riverbank fragments):

1. Work in a *metric* CRS (UTM) so areas / buffers / precision are in metres.
2. ``make_valid`` + ``set_precision(grid)`` for a robust, valid geometry.
3. ``union_all`` all water-area polygons -> a single dissolved multipolygon
   (this closes the "river drawn as many disconnected pieces" problem).
4. Seam-heal via ``buffer(+h).buffer(-h)`` to close sub-heal gaps between
   originally-adjacent polygons.
5. Drop sub-min-area / sub-min-width slivers, then reproject back to the
   original (plotting) CRS.

Linear waterways (river/stream/canal/... centerlines) are separated and returned
unchanged so the renderer can draw them as typed strokes on top of the fill.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import geopandas as gpd
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon


def iter_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[Polygon] = []
        for part in geom.geoms:
            out.extend(iter_polygons(part))
        return out
    return []


def iter_lines(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, shapely.geometry.MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[LineString] = []
        for part in geom.geoms:
            out.extend(iter_lines(part))
        return out
    return []


def _polygonal(geom):
    polys = iter_polygons(geom)
    return shapely.union_all(polys) if polys else None


def _linear(geom):
    lines = iter_lines(geom)
    return shapely.union_all(lines) if lines else None


def _metric_crs(gdf: gpd.GeoDataFrame, lon: Optional[float] = None,
                lat: Optional[float] = None) -> str:
    """A projected CRS in metres for the given data (UTM preferred)."""
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        # Fall back to a global metric CRS if UTM can't be estimated.
        try:
            import pyproj

            if lon is not None and lat is not None:
                return pyproj.CRS.from_user_input(
                    pyproj.Proj(proj="utm", zone=max(1, min(60, int(lon // 6) + 31)),
                                ellps="WGS84")
                )
        except Exception:
            pass
        return "EPSG:3857"


def clean_water_layer(
    gdf: gpd.GeoDataFrame,
    *,
    seam_heal_m: float = 0.15,
    precision_grid_m: float = 0.05,
    min_area_m2: float = 0.0,
    min_width_m: float = 0.0,
) -> Tuple[Optional[gpd.GeoDataFrame], gpd.GeoDataFrame]:
    """Dissolve + heal water-area polygons and separate linear waterways.

    Returns ``(water_areas, waterways)`` both reprojected back to the input CRS:

    - ``water_areas``: a GeoDataFrame with a single (Multi)Polygon feature (or
      an empty GeoDataFrame) - the dissolved, gap-free water body.
    - ``waterways``: a GeoDataFrame of LineString/MultiLineString features with
      a ``waterway`` column classifying each (for typed stroke widths).
    """
    if gdf is None or len(gdf) == 0:
        crs = getattr(gdf, "crs", None) or "EPSG:4326"
        empty = gpd.GeoDataFrame(geometry=[], crs=crs)
        return empty, empty

    src_crs = gdf.crs if gdf.crs is not None else "EPSG:4326"
    metric = _metric_crs(gdf)
    g = gdf.to_crs(metric).copy()
    g["geometry"] = g.geometry.map(shapely.make_valid)

    # Separate polygonal water areas from linear waterways.
    poly_parts: List[Polygon] = []
    kind_col = "waterway" if "waterway" in g.columns else None
    lines_rows = []
    for _, row in g.iterrows():
        poly = _polygonal(row.geometry)
        if poly is not None and not poly.is_empty:
            poly = shapely.set_precision(
                poly, grid_size=precision_grid_m, mode="valid_output"
            )
            poly_parts.append(poly)
        line = _linear(row.geometry)
        if line is not None and not line.is_empty and kind_col:
            kind = row.get(kind_col)
            lines_rows.append((line, kind if isinstance(kind, str) else ""))

    # Dissolve + seam-heal the water areas into one clean body.
    water_metric = MultiPolygon()
    if poly_parts:
        water = shapely.union_all(poly_parts, grid_size=precision_grid_m)
        water = _polygonal(shapely.make_valid(water)) or MultiPolygon()
        if seam_heal_m and seam_heal_m > 0:
            healed = water.buffer(seam_heal_m, join_style="round").buffer(
                -seam_heal_m, join_style="round"
            )
            water = _polygonal(shapely.make_valid(healed)) or MultiPolygon()

        kept: List[Polygon] = []
        for p in iter_polygons(water):
            a = p.area
            if min_area_m2 and a < float(min_area_m2):
                continue
            if min_width_m and min_width_m > 0:
                perim = p.length
                if perim > 0 and (a / perim) < float(min_width_m):
                    continue
            kept.append(p)
        water_metric = shapely.union_all(kept) if kept else MultiPolygon()

    water_areas = gpd.GeoDataFrame(
        geometry=[water_metric], crs=metric
    )
    if len(water_areas) and not water_areas.geometry.iloc[0].is_empty:
        water_areas = water_areas.to_crs(src_crs)
    else:
        water_areas = gpd.GeoDataFrame(geometry=[], crs=src_crs)

    waterways = gpd.GeoDataFrame(
        lines_rows, columns=["geometry", "waterway"], crs=metric
    ) if lines_rows else gpd.GeoDataFrame(
        geometry=[], data={}, crs=metric
    )
    if len(waterways):
        waterways = waterways.to_crs(src_crs)

    return water_areas, waterways
