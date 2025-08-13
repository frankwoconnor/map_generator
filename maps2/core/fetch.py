"""Data fetching and preprocessing helpers (OSM) for Map Art Generator."""
from __future__ import annotations

from typing import Any, Dict, Optional

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

from maps2.core.util import has_data, log_progress


def fetch_layer(
    query: str,
    dist: Optional[float],
    tags: Optional[Dict[str, Any]],
    is_graph: bool = False,
    custom_filter: Optional[str] = None,
    simplify_tolerance: Optional[float] = None,
    min_size_threshold: float = 0.0,
    layer_name_for_debug: Optional[str] = None,
) -> Any:
    """Fetch a street graph or GeoDataFrame for the requested layer.

    Notes:
        - When is_graph=True, returns a networkx graph.
        - Otherwise, returns a GeoDataFrame, optionally simplified and filtered by area.
        - If dist is provided, data are clipped to a circular buffer around the query point.
    """
    if is_graph:
        if dist:
            point = ox.geocode(query)
            G = ox.graph_from_point(point, dist=dist, network_type="all", custom_filter=custom_filter)
        else:
            G = ox.graph_from_place(query, network_type="all", custom_filter=custom_filter)
        return G

    # GeoDataFrame path
    if dist:
        point = ox.geocode(query)
        # osmnx 2.x: geometries_* replaced by features_*
        gdf = ox.features_from_point((point[0], point[1]), tags=tags, dist=dist)
    else:
        gdf = ox.features_from_place(query, tags=tags)

    if gdf is None or gdf.empty:
        return gdf

    # Only keep geometries
    if "geometry" in gdf.columns:
        gdf = gdf.set_geometry("geometry")

    # Optionally simplify geometries
    if simplify_tolerance and simplify_tolerance > 0 and not gdf.empty:
        try:
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance, preserve_topology=True)
        except Exception as e:
            log_progress(f"Warning: simplification failed for layer '{layer_name_for_debug}': {e}")

    # Optionally filter by min area (requires projection)
    if min_size_threshold and min_size_threshold > 0 and not gdf.empty:
        try:
            # Project to local UTM for area calculation
            original_crs = gdf.crs
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs()) if original_crs and original_crs.is_geographic else gdf
            areas = gdf_proj.geometry.area.fillna(0)
            before = len(gdf_proj)
            gdf_proj = gdf_proj.loc[areas >= float(min_size_threshold)].copy()
            after = len(gdf_proj)
            if layer_name_for_debug:
                log_progress("--------------------------------------------------")
                log_progress(f"Layer: {layer_name_for_debug}")
                log_progress(f"  Features before filtering: {before}")
                log_progress(f"  Features after filtering:  {after}")
                log_progress(f"  Features removed:          {before - after}")
                log_progress("--------------------------------------------------")
            # Reproject back if needed
            if original_crs and original_crs.is_geographic and not gdf_proj.empty:
                gdf = gdf_proj.to_crs(original_crs)
            else:
                gdf = gdf_proj
        except Exception as e:
            log_progress(f"Warning: area filtering failed for layer '{layer_name_for_debug}': {e}")

    return gdf
