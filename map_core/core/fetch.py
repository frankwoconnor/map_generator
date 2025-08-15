"""Data fetching and preprocessing helpers (OSM) for Map Art Generator."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os
import json
import hashlib
from pathlib import Path

import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd

from map_core.core.util import has_data, log_progress

# --- Logging helpers -------------------------------------------------------
def _log_gdf_summary(gdf: Any, layer_name: Optional[str] = None) -> None:
    try:
        if gdf is None or not hasattr(gdf, "empty"):
            return
        name = layer_name or "features"
        log_progress(f"[summary] Layer='{name}': count={len(gdf)}; CRS={getattr(gdf, 'crs', None)}")
        try:
            if hasattr(gdf, "geom_type"):
                types = gdf.geom_type.value_counts().to_dict()
                log_progress(f"[summary] geometry types: {types}")
        except Exception:
            pass
        try:
            cols = list(gdf.columns)
            log_progress(f"[summary] columns({len(cols)}): {cols}")
        except Exception:
            pass
        # Top categories for common OSM tags
        for tag in ["highway","building","amenity","landuse","natural","leisure","waterway","man_made","railway","aeroway","shop","tourism"]:
            try:
                if tag in gdf.columns:
                    vc = gdf[tag].value_counts().head(12).to_dict()
                    if vc:
                        log_progress(f"[summary] tag '{tag}': top={vc}")
            except Exception:
                continue
    except Exception as e:
        log_progress(f"[summary] Warning: failed to summarize GeoDataFrame: {e}")


def _log_graph_summary(G: Any, layer_name: Optional[str] = None) -> None:
    try:
        name = layer_name or "graph"
        import networkx as nx  # type: ignore
        if G is None or not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)):
            return
        log_progress(f"[summary] Layer='{name}': graph nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        # Convert edges to gdf for tag summaries
        try:
            import osmnx as _ox
            _, edges = _ox.graph_to_gdfs(G)
            _log_gdf_summary(edges, f"{name}.edges")
        except Exception as e:
            log_progress(f"[summary] Warning: failed to convert graph to gdfs: {e}")
    except Exception as e:
        log_progress(f"[summary] Warning: failed to summarize graph: {e}")

# Optional dependency for reading local PBF files
try:  # pragma: no cover
    from pyrosm import OSM  # type: ignore
except Exception:  # pragma: no cover
    OSM = None  # type: ignore


# --- OSMnx cache defaults ---------------------------------------------------
try:
    # Ensure OSMnx uses local cache where possible
    if getattr(ox.settings, 'use_cache', None) is not None:
        ox.settings.use_cache = True
    if getattr(ox.settings, 'cache_only', None) is not None:
        # Leave cache_only as default False; user can override externally
        pass
    if getattr(ox.settings, 'cache_folder', None) is not None:
        env_dir = os.environ.get('OSMNX_CACHE_FOLDER')
        cache_dir = Path(env_dir) if env_dir else Path('../cache/osmnx')
        cache_dir.mkdir(parents=True, exist_ok=True)
        ox.settings.cache_folder = str(cache_dir)
except Exception:
    pass


def _compute_cache_key(
    *,
    source: str,
    query: str,
    dist: Optional[float],
    tags: Optional[Dict[str, Any]],
    simplify_tolerance: Optional[float],
    min_size_threshold: float,
    layer_name: Optional[str],
    bbox_override: Optional[Tuple[float, float, float, float]],
) -> tuple[str, str]:
    """Return (key_json, sha1) stable key for feature-layer cache."""
    key = {
        "v": 1,
        "source": source,
        "layer": layer_name or "features",
        "query": query,
        "dist": dist,
        "tags": tags,
        "simplify_tolerance": simplify_tolerance,
        "min_size_threshold": float(min_size_threshold or 0.0),
        "bbox_override": bbox_override,
    }
    key_json = json.dumps(key, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
    return key_json, h


def _cache_paths(layer_name: str, sha1: str) -> tuple[Path, Path, Path]:
    """Return (dir, parquet_path, meta_path)."""
    d = Path("../cache/features") / (layer_name or "features")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sha1}.parquet"
    m = d / f"{sha1}.meta.json"
    return d, p, m


def fetch_layer(
    query: str,
    dist: Optional[float],
    tags: Optional[Dict[str, Any]],
    is_graph: bool = False,
    custom_filter: Optional[str] = None,
    simplify_tolerance: Optional[float] = None,
    min_size_threshold: float = 0.0,
    layer_name_for_debug: Optional[str] = None,
    pbf_path: Optional[str] = None,
    bbox_override: Optional[Tuple[float, float, float, float]] = None,
) -> Any:
    """Fetch a street graph or GeoDataFrame for the requested layer.

    Notes:
        - When is_graph=True, returns a networkx graph.
        - Otherwise, returns a GeoDataFrame, optionally simplified and filtered by area.
        - If dist is provided, data are clipped to a circular buffer around the query point.
    """
    # If a local PBF is provided and available, use it for sourcing data
    if pbf_path:
        if OSM is None:
            # Strict offline: do NOT fall back to remote if user requested local PBF
            msg = "pyrosm is not installed; required for local PBF usage. Install 'pyrosm' or remove pbf_path to allow remote fetch."
            log_progress("[fetch_layer] ERROR: " + msg)
            raise RuntimeError(msg)
        else:
            try:
                log_progress("[fetch_layer] Data source: LOCAL_PBF. pbf='{pbf}', layer='{layer}', query='{q}', dist_m={dm}, dist_km={dk}".format(
                    pbf=pbf_path,
                    layer=layer_name_for_debug or ("graph" if is_graph else "features"),
                    q=query,
                    dm=(None if dist is None else int(round(dist))),
                    dk=(None if dist is None else round(dist/1000, 3))
                ))
                # Validate file exists before attempting to open
                import os
                if not os.path.isfile(pbf_path):
                    msg = f"Local PBF not found at path: {pbf_path}"
                    log_progress("[fetch_layer] ERROR: " + msg)
                    raise RuntimeError(msg)
                # Compute bbox if a distance-limited query is provided, or use override. No network calls.
                bbox: Optional[Tuple[float, float, float, float]] = None
                if bbox_override is not None:
                    try:
                        bx = tuple(bbox_override)
                        if len(bx) == 4:
                            raw = (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))
                            # Normalize ordering using center if possible
                            lat_c = lon_c = None
                            try:
                                q = query.replace(",", " ").split()
                                if len(q) >= 2:
                                    lat_c = float(q[0]); lon_c = float(q[1])
                            except Exception:
                                pass
                            bbox_norm = raw
                            if lat_c is not None and lon_c is not None:
                                vals = list(raw)
                                lats, lons = [], []
                                for v in vals:
                                    if abs(v - lat_c) <= abs(v - lon_c):
                                        lats.append(v)
                                    else:
                                        lons.append(v)
                                if len(lats) == 2 and len(lons) == 2:
                                    lat_min, lat_max = sorted(lats)
                                    lon_min, lon_max = sorted(lons)
                                    bbox_norm = (lon_min, lat_min, lon_max, lat_max)
                            bbox = bbox_norm
                            log_progress(f"[fetch_layer] Using bbox from override: raw={raw} -> normalized={bbox}")
                    except Exception:
                        bbox = None
                elif dist:
                    # Try to parse numeric center from query: formats like "lat, lon" or "lat lon"
                    center_tuple = None
                    try:
                        q = query.replace(",", " ").split()
                        if len(q) >= 2:
                            lat = float(q[0])
                            lon = float(q[1])
                            center_tuple = (lat, lon)
                    except Exception:
                        center_tuple = None
                    if center_tuple:
                        north, south, east, west = ox.utils_geo.bbox_from_point(center_tuple, dist=dist)
                        bbox = (west, south, east, north)  # pyrosm expects (minx, miny, maxx, maxy)
                        log_progress(f"[fetch_layer] Using offline bbox from numeric center for dist {dist}m: {bbox}")
                        try:
                            w, s, e, n = bbox
                            log_progress(f"[fetch_layer] BBox size (deg): dx={e-w:.8f}, dy={n-s:.8f}")
                        except Exception:
                            pass
                    else:
                        log_progress("[fetch_layer] Offline mode: could not parse numeric center from query; proceeding without bbox (full PBF extent).")

                # Create OSM reader without constructor-level bounding to avoid pyrosm 'id' index issues on some extracts
                # We will apply spatial clipping manually after reading.
                try:
                    osm = OSM(pbf_path)
                    if bbox is not None:
                        log_progress(f"[fetch_layer] pyrosm OSM initialized (no constructor bbox). Will clip to bbox={bbox} after read.")
                    else:
                        log_progress("[fetch_layer] pyrosm OSM initialized for full PBF extent.")
                except Exception as e_init:
                    log_progress(f"[fetch_layer] ERROR: Failed to initialize pyrosm OSM reader: {e_init}")
                    raise
                try:
                    import pyrosm as _pyrosm
                    log_progress(f"[fetch_layer] pyrosm version: {_pyrosm.__version__}")
                except Exception:
                    pass

                if is_graph:
                    # Build a street network graph from PBF edges/nodes
                    # Do not pass bbox/bounding_box here; bounding applied at OSM() construction when possible
                    try:
                        net = osm.get_network(network_type="all", nodes=True)
                    except TypeError:
                        net = osm.get_network(network_type="all")
                    except Exception as e_net:
                        # Retry without nodes in case of parser/index issues
                        log_progress(f"[fetch_layer] pyrosm get_network failed (nodes=True): {e_net}. Retrying with nodes=False.")
                        net = osm.get_network(network_type="all")
                    nodes = None
                    edges = None
                    if isinstance(net, (tuple, list)) and len(net) == 2:
                        nodes, edges = net
                    else:
                        edges = net
                    # If we have a bbox and edges are a GeoDataFrame, post-filter to bbox to emulate bounding
                    if bbox is not None and edges is not None:
                        try:
                            from shapely.geometry import box as _box
                            import geopandas as _gpd
                            minx, miny, maxx, maxy = bbox
                            bbox_poly = _box(minx, miny, maxx, maxy)
                            # Log CRS and bounds pre-clip
                            try:
                                log_progress(f"[fetch_layer] Graph edges CRS={getattr(edges, 'crs', None)}; bounds before clip={getattr(edges.total_bounds, 'tolist', lambda: list(edges.total_bounds))() if hasattr(edges, 'total_bounds') else '(n/a)'}")
                            except Exception:
                                pass
                            edges = _gpd.clip(edges, bbox_poly)
                            try:
                                log_progress(f"[fetch_layer] Graph edges bounds after clip={getattr(edges.total_bounds, 'tolist', lambda: list(edges.total_bounds))() if hasattr(edges, 'total_bounds') else '(n/a)'}")
                            except Exception:
                                pass
                            if nodes is not None and hasattr(nodes, 'geometry'):
                                try:
                                    log_progress(f"[fetch_layer] Graph nodes CRS={getattr(nodes, 'crs', None)}; bounds before clip={getattr(nodes.total_bounds, 'tolist', lambda: list(nodes.total_bounds))() if hasattr(nodes, 'total_bounds') else '(n/a)'}")
                                except Exception:
                                    pass
                                nodes = _gpd.clip(nodes, bbox_poly)
                                try:
                                    log_progress(f"[fetch_layer] Graph nodes bounds after clip={getattr(nodes.total_bounds, 'tolist', lambda: list(nodes.total_bounds))() if hasattr(nodes, 'total_bounds') else '(n/a)'}")
                                except Exception:
                                    pass
                            log_progress(f"[fetch_layer] Applied post-filter clip to bbox; edges now {0 if edges is None else len(edges)} features")
                        except Exception as e:
                            log_progress(f"Warning: failed to clip to bbox: {e}")
                    try:
                        if nodes is not None:
                            # Use positional arguments for broader OSMnx compatibility
                            G = ox.graph_from_gdfs(nodes, edges)
                        else:
                            # Some OSMnx versions require both; fallback to edges-only path
                            G = ox.graph_from_gdfs(None, edges)
                        log_progress("[fetch_layer] LOCAL_PBF graph constructed from edges{n}.".format(n="+nodes" if nodes is not None else ""))
                        try:
                            _log_graph_summary(G, layer_name_for_debug or "graph")
                        except Exception:
                            pass
                        return G
                    except Exception as e:
                        log_progress(f"Warning: failed to construct graph from PBF: {e}. Falling back to edges GeoDataFrame.")
                        return edges
                else:
                    # Generic features via custom tag filter
                    data = None
                    # Normalize tags -> custom_filter: values must be lists or True
                    raw_tags = tags or {}
                    custom_filter = {}
                    try:
                        for k, v in raw_tags.items():
                            if v is True:
                                custom_filter[k] = True
                            elif isinstance(v, (list, tuple)):
                                custom_filter[k] = list(v)
                            elif isinstance(v, str):
                                custom_filter[k] = [v]
                            else:
                                custom_filter[k] = [str(v)]
                    except Exception:
                        custom_filter = raw_tags or {}
                    # Prefer newer pyrosm API and avoid relation geometry issues first
                    # Prefer including relations first: many large water bodies are relations
                    try:
                        data = osm.get_data_by_custom_criteria(custom_filter=custom_filter, keep_relations=True)
                        log_progress("[fetch_layer] pyrosm custom criteria fetched with keep_relations=True")
                    except AttributeError:
                        # Older pyrosm API name
                        try:
                            data = osm.get_data_by_custom_filter(tags=custom_filter)
                            log_progress("[fetch_layer] pyrosm legacy custom filter fetched")
                        except Exception as e_legacy:
                            log_progress(f"[fetch_layer] pyrosm legacy custom filter failed: {e_legacy}")
                            raise
                    except Exception as e_keep_true_first:
                        log_progress(f"[fetch_layer] custom criteria failed with keep_relations=True: {e_keep_true_first}; retrying with keep_relations=False")
                        try:
                            data = osm.get_data_by_custom_criteria(custom_filter=custom_filter, keep_relations=False)
                            log_progress("[fetch_layer] pyrosm custom criteria fetched with keep_relations=False")
                        except AttributeError:
                            try:
                                data = osm.get_data_by_custom_filter(tags=custom_filter)
                                log_progress("[fetch_layer] pyrosm legacy custom filter fetched (post-retry)")
                            except Exception as e2_legacy:
                                log_progress(f"[fetch_layer] pyrosm legacy custom filter failed after retry: {e2_legacy}")
                                raise
                        except Exception as e_keep_false_second:
                            log_progress(f"[fetch_layer] custom criteria also failed with keep_relations=False: {e_keep_false_second}")
                            raise
                    # Optional simplify and area filtering applied below
                    if data is None:
                        log_progress("[fetch_layer] LOCAL_PBF returned no data for layer='{layer}'.".format(layer=layer_name_for_debug or "features"))
                        return data
                    gdf = data
                    # Log CRS and bounds pre-clip for features
                    try:
                        log_progress(f"[fetch_layer] Features layer='{layer_name_for_debug or 'features'}' CRS={getattr(gdf, 'crs', None)}; bounds before clip={getattr(gdf.total_bounds, 'tolist', lambda: list(gdf.total_bounds))() if hasattr(gdf, 'total_bounds') else '(n/a)'}; count={len(gdf) if hasattr(gdf, '__len__') else 'n/a'}")
                    except Exception:
                        pass
                    # Attempt to repair invalid geometries that can arise from relations or slicing
                    try:
                        if has_data(gdf):
                            before_n = len(gdf)
                            # Drop obvious nulls first
                            gdf = gdf[gdf.geometry.notnull()].copy()
                            # Make valid where possible (Shapely 2.x)
                            try:
                                import shapely
                                def _fix_geom(g):
                                    if g is None:
                                        return None
                                    try:
                                        return shapely.make_valid(g) if hasattr(shapely, 'make_valid') else g
                                    except Exception:
                                        try:
                                            return g.buffer(0)
                                        except Exception:
                                            return None
                                gdf['geometry'] = gdf['geometry'].apply(_fix_geom)
                            except Exception:
                                # Fallback: buffer(0) heuristic
                                try:
                                    gdf['geometry'] = gdf['geometry'].buffer(0)
                                except Exception:
                                    pass
                            # Drop any that are still null/empty
                            try:
                                gdf = gdf[gdf.geometry.notnull()].copy()
                                if hasattr(gdf.geometry, 'is_empty'):
                                    gdf = gdf[~gdf.geometry.is_empty].copy()
                            except Exception:
                                pass
                            after_n = len(gdf)
                            removed = before_n - after_n
                            if removed > 0:
                                log_progress(f"[fetch_layer] Repaired geometries; removed {removed} invalid/empty features (kept {after_n}).")
                    except Exception as _e_fix:
                        log_progress(f"Warning: geometry repair step failed: {_e_fix}")
                    # If we have a bbox, clip features to it to respect distance/manual bounds
                    if bbox is not None and has_data(gdf):
                        try:
                            from shapely.geometry import box as _box
                            import geopandas as _gpd
                            minx, miny, maxx, maxy = bbox
                            bbox_poly = _box(minx, miny, maxx, maxy)
                            gdf = _gpd.clip(gdf, bbox_poly)
                            try:
                                log_progress(f"[fetch_layer] Clipped features to bbox; now {0 if gdf is None else len(gdf)} features; bounds after clip={getattr(gdf.total_bounds, 'tolist', lambda: list(gdf.total_bounds))() if hasattr(gdf, 'total_bounds') else '(n/a)'}")
                            except Exception:
                                log_progress(f"[fetch_layer] Clipped features to bbox; now {0 if gdf is None else len(gdf)} features")
                        except Exception as e:
                            log_progress(f"Warning: failed to clip features to bbox: {e}")
                    # Proceed to post-processing below
                    # Note: skips remote fetch path
                    # Optionally simplify geometries
                    if simplify_tolerance and simplify_tolerance > 0 and has_data(gdf):
                        try:
                            gdf = gdf.copy()
                            gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance, preserve_topology=True)
                        except Exception as e:
                            log_progress(f"Warning: simplification failed for layer '{layer_name_for_debug}': {e}")
                    # Optionally filter by min area
                    if min_size_threshold and min_size_threshold > 0 and has_data(gdf):
                        try:
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
                            if original_crs and original_crs.is_geographic and has_data(gdf_proj):
                                gdf = gdf_proj.to_crs(original_crs)
                            else:
                                gdf = gdf_proj
                        except Exception as e:
                            log_progress(f"Warning: area filtering failed for layer '{layer_name_for_debug}': {e}")
                    log_progress("[fetch_layer] LOCAL_PBF fetched {n} features for layer='{layer}'.".format(
                        n=(0 if gdf is None else len(gdf)), layer=layer_name_for_debug or "features"))
                    try:
                        _log_gdf_summary(gdf, layer_name_for_debug or "features")
                    except Exception:
                        pass
                    return gdf
            except Exception as e:
                msg = f"Failed to read from local PBF '{pbf_path}': {e}"
                log_progress("[fetch_layer] ERROR: " + msg)
                raise RuntimeError(msg)

    # Remote fetching via OSMnx
    log_progress("[fetch_layer] Data source: REMOTE_OSMNX. layer='{layer}', query='{q}', dist_m={dm}, dist_km={dk}".format(
        layer=layer_name_for_debug or ("graph" if is_graph else "features"),
        q=query,
        dm=(None if dist is None else int(round(dist))),
        dk=(None if dist is None else round(dist/1000, 3))
    ))
    # Log cache configuration to help diagnose external calls
    try:
        log_progress(f"[fetch_layer] OSMnx cache settings: use_cache={getattr(ox.settings, 'use_cache', None)}, cache_only={getattr(ox.settings, 'cache_only', None)}, cache_folder={getattr(ox.settings, 'cache_folder', None)}")
    except Exception:
        pass
    if is_graph:
        # Try to avoid network geocoding if numeric coords are present in query
        numeric_point = None
        try:
            q = query.replace(",", " ").split()
            if len(q) >= 2:
                lat = float(q[0]); lon = float(q[1])
                numeric_point = (lat, lon)
        except Exception:
            numeric_point = None
        if dist and numeric_point is not None:
            G = ox.graph_from_point(numeric_point, dist=dist, network_type="all", custom_filter=custom_filter)
        elif dist:
            if getattr(ox.settings, 'cache_only', False):
                log_progress("[fetch_layer] cache_only=True but geocoding is required for query -> this may fail if not cached. query='{}'".format(query))
            point = ox.geocode(query)
            G = ox.graph_from_point(point, dist=dist, network_type="all", custom_filter=custom_filter)
        else:
            if getattr(ox.settings, 'cache_only', False):
                log_progress("[fetch_layer] cache_only=True using graph_from_place; will hit cache only. query='{}'".format(query))
            G = ox.graph_from_place(query, network_type="all", custom_filter=custom_filter)
        try:
            _log_graph_summary(G, layer_name_for_debug or "graph")
        except Exception:
            pass
        return G

    # GeoDataFrame path
    # Feature cache lookup (applies to remote path only and also effectively to local PBF below when added)
    source = "LOCAL_PBF" if pbf_path else "REMOTE_OSMNX"
    key_json, key_sha1 = _compute_cache_key(
        source=source,
        query=query,
        dist=dist,
        tags=tags,
        simplify_tolerance=simplify_tolerance,
        min_size_threshold=min_size_threshold,
        layer_name=layer_name_for_debug,
        bbox_override=bbox_override,
    )
    cache_dir, parquet_path, meta_path = _cache_paths(layer_name_for_debug or "features", key_sha1)
    try:
        if parquet_path.exists():
            log_progress(f"[fetch_layer] CACHE HIT '{layer_name_for_debug or 'features'}' -> {parquet_path.name}")
            gdf = gpd.read_parquet(parquet_path)
            try:
                _log_gdf_summary(gdf, layer_name_for_debug or "features")
            except Exception:
                pass
            return gdf
        else:
            log_progress(f"[fetch_layer] CACHE MISS '{layer_name_for_debug or 'features'}'")
    except Exception as e:
        log_progress(f"[fetch_layer] Cache check failed: {e}")

    try:
        if dist:
            point = ox.geocode(query)
            # osmnx 2.x: geometries_* replaced by features_*
            gdf = ox.features_from_point((point[0], point[1]), tags=tags, dist=dist)
        else:
            gdf = ox.features_from_place(query, tags=tags)
    except Exception as e_remote:
        # Known issue: Shapely MultiPolygon creation can fail inside OSMnx polygon consolidation
        emsg = str(e_remote)
        log_progress(f"[fetch_layer] REMOTE_OSMNX failed: {e_remote}")
        # Attempt local PBF fallback with common paths if available
        try:
            fallback_paths = [
                Path('../osm-data/ireland_cork.osm.pbf'),
                Path('../osm-data/ireland.pbf'),
            ]
            fb = next((p for p in fallback_paths if p.exists()), None)
            if fb is None:
                raise FileNotFoundError("No fallback PBF found at ../osm-data/{ireland_cork.osm.pbf, ireland.pbf}")
            log_progress(f"[fetch_layer] Falling back to LOCAL_PBF at '{fb}' with keep_relations=True")
            # Re-enter local PBF logic by instantiating OSM and fetching features with same tags and bbox
            osm = OSM(str(fb)) if OSM is not None else None
            if osm is None:
                raise RuntimeError("pyrosm not available for local fallback")
            # Derive bbox from query+dist if possible
            bbox_fb = None
            if dist:
                try:
                    q = query.replace(',', ' ').split()
                    if len(q) >= 2:
                        lat = float(q[0]); lon = float(q[1])
                        n, s, e, w = ox.utils_geo.bbox_from_point((lat, lon), dist=dist)
                        bbox_fb = (w, s, e, n)
                except Exception:
                    bbox_fb = None
            # Fetch with relations first, then fallback without relations
            try:
                gdf = osm.get_data_by_custom_criteria(custom_filter=tags or {}, keep_relations=True)
            except Exception as e_rel:
                log_progress(f"[fetch_layer] LOCAL_PBF keep_relations=True failed: {e_rel}; retrying without relations")
                try:
                    gdf = osm.get_data_by_custom_criteria(custom_filter=tags or {}, keep_relations=False)
                except AttributeError:
                    # Older pyrosm
                    gdf = osm.get_data_by_custom_filter(tags=tags or {})
            # Attempt to repair invalid geometries to avoid Shapely create_collection errors
            if has_data(gdf):
                try:
                    import geopandas as _gpd
                    gdf = gdf[gdf.geometry.notnull()].copy()
                    def _fix_geom(_g):
                        if _g is None:
                            return None
                        try:
                            from shapely.validation import make_valid as _mv
                            _g2 = _mv(_g)
                        except Exception:
                            try:
                                _g2 = _g.buffer(0)
                            except Exception:
                                return None
                        try:
                            return _g2 if (_g2 is not None and not _g2.is_empty) else None
                        except Exception:
                            return None
                    gdf["geometry"] = gdf["geometry"].apply(_fix_geom)
                    gdf = gdf[gdf.geometry.notnull()]
                except Exception as e_fix:
                    log_progress(f"[fetch_layer] LOCAL_PBF geometry repair step skipped: {e_fix}")
            # Clip to bbox if derived
            if bbox_fb is not None and has_data(gdf):
                from shapely.geometry import box as _box
                import geopandas as _gpd
                gdf = _gpd.clip(gdf, _box(*bbox_fb))
            log_progress(f"[fetch_layer] LOCAL_PBF fallback returned {0 if gdf is None else len(gdf)} features")
        except Exception as e_fb:
            log_progress(f"[fetch_layer] LOCAL_PBF fallback failed: {e_fb}")
            raise

    if gdf is None or gdf.empty:
        log_progress("[fetch_layer] REMOTE_OSMNX returned no data for layer='{layer}'.".format(layer=layer_name_for_debug or "features"))
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

    # Keep only essential columns
    try:
        if hasattr(gdf, 'columns') and 'geometry' in gdf.columns and not gdf.empty:
            tag_keys = list(tags.keys()) if isinstance(tags, dict) else []
            common = [
                'name', 'height', 'levels', 'building', 'water',
                'landuse', 'natural', 'leisure', 'amenity', 'shop'
            ]
            keep = ['geometry'] + [c for c in tag_keys + common if c in gdf.columns]
            # Drop duplicates while preserving order
            seen = set(); ordered_keep = []
            for c in keep:
                if c not in seen:
                    seen.add(c); ordered_keep.append(c)
            gdf = gdf[ordered_keep].copy()
    except Exception as e:
        log_progress(f"[fetch_layer] keep-only-columns failed: {e}")

    # Persist to cache
    try:
        meta = {
            "key": json.loads(key_json),
            "rows": None if gdf is None else int(len(gdf)),
            "crs": None if gdf is None else (str(getattr(gdf, 'crs', None)) if getattr(gdf, 'crs', None) is not None else None),
        }
        gdf.to_parquet(parquet_path)
        meta_path.write_text(json.dumps(meta, indent=2))
        log_progress(f"[fetch_layer] CACHE WRITE '{layer_name_for_debug or 'features'}' -> {parquet_path.name}")
    except Exception as e:
        log_progress(f"[fetch_layer] cache write skipped: {e}")

    return gdf
