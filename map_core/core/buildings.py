"""Building-specific helpers for classification and metrics."""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from map_core.core.util import has_data, log_progress


def compute_metric(data: gpd.GeoDataFrame, metric: str) -> pd.Series:
    """Compute a metric per-building for classification.

    Supported metrics:
    - 'area': polygon area in projected meters^2
    - 'distance': centroid distance to overall centroid in projected meters

    Returns a Series aligned to data.index. Falls back to zeros on error.
    """
    try:
        if not has_data(data):
            return pd.Series([0] * len(data), index=getattr(data, 'index', None))
        original_crs = data.crs
        if original_crs and original_crs.is_geographic:
            try:
                data_proj = data.to_crs(data.estimate_utm_crs())
            except Exception as e:
                log_progress(f"Warning: Could not reproject buildings to UTM for metric '{metric}': {e}. Using original CRS.")
                data_proj = data
        else:
            data_proj = data
        if metric == 'area':
            return data_proj.geometry.area
        elif metric == 'distance':
            # For distance metric, compute distance from centroid of all buildings
            # This matches the original behavior for consistency
            center_proj = data_proj.unary_union.centroid
            return data_proj.geometry.centroid.distance(center_proj)
        else:
            log_progress(f"Warning: Unknown metric '{metric}'. Using zeros.")
            return pd.Series([0] * len(data), index=getattr(data, 'index', None))
    except Exception as e:
        log_progress(f"Warning: failed to compute buildings metric '{metric}': {e}. Using zeros.")
    return pd.Series([0] * len(data), index=getattr(data, 'index', None))
