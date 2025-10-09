import json
import os
import time
from typing import Optional, Tuple

import requests

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache"
)
CACHE_FILE = os.path.join(CACHE_DIR, "geocode.json")
USER_AGENT = "MapArtGenerator/1.0 (+https://example.local)"  # customize if desired


def _load_cache() -> dict:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.isfile(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def geocode_to_point(
    query: str, allow_online: bool = False, timeout: int = 10
) -> Optional[Tuple[float, float]]:
    """Resolve an address/place text to (lat, lon).

    - Uses local JSON cache first.
    - If allow_online is True, queries Nominatim and updates cache.
    - Returns None if not found or network disabled/unavailable.
    """
    if not query or not query.strip():
        return None

    key = query.strip()
    cache = _load_cache()
    if key in cache:
        v = cache[key]
        if isinstance(v, dict) and "lat" in v and "lon" in v:
            try:
                return float(v["lat"]), float(v["lon"])
            except Exception:
                pass

    if not allow_online:
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": key,
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": USER_AGENT,
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            # write-through cache
            cache[key] = {"lat": lat, "lon": lon, "ts": int(time.time())}
            _save_cache(cache)
            return lat, lon
    except Exception:
        return None

    return None
