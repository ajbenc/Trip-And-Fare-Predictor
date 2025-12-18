from __future__ import annotations

import requests
from typing import Optional, Tuple, Dict


class GeocodingService:
    """
    Minimal reverse geocoding via OpenStreetMap Nominatim.
    - For display only. Do not feed addresses into the model.
    - Includes a tiny in-memory cache to reduce external calls.
    """

    BASE_URL = "https://nominatim.openstreetmap.org/reverse"

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def _key(self, lat: float, lon: float) -> str:
        return f"{round(lat, 5)},{round(lon, 5)}"

    def reverse(self, lat: float, lon: float) -> Optional[str]:
        key = self._key(lat, lon)
        if key in self._cache:
            return self._cache[key]

        params = {
            "format": "jsonv2",
            "lat": str(lat),
            "lon": str(lon),
            "addressdetails": 0,
        }
        headers = {"User-Agent": "nyc-taxi-ml/1.0 (educational)"}
        try:
            r = requests.get(self.BASE_URL, params=params, headers=headers, timeout=8)
            r.raise_for_status()
            j = r.json()
            disp = j.get("display_name")
            if not disp:
                return None
            # Cache with crude eviction
            if len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = disp
            return disp
        except Exception:
            return None
