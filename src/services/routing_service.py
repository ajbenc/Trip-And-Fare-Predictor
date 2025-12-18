from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from functools import lru_cache


@dataclass
class RouteResult:
    distance_miles: float
    duration_minutes: float
    geometry: List[Tuple[float, float]]  # [(lat, lon), ...]
    steps: List[str]


class RoutingService:
    """
    Lightweight wrapper around the public OSRM API to fetch realistic driving routes.

    Notes:
    - This is used for DISPLAY ONLY by default. The prediction model continues to
      use centroid-to-centroid Haversine distance to stay consistent with training.
    - Optionally, the UI can allow using route distance for features, but it is
      off by default to avoid distribution shift.
    """

    BASE_URL = "https://router.project-osrm.org/route/v1/driving"

    def __init__(self) -> None:
        # Simple in-memory cache: key by rounded coordinates and steps flag
        self._cache: Dict[str, RouteResult] = {}

    def _key(self, pu_lat: float, pu_lng: float, do_lat: float, do_lng: float, steps: bool) -> str:
        return f"{round(pu_lat,5)},{round(pu_lng,5)}->{round(do_lat,5)},{round(do_lng,5)}|steps={int(steps)}"

    def get_route(self, pu_lat: float, pu_lng: float, do_lat: float, do_lng: float, *, steps: bool = True) -> Optional[RouteResult]:
        cache_key = self._key(pu_lat, pu_lng, do_lat, do_lng, steps)
        if cache_key in self._cache:
            return self._cache[cache_key]
        coords = f"{pu_lng},{pu_lat};{do_lng},{do_lat}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true" if steps else "false",
            "annotations": "false",
        }
        url = f"{self.BASE_URL}/{coords}"
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data or data.get("code") != "Ok" or not data.get("routes"):
                return None
            r0 = data["routes"][0]
            meters = float(r0.get("distance", 0.0))
            seconds = float(r0.get("duration", 0.0))
            distance_miles = meters * 0.000621371
            duration_minutes = seconds / 60.0
            geom = r0.get("geometry", {}).get("coordinates", [])
            # OSRM returns [lon, lat]; convert to [(lat, lon)]
            geometry = [(float(lat), float(lon)) for lon, lat in geom]

            step_texts: List[str] = []
            if steps and r0.get("legs"):
                for leg in r0["legs"]:
                    for s in leg.get("steps", []):
                        name = s.get("name") or ""
                        maneuver = s.get("maneuver", {})
                        instr = maneuver.get("instruction") or maneuver.get("type") or "Proceed"
                        step_texts.append(instr if not name else f"{instr} → {name}")

            result = RouteResult(
                distance_miles=distance_miles,
                duration_minutes=duration_minutes,
                geometry=geometry,
                steps=step_texts,
            )
            # Basic cache with crude eviction (max ~256 entries)
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
            return result
        except Exception:
            return None
