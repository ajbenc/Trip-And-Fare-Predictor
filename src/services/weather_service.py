import requests
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import time as _time


class _TTLCache:
    def __init__(self, ttl_seconds: int = 1200, max_size: int = 1024):
        self.ttl = ttl_seconds
        self.max = max_size
        self.store: Dict[Tuple, Tuple[float, dict]] = {}

    def get(self, key: Tuple):
        now = _time.time()
        v = self.store.get(key)
        if not v:
            return None
        ts, data = v
        if now - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return data

    def set(self, key: Tuple, data: dict):
        if len(self.store) >= self.max:
            # naive eviction: remove oldest
            oldest_key = min(self.store.items(), key=lambda kv: kv[1][0])[0]
            self.store.pop(oldest_key, None)
        self.store[key] = (_time.time(), data)


class WeatherService:
    """
    Minimal weather fetcher using Open-Meteo (no API key).
    Maps to the model's expected weather feature schema.
    """

    def __init__(self):
        self.cache = _TTLCache(ttl_seconds=20 * 60)

    @staticmethod
    def _round_coord(x: float) -> float:
        return round(float(x), 3)

    def get_weather_features(self, latitude: float, longitude: float, when: Optional[datetime] = None) -> Dict:
        lat = self._round_coord(latitude)
        lon = self._round_coord(longitude)
        # bucket by hour for caching
        when_local = when or datetime.now()
        hour_key = when_local.replace(minute=0, second=0, microsecond=0)
        key = (lat, lon, hour_key.isoformat())
        cached = self.cache.get(key)
        if cached:
            return cached

        # Build Open-Meteo request (forecast with hourly + current)
        # Use NYC timezone to align hours; Open-Meteo supports named timezones
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "America/New_York",
            "current_weather": "true",
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "pressure_msl",
                "wind_speed_10m",
                "cloud_cover",
                "precipitation",
                "snowfall",
                "visibility",
            ]),
        }
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=6)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Fallback to neutral defaults if API fails
            feats = self._neutral_features()
            self.cache.set(key, feats)
            return feats

        feats = self._map_to_features(data, when_local)
        self.cache.set(key, feats)
        return feats

    def _map_to_features(self, payload: dict, when_local: datetime) -> Dict:
        # Defaults
        f = self._neutral_features()
        try:
            hourly = payload.get("hourly", {})
            times = hourly.get("time", [])
            # find nearest hour index to when_local (assumes timezone in params)
            idx = None
            if times:
                # times are ISO strings
                target = when_local.replace(minute=0, second=0, microsecond=0)
                best_diff = None
                for i, t in enumerate(times):
                    try:
                        dt = datetime.fromisoformat(t)
                    except Exception:
                        continue
                    diff = abs((dt - target).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        idx = i
            # populate from hourly when available
            def at(arr_name, default=None):
                arr = hourly.get(arr_name)
                if isinstance(arr, list) and idx is not None and 0 <= idx < len(arr):
                    v = arr[idx]
                    return v if v is not None else default
                return default

            temp = at("temperature_2m", f["temperature"])  # °C
            hum = at("relative_humidity_2m", f["humidity"])  # %
            pres = at("pressure_msl", f["pressure"])  # hPa
            wind = at("wind_speed_10m", f["wind_speed"])  # m/s
            clouds = at("cloud_cover", f["clouds"])  # %
            precip = at("precipitation", f["precipitation"])  # mm
            snow = at("snowfall", f["snow"])  # cm or mm water eq depending; treat as mm-like
            vis = at("visibility", None)  # meters

            # map units and heuristics
            # convert temp °C to °F for consistency with example defaults
            temp_f = (temp * 9/5 + 32) if temp is not None else f["temperature"]
            # approximate feels_like via simple wind adjustment
            feels_like = temp_f
            if temp is not None and wind is not None:
                # convert m/s to mph
                mph = wind * 2.23694
                feels_like = temp_f - 0.7 * max(0.0, mph - 3)

            wind_mph = wind * 2.23694 if wind is not None else f["wind_speed"]

            f.update({
                "temperature": float(temp_f) if temp is not None else f["temperature"],
                "feels_like": float(feels_like),
                "humidity": float(hum) if hum is not None else f["humidity"],
                "pressure": float(pres) if pres is not None else f["pressure"],
                "wind_speed": float(wind_mph),
                "clouds": float(clouds) if clouds is not None else f["clouds"],
                "precipitation": float(precip) if precip is not None else f["precipitation"],
                "snow": float(snow) if snow is not None else f["snow"],
                "visibility_m": float(vis) if vis is not None else f.get("visibility_m", 10000.0),
            })

            # derive flags & severity
            is_raining = 1 if (precip or 0) > 0.1 else 0
            is_snowing = 1 if (snow or 0) > 0.1 else 0
            is_heavy_rain = 1 if (precip or 0) >= 5.0 else 0
            is_heavy_snow = 1 if (snow or 0) >= 5.0 else 0
            is_poor_visibility = 1 if (vis is not None and vis < 1500) else 0
            weather_severity = 1.0
            if is_heavy_rain or is_heavy_snow:
                weather_severity = 5.0
            elif is_raining or is_snowing:
                weather_severity = 3.0
            elif (clouds or 0) > 70:
                weather_severity = 2.0

            f.update({
                "is_raining": is_raining,
                "is_snowing": is_snowing,
                "is_heavy_rain": is_heavy_rain,
                "is_heavy_snow": is_heavy_snow,
                "is_extreme_weather": 1 if weather_severity >= 5.0 else 0,
                "is_poor_visibility": is_poor_visibility,
                "weather_severity": weather_severity,
                "source": "open-meteo",
                "fetched_at": when_local.isoformat(),
            })
        except Exception:
            # leave defaults
            pass
        return f

    @staticmethod
    def _neutral_features() -> Dict:
        return {
            'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
            'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
            'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
            'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
            'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0,
            'visibility_m': 10000.0, 'source': 'neutral', 'fetched_at': datetime.now(timezone.utc).isoformat()
        }
