# Moved from project root to src/domain/geo/zone_manager.py
# See README architecture section for details.

from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import json

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Optional imports for logger/config with safe fallbacks for standalone usage
try:
    from src.utils.logger import get_logger
except Exception:  # pragma: no cover - fallback for environments without src.utils
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    def get_logger(name: str):
        return logging.getLogger(name)

try:
    from src.utils.config import get_settings
except Exception:  # pragma: no cover - fallback
    def get_settings():
        class _S:  # minimal settings stub
            pass
        return _S()

logger = get_logger(__name__)
settings = get_settings()

class ZoneManager:
    # (Content unchanged; copied verbatim from original for stability.)
    def __init__(self, zone_dir: str = "Data/zones", zone_file: str = "taxi_zones.geojson"):
        self.zone_dir = Path(zone_dir)
        self.zone_file = zone_file
        self.zones_gdf = None
        self.zones = None
        self.zone_stats = None
        self.zone_coords = None
        self._sindex = None
        self._centroids_gs = None
        self.zones_gdf = self._load_zone_data()
        self.zones = self.zones_gdf
        self.zone_lookup = self._load_minimal_zones()
        self._extract_zone_coordinates()
        if self.zones_gdf is not None and not self.zones_gdf.empty:
            try:
                self._sindex = self.zones_gdf.sindex
            except Exception:
                self._sindex = None
            try:
                zones_proj = self.zones_gdf.to_crs(epsg=2263)
                centroids_proj = zones_proj.geometry.centroid
                self._centroids_gs = gpd.GeoSeries(centroids_proj, crs=zones_proj.crs).to_crs(epsg=4326)
            except Exception:
                self._centroids_gs = None

    def _load_zone_data(self):
        logger.info("Loading complete geospatial zone data and statistics.")
        zone_geo_path = self.zone_dir / self.zone_file
        if not zone_geo_path.exists():
            logger.error(f"Zone file not found at: {zone_geo_path}. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame()
        try:
            zones = gpd.read_file(zone_geo_path)
            if zones.crs is None:
                logger.warning("⚠️ CRS not defined, assuming EPSG:2263")
                zones = zones.set_crs("EPSG:2263")
            if zones.crs.to_string() != "EPSG:4326":
                logger.info(f"Reprojecting zones from {zones.crs} to EPSG:4326 (WGS84)")
                zones = zones.to_crs(epsg=4326)
            zones['LocationID'] = zones['LocationID'].astype(int)
            loaded_gdf = zones[['LocationID', 'zone', 'borough', 'geometry']]

            # Validate geometries
            invalid_geometries = loaded_gdf[~loaded_gdf.is_valid]
            if not invalid_geometries.empty:
                logger.warning(f"⚠️ Invalid geometries detected: {len(invalid_geometries)} zones")
                logger.warning(invalid_geometries)

            logger.info(f"✓ GeoDataFrame loaded and reprojected. Total: {len(loaded_gdf)} zones")
            return loaded_gdf
        except Exception as e:
            logger.error(f"Error loading GeoDataFrame from {zone_geo_path}: {e}")
            logger.warning("Map will not be available.")
            return gpd.GeoDataFrame()

    def _load_minimal_zones(self) -> Dict[int, str]:
        zone_lookup_data = {
            1: "Newark Airport",
            132: "JFK Airport",
            138: "LaGuardia Airport",
            161: "Midtown Center",
            162: "Midtown East",
            170: "Murray Hill",
            186: "Penn Station/Madison Sq West",
            229: "Sutton Place/Turtle Bay North",
            230: "Times Square/Theatre District",
            234: "Union Sq",
            236: "Upper East Side North",
            237: "Upper East Side South"
        }
        self.zone_lookup = zone_lookup_data
        return zone_lookup_data

    def _extract_zone_coordinates(self):
        if self.zones_gdf is not None and not self.zones_gdf.empty:
            try:
                zones_proj = self.zones_gdf.to_crs(epsg=2263)
                centroids_proj = zones_proj.geometry.centroid
                centroids = gpd.GeoSeries(centroids_proj, crs=zones_proj.crs).to_crs(epsg=4326)
            except Exception:
                centroids = self.zones_gdf.geometry.centroid
            self.zone_coords = {}
            for idx, row in self.zones_gdf.iterrows():
                zone_id = row['LocationID']
                c = centroids.loc[idx]
                self.zone_coords[zone_id] = (c.y, c.x)

    # ... (copy the remaining methods unchanged from original file) ...
    # For brevity in this header snippet, all methods from get_zone_from_coords
    # through calculate_haversine_distance are identical to the original.

    def get_zone_from_coords(self, latitude: float, longitude: float, nearest_if_outside: bool = True) -> Optional[int]:
        if self.zones_gdf is None or self.zones_gdf.empty:
            if nearest_if_outside:
                return self._find_nearest_zone(latitude, longitude)
            else:
                logger.warning("Zone geodataframe is empty; cannot resolve coordinates to a zone.")
                return None
        pt = Point(float(longitude), float(latitude))
        candidate_idxs = None
        try:
            if self._sindex is not None:
                candidate_idxs = list(self._sindex.query(pt))
        except Exception:
            candidate_idxs = None
        candidates = (
            self.zones_gdf.iloc[candidate_idxs]
            if candidate_idxs is not None and len(candidate_idxs) > 0
            else self.zones_gdf
        )
        for idx, row in candidates.iterrows():
            try:
                if row['geometry'].contains(pt):
                    return int(row['LocationID'])
            except Exception:
                continue
        if nearest_if_outside:
            return self._find_nearest_zone(latitude, longitude)
        return None

    def _find_nearest_zone(self, latitude: float, longitude: float) -> Optional[int]:
        try:
            all_coords = self.get_zone_coordinates()
            if not all_coords:
                return None
            if self._centroids_gs is not None and self.zones_gdf is not None and not self.zones_gdf.empty and len(self._centroids_gs) == len(self.zones_gdf):
                lats = self._centroids_gs.y.to_numpy()
                lons = self._centroids_gs.x.to_numpy()
                ids = self.zones_gdf['LocationID'].to_numpy()
            else:
                ids, lats, lons = zip(*[(zid, lat, lon) for zid, (lat, lon) in all_coords.items()])
                ids = np.array(ids)
                lats = np.array(lats)
                lons = np.array(lons)
            lat_rad = np.radians(latitude)
            lon_rad = np.radians(longitude)
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)
            dlat = lats_rad - lat_rad
            dlon = lons_rad - lon_rad
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distances_km = 6371.0 * c
            distances_miles = distances_km * 0.621371
            min_idx = int(np.argmin(distances_miles))
            nearest_zone_id = int(ids[min_idx])
            logger.info(f"Using nearest centroid fallback -> Zone {nearest_zone_id}")
            return nearest_zone_id
        except Exception as e:
            logger.warning(f"Nearest-zone calculation failed: {e}")
            return None

    def get_zone_name(self, zone_id: int) -> str:
        return self.zone_lookup.get(zone_id, f"Zone {zone_id}")

    def get_zone_coordinates(self) -> Dict[int, Tuple[float, float]]:
        if self.zone_coords:
            return self.zone_coords
        return {
            161: (40.7549, -73.9840),
            237: (40.7769, -73.9597),
            230: (40.7580, -73.9855),
            132: (40.6413, -73.7781),
            138: (40.7769, -73.8740),
            1: (40.6895, -74.1745),
        }

    def get_zones_geodataframe(self) -> gpd.GeoDataFrame:
        return self.zones_gdf

    def calculate_haversine_distance(self, pickup_zone_id: int, dropoff_zone_id: int) -> float:
        all_coords = self.get_zone_coordinates()
        pickup_coords = all_coords.get(pickup_zone_id)
        dropoff_coords = all_coords.get(dropoff_zone_id)
        if not pickup_coords or not dropoff_coords:
            logger.warning(f"Could not get coordinates for zones {pickup_zone_id} or {dropoff_zone_id}")
            return 0.0
        lat1, lon1 = pickup_coords
        lat2, lon2 = dropoff_coords
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance_km = R * c
        distance_miles = distance_km * 0.621371
        return distance_miles
