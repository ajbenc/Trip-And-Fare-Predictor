import json
import sys
from pathlib import Path

# Ensure project root on path for `src.*` imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.geo.zone_manager import ZoneManager


def main():
    zm = ZoneManager()

    # Test coordinates
    times_square = (40.7580, -73.9855)
    jfk = (40.6413, -73.7781)

    ts_zone = zm.get_zone_from_coords(*times_square)
    jfk_zone = zm.get_zone_from_coords(*jfk)

    ts_name = zm.get_zone_name(ts_zone) if ts_zone is not None else None
    jfk_name = zm.get_zone_name(jfk_zone) if jfk_zone is not None else None

    dist_miles = None
    if ts_zone is not None and jfk_zone is not None:
        dist_miles = zm.calculate_haversine_distance(ts_zone, jfk_zone)

    result = {
        "times_square": {
            "coords": times_square,
            "zone_id": ts_zone,
            "zone_name": ts_name,
        },
        "jfk": {
            "coords": jfk,
            "zone_id": jfk_zone,
            "zone_name": jfk_name,
        },
        "centroid_distance_miles": dist_miles,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
