"""Quick plausibility evaluation for duration & fare predictions.

Generates random (pickup_zone, dropoff_zone) pairs from the loaded ZoneManager,
builds feature vectors for several representative datetimes & passenger counts,
and computes simple derived metrics:
 - implied speed (mph) = estimated_distance_miles / (duration_minutes/60)
 - implied fare per mile

Flags predictions outside rough sanity ranges:
 - speed < 3 mph or > 60 mph
 - fare_per_mile < 1.5 or > 20
 - duration < 1 min or > 180 min

This is not a full accuracy test (needs real labels). It is a structural plausibility check
to catch egregious outliers or feature leakage regressions.
"""
from __future__ import annotations

import random
from datetime import datetime
from statistics import mean
from pathlib import Path
import sys
from pathlib import Path

# Ensure project root is on sys.path for local execution without installation
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.geo.zone_manager import ZoneManager  # adjusted import after reorg
from src.services.model_service import ModelService  # re-export of prediction ModelService

RANDOM_SEED = 42
N_SAMPLES = 40  # zone pair samples
DATETIMES = [
    datetime(2022, 7, 15, 8, 30),   # weekday morning rush
    datetime(2022, 7, 15, 14, 15),  # mid-afternoon
    datetime(2022, 12, 25, 10, 0),  # holiday morning
    datetime(2022, 12, 25, 21, 30), # holiday evening
    datetime(2022, 2, 10, 23, 45),  # late night winter
]
PASSENGER_COUNTS = [1, 2, 4]

def main() -> None:
    random.seed(RANDOM_SEED)
    zm = ZoneManager()
    svc = ModelService()
    if not svc.is_loaded:
        svc.load()

    zones_gdf = zm.get_zones_geodataframe()
    if zones_gdf is None or zones_gdf.empty:
        print("ERROR: Zones geodataframe is empty; aborting plausibility test.")
        return

    zone_ids = zones_gdf['LocationID'].tolist()
    samples = [tuple(random.sample(zone_ids, 2)) for _ in range(N_SAMPLES)]

    speeds = []
    fare_per_mile = []
    anomalies = []
    rows = []

    for (pu_zone, do_zone) in samples:
        est_miles = zm.calculate_haversine_distance(pu_zone, do_zone)
        # Skip zero-distance same-zone anomalies for this test
        if est_miles < 0.05:
            continue
        for dt in DATETIMES:
            for pax in PASSENGER_COUNTS:
                feats = svc.build_features(
                    pickup_zone_id=pu_zone,
                    dropoff_zone_id=do_zone,
                    pickup_datetime=dt,
                    passenger_count=pax,
                    estimated_distance=est_miles,
                    weather={  # neutral weather for plausibility
                        'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
                        'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
                        'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
                        'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
                        'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0
                    },
                    holiday={'is_holiday': 0, 'is_major_holiday': 0, 'is_holiday_week': 0},
                )
                pred = svc.predict(feats)
                dur_min = pred['duration_minutes']
                fare = pred['fare_amount']
                mph = (est_miles / (dur_min / 60)) if dur_min > 0 else float('inf')
                fpm = (fare / est_miles) if est_miles > 0 else float('inf')
                speeds.append(mph)
                fare_per_mile.append(fpm)
                row = {
                    'pickup_zone': pu_zone,
                    'dropoff_zone': do_zone,
                    'datetime': dt.isoformat(),
                    'passengers': pax,
                    'distance_miles': round(est_miles, 3),
                    'duration_min': round(dur_min, 2),
                    'fare': round(fare, 2),
                    'mph': round(mph, 2),
                    'fare_per_mile': round(fpm, 2),
                }
                # Anomaly rules
                if mph < 3 or mph > 60 or fpm < 1.5 or fpm > 20 or dur_min < 1 or dur_min > 180:
                    anomalies.append(row)
                rows.append(row)

    print("=== Prediction Plausibility Summary ===")
    print(f"Total evaluated trip variants: {len(rows)}")
    print(f"Mean implied speed (mph): {mean(speeds):.2f}")
    print(f"Mean fare per mile: ${mean(fare_per_mile):.2f}")
    print(f"Anomalies found: {len(anomalies)}")
    if anomalies:
        print("Sample anomalies (up to 5):")
        for a in anomalies[:5]:
            print(a)
    else:
        print("No anomalies flagged under current heuristic thresholds.")

    # Simple distribution buckets
    fast = sum(1 for s in speeds if s > 40)
    slow = sum(1 for s in speeds if s < 5)
    print(f"Very fast (>40 mph) fraction: {fast/len(speeds):.1%}")
    print(f"Very slow (<5 mph) fraction: {slow/len(speeds):.1%}")

if __name__ == "__main__":
    main()