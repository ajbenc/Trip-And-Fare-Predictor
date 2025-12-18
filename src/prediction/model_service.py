"""Model Service for NYC Taxi Trip Prediction

Loads the full-year LightGBM duration and fare models (56-feature pipeline) and
provides a unified interface to build features from high-level inputs:
    - pickup_zone_id, dropoff_zone_id
    - pickup_datetime (datetime)
    - passenger_count
    - estimated_distance (from route engine or centroid haversine fallback)
    - weather snapshot dict
    - holiday snapshot dict

Feature order must match the training schema (see models/lightgbm_80_20_full_year/README.md).
"""
from __future__ import annotations
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Constants defining airport & Manhattan zones (align with training)
AIRPORT_ZONES = {132, 138, 161}
MANHATTAN_ZONE_MIN = 4
MANHATTAN_ZONE_MAX = 233  # Approximate range used during training

@dataclass
class ModelArtifacts:
    duration_model: Any
    fare_model: Any
    feature_names: list

class ModelService:
    def __init__(self, models_dir: str = "models/lightgbm_80_20_full_year"):
        self.models_dir = Path(models_dir)
        self.duration_model_path = self.models_dir / "duration_model_final.pkl"
        self.fare_model_path = self.models_dir / "fare_model_final.pkl"
        self._artifacts: Optional[ModelArtifacts] = None

    def load(self) -> None:
        if not self.duration_model_path.exists() or not self.fare_model_path.exists():
            raise FileNotFoundError("Duration or fare model pickle not found in models directory.")
        with open(self.duration_model_path, "rb") as f:
            duration_model = pickle.load(f)
        with open(self.fare_model_path, "rb") as f:
            fare_model = pickle.load(f)
        # Derive feature names from model (LightGBM supports .feature_name())
        try:
            feature_names = duration_model.feature_name()
        except Exception:
            feature_names = []
        self._artifacts = ModelArtifacts(duration_model=duration_model, fare_model=fare_model, feature_names=feature_names)

    @property
    def is_loaded(self) -> bool:
        return self._artifacts is not None

    def build_features(self,
                       pickup_zone_id: int,
                       dropoff_zone_id: int,
                       pickup_datetime: datetime,
                       passenger_count: int,
                       estimated_distance: float,
                       weather: Dict[str, Any],
                       holiday: Dict[str, Any]) -> pd.DataFrame:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load() before building features.")

        # Normalize distance
        estimated_distance = float(max(0.1, min(estimated_distance, 50.0)))

        hour = pickup_datetime.hour
        weekday = pickup_datetime.weekday()
        month = pickup_datetime.month
        day = pickup_datetime.day
        dayofyear = pickup_datetime.timetuple().tm_yday
        weekofyear = pickup_datetime.isocalendar()[1]

        pickup_is_airport = int(pickup_zone_id in AIRPORT_ZONES)
        dropoff_is_airport = int(dropoff_zone_id in AIRPORT_ZONES)
        is_airport_trip = int(pickup_is_airport or dropoff_is_airport)
        pickup_is_manhattan = int(MANHATTAN_ZONE_MIN <= pickup_zone_id <= MANHATTAN_ZONE_MAX)
        dropoff_is_manhattan = int(MANHATTAN_ZONE_MIN <= dropoff_zone_id <= MANHATTAN_ZONE_MAX)
        same_location = int(pickup_zone_id == dropoff_zone_id)

        is_weekend = int(weekday >= 5)
        is_rush_hour = int((7 <= hour <= 9) or (16 <= hour <= 19))
        is_late_night = int(hour >= 22 or hour <= 4)
        is_business_hours = int((9 <= hour <= 17) and weekday < 5)

        # Interaction features rely on weather + above flags
        weather_severity = weather.get("weather_severity", 0.0)
        is_raining = int(weather.get("is_raining", 0))
        is_snowing = int(weather.get("is_snowing", 0))

        data = {
            # Location
            'PULocationID': pickup_zone_id,
            'DOLocationID': dropoff_zone_id,
            'pickup_is_airport': pickup_is_airport,
            'dropoff_is_airport': dropoff_is_airport,
            'is_airport_trip': is_airport_trip,
            'pickup_is_manhattan': pickup_is_manhattan,
            'dropoff_is_manhattan': dropoff_is_manhattan,
            'same_location': same_location,
            # Trip
            'passenger_count': passenger_count,
            'estimated_distance': estimated_distance,
            # Temporal
            'pickup_hour': hour,
            'pickup_day': day,
            'pickup_month': month,
            'pickup_weekday': weekday,
            'pickup_dayofyear': dayofyear,
            'pickup_weekofyear': weekofyear,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'is_late_night': is_late_night,
            'is_business_hours': is_business_hours,
            # Weather (subset + binary flags - expect caller to provide all needed keys)
            'temperature': weather.get('temperature', 70.0),
            'feels_like': weather.get('feels_like', weather.get('temperature', 70.0)),
            'humidity': weather.get('humidity', 50.0),
            'pressure': weather.get('pressure', 1013.0),
            'wind_speed': weather.get('wind_speed', 5.0),
            'clouds': weather.get('clouds', 30.0),
            'precipitation': weather.get('precipitation', 0.0),
            'snow': weather.get('snow', 0.0),
            'weather_severity': weather_severity,
            'is_raining': is_raining,
            'is_snowing': is_snowing,
            'is_heavy_rain': int(weather.get('is_heavy_rain', 0)),
            'is_heavy_snow': int(weather.get('is_heavy_snow', 0)),
            'is_extreme_weather': int(weather.get('is_extreme_weather', 0)),
            'is_poor_visibility': int(weather.get('is_poor_visibility', 0)),
            # Holiday
            'is_holiday': int(holiday.get('is_holiday', 0)),
            'is_major_holiday': int(holiday.get('is_major_holiday', 0)),
            'is_holiday_week': int(holiday.get('is_holiday_week', 0)),
            # Interactions
            'weather_airport_interaction': weather_severity * is_airport_trip,
            'weather_rushhour_interaction': weather_severity * is_rush_hour,
            'rushhour_airport_interaction': is_rush_hour * is_airport_trip,
            'latenight_manhattan_interaction': is_late_night * pickup_is_manhattan,
            'distance_hour_interaction': estimated_distance * hour,
            'distance_rushhour_interaction': estimated_distance * is_rush_hour,
            'holiday_airport_interaction': int(holiday.get('is_holiday', 0)) * is_airport_trip,
            'holiday_manhattan_interaction': int(holiday.get('is_holiday', 0)) * pickup_is_manhattan,
            'rain_distance_interaction': is_raining * estimated_distance,
            'snow_distance_interaction': is_snowing * estimated_distance,
            # Cyclical
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'weekday_sin': np.sin(2 * np.pi * weekday / 7),
            'weekday_cos': np.cos(2 * np.pi * weekday / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'dayofyear_sin': np.sin(2 * np.pi * dayofyear / 365),
            'dayofyear_cos': np.cos(2 * np.pi * dayofyear / 365),
        }

        df = pd.DataFrame([data])
        # Ensure feature order matches duration model if available
        if self._artifacts and self._artifacts.feature_names:
            # Some LightGBM installations may not preserve all names; fallback silently
            try:
                df = df[self._artifacts.feature_names]
            except KeyError:
                pass
        return df

    def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load() before prediction.")
        duration_pred = float(self._artifacts.duration_model.predict(features_df)[0])
        fare_pred = float(self._artifacts.fare_model.predict(features_df)[0])
        return {
            'duration_minutes': round(duration_pred, 2),
            'fare_amount': round(fare_pred, 2)
        }

    def predict_from_inputs(self, **kwargs) -> Dict[str, float]:
        df = self.build_features(**kwargs)
        return self.predict(df)

# Convenience for manual testing
if __name__ == "__main__":
    svc = ModelService()
    svc.load()
    now = datetime(2022, 7, 15, 10, 30)
    weather = {
        'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
        'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
        'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
        'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
        'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0
    }
    holiday = {'is_holiday': 0, 'is_major_holiday': 0, 'is_holiday_week': 0}
    feats = svc.build_features(
        pickup_zone_id=230,
        dropoff_zone_id=161,
        pickup_datetime=now,
        passenger_count=1,
        estimated_distance=17.5,
        weather=weather,
        holiday=holiday,
    )
    print(feats.head())
    print(svc.predict(feats))
