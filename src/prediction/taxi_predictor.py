"""
NYC Taxi Fare & Duration Prediction Service
Combines LightGBM (Fare) and MLP (Duration) models for real-time predictions
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, List


class TaxiPredictor:
    """
    Production-ready predictor combining:
    - LightGBM for fare prediction (94%+ R²)
    - MLP Neural Network for duration prediction (90%+ R²)
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize predictor with trained models
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.fare_model = None
        self.duration_model = None
        self.normalization_stats = None
        
        # Model paths
        self.fare_model_path = self.models_dir / "lightgbm" / "lightgbm_fare_amount_cleaned.txt"
        self.duration_model_path = self.models_dir / "mlp_duration_v3.keras"
        self.norm_stats_path = self.models_dir / "mlp_duration_v3_normalization.npz"
        
        print("🚕 NYC Taxi Predictor Initialized")
        print(f"📁 Models directory: {self.models_dir.absolute()}")
        
    def load_models(self):
        """Load both fare and duration models"""
        print("\n🔄 Loading models...")
        
        # Load LightGBM Fare Model
        if self.fare_model_path.exists():
            self.fare_model = lgb.Booster(model_file=str(self.fare_model_path))
            print(f"✅ Fare model loaded: {self.fare_model_path.name}")
        else:
            print(f"⚠️  Fare model not found: {self.fare_model_path}")
            
        # Load MLP Duration Model
        if self.duration_model_path.exists():
            self.duration_model = tf.keras.models.load_model(self.duration_model_path)
            print(f"✅ Duration model loaded: {self.duration_model_path.name}")
        else:
            print(f"⚠️  Duration model not found: {self.duration_model_path}")
            
        # Load normalization stats for MLP
        if self.norm_stats_path.exists():
            stats = np.load(self.norm_stats_path)
            self.normalization_stats = {
                'mean': stats['mean'],
                'std': stats['std']
            }
            print(f"✅ Normalization stats loaded")
        else:
            print(f"⚠️  Normalization stats not found: {self.norm_stats_path}")
            
        return self
    
    def prepare_features(self, trip_data: Dict) -> np.ndarray:
        """
        Prepare features from raw trip data
        
        Args:
            trip_data: Dictionary with trip information
                {
                    'pickup_datetime': '2016-06-15 14:30:00',
                    'pickup_longitude': -73.98,
                    'pickup_latitude': 40.75,
                    'dropoff_longitude': -73.95,
                    'dropoff_latitude': 40.77,
                    'passenger_count': 1
                }
        
        Returns:
            Feature array (1, n_features)
        """
        # Parse datetime
        dt = pd.to_datetime(trip_data['pickup_datetime'])
        
        # Extract temporal features
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month
        year = dt.year
        day = dt.day
        
        # Calculate distance (haversine)
        pickup_lon = trip_data['pickup_longitude']
        pickup_lat = trip_data['pickup_latitude']
        dropoff_lon = trip_data['dropoff_longitude']
        dropoff_lat = trip_data['dropoff_latitude']
        
        distance = self._haversine_distance(
            pickup_lon, pickup_lat,
            dropoff_lon, dropoff_lat
        )
        
        # Calculate direction
        direction = self._calculate_direction(
            pickup_lon, pickup_lat,
            dropoff_lon, dropoff_lat
        )
        
        passenger_count = trip_data.get('passenger_count', 1)
        
        # Build feature array (basic features - expand with weather/holiday if available)
        features = np.array([[
            pickup_lon,
            pickup_lat,
            dropoff_lon,
            dropoff_lat,
            passenger_count,
            distance,
            direction,
            hour,
            day_of_week,
            month,
            year,
            day,
            # Add weather features if available (temp, precip, etc.)
            # Add holiday features if available
        ]])
        
        return features
    
    def predict(self, trip_data: Dict) -> Dict:
        """
        Predict both fare and duration for a trip
        
        Args:
            trip_data: Dictionary with trip information
        
        Returns:
            Dictionary with predictions:
            {
                'fare': 12.50,
                'duration_minutes': 15.3,
                'confidence': 'high',
                'features': {...}
            }
        """
        if self.fare_model is None or self.duration_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Prepare features
        features = self.prepare_features(trip_data)
        
        # Predict Fare (LightGBM)
        fare_prediction = self.fare_model.predict(features)[0]
        
        # Normalize features for MLP
        if self.normalization_stats is not None:
            features_norm = (features - self.normalization_stats['mean']) / self.normalization_stats['std']
            features_norm = np.nan_to_num(features_norm, nan=0.0)
        else:
            features_norm = features
        
        # Predict Duration (MLP)
        duration_prediction = self.duration_model.predict(features_norm, verbose=0)[0][0]
        
        # Calculate confidence based on input validity
        confidence = self._calculate_confidence(trip_data, features)
        
        return {
            'fare_amount': round(float(fare_prediction), 2),
            'duration_minutes': round(float(duration_prediction), 2),
            'confidence': confidence,
            'trip_details': {
                'distance_miles': round(float(features[0][5]), 2),
                'pickup': {
                    'longitude': trip_data['pickup_longitude'],
                    'latitude': trip_data['pickup_latitude']
                },
                'dropoff': {
                    'longitude': trip_data['dropoff_longitude'],
                    'latitude': trip_data['dropoff_latitude']
                },
                'datetime': trip_data['pickup_datetime']
            }
        }
    
    def predict_batch(self, trips: List[Dict]) -> List[Dict]:
        """
        Predict fare and duration for multiple trips
        
        Args:
            trips: List of trip dictionaries
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(trip) for trip in trips]
    
    @staticmethod
    def _haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate haversine distance in miles"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        miles = 3959 * c
        return miles
    
    @staticmethod
    def _calculate_direction(lon1, lat1, lon2, lat2):
        """Calculate direction in degrees (0-360)"""
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        direction = np.arctan2(y, x)
        direction = np.degrees(direction)
        direction = (direction + 360) % 360
        return direction
    
    @staticmethod
    def _calculate_confidence(trip_data: Dict, features: np.ndarray) -> str:
        """Calculate prediction confidence based on input validity"""
        distance = features[0][5]
        
        # Check for reasonable values
        if distance < 0.1 or distance > 50:
            return "low"
        
        if not (-74.05 <= trip_data['pickup_longitude'] <= -73.75):
            return "low"
        if not (40.60 <= trip_data['pickup_latitude'] <= 40.90):
            return "low"
        
        passenger_count = trip_data.get('passenger_count', 1)
        if passenger_count < 1 or passenger_count > 6:
            return "medium"
        
        if distance < 20 and 0.5 < distance:
            return "high"
        else:
            return "medium"


# Example usage for testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = TaxiPredictor(models_dir="../../models")
    predictor.load_models()
    
    # Example trip
    sample_trip = {
        'pickup_datetime': '2016-06-15 14:30:00',
        'pickup_longitude': -73.982,
        'pickup_latitude': 40.767,
        'dropoff_longitude': -73.958,
        'dropoff_latitude': 40.778,
        'passenger_count': 1
    }
    
    # Get prediction
    result = predictor.predict(sample_trip)
    
    print("\n" + "="*60)
    print("🚕 NYC TAXI TRIP PREDICTION")
    print("="*60)
    print(f"💵 Estimated Fare:     ${result['fare_amount']:.2f}")
    print(f"⏱️  Estimated Duration:  {result['duration_minutes']:.1f} minutes")
    print(f"📏 Distance:           {result['trip_details']['distance_miles']:.2f} miles")
    print(f"🎯 Confidence:         {result['confidence']}")
    print("="*60)
