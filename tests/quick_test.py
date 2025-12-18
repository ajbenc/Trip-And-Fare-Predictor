# -*- coding: utf-8 -*-
"""
Quick API Test
=============
Simple test of the API with one prediction.
"""

import requests
import numpy as np

BASE_URL = "http://localhost:8000"

# Example: Manhattan to JFK Airport
payload = {
    "pickup_hour": 14,
    "pickup_day": 2,
    "pickup_month": 5,
    "pickup_weekday": 2,  # Wednesday
    "is_weekend": 0,
    "is_night": 0,
    "is_morning_rush": 0,
    "is_evening_rush": 0,
    "hour_sin": float(np.sin(2 * np.pi * 14 / 24)),
    "hour_cos": float(np.cos(2 * np.pi * 14 / 24)),
    "day_sin": float(np.sin(2 * np.pi * 2 / 7)),
    "day_cos": float(np.cos(2 * np.pi * 2 / 7)),
    "pickup_dayofyear": 135,
    "pickup_weekofyear": 20,
    "is_holiday": 0,
    "pickup_is_airport": 0,
    "dropoff_is_airport": 1,
    "is_airport_trip": 1,
    "pickup_is_manhattan": 1,
    "dropoff_is_manhattan": 0,
    "pickup_borough": 1,
    "dropoff_borough": 4,
    "is_cross_borough": 1,
    "pickup_is_popular": 1,
    "dropoff_is_popular": 1,
    "both_manhattan": 0,
    "manhattan_to_airport": 1,
    "airport_to_manhattan": 0,
    "typical_distance": 15.2,
    "typical_duration": 35.5,
    "typical_fare": 52.0,
    "route_popularity": 1250,
    "route_efficiency": 0.85,
    "rush_airport": 0,
    "weekend_night": 0,
    "distance_hour_interaction": 15.2 * 14,
    "cross_borough_rush": 0,
    "long_trip_night": 0,
    "PULocationID": 161,
    "DOLocationID": 132
}

print("=" * 70)
print("NYC TAXI PREDICTION API - QUICK TEST")
print("=" * 70)
print("\nTrip: Midtown Manhattan → JFK Airport")
print("Time: Wednesday, 2:00 PM (May)")
print(f"Typical Distance: {payload['typical_distance']} miles")
print(f"Typical Duration: {payload['typical_duration']} minutes")
print(f"Typical Fare: ${payload['typical_fare']}")

try:
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ PREDICTION SUCCESSFUL!")
        print(f"   Predicted Fare: ${data['fare_amount']:.2f}")
        print(f"   Predicted Duration: {data['trip_duration']:.2f} minutes")
        print(f"   Model: {data['model_used']}")
        print(f"   Confidence: {data['confidence']}")
    else:
        print(f"\n❌ ERROR: Status code {response.status_code}")
        print(f"   {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Cannot connect to API server!")
    print("   Make sure the server is running at http://localhost:8000")

print("=" * 70)
