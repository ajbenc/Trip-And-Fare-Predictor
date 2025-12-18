# -*- coding: utf-8 -*-
"""
API Test Script
===============
Test the NYC Taxi Prediction API with sample requests.

Author: Julian
Date: October 2025
"""

import requests
import json
from datetime import datetime
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def print_separator(title=""):
    """Print a formatted separator."""
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)
    print()

def test_health_check():
    """Test health check endpoint."""
    print_separator("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_model_info():
    """Test model information endpoint."""
    print_separator("TEST 2: Model Information")
    
    response = requests.get(f"{BASE_URL}/models")
    data = response.json()
    
    print(f"Status Code: {response.status_code}")
    print(f"\nFare Model:")
    print(f"  Type: {data['fare_model']['type']}")
    print(f"  Accuracy: {data['fare_model']['accuracy']}")
    print(f"  RMSE: {data['fare_model']['rmse']}")
    
    print(f"\nDuration Model:")
    print(f"  Type: {data['duration_model']['type']}")
    print(f"  Accuracy: {data['duration_model']['accuracy']}")
    print(f"  RMSE: {data['duration_model']['rmse']}")
    
    print(f"\nTotal Features: {data['total_features']}")
    
    return response.status_code == 200

def test_single_prediction():
    """Test single trip prediction."""
    print_separator("TEST 3: Single Trip Prediction")
    
    # Example: Manhattan to JFK Airport on Wednesday afternoon
    payload = {
        "pickup_hour": 14,
        "pickup_day": 2,`r`n        "pickup_weekday": 2,  # Wednesday
        "pickup_month": 5,
        "is_weekend": 0,
        "is_night": 0,
        "is_morning_rush": 0,
        "is_evening_rush": 0,
        "hour_sin": np.sin(2 * np.pi * 14 / 24),
        "hour_cos": np.cos(2 * np.pi * 14 / 24),
        "day_sin": np.sin(2 * np.pi * 2 / 7),
        "day_cos": np.cos(2 * np.pi * 2 / 7),
        "pickup_dayofyear": 135,
        "pickup_weekofyear": 20,
        "is_holiday": 0,
        "pickup_is_airport": 0,
        "dropoff_is_airport": 1,
        "is_airport_trip": 1,
        "pickup_is_manhattan": 1,
        "dropoff_is_manhattan": 0,
        "pickup_borough": 1,  # Manhattan
        "dropoff_borough": 4,  # Queens (JFK)
        "is_cross_borough": 1,
        "both_manhattan": 0,
        "manhattan_to_airport": 1,
        "airport_to_manhattan": 0,
        "PULocationID": 161,  # Midtown Manhattan
        "DOLocationID": 132,  # JFK Airport
        "typical_distance": 15.2,
        "typical_duration": 35.5,
        "typical_fare": 52.0,
        "route_popularity": 1250,
        "route_efficiency": 0.85,
        "pickup_is_popular": 1,
        "dropoff_is_popular": 1,
        "rush_airport": 0,
        "weekend_night": 0,
        "distance_hour_interaction": 15.2 * 14,
        "cross_borough_rush": 0,
        "long_trip_night": 0
    }
    
    print("Trip Details:")
    print(f"  Route: Midtown Manhattan → JFK Airport")
    print(f"  Time: Wednesday, 2:00 PM (May)")
    print(f"  Typical Distance: {payload['typical_distance']} miles")
    print(f"  Typical Duration: {payload['typical_duration']} minutes")
    print(f"  Typical Fare: ${payload['typical_fare']}")
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    data = response.json()
    
    print(f"\nPrediction Results:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Predicted Fare: ${data['fare_amount']:.2f}")
    print(f"  Predicted Duration: {data['trip_duration']:.2f} minutes")
    print(f"  Model Used: {data['model_used']}")
    print(f"  Confidence: {data['confidence']}")
    
    return response.status_code == 200

def test_short_trip():
    """Test prediction for a short Manhattan trip."""
    print_separator("TEST 4: Short Manhattan Trip")
    
    # Example: Short trip within Manhattan during evening rush
    payload = {
        "pickup_hour": 18,
        "pickup_day": 4,`r`n        "pickup_weekday": 4,  # Friday
        "pickup_month": 5,
        "is_weekend": 0,
        "is_night": 0,
        "is_morning_rush": 0,
        "is_evening_rush": 1,
        "hour_sin": np.sin(2 * np.pi * 18 / 24),
        "hour_cos": np.cos(2 * np.pi * 18 / 24),
        "day_sin": np.sin(2 * np.pi * 4 / 7),
        "day_cos": np.cos(2 * np.pi * 4 / 7),
        "pickup_dayofyear": 137,
        "pickup_weekofyear": 20,
        "is_holiday": 0,
        "pickup_is_airport": 0,
        "dropoff_is_airport": 0,
        "is_airport_trip": 0,
        "pickup_is_manhattan": 1,
        "dropoff_is_manhattan": 1,
        "pickup_borough": 1,
        "dropoff_borough": 1,
        "is_cross_borough": 0,
        "both_manhattan": 1,
        "manhattan_to_airport": 0,
        "airport_to_manhattan": 0,
        "PULocationID": 237,  # Upper West Side
        "DOLocationID": 161,  # Midtown
        "typical_distance": 2.1,
        "typical_duration": 8.5,
        "typical_fare": 12.0,
        "route_popularity": 3450,
        "route_efficiency": 0.92,
        "pickup_is_popular": 1,
        "dropoff_is_popular": 1,
        "rush_airport": 0,
        "weekend_night": 0,
        "distance_hour_interaction": 2.1 * 18,
        "cross_borough_rush": 0,
        "long_trip_night": 0
    }
    
    print("Trip Details:")
    print(f"  Route: Upper West Side → Midtown Manhattan")
    print(f"  Time: Friday, 6:00 PM (Evening Rush)")
    print(f"  Typical Distance: {payload['typical_distance']} miles")
    print(f"  Typical Duration: {payload['typical_duration']} minutes")
    print(f"  Typical Fare: ${payload['typical_fare']}")
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    data = response.json()
    
    print(f"\nPrediction Results:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Predicted Fare: ${data['fare_amount']:.2f}")
    print(f"  Predicted Duration: {data['trip_duration']:.2f} minutes")
    print(f"  Model Used: {data['model_used']}")
    print(f"  Confidence: {data['confidence']}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_separator("TEST 5: Batch Prediction (3 trips)")
    
    # Create 3 different trip scenarios
    trips = []
    
    # Trip 1: Morning rush Manhattan
    trip1 = {
        "pickup_hour": 8, "pickup_day": 1,`r`n        "pickup_weekday": 1, "pickup_month": 5,
        "is_weekend": 0, "is_night": 0, "is_morning_rush": 1,
        "is_evening_rush": 0, "hour_sin": 0.707, "hour_cos": 0.707,
        "day_sin": 0.433, "day_cos": -0.901, "pickup_dayofyear": 134,
        "pickup_weekofyear": 20, "is_holiday": 0,
        "pickup_is_airport": 0, "dropoff_is_airport": 0,
        "is_airport_trip": 0, "pickup_is_manhattan": 1,
        "dropoff_is_manhattan": 1, "pickup_borough": 1,
        "dropoff_borough": 1, "is_cross_borough": 0,
        "both_manhattan": 1, "manhattan_to_airport": 0,
        "airport_to_manhattan": 0, "PULocationID": 142,
        "DOLocationID": 236, "typical_distance": 3.2,
        "typical_duration": 12.5, "typical_fare": 15.0,
        "route_popularity": 2100, "route_efficiency": 0.88,
        "pickup_is_popular": 1, "dropoff_is_popular": 1,
        "rush_airport": 0, "weekend_night": 0,
        "distance_hour_interaction": 25.6, "cross_borough_rush": 0,
        "long_trip_night": 0
    }
    
    # Trip 2: Weekend night Brooklyn to Manhattan
    trip2 = {
        "pickup_hour": 23, "pickup_day": 5,`r`n        "pickup_weekday": 5, "pickup_month": 5,
        "is_weekend": 1, "is_night": 1, "is_morning_rush": 0,
        "is_evening_rush": 0, "hour_sin": -0.259, "hour_cos": -0.966,
        "day_sin": -0.975, "day_cos": -0.223, "pickup_dayofyear": 138,
        "pickup_weekofyear": 20, "is_holiday": 0,
        "pickup_is_airport": 0, "dropoff_is_airport": 0,
        "is_airport_trip": 0, "pickup_is_manhattan": 0,
        "dropoff_is_manhattan": 1, "pickup_borough": 3,
        "dropoff_borough": 1, "is_cross_borough": 1,
        "both_manhattan": 0, "manhattan_to_airport": 0,
        "airport_to_manhattan": 0, "PULocationID": 61,
        "DOLocationID": 186, "typical_distance": 6.8,
        "typical_duration": 18.2, "typical_fare": 24.0,
        "route_popularity": 890, "route_efficiency": 0.79,
        "pickup_is_popular": 0, "dropoff_is_popular": 1,
        "rush_airport": 0, "weekend_night": 1,
        "distance_hour_interaction": 156.4, "cross_borough_rush": 0,
        "long_trip_night": 0
    }
    
    # Trip 3: Airport trip
    trip3 = {
        "pickup_hour": 10, "pickup_day": 0,`r`n        "pickup_weekday": 0, "pickup_month": 5,
        "is_weekend": 0, "is_night": 0, "is_morning_rush": 0,
        "is_evening_rush": 0, "hour_sin": 0.866, "hour_cos": -0.5,
        "day_sin": 0.0, "day_cos": 1.0, "pickup_dayofyear": 133,
        "pickup_weekofyear": 20, "is_holiday": 0,
        "pickup_is_airport": 1, "dropoff_is_airport": 0,
        "is_airport_trip": 1, "pickup_is_manhattan": 0,
        "dropoff_is_manhattan": 1, "pickup_borough": 4,
        "dropoff_borough": 1, "is_cross_borough": 1,
        "both_manhattan": 0, "manhattan_to_airport": 0,
        "airport_to_manhattan": 1, "PULocationID": 132,
        "DOLocationID": 237, "typical_distance": 16.5,
        "typical_duration": 38.0, "typical_fare": 55.0,
        "route_popularity": 1580, "route_efficiency": 0.83,
        "pickup_is_popular": 1, "dropoff_is_popular": 1,
        "rush_airport": 0, "weekend_night": 0,
        "distance_hour_interaction": 165.0, "cross_borough_rush": 0,
        "long_trip_night": 0
    }
    
    trips = [trip1, trip2, trip3]
    
    print("Testing 3 different trip scenarios:")
    print("  1. Manhattan → Manhattan (Morning Rush)")
    print("  2. Brooklyn → Manhattan (Weekend Night)")
    print("  3. JFK Airport → Manhattan (Monday Morning)")
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=trips)
    data = response.json()
    
    print(f"\nBatch Prediction Results:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Total Predictions: {data['count']}")
    
    for i, pred in enumerate(data['predictions'], 1):
        print(f"\n  Trip {i}:")
        print(f"    Predicted Fare: ${pred['fare_amount']:.2f}")
        print(f"    Predicted Duration: {pred['trip_duration']:.2f} minutes")
    
    return response.status_code == 200

def main():
    """Run all API tests."""
    print("\n" + "="*70)
    print("  NYC TAXI PREDICTION API - TEST SUITE")
    print("="*70)
    print(f"\n  Testing API at: {BASE_URL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    results = []
    
    try:
        results.append(("Health Check", test_health_check()))
        results.append(("Model Info", test_model_info()))
        results.append(("Single Prediction", test_single_prediction()))
        results.append(("Short Trip", test_short_trip()))
        results.append(("Batch Prediction", test_batch_prediction()))
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server!")
        print("   Make sure the server is running at http://localhost:8000")
        print("   Start it with: python -m uvicorn api.app:app --reload")
        return
    
    # Print summary
    print_separator("TEST SUMMARY")
    
    passed = sum(results)
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed successfully!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
