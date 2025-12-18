# -*- coding: utf-8 -*-
"""
NYC Taxi Prediction API
=======================
FastAPI application for real-time fare and duration predictions.

Uses best performing models:
- Fare Amount: XGBoost (R² = 94.31%)
- Trip Duration: XGBoost (R² = 86.09%)

Author: Julian
Date: October 2025
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Trip Prediction API",
    description="Predict taxi fares and trip durations using advanced ML models",
    version="1.0.0"
)

# Global variables for models
fare_model = None
duration_model = None
feature_names = None

# Feature names (40 features) - MUST MATCH TRAINING DATA ORDER
FEATURE_NAMES = [
    'pickup_hour', 'pickup_day', 'pickup_month', 'pickup_weekday', 'is_weekend',
    'is_night', 'is_morning_rush', 'is_evening_rush', 'hour_sin', 'hour_cos',
    'day_sin', 'day_cos', 'pickup_dayofyear', 'pickup_weekofyear', 'is_holiday',
    'pickup_is_airport', 'dropoff_is_airport', 'is_airport_trip',
    'pickup_is_manhattan', 'dropoff_is_manhattan', 'pickup_borough',
    'dropoff_borough', 'is_cross_borough', 'pickup_is_popular',
    'dropoff_is_popular', 'both_manhattan', 'manhattan_to_airport',
    'airport_to_manhattan', 'typical_distance', 'typical_duration',
    'typical_fare', 'route_popularity', 'route_efficiency', 'rush_airport',
    'weekend_night', 'distance_hour_interaction', 'cross_borough_rush',
    'long_trip_night', 'PULocationID', 'DOLocationID'
]


class TripFeatures(BaseModel):
    """Input features for trip prediction."""
    
    # Temporal features
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    pickup_day: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    pickup_month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    pickup_weekday: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, 0 otherwise")
    is_night: int = Field(..., ge=0, le=1, description="1 if night (0-6 or 22-23), 0 otherwise")
    is_morning_rush: int = Field(..., ge=0, le=1, description="1 if morning rush (7-9), 0 otherwise")
    is_evening_rush: int = Field(..., ge=0, le=1, description="1 if evening rush (17-19), 0 otherwise")
    hour_sin: float = Field(..., description="Sin encoding of hour")
    hour_cos: float = Field(..., description="Cos encoding of hour")
    day_sin: float = Field(..., description="Sin encoding of day")
    day_cos: float = Field(..., description="Cos encoding of day")
    pickup_dayofyear: int = Field(..., ge=1, le=366, description="Day of year (1-366)")
    pickup_weekofyear: int = Field(..., ge=1, le=53, description="Week of year (1-53)")
    is_holiday: int = Field(..., ge=0, le=1, description="1 if holiday, 0 otherwise")
    
    # Location features
    pickup_is_airport: int = Field(..., ge=0, le=1)
    dropoff_is_airport: int = Field(..., ge=0, le=1)
    is_airport_trip: int = Field(..., ge=0, le=1)
    pickup_is_manhattan: int = Field(..., ge=0, le=1)
    dropoff_is_manhattan: int = Field(..., ge=0, le=1)
    pickup_borough: int = Field(..., ge=0, le=5, description="Borough ID (0-5)")
    dropoff_borough: int = Field(..., ge=0, le=5, description="Borough ID (0-5)")
    is_cross_borough: int = Field(..., ge=0, le=1)
    both_manhattan: int = Field(..., ge=0, le=1)
    manhattan_to_airport: int = Field(..., ge=0, le=1)
    airport_to_manhattan: int = Field(..., ge=0, le=1)
    PULocationID: int = Field(..., ge=1, le=265, description="Pickup location ID")
    DOLocationID: int = Field(..., ge=1, le=265, description="Dropoff location ID")
    
    # Route intelligence
    typical_distance: float = Field(..., gt=0, description="Typical distance for this route (miles)")
    typical_duration: float = Field(..., gt=0, description="Typical duration for this route (minutes)")
    typical_fare: float = Field(..., gt=0, description="Typical fare for this route ($)")
    route_popularity: int = Field(..., ge=0, description="Number of trips on this route")
    route_efficiency: float = Field(..., description="Route efficiency score")
    
    # Popularity
    pickup_is_popular: int = Field(..., ge=0, le=1)
    dropoff_is_popular: int = Field(..., ge=0, le=1)
    
    # Interactions
    rush_airport: int = Field(..., ge=0, le=1)
    weekend_night: int = Field(..., ge=0, le=1)
    distance_hour_interaction: float = Field(...)
    cross_borough_rush: int = Field(..., ge=0, le=1)
    long_trip_night: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_hour": 14, "pickup_day": 2, "pickup_month": 5,
                "pickup_weekday": 2, "is_weekend": 0, "is_night": 0,
                "is_morning_rush": 0, "is_evening_rush": 0, "hour_sin": 0.866,
                "hour_cos": -0.5, "day_sin": 0.781, "day_cos": -0.625,
                "pickup_dayofyear": 135, "pickup_weekofyear": 20, "is_holiday": 0,
                "pickup_is_airport": 0, "dropoff_is_airport": 1,
                "is_airport_trip": 1, "pickup_is_manhattan": 1,
                "dropoff_is_manhattan": 0, "pickup_borough": 1,
                "dropoff_borough": 4, "is_cross_borough": 1,
                "pickup_is_popular": 1, "dropoff_is_popular": 1,
                "both_manhattan": 0, "manhattan_to_airport": 1,
                "airport_to_manhattan": 0, "typical_distance": 15.2,
                "typical_duration": 35.5, "typical_fare": 52.0,
                "route_popularity": 1250, "route_efficiency": 0.85,
                "rush_airport": 0, "weekend_night": 0,
                "distance_hour_interaction": 212.8, "cross_borough_rush": 0,
                "long_trip_night": 0, "PULocationID": 161, "DOLocationID": 132
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    fare_amount: float = Field(..., description="Predicted fare amount ($)")
    trip_duration: float = Field(..., description="Predicted trip duration (minutes)")
    model_used: str = Field(default="XGBoost", description="Model used for prediction")
    confidence: str = Field(default="High", description="Confidence level")


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global fare_model, duration_model
    
    try:
        models_dir = project_root / "src" / "models" / "advanced"
        
        fare_model_path = models_dir / "xgboost_fare_amount.pkl"
        duration_model_path = models_dir / "xgboost_trip_duration.pkl"
        
        fare_model = joblib.load(fare_model_path)
        duration_model = joblib.load(duration_model_path)
        
        print("✅ Models loaded successfully!")
        print(f"   Fare Model: {fare_model_path}")
        print(f"   Duration Model: {duration_model_path}")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NYC Taxi Trip Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict fare and duration",
            "/health": "GET - Health check",
            "/models": "GET - Model information",
            "/docs": "Swagger UI documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": fare_model is not None and duration_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
async def get_model_info():
    """Get information about loaded models."""
    return {
        "fare_model": {
            "type": "XGBoost Regressor",
            "accuracy": "94.31% R²",
            "rmse": "$2.95",
            "features": 40
        },
        "duration_model": {
            "type": "XGBoost Regressor",
            "accuracy": "86.09% R²",
            "rmse": "4.88 minutes",
            "features": 40
        },
        "total_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_trip(features: TripFeatures):
    """
    Predict fare amount and trip duration for a taxi trip.
    
    Args:
        features: Trip features (40 features total)
    
    Returns:
        PredictionResponse with fare and duration predictions
    """
    try:
        # Convert features to array
        feature_dict = features.dict()
        feature_array = np.array([[feature_dict[name] for name in FEATURE_NAMES]])
        
        # Make predictions
        fare_prediction = fare_model.predict(feature_array)[0]
        duration_prediction = duration_model.predict(feature_array)[0]
        
        # Ensure positive predictions
        fare_prediction = max(2.5, fare_prediction)  # Minimum fare
        duration_prediction = max(1.0, duration_prediction)  # Minimum duration
        
        return PredictionResponse(
            fare_amount=round(fare_prediction, 2),
            trip_duration=round(duration_prediction, 2),
            model_used="XGBoost",
            confidence="High" if fare_prediction < 100 else "Medium"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(trips: List[TripFeatures]):
    """
    Predict fare and duration for multiple trips.
    
    Args:
        trips: List of trip features
    
    Returns:
        List of predictions
    """
    try:
        predictions = []
        
        for trip_features in trips:
            feature_dict = trip_features.dict()
            feature_array = np.array([[feature_dict[name] for name in FEATURE_NAMES]])
            
            fare_pred = fare_model.predict(feature_array)[0]
            duration_pred = duration_model.predict(feature_array)[0]
            
            predictions.append({
                "fare_amount": round(max(2.5, fare_pred), 2),
                "trip_duration": round(max(1.0, duration_pred), 2),
                "model_used": "XGBoost"
            })
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("NYC TAXI PREDICTION API")
    print("="*70)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("API Root: http://localhost:8000")
    print("\nPress CTRL+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
