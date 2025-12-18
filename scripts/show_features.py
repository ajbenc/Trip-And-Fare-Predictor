"""
Script to display all features used by the NYC Taxi prediction models
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.prediction.model_service import ModelService

def main():
    print("=" * 80)
    print("NYC TAXI TRIP PREDICTION - FEATURE LIST")
    print("=" * 80)
    
    # Initialize and load models
    print("\n📦 Loading models...")
    model_service = ModelService(models_dir="models/lightgbm_80_20_full_year")
    
    try:
        model_service.load()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Get feature names from the model
    if model_service._artifacts and model_service._artifacts.feature_names:
        feature_names = model_service._artifacts.feature_names
        print(f"\n✅ Total features used by models: {len(feature_names)}")
    else:
        print("\n⚠️  Feature names not available from model, using build_features schema...")
        feature_names = None
    
    # Build sample features to show the structure
    print("\n🔧 Building sample feature set to demonstrate structure...")
    
    sample_weather = {
        'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
        'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
        'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
        'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
        'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0
    }
    
    sample_holiday = {
        'is_holiday': 0, 'is_major_holiday': 0, 'is_holiday_week': 0
    }
    
    sample_features = model_service.build_features(
        pickup_zone_id=161,  # Midtown
        dropoff_zone_id=236,  # Upper East Side
        pickup_datetime=datetime(2022, 7, 15, 10, 30),
        passenger_count=1,
        estimated_distance=3.2,
        weather=sample_weather,
        holiday=sample_holiday
    )
    
    # Print all features by category
    print("\n" + "=" * 80)
    print("COMPLETE FEATURE LIST (56 FEATURES)")
    print("=" * 80)
    
    # Get all columns from the sample
    all_features = sample_features.columns.tolist()
    
    # Categorize features
    location_features = [f for f in all_features if any(x in f for x in ['Location', 'airport', 'manhattan', 'same_location'])]
    temporal_features = [f for f in all_features if any(x in f for x in ['pickup_', 'weekend', 'rush_hour', 'late_night', 'business_hours', 'sin', 'cos']) and 'interaction' not in f]
    distance_features = [f for f in all_features if 'distance' in f and 'interaction' not in f]
    weather_features = [f for f in all_features if any(x in f for x in ['temperature', 'feels_like', 'humidity', 'pressure', 'wind', 'clouds', 'precipitation', 'snow', 'weather', 'rain', 'visibility']) and 'interaction' not in f]
    holiday_features = [f for f in all_features if 'holiday' in f and 'interaction' not in f]
    interaction_features = [f for f in all_features if 'interaction' in f]
    passenger_features = [f for f in all_features if 'passenger' in f]
    
    # Print by category
    print(f"\n🗺️  LOCATION FEATURES ({len(location_features)}):")
    for i, feat in enumerate(location_features, 1):
        value = sample_features[feat].iloc[0]
        print(f"   {i:2d}. {feat:40s} = {value}")
    
    print(f"\n⏰ TEMPORAL FEATURES ({len(temporal_features)}):")
    for i, feat in enumerate(temporal_features, 1):
        value = sample_features[feat].iloc[0]
        if 'sin' in feat or 'cos' in feat:
            print(f"   {i:2d}. {feat:40s} = {value:.4f}")
        else:
            print(f"   {i:2d}. {feat:40s} = {value}")
    
    print(f"\n📏 DISTANCE FEATURES ({len(distance_features)}):")
    for i, feat in enumerate(distance_features, 1):
        value = sample_features[feat].iloc[0]
        print(f"   {i:2d}. {feat:40s} = {value}")
    
    print(f"\n🌤️  WEATHER FEATURES ({len(weather_features)}):")
    for i, feat in enumerate(weather_features, 1):
        value = sample_features[feat].iloc[0]
        print(f"   {i:2d}. {feat:40s} = {value}")
    
    print(f"\n🎉 HOLIDAY FEATURES ({len(holiday_features)}):")
    for i, feat in enumerate(holiday_features, 1):
        value = sample_features[feat].iloc[0]
        print(f"   {i:2d}. {feat:40s} = {value}")
    
    print(f"\n🔗 INTERACTION FEATURES ({len(interaction_features)}):")
    for i, feat in enumerate(interaction_features, 1):
        value = sample_features[feat].iloc[0]
        print(f"   {i:2d}. {feat:40s} = {value:.4f}")
    
    print(f"\n👥 PASSENGER FEATURES ({len(passenger_features)}):")
    for i, feat in enumerate(passenger_features, 1):
        value = sample_features[feat].iloc[0]
        print(f"   {i:2d}. {feat:40s} = {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    print(f"   Location Features:     {len(location_features):2d}")
    print(f"   Temporal Features:     {len(temporal_features):2d}")
    print(f"   Distance Features:     {len(distance_features):2d}")
    print(f"   Weather Features:      {len(weather_features):2d}")
    print(f"   Holiday Features:      {len(holiday_features):2d}")
    print(f"   Interaction Features:  {len(interaction_features):2d}")
    print(f"   Passenger Features:    {len(passenger_features):2d}")
    print("   " + "-" * 76)
    print(f"   TOTAL FEATURES:        {len(all_features):2d}")
    print("=" * 80)
    
    # Make a sample prediction
    print("\n🔮 Making sample prediction...")
    prediction = model_service.predict(sample_features)
    print(f"   Duration: {prediction['duration_minutes']:.2f} minutes")
    print(f"   Fare: ${prediction['fare_amount']:.2f}")
    print("\n✅ Feature display complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
