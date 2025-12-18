# NYC Taxi Trip Prediction Models - Full Year (2022)

## Overview

These are production-ready LightGBM models trained on **all 12 months** of 2022 NYC Yellow Taxi data (36.6M trips) for predicting trip duration and fare amount.

## Model Performance

### Duration Prediction Model
- **Test R²**: 0.8796
- **Test RMSE**: 3.58 minutes
- **Test MAE**: 2.46 minutes
- **Training Data**: 29.2M samples
- **Test Data**: 7.3M samples

### Fare Prediction Model
- **Test R²**: 0.9437
- **Test RMSE**: $3.04
- **Test MAE**: $1.36
- **Training Data**: 29.2M samples
- **Test Data**: 7.3M samples

## Files

```
lightgbm_80_20_full_year/
├── duration_model_final.pkl    # Duration prediction model (pickle)
├── fare_model_final.pkl         # Fare prediction model (pickle)
├── model_metrics.json           # Training metrics and performance
└── README.md                    # This file
```

## Required Features (56 total)

The models require exactly **56 features** in this order:

### 1. Location Features (6)
- `PULocationID` - Pickup location zone ID (1-263)
- `DOLocationID` - Dropoff location zone ID (1-263)
- `pickup_is_airport` - Is pickup at airport? (0/1)
- `dropoff_is_airport` - Is dropoff at airport? (0/1)
- `is_airport_trip` - Is either pickup or dropoff at airport? (0/1)
- `pickup_is_manhattan` - Is pickup in Manhattan? (0/1)
- `dropoff_is_manhattan` - Is dropoff in Manhattan? (0/1)
- `same_location` - Same pickup and dropoff location? (0/1)

### 2. Trip Features (2)
- `passenger_count` - Number of passengers (1-6)
- `estimated_distance` - Estimated trip distance in miles (0.1-50.0)

### 3. Temporal Features (10)
- `pickup_hour` - Hour of day (0-23)
- `pickup_day` - Day of month (1-31)
- `pickup_month` - Month (1-12)
- `pickup_weekday` - Day of week (0=Monday, 6=Sunday)
- `pickup_dayofyear` - Day of year (1-365)
- `pickup_weekofyear` - Week of year (1-52)
- `is_weekend` - Is weekend? (0/1)
- `is_rush_hour` - Is rush hour (7-9am or 4-7pm)? (0/1)
- `is_late_night` - Is late night (10pm-4am)? (0/1)
- `is_business_hours` - Is business hours (9am-5pm weekday)? (0/1)

### 4. Weather Features (16)
- `temperature` - Temperature in Fahrenheit
- `feels_like` - Feels like temperature in Fahrenheit
- `humidity` - Humidity percentage (0-100)
- `pressure` - Atmospheric pressure (hPa)
- `wind_speed` - Wind speed (mph)
- `clouds` - Cloud coverage percentage (0-100)
- `precipitation` - Precipitation in inches
- `snow` - Snowfall in inches
- `weather_severity` - Weather severity score (0-10)
- `is_raining` - Is it raining? (0/1)
- `is_snowing` - Is it snowing? (0/1)
- `is_heavy_rain` - Heavy rain? (0/1)
- `is_heavy_snow` - Heavy snow? (0/1)
- `is_extreme_weather` - Extreme weather? (0/1)
- `is_poor_visibility` - Poor visibility? (0/1)

### 5. Holiday Features (3)
- `is_holiday` - Is it a holiday? (0/1)
- `is_major_holiday` - Is it a major holiday? (0/1)
- `is_holiday_week` - Within 3 days of major holiday? (0/1)

### 6. Interaction Features (8)
- `weather_airport_interaction` - weather_severity × is_airport_trip
- `weather_rushhour_interaction` - weather_severity × is_rush_hour
- `rushhour_airport_interaction` - is_rush_hour × is_airport_trip
- `latenight_manhattan_interaction` - is_late_night × pickup_is_manhattan
- `distance_hour_interaction` - estimated_distance × pickup_hour
- `distance_rushhour_interaction` - estimated_distance × is_rush_hour
- `holiday_airport_interaction` - is_holiday × is_airport_trip
- `holiday_manhattan_interaction` - is_holiday × pickup_is_manhattan
- `rain_distance_interaction` - is_raining × estimated_distance
- `snow_distance_interaction` - is_snowing × estimated_distance

### 7. Cyclical Encoding Features (8)
- `hour_sin` - sin(2π × hour / 24)
- `hour_cos` - cos(2π × hour / 24)
- `weekday_sin` - sin(2π × weekday / 7)
- `weekday_cos` - cos(2π × weekday / 7)
- `month_sin` - sin(2π × month / 12)
- `month_cos` - cos(2π × month / 12)
- `dayofyear_sin` - sin(2π × dayofyear / 365)
- `dayofyear_cos` - cos(2π × dayofyear / 365)

## Installation

```bash
pip install lightgbm pandas numpy scikit-learn
```

## Usage

### Basic Example

```python
import pickle
import pandas as pd
import numpy as np

# Load models
with open('models/lightgbm_80_20_full_year/duration_model_final.pkl', 'rb') as f:
    duration_model = pickle.load(f)

with open('models/lightgbm_80_20_full_year/fare_model_final.pkl', 'rb') as f:
    fare_model = pickle.load(f)

# Prepare input features (example)
features = pd.DataFrame([{
    # Location
    'PULocationID': 161,  # JFK Airport
    'DOLocationID': 230,  # Times Square
    'pickup_is_airport': 1,
    'dropoff_is_airport': 0,
    'is_airport_trip': 1,
    'pickup_is_manhattan': 0,
    'dropoff_is_manhattan': 1,
    'same_location': 0,
    
    # Trip
    'passenger_count': 2,
    'estimated_distance': 17.5,  # miles
    
    # Temporal
    'pickup_hour': 14,
    'pickup_day': 15,
    'pickup_month': 7,
    'pickup_weekday': 4,  # Friday
    'pickup_dayofyear': 196,
    'pickup_weekofyear': 28,
    'is_weekend': 0,
    'is_rush_hour': 0,
    'is_late_night': 0,
    'is_business_hours': 1,
    
    # Weather (good weather)
    'temperature': 78.0,
    'feels_like': 80.0,
    'humidity': 65.0,
    'pressure': 1013.0,
    'wind_speed': 8.0,
    'clouds': 20.0,
    'precipitation': 0.0,
    'snow': 0.0,
    'weather_severity': 1.0,
    'is_raining': 0,
    'is_snowing': 0,
    'is_heavy_rain': 0,
    'is_heavy_snow': 0,
    'is_extreme_weather': 0,
    'is_poor_visibility': 0,
    
    # Holiday
    'is_holiday': 0,
    'is_major_holiday': 0,
    'is_holiday_week': 0,
    
    # Interactions
    'weather_airport_interaction': 1.0,  # 1.0 * 1
    'weather_rushhour_interaction': 0.0,
    'rushhour_airport_interaction': 0,
    'latenight_manhattan_interaction': 0,
    'distance_hour_interaction': 245.0,  # 17.5 * 14
    'distance_rushhour_interaction': 0.0,
    'holiday_airport_interaction': 0,
    'holiday_manhattan_interaction': 0,
    'rain_distance_interaction': 0.0,
    'snow_distance_interaction': 0.0,
    
    # Cyclical
    'hour_sin': np.sin(2 * np.pi * 14 / 24),
    'hour_cos': np.cos(2 * np.pi * 14 / 24),
    'weekday_sin': np.sin(2 * np.pi * 4 / 7),
    'weekday_cos': np.cos(2 * np.pi * 4 / 7),
    'month_sin': np.sin(2 * np.pi * 7 / 12),
    'month_cos': np.cos(2 * np.pi * 7 / 12),
    'dayofyear_sin': np.sin(2 * np.pi * 196 / 365),
    'dayofyear_cos': np.cos(2 * np.pi * 196 / 365),
}])

# Make predictions
duration_pred = duration_model.predict(features)[0]
fare_pred = fare_model.predict(features)[0]

print(f"Predicted Duration: {duration_pred:.2f} minutes")
print(f"Predicted Fare: ${fare_pred:.2f}")
```

Output:
```
Predicted Duration: 45.23 minutes
Predicted Fare: $68.50
```

### Using with Feature Engineering Pipeline

```python
import pickle
import pandas as pd
from datetime import datetime

# Load models
with open('models/lightgbm_80_20_full_year/duration_model_final.pkl', 'rb') as f:
    duration_model = pickle.load(f)

with open('models/lightgbm_80_20_full_year/fare_model_final.pkl', 'rb') as f:
    fare_model = pickle.load(f)

def create_features(pickup_location_id, dropoff_location_id, 
                   passenger_count, estimated_distance,
                   pickup_datetime, weather_data, holiday_data):
    """
    Create all 56 features from basic inputs.
    
    Args:
        pickup_location_id: NYC taxi zone ID for pickup
        dropoff_location_id: NYC taxi zone ID for dropoff
        passenger_count: Number of passengers
        estimated_distance: Estimated trip distance in miles
        pickup_datetime: datetime object for pickup time
        weather_data: dict with weather info
        holiday_data: dict with holiday info
    
    Returns:
        DataFrame with 56 features
    """
    
    features = {}
    
    # Location features
    airport_zones = [132, 138, 161]
    manhattan_zones = list(range(4, 234))
    
    features['PULocationID'] = pickup_location_id
    features['DOLocationID'] = dropoff_location_id
    features['pickup_is_airport'] = int(pickup_location_id in airport_zones)
    features['dropoff_is_airport'] = int(dropoff_location_id in airport_zones)
    features['is_airport_trip'] = int(features['pickup_is_airport'] or features['dropoff_is_airport'])
    features['pickup_is_manhattan'] = int(pickup_location_id in manhattan_zones)
    features['dropoff_is_manhattan'] = int(dropoff_location_id in manhattan_zones)
    features['same_location'] = int(pickup_location_id == dropoff_location_id)
    
    # Trip features
    features['passenger_count'] = passenger_count
    features['estimated_distance'] = max(0.1, min(estimated_distance, 50.0))
    
    # Temporal features
    features['pickup_hour'] = pickup_datetime.hour
    features['pickup_day'] = pickup_datetime.day
    features['pickup_month'] = pickup_datetime.month
    features['pickup_weekday'] = pickup_datetime.weekday()
    features['pickup_dayofyear'] = pickup_datetime.timetuple().tm_yday
    features['pickup_weekofyear'] = pickup_datetime.isocalendar()[1]
    features['is_weekend'] = int(features['pickup_weekday'] >= 5)
    features['is_rush_hour'] = int((7 <= features['pickup_hour'] <= 9) or 
                                   (16 <= features['pickup_hour'] <= 19))
    features['is_late_night'] = int((features['pickup_hour'] >= 22) or 
                                    (features['pickup_hour'] <= 4))
    features['is_business_hours'] = int((9 <= features['pickup_hour'] <= 17) and 
                                        (features['pickup_weekday'] < 5))
    
    # Weather features
    features.update(weather_data)
    
    # Holiday features
    features.update(holiday_data)
    
    # Interaction features
    features['weather_airport_interaction'] = weather_data['weather_severity'] * features['is_airport_trip']
    features['weather_rushhour_interaction'] = weather_data['weather_severity'] * features['is_rush_hour']
    features['rushhour_airport_interaction'] = features['is_rush_hour'] * features['is_airport_trip']
    features['latenight_manhattan_interaction'] = features['is_late_night'] * features['pickup_is_manhattan']
    features['distance_hour_interaction'] = features['estimated_distance'] * features['pickup_hour']
    features['distance_rushhour_interaction'] = features['estimated_distance'] * features['is_rush_hour']
    features['holiday_airport_interaction'] = holiday_data['is_holiday'] * features['is_airport_trip']
    features['holiday_manhattan_interaction'] = holiday_data['is_holiday'] * features['pickup_is_manhattan']
    features['rain_distance_interaction'] = weather_data['is_raining'] * features['estimated_distance']
    features['snow_distance_interaction'] = weather_data['is_snowing'] * features['estimated_distance']
    
    # Cyclical encoding
    import numpy as np
    features['hour_sin'] = np.sin(2 * np.pi * features['pickup_hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['pickup_hour'] / 24)
    features['weekday_sin'] = np.sin(2 * np.pi * features['pickup_weekday'] / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * features['pickup_weekday'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * features['pickup_month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['pickup_month'] / 12)
    features['dayofyear_sin'] = np.sin(2 * np.pi * features['pickup_dayofyear'] / 365)
    features['dayofyear_cos'] = np.cos(2 * np.pi * features['pickup_dayofyear'] / 365)
    
    return pd.DataFrame([features])

# Example usage
weather = {
    'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
    'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
    'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
    'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
    'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0
}

holiday = {
    'is_holiday': 0, 'is_major_holiday': 0, 'is_holiday_week': 0
}

X = create_features(
    pickup_location_id=230,  # Times Square
    dropoff_location_id=161,  # JFK
    passenger_count=1,
    estimated_distance=17.5,
    pickup_datetime=datetime(2022, 7, 15, 10, 30),
    weather_data=weather,
    holiday_data=holiday
)

duration = duration_model.predict(X)[0]
fare = fare_model.predict(X)[0]

print(f"Trip Duration: {duration:.1f} minutes")
print(f"Trip Fare: ${fare:.2f}")
```

## Feature Engineering Notes

### Estimated Distance
- In production, use a route planning API (Google Maps, MapBox) to get estimated distance
- For testing, you can use historical average distance for the route pair
- **IMPORTANT**: Clip values between 0.1 and 50.0 miles

### Weather Data
- Use real-time weather API (OpenWeatherMap, Weather.com)
- Round pickup time to nearest hour for weather lookup
- Fill missing values: median for numeric, 0 for binary flags

### Holiday Data
- Maintain a calendar of holidays (federal + NYC-specific)
- Mark major holidays (Thanksgiving, Christmas, New Year's, July 4th)
- Holiday week = ±3 days around major holiday

### Zone IDs
- NYC has 263 taxi zones (TLC Taxi Zones shapefile)
- Airport zones: JFK (132, 138), LaGuardia (161)
- Manhattan: zones 4-233 (approximately)

## Model Details

### Training Configuration

**Duration Model:**
- Algorithm: LightGBM Gradient Boosting
- Trees: 1000
- Max Depth: 12
- Learning Rate: 0.05
- Num Leaves: 127
- L1/L2 Regularization: 0.1

**Fare Model:**
- Algorithm: LightGBM Gradient Boosting
- Trees: 500
- Max Depth: 10
- Learning Rate: 0.05
- Num Leaves: 63
- L1/L2 Regularization: 0.1

### Training Dataset
- **Total Samples**: 36,556,803 trips
- **Training Set**: 29,245,436 trips (80%)
- **Test Set**: 7,311,367 trips (20%)
- **Time Period**: All 12 months of 2022 (Jan-Dec)
- **Source**: NYC TLC Yellow Taxi Trip Records

## Important Notes

1. **Feature Order Matters**: The 56 features must be in the exact order listed above
2. **No Data Leakage**: All features must be available at prediction time (pickup)
3. **Value Ranges**: Ensure all features are within valid ranges
4. **Missing Values**: Handle missing weather/holiday data with appropriate defaults
5. **Zone ID Validation**: Use only valid NYC taxi zone IDs (1-263)

## Troubleshooting

### Error: "Feature names mismatch"
- Ensure DataFrame has exactly 56 columns in correct order
- Check feature names match exactly (case-sensitive)

### Error: "Expected 56 features, got X"
- Count your features - must be exactly 56
- Check for duplicate or missing features

### Poor Predictions
- Verify estimated_distance is reasonable (0.1-50 miles)
- Check weather features are filled (no NaN values)
- Ensure cyclical features are calculated correctly


