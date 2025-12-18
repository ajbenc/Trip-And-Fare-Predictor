"""
NYC 2022 Weather Data Downloader
=================================
Downloads ALL weather data for 2022 in ONE API call using Open-Meteo.
This is a one-time download - the data is saved and reused forever.

Open-Meteo API: https://open-meteo.com/
- 100% FREE
- No API key required
- Can fetch entire year in ONE request

Author: NYC Taxi ML Team
Date: 2024
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime


def download_2022_weather():
    """
    Download complete 2022 weather data for NYC in ONE API call.
    """
    print("="*80)
    print("🌤️ NYC 2022 WEATHER DATA DOWNLOADER")
    print("="*80)
    print()
    print("📍 Location: New York City (Central Park)")
    print("📅 Period: January 1, 2022 - December 31, 2022")
    print("🌐 API: Open-Meteo (FREE - No API key needed!)")
    print()
    
    # NYC coordinates (Central Park)
    lat = 40.7829
    lon = -73.9654
    
    # API parameters - request ENTIRE year in ONE call!
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': '2022-01-01',
        'end_date': '2022-12-31',
        'hourly': 'temperature_2m,relativehumidity_2m,precipitation,snowfall,windspeed_10m,cloudcover,visibility,pressure_msl,weathercode',
        'temperature_unit': 'fahrenheit',
        'windspeed_unit': 'mph',
        'precipitation_unit': 'inch',
        'timezone': 'America/New_York'
    }
    
    print("🌐 Fetching weather data from Open-Meteo API...")
    print("   (This may take 10-30 seconds for entire year)")
    print()
    
    try:
        # Single API call for entire year!
        response = requests.get(
            'https://archive-api.open-meteo.com/v1/archive',
            params=params,
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ Successfully fetched weather data!")
            print()
            
            # Parse response
            data = response.json()
            hourly = data['hourly']
            
            print("📊 Processing hourly weather records...")
            
            # Create DataFrame
            weather_records = []
            times = hourly['time']
            
            for i, time_str in enumerate(times):
                # Parse datetime
                dt = datetime.fromisoformat(time_str)
                
                # Weather code to description
                weather_code = hourly['weathercode'][i]
                weather_desc = _weather_code_to_description(weather_code)
                
                # Create record
                record = {
                    'datetime': dt,
                    'temperature': hourly['temperature_2m'][i],
                    'feels_like': hourly['temperature_2m'][i],  # Approximation
                    'humidity': hourly['relativehumidity_2m'][i],
                    'pressure': hourly['pressure_msl'][i] * 0.02953 if hourly['pressure_msl'][i] else None,  # hPa to inHg
                    'wind_speed': hourly['windspeed_10m'][i],
                    'clouds': hourly['cloudcover'][i],
                    'visibility': hourly['visibility'][i] / 1609.34 if hourly['visibility'][i] else None,  # meters to miles
                    'weather_main': weather_desc['main'],
                    'weather_description': weather_desc['description'],
                    'precipitation': hourly['precipitation'][i],  # Already in inches
                    'snow': hourly['snowfall'][i],  # Already in inches
                }
                
                weather_records.append(record)
            
            # Create DataFrame
            weather_df = pd.DataFrame(weather_records)
            
            print(f"✅ Created DataFrame with {len(weather_df):,} hourly records")
            print()
            
            # Add derived features
            print("🔧 Creating derived weather features...")
            weather_df = _add_derived_features(weather_df)
            print("✅ Added derived features")
            print()
            
            # Save to parquet
            output_dir = Path('data/external')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'weather_2022.parquet'
            
            weather_df.to_parquet(output_file, index=False)
            
            print("="*80)
            print("💾 WEATHER DATA SAVED!")
            print("="*80)
            print(f"📁 File: {output_file}")
            print(f"📊 Records: {len(weather_df):,} hours")
            print(f"📅 Period: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
            print()
            
            # Print summary statistics
            _print_summary(weather_df)
            
            print("="*80)
            print("✅ SUCCESS! Weather data ready for ML training")
            print("="*80)
            print()
            print("🎯 Next steps:")
            print("   1. Create holiday calendar: python src/data/create_holiday_calendar.py")
            print("   2. Feature engineering: Merge taxi + weather + holidays")
            print("   3. Train models with enhanced features")
            
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error downloading weather data: {e}")


def _weather_code_to_description(code: int) -> dict:
    """Convert WMO weather code to description."""
    code_map = {
        0: ('Clear', 'clear sky'),
        1: ('Clear', 'mainly clear'),
        2: ('Clouds', 'partly cloudy'),
        3: ('Clouds', 'overcast'),
        45: ('Fog', 'fog'),
        48: ('Fog', 'depositing rime fog'),
        51: ('Drizzle', 'light drizzle'),
        53: ('Drizzle', 'moderate drizzle'),
        55: ('Drizzle', 'dense drizzle'),
        61: ('Rain', 'slight rain'),
        63: ('Rain', 'moderate rain'),
        65: ('Rain', 'heavy rain'),
        71: ('Snow', 'slight snow'),
        73: ('Snow', 'moderate snow'),
        75: ('Snow', 'heavy snow'),
        80: ('Rain', 'slight rain showers'),
        81: ('Rain', 'moderate rain showers'),
        82: ('Rain', 'violent rain showers'),
        95: ('Thunderstorm', 'thunderstorm'),
        96: ('Thunderstorm', 'thunderstorm with hail'),
    }
    
    result = code_map.get(code, ('Unknown', 'unknown'))
    return {'main': result[0], 'description': result[1]}


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived weather features for ML model."""
    import numpy as np
    
    # Temperature categories
    df['temp_category'] = pd.cut(
        df['temperature'],
        bins=[-np.inf, 32, 50, 65, 80, np.inf],
        labels=['freezing', 'cold', 'cool', 'warm', 'hot']
    )
    
    # Weather severity score (0-10)
    df['weather_severity'] = 0
    df.loc[df['precipitation'] > 0, 'weather_severity'] += 2
    df.loc[df['precipitation'] > 0.1, 'weather_severity'] += 2
    df.loc[df['snow'] > 0, 'weather_severity'] += 3
    df.loc[df['snow'] > 1, 'weather_severity'] += 2
    df.loc[df['wind_speed'] > 20, 'weather_severity'] += 1
    df.loc[df['visibility'] < 5, 'weather_severity'] += 2
    
    # Boolean flags
    df['is_raining'] = (df['precipitation'] > 0).astype(int)
    df['is_snowing'] = (df['snow'] > 0).astype(int)
    df['is_heavy_rain'] = (df['precipitation'] > 0.1).astype(int)
    df['is_heavy_snow'] = (df['snow'] > 1).astype(int)
    df['is_extreme_weather'] = (df['weather_severity'] >= 5).astype(int)
    df['is_poor_visibility'] = (df['visibility'] < 5).astype(int)
    
    return df


def _print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("📊 WEATHER DATA SUMMARY - 2022")
    print("="*80)
    print()
    
    print(f"📅 Date Range:")
    print(f"   Start: {df['datetime'].min()}")
    print(f"   End: {df['datetime'].max()}")
    print(f"   Total Hours: {len(df):,}")
    print()
    
    print(f"🌡️ Temperature:")
    print(f"   Mean: {df['temperature'].mean():.1f}°F")
    print(f"   Min: {df['temperature'].min():.1f}°F")
    print(f"   Max: {df['temperature'].max():.1f}°F")
    print()
    
    print(f"🌧️ Precipitation:")
    rainy_hours = df['is_raining'].sum()
    snowy_hours = df['is_snowing'].sum()
    print(f"   Rainy hours: {rainy_hours:,} ({rainy_hours/len(df)*100:.1f}%)")
    print(f"   Snowy hours: {snowy_hours:,} ({snowy_hours/len(df)*100:.1f}%)")
    print(f"   Total precipitation: {df['precipitation'].sum():.1f} inches")
    print(f"   Total snowfall: {df['snow'].sum():.1f} inches")
    print()
    
    print(f"💨 Wind:")
    print(f"   Mean speed: {df['wind_speed'].mean():.1f} mph")
    print(f"   Max speed: {df['wind_speed'].max():.1f} mph")
    print()
    
    print(f"⚠️ Extreme Weather:")
    extreme_hours = df['is_extreme_weather'].sum()
    print(f"   Extreme weather hours: {extreme_hours:,} ({extreme_hours/len(df)*100:.1f}%)")
    print()
    
    # Monthly breakdown
    print(f"📆 Monthly Weather Patterns:")
    df['month'] = df['datetime'].dt.month
    monthly = df.groupby('month').agg({
        'temperature': 'mean',
        'precipitation': 'sum',
        'snow': 'sum',
        'is_extreme_weather': 'sum'
    }).round(1)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print()
    print("Month  Avg Temp  Precipitation  Snowfall  Extreme Hours")
    print("-" * 60)
    for month in range(1, 13):
        if month in monthly.index:
            row = monthly.loc[month]
            print(f"{month_names[month-1]:>5}  {row['temperature']:>7.1f}°F  {row['precipitation']:>12.1f}\"  {row['snow']:>8.1f}\"  {int(row['is_extreme_weather']):>13}")
    print()


if __name__ == '__main__':
    download_2022_weather()
