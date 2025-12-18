"""
NYC Taxi Trip Weather Data Fetcher
===================================
Fetches historical weather data from Open-Meteo API for 2022.
This script extracts unique datetime hours from cleaned taxi data and fetches corresponding weather information.

Open-Meteo API: https://open-meteo.com/
- 100% FREE for historical data
- No API key required
- 10,000 calls/day limit
- Historical data available back to 1940

Weather Features Collected:
- Temperature (°F)
- Feels like temperature (°F)
- Precipitation (inches)
- Snow (inches)
- Wind speed (mph)
- Humidity (%)
- Visibility (miles)
- Pressure (inHg)
- Weather condition (Clear, Rain, Snow, etc.)
- Cloud coverage (%)

Author: NYC Taxi ML Team
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from tqdm import tqdm
import os
from typing import Dict, List, Optional


class WeatherDataFetcher:
    """
    Fetches historical weather data from Open-Meteo API.
    Handles rate limiting, retries, and data persistence.
    """
    
    def __init__(self, cache_file: str = 'data/external/weather_cache.json'):
        """
        Initialize weather data fetcher.
        
        Args:
            cache_file: Path to cache file for storing fetched data
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # NYC coordinates (Central Park)
        self.lat = 40.7829
        self.lon = -73.9654
        
        # API endpoints (Open-Meteo - FREE!)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Rate limiting (be respectful: 10,000/day, ~2 calls/sec safe rate)
        self.calls_per_second = 2
        self.last_call_time = 0
        
        # Load cache
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load cached weather data from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save weather data cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last_call < min_interval:
            sleep_time = min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def fetch_weather_for_date(self, date: datetime.date, max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch weather data for an entire day (24 hours) from Open-Meteo.
        
        Args:
            date: Date to fetch weather for
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with hourly weather data for the day or None if failed
        """
        # Check cache first
        cache_key = date.strftime('%Y-%m-%d')
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Fetch from API (Open-Meteo format)
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': date.strftime('%Y-%m-%d'),
            'end_date': date.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relativehumidity_2m,precipitation,snowfall,windspeed_10m,cloudcover,visibility,pressure_msl,weathercode',
            'temperature_unit': 'fahrenheit',
            'windspeed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': 'America/New_York'
        }
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(self.base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    weather_data = self._parse_weather_response(data)
                    
                    # Cache successful response
                    self.cache[cache_key] = weather_data
                    
                    # Save cache periodically (every 10 days)
                    if len(self.cache) % 10 == 0:
                        self._save_cache()
                    
                    return weather_data
                
                elif response.status_code == 429:  # Rate limit exceeded
                    print(f"⚠️ Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                else:
                    print(f"⚠️ API error {response.status_code}: {response.text}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                print(f"⚠️ Error fetching weather (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        
        return None
    
    def _parse_weather_response(self, data: Dict) -> Dict:
        """
        Parse Open-Meteo API response into hourly weather data.
        
        Args:
            data: Raw API response from Open-Meteo
            
        Returns:
            Dictionary mapping hour -> weather data
        """
        try:
            hourly = data['hourly']
            times = hourly['time']
            
            # Create hourly weather dictionary
            hourly_weather = {}
            
            for i, time_str in enumerate(times):
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                
                # Weather code to description mapping
                weather_code = hourly['weathercode'][i]
                weather_desc = self._weather_code_to_description(weather_code)
                
                hourly_weather[dt.strftime('%Y-%m-%d %H:00:00')] = {
                    'temperature': hourly['temperature_2m'][i],
                    'feels_like': hourly['temperature_2m'][i],  # Open-Meteo doesn't have feels_like directly
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
            
            return hourly_weather
            
        except Exception as e:
            print(f"⚠️ Error parsing weather response: {e}")
            return {}
    
    def _weather_code_to_description(self, code: int) -> Dict[str, str]:
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
    
    def fetch_weather_for_dataset(self, cleaned_data_dir: str, output_file: str):
        """
        Extract unique dates from cleaned taxi data and fetch weather.
        
        Args:
            cleaned_data_dir: Directory containing cleaned monthly parquet files
            output_file: Path to save weather data
        """
        print("="*80)
        print("🌤️ NYC TAXI WEATHER DATA FETCHER (Open-Meteo - FREE!)")
        print("="*80)
        print()
        
        # Step 1: Extract unique dates
        print("📅 Step 1: Extracting unique dates from taxi data...")
        unique_dates = self._extract_unique_dates(cleaned_data_dir)
        print(f"✅ Found {len(unique_dates):,} unique dates to fetch weather for")
        print()
        
        # Step 2: Estimate API calls needed
        cached_count = sum(1 for date in unique_dates if date.strftime('%Y-%m-%d') in self.cache)
        needed_count = len(unique_dates) - cached_count
        
        print(f"📊 API Call Estimate:")
        print(f"   Total dates: {len(unique_dates):,}")
        print(f"   Cached: {cached_count:,}")
        print(f"   Need to fetch: {needed_count:,}")
        
        if needed_count > 0:
            time_estimate = needed_count / self.calls_per_second  # seconds
            print(f"   Estimated time: ~{time_estimate/60:.1f} minutes")
            print(f"   API: Open-Meteo (100% FREE, no API key needed!)")
        print()
        
        # Step 3: Fetch weather data
        print("🌐 Step 2: Fetching weather data from Open-Meteo API...")
        all_weather_data = {}
        
        for date in tqdm(unique_dates, desc="Fetching weather"):
            daily_weather = self.fetch_weather_for_date(date)
            
            if daily_weather:
                all_weather_data.update(daily_weather)
        
        # Save final cache
        self._save_cache()
        print(f"✅ Fetched weather for {len(all_weather_data):,} hours")
        print()
        
        # Step 4: Create DataFrame and save
        print("💾 Step 3: Creating weather dataset...")
        
        # Convert dict to list of records
        weather_records = []
        for dt_str, weather_data in all_weather_data.items():
            weather_record = {
                'datetime': pd.to_datetime(dt_str),
                **weather_data
            }
            weather_records.append(weather_record)
        
        weather_df = pd.DataFrame(weather_records)
        
        # Sort by datetime
        weather_df = weather_df.sort_values('datetime').reset_index(drop=True)
        
        # Add derived features
        weather_df = self._add_derived_features(weather_df)
        
        # Save to parquet
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weather_df.to_parquet(output_path, index=False)
        
        print(f"✅ Saved weather data to: {output_file}")
        print()
        
        # Print summary statistics
        self._print_weather_summary(weather_df)
        
        print("="*80)
        print("✅ WEATHER DATA FETCH COMPLETE!")
        print("="*80)
    
    def _extract_unique_dates(self, cleaned_data_dir: str) -> List[datetime.date]:
        """
        Extract unique dates from all cleaned taxi files.
        
        Args:
            cleaned_data_dir: Directory with cleaned parquet files
            
        Returns:
            Sorted list of unique dates
        """
        data_dir = Path(cleaned_data_dir)
        cleaned_files = sorted(data_dir.glob('cleaned_2022-*.parquet'))
        
        unique_dates_set = set()
        
        for file in cleaned_files:
            df = pd.read_parquet(file, columns=['tpep_pickup_datetime'])
            
            # Extract date only
            df['date'] = df['tpep_pickup_datetime'].dt.date
            
            # Add to set
            unique_dates_set.update(df['date'].unique())
        
        # Convert to sorted list
        unique_dates = sorted(list(unique_dates_set))
        
        return unique_dates
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived weather features for ML model.
        
        Args:
            df: Weather DataFrame
            
        Returns:
            DataFrame with additional features
        """
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
    
    def _print_weather_summary(self, df: pd.DataFrame):
        """Print summary statistics of weather data."""
        print("📊 WEATHER DATA SUMMARY")
        print("="*80)
        print()
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print()
        
        print("🌡️ Temperature:")
        print(f"   Mean: {df['temperature'].mean():.1f}°F")
        print(f"   Min: {df['temperature'].min():.1f}°F")
        print(f"   Max: {df['temperature'].max():.1f}°F")
        print()
        
        print("🌧️ Precipitation:")
        print(f"   Total rainy hours: {df['is_raining'].sum():,} ({df['is_raining'].mean()*100:.1f}%)")
        print(f"   Total snowy hours: {df['is_snowing'].sum():,} ({df['is_snowing'].mean()*100:.1f}%)")
        print(f"   Mean precipitation: {df['precipitation'].mean():.3f} inches/hour")
        print()
        
        print("💨 Wind:")
        print(f"   Mean speed: {df['wind_speed'].mean():.1f} mph")
        print(f"   Max speed: {df['wind_speed'].max():.1f} mph")
        print()
        
        print("👁️ Visibility:")
        print(f"   Mean: {df['visibility'].mean():.1f} miles")
        print(f"   Poor visibility hours: {df['is_poor_visibility'].sum():,} ({df['is_poor_visibility'].mean()*100:.1f}%)")
        print()
        
        print("⚠️ Extreme Weather:")
        print(f"   Extreme weather hours: {df['is_extreme_weather'].sum():,} ({df['is_extreme_weather'].mean()*100:.1f}%)")
        print()


def main():
    """Main execution function."""
    
    print("="*80)
    print("🌟 Using Open-Meteo API (100% FREE - No API Key Needed!)")
    print("="*80)
    print()
    
    # Initialize fetcher (no API key needed for Open-Meteo!)
    fetcher = WeatherDataFetcher(
        cache_file='data/external/weather_cache.json'
    )
    
    # Fetch weather for all cleaned taxi data
    fetcher.fetch_weather_for_dataset(
        cleaned_data_dir='data/processed',
        output_file='data/external/weather_2022.parquet'
    )


if __name__ == '__main__':
    main()
