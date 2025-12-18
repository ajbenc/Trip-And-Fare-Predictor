"""
Enhanced Feature Engineering for NYC Taxi Trips
================================================
Merges taxi trip data with weather and holiday information to create
a comprehensive feature set for ML training.

Features Created:
- Original taxi features (47)
- Weather features (10-12)
- Holiday features (6-8)
- Interaction features (8-10)
- Total: ~65-75 features

Author: NYC Taxi ML Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering with weather and holiday data.
    """
    
    def __init__(self, 
                 processed_dir: str = 'data/processed',
                 weather_file: str = 'data/external/weather_2022.parquet',
                 holiday_file: str = 'data/external/holidays_2022.csv',
                 output_dir: str = 'data/processed/enhanced'):
        """
        Initialize enhanced feature engineer.
        
        Args:
            processed_dir: Directory with cleaned taxi data
            weather_file: Path to weather data
            holiday_file: Path to holiday calendar
            output_dir: Directory to save enhanced features
        """
        self.processed_dir = Path(processed_dir)
        self.weather_file = Path(weather_file)
        self.holiday_file = Path(holiday_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load external data
        self.weather_df = None
        self.holiday_df = None
        
    def load_external_data(self):
        """Load weather and holiday data."""
        print("📦 Loading external data...")
        
        # Load weather data
        print(f"   Loading weather data from {self.weather_file}...")
        self.weather_df = pd.read_parquet(self.weather_file)
        self.weather_df['datetime'] = pd.to_datetime(self.weather_df['datetime'])
        print(f"   ✅ Loaded {len(self.weather_df):,} weather records")
        
        # Load holiday data
        print(f"   Loading holiday calendar from {self.holiday_file}...")
        self.holiday_df = pd.read_csv(self.holiday_file)
        self.holiday_df['date'] = pd.to_datetime(self.holiday_df['date'])
        print(f"   ✅ Loaded {len(self.holiday_df):,} holiday dates")
        print()
        
    def process_all_months(self):
        """Process all monthly files and create enhanced features."""
        print("="*80)
        print("🔧 ENHANCED FEATURE ENGINEERING")
        print("="*80)
        print()
        
        # Load external data first
        self.load_external_data()
        
        # Get all cleaned monthly files
        cleaned_files = sorted(self.processed_dir.glob('cleaned_2022-*.parquet'))
        
        if not cleaned_files:
            print("❌ No cleaned files found!")
            return
        
        print(f"📁 Found {len(cleaned_files)} monthly files to process")
        print()
        
        # Process each month
        total_records = 0
        feature_counts = {}
        
        for i, file in enumerate(cleaned_files, 1):
            month_name = file.stem.replace('cleaned_', '')
            print(f"Processing {i}/{len(cleaned_files)}: {month_name}")
            print("-" * 80)
            
            # Load taxi data
            df = pd.read_parquet(file)
            original_count = len(df)
            print(f"   📊 Loaded {original_count:,} taxi trips")
            
            # Create enhanced features
            df = self.create_enhanced_features(df)
            
            # Save enhanced file
            output_file = self.output_dir / f'enhanced_{month_name}.parquet'
            df.to_parquet(output_file, index=False)
            
            total_records += len(df)
            feature_counts[month_name] = len(df.columns)
            
            print(f"   ✅ Created {len(df.columns)} features")
            print(f"   💾 Saved to: {output_file}")
            print()
        
        # Print summary
        self._print_summary(total_records, feature_counts)
        
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all enhanced features for taxi trips.
        
        Args:
            df: Taxi trip DataFrame
            
        Returns:
            DataFrame with enhanced features
        """
        # Ensure datetime columns
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
        # 1. Merge with weather data
        print("   🌤️ Merging weather data...")
        df = self._merge_weather(df)
        
        # 2. Merge with holiday data
        print("   🎉 Merging holiday data...")
        df = self._merge_holidays(df)
        
        # 3. Create interaction features
        print("   🔗 Creating interaction features...")
        df = self._create_interaction_features(df)
        
        # 4. Create cyclical features
        print("   🔄 Creating cyclical features...")
        df = self._create_cyclical_features(df)
        
        return df
    
    def _merge_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data with taxi trips."""
        # Round pickup datetime to nearest hour
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')
        
        # Merge with weather data
        df = df.merge(
            self.weather_df,
            left_on='pickup_hour',
            right_on='datetime',
            how='left',
            suffixes=('', '_weather')
        )
        
        # Drop redundant columns
        df = df.drop(columns=['datetime', 'pickup_hour'], errors='ignore')
        
        # Fill missing weather values with reasonable defaults
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation', 
                       'snow', 'weather_severity', 'is_raining', 'is_snowing',
                       'is_extreme_weather']
        
        for col in weather_cols:
            if col in df.columns:
                if col.startswith('is_'):
                    df[col] = df[col].fillna(0).astype(int)
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _merge_holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge holiday data with taxi trips."""
        # Extract date from pickup datetime
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
        
        # Create holiday lookup
        holiday_dates = set(self.holiday_df['date'].dt.date)
        major_holiday_dates = set(self.holiday_df[self.holiday_df['major'] == True]['date'].dt.date)
        
        # Create holiday features
        df['is_holiday'] = df['pickup_date'].isin(holiday_dates).astype(int)
        df['is_major_holiday'] = df['pickup_date'].isin(major_holiday_dates).astype(int)
        
        # Days before/after holiday (within 3 days)
        df['days_to_holiday'] = 0
        df['days_from_holiday'] = 0
        
        # For each trip, find closest holiday
        for idx, row in df.sample(min(1000, len(df))).iterrows():  # Sample for speed
            pickup_date = row['pickup_date']
            
            # Find days to next holiday
            future_holidays = [h for h in holiday_dates if h > pickup_date]
            if future_holidays:
                next_holiday = min(future_holidays)
                days_to = (next_holiday - pickup_date).days
                if days_to <= 3:
                    df.at[idx, 'days_to_holiday'] = days_to
            
            # Find days from last holiday
            past_holidays = [h for h in holiday_dates if h < pickup_date]
            if past_holidays:
                last_holiday = max(past_holidays)
                days_from = (pickup_date - last_holiday).days
                if days_from <= 3:
                    df.at[idx, 'days_from_holiday'] = days_from
        
        # Holiday week (week containing a major holiday)
        df['is_holiday_week'] = 0
        for holiday_date in major_holiday_dates:
            # Mark entire week as holiday week
            week_start = holiday_date - pd.Timedelta(days=3)
            week_end = holiday_date + pd.Timedelta(days=3)
            mask = (df['pickup_date'] >= week_start) & (df['pickup_date'] <= week_end)
            df.loc[mask, 'is_holiday_week'] = 1
        
        # Drop temporary column
        df = df.drop(columns=['pickup_date'], errors='ignore')
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between weather, holidays, and trip characteristics."""
        
        # Weather × Distance
        if 'weather_severity' in df.columns and 'trip_distance' in df.columns:
            df['weather_distance_interaction'] = df['weather_severity'] * df['trip_distance']
        
        # Weather × Hour (rush hour in bad weather)
        if 'weather_severity' in df.columns and 'pickup_hour' in df.columns:
            df['weather_rushhour_interaction'] = df['weather_severity'] * df['is_rush_hour']
        
        # Rain × Airport (airport trips in rain)
        if 'is_raining' in df.columns and 'is_airport_trip' in df.columns:
            df['rain_airport_interaction'] = df['is_raining'] * df['is_airport_trip']
        
        # Snow × Duration (snow slows down trips)
        if 'is_snowing' in df.columns and 'trip_duration' in df.columns:
            df['snow_duration_interaction'] = df['is_snowing'] * df['trip_duration']
        
        # Temperature × Demand (extreme temps increase taxi demand)
        if 'temperature' in df.columns:
            df['temp_demand_score'] = 0
            df.loc[df['temperature'] < 32, 'temp_demand_score'] = 1  # Freezing
            df.loc[df['temperature'] > 85, 'temp_demand_score'] = 1  # Hot
        
        # Holiday × Hour (holiday rush hours different)
        if 'is_holiday' in df.columns and 'pickup_hour' in df.columns:
            df['holiday_hour_interaction'] = df['is_holiday'] * df['pickup_hour']
        
        # Holiday × Distance (longer trips on holidays)
        if 'is_holiday' in df.columns and 'trip_distance' in df.columns:
            df['holiday_distance_interaction'] = df['is_holiday'] * df['trip_distance']
        
        # Extreme Weather × Distance
        if 'is_extreme_weather' in df.columns and 'trip_distance' in df.columns:
            df['extreme_weather_distance'] = df['is_extreme_weather'] * df['trip_distance']
        
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encodings for time features."""
        
        # Hour of day (0-23) → cyclical
        if 'pickup_hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
        
        # Day of week (0-6) → cyclical
        if 'pickup_weekday' in df.columns:
            df['weekday_sin'] = np.sin(2 * np.pi * df['pickup_weekday'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['pickup_weekday'] / 7)
        
        # Month (1-12) → cyclical
        if 'pickup_month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['pickup_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['pickup_month'] / 12)
        
        # Day of year (1-365) → cyclical (captures seasonal patterns)
        if 'pickup_dayofyear' in df.columns:
            df['dayofyear_sin'] = np.sin(2 * np.pi * df['pickup_dayofyear'] / 365)
            df['dayofyear_cos'] = np.cos(2 * np.pi * df['pickup_dayofyear'] / 365)
        
        return df
    
    def _print_summary(self, total_records: int, feature_counts: dict):
        """Print processing summary."""
        print("="*80)
        print("✅ ENHANCED FEATURE ENGINEERING COMPLETE!")
        print("="*80)
        print()
        
        print(f"📊 Summary:")
        print(f"   Total records processed: {total_records:,}")
        print(f"   Average features per month: {np.mean(list(feature_counts.values())):.0f}")
        print(f"   Files saved to: {self.output_dir}")
        print()
        
        print(f"🔢 Feature Breakdown:")
        sample_count = list(feature_counts.values())[0]
        print(f"   Total features: ~{sample_count}")
        print(f"   - Original taxi features: ~47")
        print(f"   - Weather features: ~10-12")
        print(f"   - Holiday features: ~6-8")
        print(f"   - Interaction features: ~8-10")
        print(f"   - Cyclical features: ~8")
        print()
        
        print("🎯 Next steps:")
        print("   1. Create temporal split (train/val/test)")
        print("   2. Train models with enhanced features")
        print("   3. Evaluate performance improvements")
        print()


def main():
    """Main execution function."""
    engineer = EnhancedFeatureEngineer()
    engineer.process_all_months()


if __name__ == '__main__':
    main()
