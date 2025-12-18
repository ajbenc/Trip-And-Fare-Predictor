"""
Comprehensive preprocessing pipeline for full year NYC Taxi data (2022)

This script processes all raw monthly parquet files:
1. Cleans and filters outliers
2. Creates basic features
3. Handles missing values
4. Saves cleaned monthly files
5. Generates data quality report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TaxiDataPreprocessor:
    """Preprocessor for NYC TLC Yellow Taxi data"""
    
    def __init__(self, raw_dir='data/raw', output_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds (same as original preprocessing)
        self.thresholds = {
            'fare_min': 2.5,
            'fare_max': 250,
            'distance_min': 0.1,
            'distance_max': 50,
            'duration_min': 1,
            'duration_max': 180,
            'passenger_min': 1,
            'passenger_max': 6,
            'speed_max': 100  # mph
        }
        
        # Statistics tracking
        self.stats = {
            'month': [],
            'raw_count': [],
            'after_cleaning': [],
            'removed_count': [],
            'removed_pct': [],
            'avg_fare': [],
            'avg_duration': [],
            'avg_distance': []
        }
    
    def load_raw_month(self, filename):
        """Load raw monthly data"""
        filepath = self.raw_dir / filename
        print(f"\n📂 Loading: {filename}")
        df = pd.read_parquet(filepath, engine='pyarrow')
        print(f"   Raw records: {len(df):,}")
        return df
    
    def calculate_trip_duration(self, df):
        """Calculate trip duration in minutes"""
        df['trip_duration'] = (
            df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
        ).dt.total_seconds() / 60
        return df
    
    def remove_outliers(self, df):
        """Remove outliers based on business rules"""
        initial_count = len(df)
        
        # 1. Remove missing values and zero/null passengers (IMPORTANT!)
        df = df.dropna(subset=['fare_amount', 'trip_distance', 'trip_duration', 'passenger_count'])
        df = df[df['passenger_count'] > 0]  # Remove zero passenger trips
        
        # 2. Fare amount filters
        df = df[
            (df['fare_amount'] >= self.thresholds['fare_min']) &
            (df['fare_amount'] <= self.thresholds['fare_max'])
        ]
        
        # 3. Distance filters
        df = df[
            (df['trip_distance'] >= self.thresholds['distance_min']) &
            (df['trip_distance'] <= self.thresholds['distance_max'])
        ]
        
        # 4. Duration filters
        df = df[
            (df['trip_duration'] >= self.thresholds['duration_min']) &
            (df['trip_duration'] <= self.thresholds['duration_max'])
        ]
        
        # 5. Passenger count filters (valid range: 1-6)
        df = df[
            (df['passenger_count'] >= self.thresholds['passenger_min']) &
            (df['passenger_count'] <= self.thresholds['passenger_max'])
        ]
        
        # 6. Remove zero distance with non-zero duration
        df = df[~((df['trip_distance'] < 0.1) & (df['trip_duration'] > 5))]
        
        # 7. Remove unrealistic speeds (>100 mph)
        df['speed'] = (df['trip_distance'] / (df['trip_duration'] / 60))
        df = df[df['speed'] <= self.thresholds['speed_max']]
        df = df.drop('speed', axis=1)
        
        # 8. Remove same pickup/dropoff with significant fare
        df = df[~((df['PULocationID'] == df['DOLocationID']) & (df['trip_distance'] < 0.1))]
        
        removed_count = initial_count - len(df)
        removed_pct = (removed_count / initial_count) * 100
        
        print(f"   ❌ Removed {removed_count:,} outliers ({removed_pct:.1f}%)")
        print(f"   ✅ Kept {len(df):,} clean records ({100-removed_pct:.1f}%)")
        
        return df, removed_count, removed_pct
    
    def create_basic_features(self, df):
        """Create basic temporal features"""
        # Temporal features
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
        
        # Day of year and week of year (useful for seasonality)
        df['pickup_dayofyear'] = df['tpep_pickup_datetime'].dt.dayofyear
        df['pickup_weekofyear'] = df['tpep_pickup_datetime'].dt.isocalendar().week
        
        # Weekend flag
        df['is_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
        
        return df
    
    def select_relevant_columns(self, df):
        """Keep only relevant columns for modeling"""
        columns_to_keep = [
            # Datetime
            'tpep_pickup_datetime',
            'tpep_dropoff_datetime',
            
            # Location
            'PULocationID',
            'DOLocationID',
            
            # Trip characteristics
            'trip_distance',
            'passenger_count',
            
            # Targets
            'fare_amount',
            'trip_duration',
            
            # Basic features
            'pickup_hour',
            'pickup_day',
            'pickup_month',
            'pickup_weekday',
            'pickup_date',
            'pickup_dayofyear',
            'pickup_weekofyear',
            'is_weekend'
        ]
        
        return df[columns_to_keep]
    
    def process_month(self, filename):
        """Process a single month"""
        # Load
        df = self.load_raw_month(filename)
        raw_count = len(df)
        
        # Calculate duration
        df = self.calculate_trip_duration(df)
        
        # Remove outliers
        df, removed_count, removed_pct = self.remove_outliers(df)
        
        # Create basic features
        df = self.create_basic_features(df)
        
        # Select columns
        df = self.select_relevant_columns(df)
        
        # Calculate statistics
        avg_fare = df['fare_amount'].mean()
        avg_duration = df['trip_duration'].mean()
        avg_distance = df['trip_distance'].mean()
        
        # Store stats
        month_name = filename.replace('yellow_tripdata_', '').replace('.parquet', '')
        self.stats['month'].append(month_name)
        self.stats['raw_count'].append(raw_count)
        self.stats['after_cleaning'].append(len(df))
        self.stats['removed_count'].append(removed_count)
        self.stats['removed_pct'].append(removed_pct)
        self.stats['avg_fare'].append(avg_fare)
        self.stats['avg_duration'].append(avg_duration)
        self.stats['avg_distance'].append(avg_distance)
        
        print(f"   📊 Avg Fare: ${avg_fare:.2f} | Avg Duration: {avg_duration:.1f} min | Avg Distance: {avg_distance:.2f} mi")
        
        return df, month_name
    
    def save_processed_month(self, df, month_name):
        """Save processed monthly data"""
        output_file = self.output_dir / f'cleaned_{month_name}.parquet'
        df.to_parquet(output_file, engine='pyarrow', index=False)
        print(f"   💾 Saved to: {output_file}")
        return output_file
    
    def generate_quality_report(self):
        """Generate data quality report"""
        stats_df = pd.DataFrame(self.stats)
        
        print("\n" + "="*80)
        print("📊 DATA QUALITY REPORT - FULL YEAR 2022")
        print("="*80)
        
        print(f"\n{'Month':<12} {'Raw Count':>12} {'Cleaned':>12} {'Removed':>12} {'Removed %':>10} {'Avg Fare':>10} {'Avg Dur':>10} {'Avg Dist':>10}")
        print("-"*106)
        
        for _, row in stats_df.iterrows():
            print(f"{row['month']:<12} {row['raw_count']:>12,} {row['after_cleaning']:>12,} "
                  f"{row['removed_count']:>12,} {row['removed_pct']:>9.1f}% "
                  f"${row['avg_fare']:>9.2f} {row['avg_duration']:>9.1f}m {row['avg_distance']:>9.2f}mi")
        
        # Summary
        print("-"*106)
        total_raw = stats_df['raw_count'].sum()
        total_cleaned = stats_df['after_cleaning'].sum()
        total_removed = stats_df['removed_count'].sum()
        avg_removal_pct = stats_df['removed_pct'].mean()
        
        print(f"{'TOTAL':<12} {total_raw:>12,} {total_cleaned:>12,} {total_removed:>12,} {avg_removal_pct:>9.1f}%")
        
        print(f"\n✅ Successfully processed {len(stats_df)} months")
        print(f"✅ Total clean records: {total_cleaned:,}")
        print(f"✅ Average retention rate: {100-avg_removal_pct:.1f}%")
        
        # Seasonal insights
        print(f"\n🌡️ SEASONAL PATTERNS:")
        stats_df['season'] = stats_df['month'].apply(self._get_season)
        seasonal = stats_df.groupby('season').agg({
            'after_cleaning': 'sum',
            'avg_fare': 'mean',
            'avg_duration': 'mean',
            'avg_distance': 'mean'
        })
        
        for season, row in seasonal.iterrows():
            print(f"   {season:<10} Trips: {row['after_cleaning']:>10,} | "
                  f"Fare: ${row['avg_fare']:>6.2f} | "
                  f"Duration: {row['avg_duration']:>5.1f}m | "
                  f"Distance: {row['avg_distance']:>5.2f}mi")
        
        # Save report
        report_file = self.output_dir / 'data_quality_report.csv'
        stats_df.to_csv(report_file, index=False)
        print(f"\n💾 Report saved to: {report_file}")
        
        return stats_df
    
    def _get_season(self, month_str):
        """Get season from month string"""
        month_num = int(month_str.split('-')[1])
        if month_num in [12, 1, 2]:
            return 'Winter'
        elif month_num in [3, 4, 5]:
            return 'Spring'
        elif month_num in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def process_all_months(self):
        """Process all raw monthly files"""
        print("\n" + "="*80)
        print("🚀 STARTING FULL YEAR PREPROCESSING - 2022")
        print("="*80)
        print(f"📁 Raw data directory: {self.raw_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Find all raw files
        raw_files = sorted([f.name for f in self.raw_dir.glob('yellow_tripdata_2022-*.parquet')])
        
        if not raw_files:
            print("❌ No raw files found!")
            return None
        
        print(f"\n✅ Found {len(raw_files)} monthly files to process")
        
        # Process each month
        processed_files = []
        start_time = datetime.now()
        
        for i, filename in enumerate(raw_files, 1):
            print(f"\n{'='*80}")
            print(f"Processing {i}/{len(raw_files)}: {filename}")
            print(f"{'='*80}")
            
            try:
                df, month_name = self.process_month(filename)
                output_file = self.save_processed_month(df, month_name)
                processed_files.append(output_file)
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
        
        # Generate report
        stats_df = self.generate_quality_report()
        
        # Time elapsed
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n⏱️ Total processing time: {elapsed/60:.1f} minutes")
        print(f"⚡ Average time per month: {elapsed/len(raw_files):.1f} seconds")
        
        return processed_files, stats_df


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("NYC TAXI DATA PREPROCESSING PIPELINE")
    print("="*80)
    print("This script will:")
    print("  1. Load all raw monthly parquet files")
    print("  2. Clean and filter outliers")
    print("  3. Create basic temporal features")
    print("  4. Save cleaned monthly files")
    print("  5. Generate data quality report")
    print("="*80)
    
    input("\nPress Enter to start preprocessing...")
    
    # Initialize preprocessor
    preprocessor = TaxiDataPreprocessor(
        raw_dir='data/raw',
        output_dir='data/processed'
    )
    
    # Process all months
    processed_files, stats_df = preprocessor.process_all_months()
    
    if processed_files:
        print("\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\n📂 Processed files saved to: data/processed/")
        print(f"📊 Quality report saved to: data/processed/data_quality_report.csv")
        print(f"\n🎯 Next steps:")
        print(f"   1. Run feature engineering: python src/feature_engineering/engineer_full_year_features.py")
        print(f"   2. Split data: python src/data/split_temporal.py")
        print(f"   3. Train models: python src/training/train_full_year_models.py")
    else:
        print("\n❌ No files were processed. Please check the raw data directory.")


if __name__ == "__main__":
    main()
