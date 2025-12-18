"""
Clean outliers from trip duration data by capping extreme long trips
Strategy: Cap trips at 95th percentile and average very long distances
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_and_clean_outliers():
    """Analyze outliers and create cleaned datasets."""
    
    data_dir = Path("data/splits")
    output_dir = Path("data/splits_cleaned")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧹 OUTLIER CLEANING: Trip Duration & Distance")
    print("="*80)
    
    # ===== PHASE 1: ANALYZE CURRENT OUTLIERS =====
    print("\n📊 PHASE 1: Analyzing current outliers across all splits...")
    
    all_durations = []
    all_distances = []
    all_fares = []
    
    for split in ['train', 'val', 'test']:
        print(f"\n   Loading {split} data...")
        
        # Load features to get distance
        X_files = sorted((data_dir / split).glob('features_*_X.parquet'))
        y_duration_files = sorted((data_dir / split).glob('features_*_y_duration.parquet'))
        y_fare_files = sorted((data_dir / split).glob('features_*_y_fare.parquet'))
        
        for X_file, y_dur_file, y_fare_file in zip(X_files, y_duration_files, y_fare_files):
            X = pd.read_parquet(X_file)
            y_duration = pd.read_parquet(y_dur_file).values.ravel()
            y_fare = pd.read_parquet(y_fare_file).values.ravel()
            
            all_distances.extend(X['estimated_distance'].values)
            all_durations.extend(y_duration)
            all_fares.extend(y_fare)
            
            print(f"      {X_file.name}: {len(X):,} trips")
    
    all_distances = np.array(all_distances)
    all_durations = np.array(all_durations)
    all_fares = np.array(all_fares)
    
    print(f"\n   Total trips analyzed: {len(all_durations):,}")
    
    # Calculate percentiles
    print(f"\n   TRIP DURATION STATISTICS:")
    print(f"   {'Metric':<20} {'Value':<15}")
    print("   " + "-"*35)
    print(f"   {'Mean':<20} {all_durations.mean():<15.2f} min")
    print(f"   {'Median':<20} {np.median(all_durations):<15.2f} min")
    print(f"   {'Std Dev':<20} {all_durations.std():<15.2f} min")
    print(f"   {'Min':<20} {all_durations.min():<15.2f} min")
    print(f"   {'Max':<20} {all_durations.max():<15.2f} min")
    
    print(f"\n   TRIP DURATION PERCENTILES:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(all_durations, p)
        count = (all_durations > val).sum()
        pct = count / len(all_durations) * 100
        print(f"   {p:>5.1f}th: {val:>6.2f} min  ({count:>8,} trips > this, {pct:>5.2f}%)")
    
    print(f"\n   DISTANCE STATISTICS:")
    print(f"   {'Metric':<20} {'Value':<15}")
    print("   " + "-"*35)
    print(f"   {'Mean':<20} {all_distances.mean():<15.2f} mi")
    print(f"   {'Median':<20} {np.median(all_distances):<15.2f} mi")
    print(f"   {'Max':<20} {all_distances.max():<15.2f} mi")
    
    print(f"\n   DISTANCE PERCENTILES:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(all_distances, p)
        count = (all_distances > val).sum()
        pct = count / len(all_distances) * 100
        print(f"   {p:>5.1f}th: {val:>6.2f} mi  ({count:>8,} trips > this, {pct:>5.2f}%)")
    
    # ===== PHASE 2: DEFINE CLEANING THRESHOLDS =====
    print("\n" + "="*80)
    print("📏 PHASE 2: Defining cleaning thresholds...")
    print("="*80)
    
    # Use 95th percentile as cap for duration (removes top 5% outliers)
    duration_cap = np.percentile(all_durations, 95)
    
    # For very long distances (>20 mi), we'll cap them at 99th percentile
    distance_cap = np.percentile(all_distances, 99)
    
    print(f"\n   CLEANING STRATEGY:")
    print(f"   ✂️  Duration cap: {duration_cap:.2f} minutes (95th percentile)")
    print(f"   ✂️  Distance cap: {distance_cap:.2f} miles (99th percentile)")
    
    # Calculate impact
    duration_outliers = (all_durations > duration_cap).sum()
    distance_outliers = (all_distances > distance_cap).sum()
    
    print(f"\n   IMPACT:")
    print(f"   Duration outliers to cap: {duration_outliers:,} ({duration_outliers/len(all_durations)*100:.2f}%)")
    print(f"   Distance outliers to cap: {distance_outliers:,} ({distance_outliers/len(all_distances)*100:.2f}%)")
    
    # ===== PHASE 3: CLEAN AND SAVE DATA =====
    print("\n" + "="*80)
    print("🔧 PHASE 3: Cleaning and saving data...")
    print("="*80)
    
    total_trips_processed = 0
    total_duration_capped = 0
    total_distance_capped = 0
    
    for split in ['train', 'val', 'test']:
        print(f"\n   Processing {split} split...")
        
        split_dir = data_dir / split
        output_split_dir = output_dir / split
        output_split_dir.mkdir(exist_ok=True)
        
        X_files = sorted(split_dir.glob('features_*_X.parquet'))
        y_duration_files = sorted(split_dir.glob('features_*_y_duration.parquet'))
        y_fare_files = sorted(split_dir.glob('features_*_y_fare.parquet'))
        
        for X_file, y_dur_file, y_fare_file in zip(X_files, y_duration_files, y_fare_files):
            # Load data
            X = pd.read_parquet(X_file)
            y_duration = pd.read_parquet(y_dur_file).values.ravel()
            y_fare = pd.read_parquet(y_fare_file).values.ravel()
            
            original_count = len(X)
            
            # Track capping
            duration_capped_mask = y_duration > duration_cap
            distance_capped_mask = X['estimated_distance'] > distance_cap
            
            duration_capped_count = duration_capped_mask.sum()
            distance_capped_count = distance_capped_mask.sum()
            
            # Apply capping
            y_duration_cleaned = np.clip(y_duration, None, duration_cap)
            X_cleaned = X.copy()
            X_cleaned.loc[distance_capped_mask, 'estimated_distance'] = distance_cap
            
            # Recalculate distance-based interaction features with capped distance
            if 'distance_hour_interaction' in X_cleaned.columns:
                X_cleaned['distance_hour_interaction'] = X_cleaned['estimated_distance'] * X_cleaned['pickup_hour']
            if 'distance_rushhour_interaction' in X_cleaned.columns:
                X_cleaned['distance_rushhour_interaction'] = X_cleaned['estimated_distance'] * X_cleaned['is_rush_hour']
            if 'snow_distance_interaction' in X_cleaned.columns:
                if 'is_snowing' in X_cleaned.columns:
                    X_cleaned['snow_distance_interaction'] = X_cleaned['is_snowing'] * X_cleaned['estimated_distance']
            if 'rain_distance_interaction' in X_cleaned.columns:
                if 'is_raining' in X_cleaned.columns:
                    X_cleaned['rain_distance_interaction'] = X_cleaned['is_raining'] * X_cleaned['estimated_distance']
            
            # Save cleaned data
            X_cleaned.to_parquet(output_split_dir / X_file.name)
            pd.DataFrame(y_duration_cleaned, columns=['trip_duration']).to_parquet(
                output_split_dir / y_dur_file.name
            )
            # Fare doesn't need cleaning - copy as is
            pd.DataFrame(y_fare, columns=['fare_amount']).to_parquet(
                output_split_dir / y_fare_file.name
            )
            
            total_trips_processed += original_count
            total_duration_capped += duration_capped_count
            total_distance_capped += distance_capped_count
            
            print(f"      {X_file.name}:")
            print(f"         Trips: {original_count:,}")
            print(f"         Duration capped: {duration_capped_count:,} ({duration_capped_count/original_count*100:.2f}%)")
            print(f"         Distance capped: {distance_capped_count:,} ({distance_capped_count/original_count*100:.2f}%)")
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\n   Total trips processed: {total_trips_processed:,}")
    print(f"\n   Duration outliers capped:")
    print(f"      Count: {total_duration_capped:,}")
    print(f"      Percentage: {total_duration_capped/total_trips_processed*100:.2f}%")
    print(f"      Cap value: {duration_cap:.2f} minutes")
    
    print(f"\n   Distance outliers capped:")
    print(f"      Count: {total_distance_capped:,}")
    print(f"      Percentage: {total_distance_capped/total_trips_processed*100:.2f}%")
    print(f"      Cap value: {distance_cap:.2f} miles")
    
    print(f"\n   Weather & Holiday features: KEPT ✓")
    print(f"   Cleaned data saved to: {output_dir}/")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEP: Train LightGBM models on cleaned data")
    print("="*80)
    
    return {
        'duration_cap': float(duration_cap),
        'distance_cap': float(distance_cap),
        'total_trips': int(total_trips_processed),
        'duration_capped_count': int(total_duration_capped),
        'distance_capped_count': int(total_distance_capped)
    }

if __name__ == "__main__":
    results = analyze_and_clean_outliers()
    
    # Save cleaning parameters
    import json
    with open('data/splits_cleaned/cleaning_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Cleaning parameters saved to: data/splits_cleaned/cleaning_params.json")
