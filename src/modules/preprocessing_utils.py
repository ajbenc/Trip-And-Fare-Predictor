# -*- coding: utf-8 -*-
"""
Preprocessing Utilities and Helper Functions
============================================

Helper functions for data preprocessing tasks.
"""

import sys
import io
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path


def load_processed_data(processed_dir='../data/processed', dataset='train'):
    """
    Load preprocessed train or test data.
    
    Args:
        processed_dir (str): Directory containing processed data
        dataset (str): 'train', 'test', or 'full'
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    file_map = {
        'train': 'train.parquet',
        'test': 'test.parquet',
        'full': 'processed_full.parquet'
    }
    
    file_path = Path(processed_dir) / file_map.get(dataset, 'train.parquet')
    
    # If relative path doesn't exist, try from script location
    if not file_path.exists():
        # Go up from src/modules to project root, then to data/processed
        script_dir = Path(__file__).parent.parent.parent
        file_path = script_dir / 'data' / 'processed' / file_map.get(dataset, 'train.parquet')
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    print(f"Loaded {dataset} dataset: {len(df):,} rows × {df.shape[1]} columns")
    
    return df


def prepare_features_and_target(df, target='fare_amount', feature_list=None):
    """
    Prepare feature matrix X and target vector y.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        target (str): Target column name ('fare_amount' or 'trip_duration')
        feature_list (list, optional): List of feature columns to use
        
    Returns:
        tuple: (X, y, feature_names)
    """
    if feature_list is None:
        # Auto-select all features except targets
        exclude_cols = ['fare_amount', 'trip_duration']
        feature_list = [col for col in df.columns if col not in exclude_cols]
    
    # Filter features that exist in dataframe
    available_features = [col for col in feature_list if col in df.columns]
    
    # Prepare X and y
    X = df[available_features].copy()
    y = df[target].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(X.median())
    
    print(f"✅ Prepared features and target:")
    print(f"   • Features: {len(available_features)} columns")
    print(f"   • Target: {target}")
    print(f"   • Samples: {len(X):,}")
    
    return X, y, available_features


def get_data_summary(df):
    """
    Print summary statistics for a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to summarize
    """
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'fare_amount' in df.columns:
        print(f"\nFare Amount:")
        print(f"  Mean: ${df['fare_amount'].mean():.2f}")
        print(f"  Median: ${df['fare_amount'].median():.2f}")
        print(f"  Std: ${df['fare_amount'].std():.2f}")
        print(f"  Range: ${df['fare_amount'].min():.2f} - ${df['fare_amount'].max():.2f}")
    
    if 'trip_duration' in df.columns:
        print(f"\nTrip Duration:")
        print(f"  Mean: {df['trip_duration'].mean()/60:.2f} min")
        print(f"  Median: {df['trip_duration'].median()/60:.2f} min")
        print(f"  Std: {df['trip_duration'].std()/60:.2f} min")
        print(f"  Range: {df['trip_duration'].min()/60:.2f} - {df['trip_duration'].max()/60:.2f} min")
    
    if 'trip_distance' in df.columns:
        print(f"\nTrip Distance:")
        print(f"  Mean: {df['trip_distance'].mean():.2f} miles")
        print(f"  Median: {df['trip_distance'].median():.2f} miles")
    
    print("="*70)


if __name__ == "__main__":
    # Example: Load and prepare data
    train_df = load_processed_data(dataset='train')
    get_data_summary(train_df)
    
    # Prepare for fare prediction
    X, y, features = prepare_features_and_target(train_df, target='fare_amount')
    print(f"\n Ready for modeling with {len(features)} features")
