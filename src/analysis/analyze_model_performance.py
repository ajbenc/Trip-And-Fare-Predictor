"""
Comprehensive model analysis:
1. Feature importance analysis
2. Error pattern examination
3. LightGBM vs XGBoost comparison
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import json

class ModelAnalyzer:
    def __init__(self):
        self.data_dir = Path("data/splits")
        self.output_dir = Path("analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, split='val'):
        """Load validation data for analysis."""
        print(f"\n📥 Loading {split} data...")
        
        # Load features
        X_files = sorted((self.data_dir / split).glob('features_*_X.parquet'))
        X = pd.concat([pd.read_parquet(f) for f in X_files], ignore_index=True)
        
        # Load targets
        y_fare_files = sorted((self.data_dir / split).glob('features_*_y_fare.parquet'))
        y_fare = pd.concat([pd.read_parquet(f) for f in y_fare_files], ignore_index=True).values.ravel()
        
        y_duration_files = sorted((self.data_dir / split).glob('features_*_y_duration.parquet'))
        y_duration = pd.concat([pd.read_parquet(f) for f in y_duration_files], ignore_index=True).values.ravel()
        
        print(f"   Loaded: {len(X):,} samples, {len(X.columns)} features")
        return X, y_fare, y_duration
    
    def analyze_feature_importance(self, model_path, X, model_type='xgboost', target='fare'):
        """Extract and analyze feature importance."""
        print(f"\n🔍 Analyzing feature importance for {target} ({model_type})...")
        
        # Load model
        if model_type == 'xgboost':
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            importance = model.feature_importances_
        else:  # lightgbm
            model = lgb.Booster(model_file=str(model_path))
            importance = model.feature_importance(importance_type='gain')
        
        # Create importance dataframe
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Normalize to percentages
        feature_importance['importance_pct'] = (
            feature_importance['importance'] / feature_importance['importance'].sum() * 100
        )
        
        # Print top 20
        print(f"\n   TOP 20 MOST IMPORTANT FEATURES:")
        print(f"   {'Rank':<5} {'Feature':<35} {'Importance %':<12}")
        print("   " + "="*55)
        for idx, row in feature_importance.head(20).iterrows():
            print(f"   {feature_importance.index.get_loc(idx)+1:<5} {row['feature']:<35} {row['importance_pct']:>10.2f}%")
        
        # Categorize features
        categories = {
            'Core': ['PULocationID', 'DOLocationID', 'passenger_count', 'estimated_distance'],
            'Temporal': ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'is_rush_hour', 
                        'is_late_night', 'is_business_hours'],
            'Cyclical': [col for col in X.columns if '_sin' in col or '_cos' in col],
            'Location': ['is_airport_pickup', 'is_airport_dropoff', 'is_manhattan_pickup', 'is_manhattan_dropoff',
                        'pickup_is_airport', 'dropoff_is_airport'],
            'Weather': [col for col in X.columns if 'temperature' in col or 'precipitation' in col or 
                       'snow' in col or 'wind' in col or 'weather' in col],
            'Holiday': [col for col in X.columns if 'holiday' in col],
            'Interaction': [col for col in X.columns if '_x_' in col]
        }
        
        category_importance = {}
        for cat, features in categories.items():
            cat_features = [f for f in features if f in feature_importance['feature'].values]
            cat_importance = feature_importance[feature_importance['feature'].isin(cat_features)]['importance_pct'].sum()
            category_importance[cat] = cat_importance
        
        print(f"\n   FEATURE IMPORTANCE BY CATEGORY:")
        print(f"   {'Category':<20} {'Total Importance %':<15}")
        print("   " + "="*40)
        for cat, imp in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat:<20} {imp:>13.2f}%")
        
        # Save detailed results
        feature_importance.to_csv(
            self.output_dir / f'{target}_{model_type}_feature_importance.csv', 
            index=False
        )
        
        return feature_importance, category_importance
    
    def analyze_error_patterns(self, model_path, X, y_true, model_type='xgboost', target='fare'):
        """Analyze where the model makes errors."""
        print(f"\n📊 Analyzing error patterns for {target} ({model_type})...")
        
        # Load model and predict
        if model_type == 'xgboost':
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
        else:
            model = lgb.Booster(model_file=str(model_path))
        
        y_pred = model.predict(X)
        
        # Calculate errors
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        pct_errors = (abs_errors / y_true) * 100
        
        # Overall statistics
        print(f"\n   OVERALL ERROR STATISTICS:")
        print(f"   R² Score:         {r2_score(y_true, y_pred):.4f} ({r2_score(y_true, y_pred)*100:.2f}%)")
        print(f"   MAE:              {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"   RMSE:             {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        print(f"   Mean Error:       {errors.mean():.2f} (bias)")
        print(f"   Median Abs Error: {np.median(abs_errors):.2f}")
        print(f"   Mean % Error:     {pct_errors.mean():.1f}%")
        
        # Error distribution
        print(f"\n   ERROR DISTRIBUTION:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(abs_errors, p)
            print(f"   {p}th percentile:  {val:.2f}")
        
        # Analyze errors by feature ranges
        print(f"\n   ERROR ANALYSIS BY KEY FEATURES:")
        
        # Distance-based errors
        if 'estimated_distance' in X.columns:
            distance_ranges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
            print(f"\n   By Distance:")
            for low, high in distance_ranges:
                mask = (X['estimated_distance'] >= low) & (X['estimated_distance'] < high)
                if mask.sum() > 0:
                    mae = abs_errors[mask].mean()
                    count = mask.sum()
                    print(f"     {low:>3.0f}-{high:<3.0f} mi: MAE={mae:>6.2f}, n={count:>8,}")
        
        # Time-based errors
        if 'hour' in X.columns:
            print(f"\n   By Hour of Day:")
            hour_errors = []
            for hour in range(24):
                mask = X['hour'] == hour
                if mask.sum() > 0:
                    mae = abs_errors[mask].mean()
                    count = mask.sum()
                    hour_errors.append((hour, mae, count))
            
            # Show peak error hours
            hour_errors.sort(key=lambda x: x[1], reverse=True)
            print(f"     Worst hours (highest MAE):")
            for hour, mae, count in hour_errors[:5]:
                print(f"       Hour {hour:>2d}:00 - MAE={mae:>6.2f}, n={count:>8,}")
        
        # Weather-based errors (if available)
        if 'temperature' in X.columns:
            print(f"\n   By Temperature:")
            temp_ranges = [(-10, 32), (32, 50), (50, 70), (70, 85), (85, 110)]
            for low, high in temp_ranges:
                mask = (X['temperature'] >= low) & (X['temperature'] < high)
                if mask.sum() > 0:
                    mae = abs_errors[mask].mean()
                    count = mask.sum()
                    print(f"     {low:>3.0f}-{high:<3.0f}°F: MAE={mae:>6.2f}, n={count:>8,}")
        
        # Large error analysis
        large_error_threshold = np.percentile(abs_errors, 95)
        large_error_mask = abs_errors > large_error_threshold
        print(f"\n   LARGE ERRORS (>{large_error_threshold:.2f}, top 5%):")
        print(f"     Count: {large_error_mask.sum():,} ({large_error_mask.sum()/len(abs_errors)*100:.1f}%)")
        
        # Analyze characteristics of large errors
        if large_error_mask.sum() > 0:
            large_error_data = X[large_error_mask]
            print(f"\n     Characteristics of large errors:")
            if 'estimated_distance' in X.columns:
                print(f"       Avg distance: {large_error_data['estimated_distance'].mean():.2f} mi "
                      f"(overall: {X['estimated_distance'].mean():.2f} mi)")
            if 'hour' in X.columns:
                print(f"       Most common hour: {large_error_data['hour'].mode()[0]}")
            if 'is_rush_hour' in X.columns:
                print(f"       Rush hour %: {large_error_data['is_rush_hour'].mean()*100:.1f}% "
                      f"(overall: {X['is_rush_hour'].mean()*100:.1f}%)")
        
        # Save error analysis
        error_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'error': errors,
            'abs_error': abs_errors,
            'pct_error': pct_errors
        })
        error_df.to_csv(
            self.output_dir / f'{target}_{model_type}_error_analysis.csv',
            index=False
        )
        
        return error_df
    
    def compare_models(self, X_train, y_train, X_val, y_val, target='fare'):
        """Compare XGBoost vs LightGBM performance."""
        print(f"\n{'='*80}")
        print(f"🏆 MODEL COMPARISON: XGBoost vs LightGBM ({target})")
        print(f"{'='*80}")
        
        # Common parameters
        n_estimators = 200
        max_depth = 8
        learning_rate = 0.05
        
        results = {}
        
        # Train XGBoost
        print(f"\n🔵 Training XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
        
        start = time.time()
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        xgb_train_time = time.time() - start
        
        xgb_pred_train = xgb_model.predict(X_train)
        xgb_pred_val = xgb_model.predict(X_val)
        
        results['xgboost'] = {
            'train_r2': r2_score(y_train, xgb_pred_train),
            'train_mae': mean_absolute_error(y_train, xgb_pred_train),
            'val_r2': r2_score(y_val, xgb_pred_val),
            'val_mae': mean_absolute_error(y_val, xgb_pred_val),
            'train_time': xgb_train_time
        }
        
        print(f"   Train R²:  {results['xgboost']['train_r2']:.4f} ({results['xgboost']['train_r2']*100:.2f}%)")
        print(f"   Val R²:    {results['xgboost']['val_r2']:.4f} ({results['xgboost']['val_r2']*100:.2f}%)")
        print(f"   Val MAE:   {results['xgboost']['val_mae']:.2f}")
        print(f"   Time:      {xgb_train_time:.1f}s")
        
        # Train LightGBM
        print(f"\n🟢 Training LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 2**max_depth,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        start = time.time()
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train)
        lgb_train_time = time.time() - start
        
        lgb_pred_train = lgb_model.predict(X_train)
        lgb_pred_val = lgb_model.predict(X_val)
        
        results['lightgbm'] = {
            'train_r2': r2_score(y_train, lgb_pred_train),
            'train_mae': mean_absolute_error(y_train, lgb_pred_train),
            'val_r2': r2_score(y_val, lgb_pred_val),
            'val_mae': mean_absolute_error(y_val, lgb_pred_val),
            'train_time': lgb_train_time
        }
        
        print(f"   Train R²:  {results['lightgbm']['train_r2']:.4f} ({results['lightgbm']['train_r2']*100:.2f}%)")
        print(f"   Val R²:    {results['lightgbm']['val_r2']:.4f} ({results['lightgbm']['val_r2']*100:.2f}%)")
        print(f"   Val MAE:   {results['lightgbm']['val_mae']:.2f}")
        print(f"   Time:      {lgb_train_time:.1f}s")
        
        # Comparison
        print(f"\n{'='*80}")
        print(f"📊 COMPARISON SUMMARY ({target})")
        print(f"{'='*80}")
        
        r2_diff = results['lightgbm']['val_r2'] - results['xgboost']['val_r2']
        mae_diff = results['xgboost']['val_mae'] - results['lightgbm']['val_mae']
        time_diff = results['xgboost']['train_time'] - results['lightgbm']['train_time']
        
        print(f"\nValidation R² (higher is better):")
        print(f"  XGBoost:  {results['xgboost']['val_r2']:.4f} ({results['xgboost']['val_r2']*100:.2f}%)")
        print(f"  LightGBM: {results['lightgbm']['val_r2']:.4f} ({results['lightgbm']['val_r2']*100:.2f}%)")
        print(f"  Winner:   {'LightGBM' if r2_diff > 0 else 'XGBoost'} by {abs(r2_diff)*100:.2f} percentage points")
        
        print(f"\nValidation MAE (lower is better):")
        print(f"  XGBoost:  {results['xgboost']['val_mae']:.2f}")
        print(f"  LightGBM: {results['lightgbm']['val_mae']:.2f}")
        print(f"  Winner:   {'LightGBM' if mae_diff > 0 else 'XGBoost'} by {abs(mae_diff):.2f}")
        
        print(f"\nTraining Time (lower is better):")
        print(f"  XGBoost:  {results['xgboost']['train_time']:.1f}s")
        print(f"  LightGBM: {results['lightgbm']['train_time']:.1f}s")
        print(f"  Winner:   {'LightGBM' if time_diff > 0 else 'XGBoost'} by {abs(time_diff):.1f}s")
        
        # Determine overall winner
        xgb_wins = 0
        lgb_wins = 0
        
        if results['xgboost']['val_r2'] > results['lightgbm']['val_r2']:
            xgb_wins += 1
        else:
            lgb_wins += 1
        
        if results['xgboost']['val_mae'] < results['lightgbm']['val_mae']:
            xgb_wins += 1
        else:
            lgb_wins += 1
        
        if results['xgboost']['train_time'] < results['lightgbm']['train_time']:
            xgb_wins += 1
        else:
            lgb_wins += 1
        
        print(f"\n🏆 OVERALL WINNER: {'XGBoost' if xgb_wins > lgb_wins else 'LightGBM'}")
        print(f"   XGBoost wins: {xgb_wins}/3 metrics")
        print(f"   LightGBM wins: {lgb_wins}/3 metrics")
        
        return results, xgb_model, lgb_model
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("="*80)
        print("🔬 COMPREHENSIVE MODEL ANALYSIS")
        print("="*80)
        
        # Load validation data
        X_val, y_fare_val, y_duration_val = self.load_data('val')
        
        # Sample for faster model comparison (use 500K samples)
        sample_size = 500000
        print(f"\n📊 Sampling {sample_size:,} samples for model comparison...")
        indices = np.random.choice(len(X_val), min(sample_size, len(X_val)), replace=False)
        X_sample = X_val.iloc[indices]
        y_fare_sample = y_fare_val[indices]
        y_duration_sample = y_duration_val[indices]
        
        # Load training data for model comparison
        print(f"\n📥 Loading training data for model comparison...")
        X_train_files = sorted((self.data_dir / 'train').glob('features_*_X.parquet'))
        y_fare_train_files = sorted((self.data_dir / 'train').glob('features_*_y_fare.parquet'))
        y_duration_train_files = sorted((self.data_dir / 'train').glob('features_*_y_duration.parquet'))
        
        # Sample from training (2M samples)
        train_sample_size = 2000000
        print(f"   Sampling {train_sample_size:,} training samples...")
        X_train_list = []
        y_fare_train_list = []
        y_duration_train_list = []
        
        for X_file, y_fare_file, y_duration_file in zip(X_train_files, y_fare_train_files, y_duration_train_files):
            X_month = pd.read_parquet(X_file)
            y_fare_month = pd.read_parquet(y_fare_file).values.ravel()
            y_duration_month = pd.read_parquet(y_duration_file).values.ravel()
            
            # Sample proportionally from each month
            month_sample = min(train_sample_size // 9, len(X_month))
            idx = np.random.choice(len(X_month), month_sample, replace=False)
            
            X_train_list.append(X_month.iloc[idx])
            y_fare_train_list.append(y_fare_month[idx])
            y_duration_train_list.append(y_duration_month[idx])
        
        X_train_sample = pd.concat(X_train_list, ignore_index=True)
        y_fare_train_sample = np.concatenate(y_fare_train_list)
        y_duration_train_sample = np.concatenate(y_duration_train_list)
        
        print(f"   Train: {len(X_train_sample):,} samples")
        print(f"   Val: {len(X_sample):,} samples")
        
        # ===== FARE MODEL ANALYSIS =====
        print("\n" + "="*80)
        print("💰 FARE AMOUNT MODEL ANALYSIS")
        print("="*80)
        
        # 1. Feature importance (existing XGBoost model)
        fare_model_path = Path("models/full_training/fare_full_model.json")
        if fare_model_path.exists():
            fare_importance, fare_cat_importance = self.analyze_feature_importance(
                fare_model_path, X_val, 'xgboost', 'fare'
            )
            
            # 2. Error patterns
            fare_errors = self.analyze_error_patterns(
                fare_model_path, X_val, y_fare_val, 'xgboost', 'fare'
            )
        
        # 3. Model comparison
        fare_comparison, fare_xgb, fare_lgb = self.compare_models(
            X_train_sample, y_fare_train_sample, X_sample, y_fare_sample, 'fare'
        )
        
        # ===== DURATION MODEL ANALYSIS =====
        print("\n" + "="*80)
        print("⏱️  TRIP DURATION MODEL ANALYSIS")
        print("="*80)
        
        # 1. Feature importance (existing XGBoost model)
        duration_model_path = Path("models/full_training/duration_full_model.json")
        if duration_model_path.exists():
            duration_importance, duration_cat_importance = self.analyze_feature_importance(
                duration_model_path, X_val, 'xgboost', 'duration'
            )
            
            # 2. Error patterns
            duration_errors = self.analyze_error_patterns(
                duration_model_path, X_val, y_duration_val, 'xgboost', 'duration'
            )
        
        # 3. Model comparison
        duration_comparison, duration_xgb, duration_lgb = self.compare_models(
            X_train_sample, y_duration_train_sample, X_sample, y_duration_sample, 'duration'
        )
        
        # ===== FINAL SUMMARY =====
        print("\n" + "="*80)
        print("📋 FINAL ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n💰 FARE MODEL:")
        print(f"   Current (XGBoost): R² = {fare_comparison['xgboost']['val_r2']*100:.2f}%")
        print(f"   LightGBM:          R² = {fare_comparison['lightgbm']['val_r2']*100:.2f}%")
        print(f"   Recommendation:    Use {'LightGBM' if fare_comparison['lightgbm']['val_r2'] > fare_comparison['xgboost']['val_r2'] else 'XGBoost'}")
        
        print("\n⏱️  DURATION MODEL:")
        print(f"   Current (XGBoost): R² = {duration_comparison['xgboost']['val_r2']*100:.2f}%")
        print(f"   LightGBM:          R² = {duration_comparison['lightgbm']['val_r2']*100:.2f}%")
        print(f"   Recommendation:    Use {'LightGBM' if duration_comparison['lightgbm']['val_r2'] > duration_comparison['xgboost']['val_r2'] else 'XGBoost'}")
        
        print("\n📁 Results saved to: analysis_results/")
        print("   - Feature importance CSVs")
        print("   - Error analysis CSVs")
        print("   - Model comparison results")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)

if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer.run_full_analysis()
