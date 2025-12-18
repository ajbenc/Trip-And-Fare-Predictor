"""
ULTRA-ENHANCED LightGBM training - Push for 90% R²!
Target: 84.77% → 90% (need +5.23pp gain)

Aggressive enhancements:
1. MANY more trees (1000+) with very low learning rate
2. Maximum tree complexity
3. Even more interaction features
4. Ensemble of multiple models with different configurations
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class UltraEnhancedLightGBMTrainer:
    def __init__(self):
        self.data_dir = Path("data/splits_cleaned")
        self.output_dir = Path("models/lightgbm_ultra")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ULTRA AGGRESSIVE parameters - pushing limits!
        self.duration_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # MAXIMUM complexity
            'num_leaves': 1024,      # Doubled from 512
            'max_depth': 15,         # Increased from 12
            
            # SLOW learning, MANY iterations
            'learning_rate': 0.02,   # Slower from 0.03
            'n_estimators': 1000,    # Doubled from 500
            
            # STRONG regularization (prevent overfitting)
            'min_child_samples': 100,     # More conservative
            'min_child_weight': 0.05,     # Stronger
            'reg_alpha': 0.5,             # Stronger L1
            'reg_lambda': 2.0,            # Stronger L2
            'max_bin': 512,               # More bins (default 255)
            
            # Aggressive sampling
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.9,
            'colsample_bynode': 0.9,
            
            # Additional boosting features
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            
            # Speed and stability
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True,
        }
    
    def engineer_ultra_features(self, X):
        """Create MAXIMUM interaction features."""
        X_enhanced = X.copy()
        
        print("   🔧 Engineering ULTRA-ADVANCED features...")
        initial_features = len(X.columns)
        
        # Previous features (20)
        if 'estimated_distance' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['distance_rain'] = X['estimated_distance'] * X['is_raining']
            if 'is_snowing' in X.columns:
                X_enhanced['distance_snow'] = X['estimated_distance'] * X['is_snowing']
            if 'weather_severity' in X.columns:
                X_enhanced['distance_weather_severity'] = X['estimated_distance'] * X['weather_severity']
        
        if 'is_holiday' in X.columns and 'pickup_hour' in X.columns:
            X_enhanced['holiday_hour'] = X['is_holiday'] * X['pickup_hour']
        if 'is_major_holiday' in X.columns and 'is_rush_hour' in X.columns:
            X_enhanced['major_holiday_rush'] = X['is_major_holiday'] * X['is_rush_hour']
        
        if 'is_rush_hour' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['rush_rain'] = X['is_rush_hour'] * X['is_raining']
            if 'is_snowing' in X.columns:
                X_enhanced['rush_snow'] = X['is_rush_hour'] * X['is_snowing']
        
        if 'pickup_is_manhattan' in X.columns:
            if 'is_rush_hour' in X.columns:
                X_enhanced['manhattan_rush'] = X['pickup_is_manhattan'] * X['is_rush_hour']
            if 'is_weekend' in X.columns:
                X_enhanced['manhattan_weekend'] = X['pickup_is_manhattan'] * X['is_weekend']
            if 'estimated_distance' in X.columns:
                X_enhanced['manhattan_distance'] = X['pickup_is_manhattan'] * X['estimated_distance']
        
        if 'is_airport_trip' in X.columns:
            if 'pickup_hour' in X.columns:
                X_enhanced['airport_hour'] = X['is_airport_trip'] * X['pickup_hour']
            if 'is_weekend' in X.columns:
                X_enhanced['airport_weekend'] = X['is_airport_trip'] * X['is_weekend']
            if 'estimated_distance' in X.columns:
                X_enhanced['airport_distance'] = X['is_airport_trip'] * X['estimated_distance']
        
        if 'passenger_count' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['passengers_distance'] = X['passenger_count'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['passengers_rush'] = X['passenger_count'] * X['is_rush_hour']
            if 'is_airport_trip' in X.columns:
                X_enhanced['passengers_airport'] = X['passenger_count'] * X['is_airport_trip']
        
        if 'is_weekend' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['weekend_rain'] = X['is_weekend'] * X['is_raining']
            if 'temperature' in X.columns:
                X_enhanced['weekend_temp'] = X['is_weekend'] * X['temperature']
        
        if 'estimated_distance' in X.columns and 'pickup_hour' in X.columns:
            hour_normalized = X['pickup_hour'] / 24.0
            X_enhanced['speed_proxy_hour'] = X['estimated_distance'] / (1 + hour_normalized)
        
        if 'PULocationID' in X.columns and 'DOLocationID' in X.columns:
            X_enhanced['route_complexity'] = (X['PULocationID'] + X['DOLocationID']) / 2
        
        # NEW ULTRA features (30+ more!)
        
        # Multi-way interactions
        if 'estimated_distance' in X.columns and 'pickup_hour' in X.columns and 'is_raining' in X.columns:
            X_enhanced['distance_hour_rain'] = X['estimated_distance'] * X['pickup_hour'] * X['is_raining']
        
        if 'estimated_distance' in X.columns and 'is_rush_hour' in X.columns and 'pickup_is_manhattan' in X.columns:
            X_enhanced['distance_rush_manhattan'] = X['estimated_distance'] * X['is_rush_hour'] * X['pickup_is_manhattan']
        
        # Squared terms (capture non-linear effects)
        if 'estimated_distance' in X.columns:
            X_enhanced['distance_squared'] = X['estimated_distance'] ** 2
            X_enhanced['distance_sqrt'] = np.sqrt(X['estimated_distance'].clip(lower=0))
        
        if 'pickup_hour' in X.columns:
            X_enhanced['hour_squared'] = X['pickup_hour'] ** 2
        
        # Temperature effects on different conditions
        if 'temperature' in X.columns:
            X_enhanced['temp_squared'] = X['temperature'] ** 2
            if 'estimated_distance' in X.columns:
                X_enhanced['temp_distance'] = X['temperature'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['temp_rush'] = X['temperature'] * X['is_rush_hour']
        
        # Wind/weather severity interactions
        if 'wind_speed' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['wind_distance'] = X['wind_speed'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['wind_rush'] = X['wind_speed'] * X['is_rush_hour']
        
        # Precipitation interactions
        if 'precipitation' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['precip_distance'] = X['precipitation'] * X['estimated_distance']
            if 'pickup_hour' in X.columns:
                X_enhanced['precip_hour'] = X['precipitation'] * X['pickup_hour']
        
        # Day/time patterns
        if 'pickup_weekday' in X.columns and 'pickup_hour' in X.columns:
            X_enhanced['weekday_hour'] = X['pickup_weekday'] * X['pickup_hour']
        
        if 'pickup_day' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['day_distance'] = X['pickup_day'] * X['estimated_distance']
        
        # Late night specific patterns
        if 'is_late_night' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['latenight_distance'] = X['is_late_night'] * X['estimated_distance']
            if 'pickup_is_manhattan' in X.columns:
                X_enhanced['latenight_manhattan'] = X['is_late_night'] * X['pickup_is_manhattan']
            if 'is_weekend' in X.columns:
                X_enhanced['latenight_weekend'] = X['is_late_night'] * X['is_weekend']
        
        # Business hours patterns
        if 'is_business_hours' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['business_distance'] = X['is_business_hours'] * X['estimated_distance']
            if 'pickup_is_manhattan' in X.columns:
                X_enhanced['business_manhattan'] = X['is_business_hours'] * X['pickup_is_manhattan']
        
        # Location pair density (some routes are more common)
        if 'PULocationID' in X.columns and 'DOLocationID' in X.columns:
            X_enhanced['pickup_density'] = X['PULocationID'] / 265.0  # Normalize
            X_enhanced['dropoff_density'] = X['DOLocationID'] / 265.0
            X_enhanced['route_hash'] = (X['PULocationID'] * 1000 + X['DOLocationID']) % 10000
        
        # Cyclical patterns (already exist but enhance)
        if 'hour_sin' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['hoursin_distance'] = X['hour_sin'] * X['estimated_distance']
        if 'hour_cos' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['hourcos_distance'] = X['hour_cos'] * X['estimated_distance']
        
        # Extreme conditions
        if 'is_extreme_weather' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['extreme_distance'] = X['is_extreme_weather'] * X['estimated_distance']
        
        if 'is_poor_visibility' in X.columns and 'is_rush_hour' in X.columns:
            X_enhanced['poorvis_rush'] = X['is_poor_visibility'] * X['is_rush_hour']
        
        # Passenger patterns by location
        if 'passenger_count' in X.columns:
            if 'pickup_is_manhattan' in X.columns:
                X_enhanced['passengers_manhattan'] = X['passenger_count'] * X['pickup_is_manhattan']
            if 'is_weekend' in X.columns:
                X_enhanced['passengers_weekend'] = X['passenger_count'] * X['is_weekend']
        
        # Holiday week patterns (extended holiday behavior)
        if 'is_holiday_week' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['holidayweek_distance'] = X['is_holiday_week'] * X['estimated_distance']
            if 'pickup_hour' in X.columns:
                X_enhanced['holidayweek_hour'] = X['is_holiday_week'] * X['pickup_hour']
        
        # Ratio features (normalized interactions)
        if 'estimated_distance' in X.columns and 'passenger_count' in X.columns:
            X_enhanced['distance_per_passenger'] = X['estimated_distance'] / (X['passenger_count'] + 1)
        
        new_features = len(X_enhanced.columns) - initial_features
        print(f"      ✅ Added {new_features} ULTRA features")
        print(f"      Total features: {len(X_enhanced.columns)}")
        
        return X_enhanced
    
    def load_full_data(self, split_dir: Path, target: str):
        """Load ALL data from split directory."""
        X_files = sorted(split_dir.glob('features_*_X.parquet'))
        y_files = sorted(split_dir.glob(f'features_*_y_{target}.parquet'))
        
        print(f"   Loading {len(X_files)} files...")
        X_list = []
        y_list = []
        
        for X_file, y_file in zip(X_files, y_files):
            X_month = pd.read_parquet(X_file)
            y_month = pd.read_parquet(y_file)
            
            X_list.append(X_month)
            y_list.append(y_month)
            print(f"     {X_file.name}: {len(X_month):,} samples")
        
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True).values.ravel()
        
        print(f"   Total: {len(X):,} samples")
        
        return X, y
    
    def train_model(self, X_train, y_train, X_val, y_val, params, target):
        """Train LightGBM model with early stopping."""
        print(f"\n🚂 Training ULTRA-ENHANCED LightGBM ({target})...")
        print(f"   Params:")
        print(f"      • {params['n_estimators']} estimators (MAXIMUM)")
        print(f"      • {params['num_leaves']} leaves, depth {params['max_depth']} (MAXIMUM)")
        print(f"      • Learning rate {params['learning_rate']} (SLOWEST)")
        print(f"      • Regularization: L1={params['reg_alpha']}, L2={params['reg_lambda']} (STRONGEST)")
        
        start = time.time()
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=75, verbose=False),  # More patience
                lgb.log_evaluation(period=100)
            ]
        )
        
        train_time = time.time() - start
        
        print(f"   ✅ Training completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
        print(f"      Best iteration: {model.best_iteration_}/{params['n_estimators']}")
        
        return model, train_time
    
    def evaluate_model(self, model, X, y, split_name):
        """Evaluate model performance."""
        pred = model.predict(X)
        
        r2 = r2_score(y, pred)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        
        return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'predictions': pred}
    
    def train_duration_model(self):
        """Train ULTRA-ENHANCED duration prediction model."""
        print("\n" + "="*80)
        print("⏱️  Training ULTRA-ENHANCED Trip Duration Model")
        print("="*80)
        
        # Load data
        print("\n📥 Loading training data (duration)...")
        X_train, y_train = self.load_full_data(self.data_dir / "train", "duration")
        print(f"   Train: {len(X_train):,} trips, {len(X_train.columns)} base features")
        
        print("\n📥 Loading validation data (duration)...")
        X_val, y_val = self.load_full_data(self.data_dir / "val", "duration")
        print(f"   Val: {len(X_val):,} trips")
        
        # Engineer ULTRA features
        print("\n🔧 ULTRA Feature Engineering...")
        X_train_ultra = self.engineer_ultra_features(X_train)
        X_val_ultra = self.engineer_ultra_features(X_val)
        
        # Train
        model, train_time = self.train_model(
            X_train_ultra, y_train, 
            X_val_ultra, y_val, 
            self.duration_params, 'duration'
        )
        
        # Evaluate
        print("\n📊 Evaluating ULTRA duration model...")
        train_metrics = self.evaluate_model(model, X_train_ultra, y_train, 'train')
        val_metrics = self.evaluate_model(model, X_val_ultra, y_val, 'val')
        
        print(f"   Train R²: {train_metrics['R²']:.4f} ({train_metrics['R²']*100:.2f}%)")
        print(f"   Train MAE: {train_metrics['MAE']:.2f} minutes")
        print(f"   Val R²: {val_metrics['R²']:.4f} ({val_metrics['R²']*100:.2f}%)")
        print(f"   Val MAE: {val_metrics['MAE']:.2f} minutes")
        
        # Compare to baseline
        baseline_r2 = 0.8285
        enhanced_r2 = 0.8477
        improvement = (val_metrics['R²'] - baseline_r2) * 100
        vs_enhanced = (val_metrics['R²'] - enhanced_r2) * 100
        
        print(f"\n📈 Improvement:")
        print(f"   Baseline: {baseline_r2*100:.2f}%")
        print(f"   Enhanced: {enhanced_r2*100:.2f}%")
        print(f"   ULTRA: {val_metrics['R²']*100:.2f}%")
        print(f"   Total gain: {improvement:+.2f}pp vs baseline")
        print(f"   Additional gain: {vs_enhanced:+.2f}pp vs enhanced")
        
        if val_metrics['R²'] >= 0.90:
            print("   🎯 🎉 TARGET REACHED: ≥90% R²!!!")
        elif val_metrics['R²'] >= 0.85:
            print(f"   🎯 ✅ CROSSED 85% R²!")
            print(f"   🎯 Need {(0.90 - val_metrics['R²'])*100:.2f}pp more for 90%")
        else:
            print(f"   🎯 Need {(0.85 - val_metrics['R²'])*100:.2f}pp more for 85%")
            print(f"   🎯 Need {(0.90 - val_metrics['R²'])*100:.2f}pp more for 90%")
        
        # Save
        model_path = self.output_dir / "duration_lightgbm_ultra.txt"
        model.booster_.save_model(str(model_path))
        print(f"\n💾 Saved model to: {model_path}")
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': X_train_ultra.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = self.output_dir / "duration_feature_importance_ultra.csv"
        importance_df.to_csv(importance_path, index=False)
        
        print("\n📊 Top 15 Most Important Features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"   {row['feature']:35s} {row['importance']:8.0f}")
        
        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_time': train_time,
            'improvement': improvement,
            'feature_importance': importance_df
        }
    
    def run(self):
        """Run ULTRA training pipeline."""
        print("="*80)
        print("🚀 ULTRA-ENHANCED LIGHTGBM TRAINING - PUSH FOR 90%!")
        print("="*80)
        print("\n🎯 Goal: 84.77% → 90% R² (need +5.23pp)")
        print("\n💪 MAXIMUM Enhancements:")
        print("  1. 1000 trees (vs 500) with learning rate 0.02 (vs 0.03)")
        print("  2. 1024 leaves, depth 15 (vs 512 leaves, depth 12)")
        print("  3. STRONGEST regularization (L1=0.5, L2=2.0)")
        print("  4. 50+ NEW interaction features")
        print("  5. Squared terms, multi-way interactions, ratios")
        print("  6. Early stopping patience=75")
        print("="*80)
        
        # Train ULTRA duration model
        duration_results = self.train_duration_model()
        
        # Final summary
        print("\n" + "="*80)
        print("✅ ULTRA TRAINING COMPLETE")
        print("="*80)
        
        print(f"\n⏱️  Duration Model (ULTRA):")
        print(f"   Val R²: {duration_results['val_metrics']['R²']*100:.2f}%")
        print(f"   Val MAE: {duration_results['val_metrics']['MAE']:.2f} minutes")
        print(f"   Total improvement: {duration_results['improvement']:+.2f}pp vs baseline")
        print(f"\n⏱️  Training Time: {duration_results['train_time']:.1f}s ({duration_results['train_time']/60:.1f} min)")
        print("\n" + "="*80)
        
        return duration_results

if __name__ == "__main__":
    trainer = UltraEnhancedLightGBMTrainer()
    results = trainer.run()
