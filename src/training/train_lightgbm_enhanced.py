"""
Enhanced LightGBM training with advanced techniques to boost duration R² by 3%+
Target: 82.85% → 85-86% R²

Improvements:
1. Hyperparameter optimization (more trees, better learning rate)
2. Feature engineering (interaction features)
3. Better regularization
4. Early stopping with larger patience
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class EnhancedLightGBMTrainer:
    def __init__(self):
        self.data_dir = Path("data/splits_cleaned")
        self.output_dir = Path("models/lightgbm_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced parameters for duration prediction
        # Focus: More trees, lower learning rate, stronger regularization
        self.duration_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # Tree structure - INCREASED complexity
            'num_leaves': 512,  # Increased from 256
            'max_depth': 12,    # Increased from 8
            
            # Learning - MORE ITERATIONS, LOWER RATE
            'learning_rate': 0.03,  # Reduced from 0.05 for better convergence
            'n_estimators': 500,     # Increased from 200
            
            # Regularization - STRONGER
            'min_child_samples': 50,      # Increased from 30
            'min_child_weight': 0.01,     # Added
            'reg_alpha': 0.1,             # L1 regularization
            'reg_lambda': 1.0,            # L2 regularization
            
            # Sampling - BETTER GENERALIZATION
            'subsample': 0.85,            # Increased from 0.8
            'subsample_freq': 1,          # Enable subsampling every iteration
            'colsample_bytree': 0.85,     # Increased from 0.8
            'colsample_bynode': 0.85,     # Added - column sampling per node
            
            # Speed and stability
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True,       # Faster training
        }
        
        # Keep fare params same (already 94%+)
        self.fare_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 256,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def engineer_interaction_features(self, X):
        """Create interaction features to capture complex patterns."""
        X_enhanced = X.copy()
        
        print("   🔧 Engineering advanced interaction features...")
        initial_features = len(X.columns)
        
        # NEW: Distance-weather interactions (weather affects speed!)
        if 'estimated_distance' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['distance_rain'] = X['estimated_distance'] * X['is_raining']
            if 'is_snowing' in X.columns:
                X_enhanced['distance_snow'] = X['estimated_distance'] * X['is_snowing']
            if 'weather_severity' in X.columns:
                X_enhanced['distance_weather_severity'] = X['estimated_distance'] * X['weather_severity']
        
        # NEW: Holiday-time interactions (different patterns on holidays)
        if 'is_holiday' in X.columns and 'pickup_hour' in X.columns:
            X_enhanced['holiday_hour'] = X['is_holiday'] * X['pickup_hour']
        if 'is_major_holiday' in X.columns and 'is_rush_hour' in X.columns:
            X_enhanced['major_holiday_rush'] = X['is_major_holiday'] * X['is_rush_hour']
        
        # NEW: Weather-time interactions (weather affects different times differently)
        if 'is_rush_hour' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['rush_rain'] = X['is_rush_hour'] * X['is_raining']
            if 'is_snowing' in X.columns:
                X_enhanced['rush_snow'] = X['is_rush_hour'] * X['is_snowing']
        
        # NEW: Manhattan-specific interactions (manhattan traffic is different)
        if 'pickup_is_manhattan' in X.columns:
            if 'is_rush_hour' in X.columns:
                X_enhanced['manhattan_rush'] = X['pickup_is_manhattan'] * X['is_rush_hour']
            if 'is_weekend' in X.columns:
                X_enhanced['manhattan_weekend'] = X['pickup_is_manhattan'] * X['is_weekend']
            if 'estimated_distance' in X.columns:
                X_enhanced['manhattan_distance'] = X['pickup_is_manhattan'] * X['estimated_distance']
        
        # NEW: Airport-specific interactions (airport trips have different patterns)
        if 'is_airport_trip' in X.columns:
            if 'pickup_hour' in X.columns:
                X_enhanced['airport_hour'] = X['is_airport_trip'] * X['pickup_hour']
            if 'is_weekend' in X.columns:
                X_enhanced['airport_weekend'] = X['is_airport_trip'] * X['is_weekend']
            if 'estimated_distance' in X.columns:
                X_enhanced['airport_distance'] = X['is_airport_trip'] * X['estimated_distance']
        
        # NEW: Passenger-context interactions
        if 'passenger_count' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['passengers_distance'] = X['passenger_count'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['passengers_rush'] = X['passenger_count'] * X['is_rush_hour']
            if 'is_airport_trip' in X.columns:
                X_enhanced['passengers_airport'] = X['passenger_count'] * X['is_airport_trip']
        
        # NEW: Weekend-weather (different behavior)
        if 'is_weekend' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['weekend_rain'] = X['is_weekend'] * X['is_raining']
            if 'temperature' in X.columns:
                X_enhanced['weekend_temp'] = X['is_weekend'] * X['temperature']
        
        # NEW: Speed proxies (distance divided by time indicators)
        if 'estimated_distance' in X.columns and 'pickup_hour' in X.columns:
            # Normalize hour to 0-1 range to avoid division issues
            hour_normalized = X['pickup_hour'] / 24.0
            X_enhanced['speed_proxy_hour'] = X['estimated_distance'] / (1 + hour_normalized)
        
        # NEW: Location pair efficiency (some routes are consistently faster/slower)
        if 'PULocationID' in X.columns and 'DOLocationID' in X.columns:
            # Create a simple route identifier
            X_enhanced['route_complexity'] = (X['PULocationID'] + X['DOLocationID']) / 2
        
        new_features = len(X_enhanced.columns) - initial_features
        print(f"      ✅ Added {new_features} NEW interaction features")
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
        print(f"\n🚂 Training Enhanced LightGBM ({target})...")
        print(f"   Params:")
        print(f"      • {params['n_estimators']} estimators")
        print(f"      • {params['num_leaves']} leaves, depth {params['max_depth']}")
        print(f"      • Learning rate {params['learning_rate']}")
        print(f"      • Regularization: L1={params.get('reg_alpha', 0)}, L2={params.get('reg_lambda', 0)}")
        
        start = time.time()
        
        # Use early stopping for better generalization
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
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
    
    def train_fare_model(self):
        """Train fare prediction model (baseline)."""
        print("\n" + "="*80)
        print("💰 Training Fare Amount Model (baseline)")
        print("="*80)
        
        # Load data
        print("\n📥 Loading training data (fare)...")
        X_train, y_train = self.load_full_data(self.data_dir / "train", "fare")
        print(f"   Train: {len(X_train):,} trips, {len(X_train.columns)} features")
        
        print("\n📥 Loading validation data (fare)...")
        X_val, y_val = self.load_full_data(self.data_dir / "val", "fare")
        print(f"   Val: {len(X_val):,} trips")
        
        # Train
        model, train_time = self.train_model(X_train, y_train, X_val, y_val, self.fare_params, 'fare')
        
        # Evaluate
        print("\n📊 Evaluating fare model...")
        train_metrics = self.evaluate_model(model, X_train, y_train, 'train')
        val_metrics = self.evaluate_model(model, X_val, y_val, 'val')
        
        print(f"   Train R²: {train_metrics['R²']:.4f} ({train_metrics['R²']*100:.2f}%)")
        print(f"   Train MAE: ${train_metrics['MAE']:.2f}")
        print(f"   Val R²: {val_metrics['R²']:.4f} ({val_metrics['R²']*100:.2f}%)")
        print(f"   Val MAE: ${val_metrics['MAE']:.2f}")
        
        # Save
        model_path = self.output_dir / "fare_lightgbm.txt"
        model.booster_.save_model(str(model_path))
        print(f"\n💾 Saved model to: {model_path}")
        
        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_time': train_time,
            'X_train': X_train,
            'X_val': X_val
        }
    
    def train_duration_model(self, X_train, X_val):
        """Train ENHANCED duration prediction model."""
        print("\n" + "="*80)
        print("⏱️  Training ENHANCED Trip Duration Model")
        print("="*80)
        
        # Load targets
        print("\n📥 Loading training data (duration)...")
        _, y_train = self.load_full_data(self.data_dir / "train", "duration")
        print(f"   Train: {len(X_train):,} trips")
        
        print("\n📥 Loading validation data (duration)...")
        _, y_val = self.load_full_data(self.data_dir / "val", "duration")
        print(f"   Val: {len(X_val):,} trips")
        
        # Engineer interaction features
        print("\n🔧 Feature Engineering...")
        X_train_enhanced = self.engineer_interaction_features(X_train)
        X_val_enhanced = self.engineer_interaction_features(X_val)
        
        # Train
        model, train_time = self.train_model(
            X_train_enhanced, y_train, 
            X_val_enhanced, y_val, 
            self.duration_params, 'duration'
        )
        
        # Evaluate
        print("\n📊 Evaluating enhanced duration model...")
        train_metrics = self.evaluate_model(model, X_train_enhanced, y_train, 'train')
        val_metrics = self.evaluate_model(model, X_val_enhanced, y_val, 'val')
        
        print(f"   Train R²: {train_metrics['R²']:.4f} ({train_metrics['R²']*100:.2f}%)")
        print(f"   Train MAE: {train_metrics['MAE']:.2f} minutes")
        print(f"   Val R²: {val_metrics['R²']:.4f} ({val_metrics['R²']*100:.2f}%)")
        print(f"   Val MAE: {val_metrics['MAE']:.2f} minutes")
        
        # Compare to baseline
        baseline_r2 = 0.8285  # From previous run
        improvement = (val_metrics['R²'] - baseline_r2) * 100
        
        print(f"\n📈 Improvement vs Baseline:")
        print(f"   Baseline: {baseline_r2*100:.2f}%")
        print(f"   Enhanced: {val_metrics['R²']*100:.2f}%")
        print(f"   Gain: {improvement:+.2f} percentage points")
        
        if val_metrics['R²'] >= 0.85:
            print("   🎯 ✅ TARGET REACHED: ≥85% R²!")
        else:
            print(f"   🎯 ⚠️  Need {(0.85 - val_metrics['R²'])*100:.2f}pp more to reach 85%")
        
        # Save
        model_path = self.output_dir / "duration_lightgbm_enhanced.txt"
        model.booster_.save_model(str(model_path))
        print(f"\n💾 Saved model to: {model_path}")
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': X_train_enhanced.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = self.output_dir / "duration_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"💾 Saved feature importance to: {importance_path}")
        
        print("\n📊 Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']:30s} {row['importance']:8.0f}")
        
        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_time': train_time,
            'improvement': improvement,
            'feature_importance': importance_df
        }
    
    def run(self):
        """Run full training pipeline."""
        print("="*80)
        print("🚀 ENHANCED LIGHTGBM TRAINING")
        print("="*80)
        print("\n🎯 Goal: Improve Duration R² from 82.85% to 85-86%")
        print("\n✨ Enhancements:")
        print("  1. More trees (200 → 500) with lower learning rate (0.05 → 0.03)")
        print("  2. Deeper trees (depth 8 → 12) with more leaves (256 → 512)")
        print("  3. Stronger regularization (L1 + L2)")
        print("  4. Interaction features (time × location, distance × rush hour, etc.)")
        print("  5. Early stopping with patience=50")
        print("="*80)
        
        # Train fare model (baseline)
        fare_results = self.train_fare_model()
        
        # Train ENHANCED duration model
        duration_results = self.train_duration_model(
            fare_results['X_train'], 
            fare_results['X_val']
        )
        
        # Final summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        
        print(f"\n💰 Fare Model:")
        print(f"   Val R²: {fare_results['val_metrics']['R²']*100:.2f}%")
        print(f"   Val MAE: ${fare_results['val_metrics']['MAE']:.2f}")
        
        print(f"\n⏱️  Duration Model (ENHANCED):")
        print(f"   Val R²: {duration_results['val_metrics']['R²']*100:.2f}%")
        print(f"   Val MAE: {duration_results['val_metrics']['MAE']:.2f} minutes")
        print(f"   Improvement: {duration_results['improvement']:+.2f}pp vs baseline")
        
        total_time = fare_results['train_time'] + duration_results['train_time']
        print(f"\n⏱️  Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        print("\n" + "="*80)
        
        return {
            'fare': fare_results,
            'duration': duration_results
        }

if __name__ == "__main__":
    trainer = EnhancedLightGBMTrainer()
    results = trainer.run()
