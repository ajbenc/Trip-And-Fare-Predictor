"""
Test Enhanced LightGBM V2 on December Test Set
This is the 84.77% validation model - let's see if it generalizes better!
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

class EnhancedTestEvaluator:
    def __init__(self):
        self.data_dir = Path("Data/splits_cleaned/test")
        self.model_path = Path("models/lightgbm_enhanced/duration_lightgbm_enhanced.txt")
        
    def engineer_enhanced_features(self, X):
        """Apply Enhanced V2 feature engineering (20 interaction features)."""
        X_enhanced = X.copy()
        
        # Weather interactions
        if 'estimated_distance' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['distance_rain'] = X['estimated_distance'] * X['is_raining']
            if 'is_snowing' in X.columns:
                X_enhanced['distance_snow'] = X['estimated_distance'] * X['is_snowing']
            if 'weather_severity' in X.columns:
                X_enhanced['distance_weather_severity'] = X['estimated_distance'] * X['weather_severity']
        
        # Holiday interactions
        if 'is_holiday' in X.columns and 'pickup_hour' in X.columns:
            X_enhanced['holiday_hour'] = X['is_holiday'] * X['pickup_hour']
        if 'is_major_holiday' in X.columns and 'is_rush_hour' in X.columns:
            X_enhanced['major_holiday_rush'] = X['is_major_holiday'] * X['is_rush_hour']
        
        # Rush hour weather
        if 'is_rush_hour' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['rush_rain'] = X['is_rush_hour'] * X['is_raining']
            if 'is_snowing' in X.columns:
                X_enhanced['rush_snow'] = X['is_rush_hour'] * X['is_snowing']
        
        # Manhattan interactions
        if 'pickup_is_manhattan' in X.columns:
            if 'is_rush_hour' in X.columns:
                X_enhanced['manhattan_rush'] = X['pickup_is_manhattan'] * X['is_rush_hour']
            if 'is_weekend' in X.columns:
                X_enhanced['manhattan_weekend'] = X['pickup_is_manhattan'] * X['is_weekend']
            if 'estimated_distance' in X.columns:
                X_enhanced['manhattan_distance'] = X['pickup_is_manhattan'] * X['estimated_distance']
        
        # Airport interactions
        if 'is_airport_trip' in X.columns:
            if 'pickup_hour' in X.columns:
                X_enhanced['airport_hour'] = X['is_airport_trip'] * X['pickup_hour']
            if 'is_weekend' in X.columns:
                X_enhanced['airport_weekend'] = X['is_airport_trip'] * X['is_weekend']
            if 'estimated_distance' in X.columns:
                X_enhanced['airport_distance'] = X['is_airport_trip'] * X['estimated_distance']
        
        # Passenger interactions
        if 'passenger_count' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['passengers_distance'] = X['passenger_count'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['passengers_rush'] = X['passenger_count'] * X['is_rush_hour']
            if 'is_airport_trip' in X.columns:
                X_enhanced['passengers_airport'] = X['passenger_count'] * X['is_airport_trip']
        
        # Weekend interactions
        if 'is_weekend' in X.columns:
            if 'is_raining' in X.columns:
                X_enhanced['weekend_rain'] = X['is_weekend'] * X['is_raining']
            if 'temperature' in X.columns:
                X_enhanced['weekend_temp'] = X['is_weekend'] * X['temperature']
        
        # Speed proxy
        if 'estimated_distance' in X.columns and 'pickup_hour' in X.columns:
            hour_normalized = X['pickup_hour'] / 24.0
            X_enhanced['speed_proxy_hour'] = X['estimated_distance'] / (1 + hour_normalized)
        
        # Route complexity
        if 'PULocationID' in X.columns and 'DOLocationID' in X.columns:
            X_enhanced['route_complexity'] = (X['PULocationID'] + X['DOLocationID']) / 2
        
        return X_enhanced
    
    def load_test_data(self):
        """Load December test data."""
        print("\n📥 Loading TEST data (December)...")
        
        X_files = sorted(self.data_dir.glob('features_*_X.parquet'))
        y_files = sorted(self.data_dir.glob('features_*_y_duration.parquet'))
        
        X_list = []
        y_list = []
        
        for X_file, y_file in zip(X_files, y_files):
            X_month = pd.read_parquet(X_file)
            y_month = pd.read_parquet(y_file)
            
            X_list.append(X_month)
            y_list.append(y_month)
        
        X_test = pd.concat(X_list, ignore_index=True)
        y_test = pd.concat(y_list, ignore_index=True).values.ravel()
        
        print(f"   Total TEST samples: {len(X_test):,}")
        
        return X_test, y_test
    
    def evaluate(self):
        """Run evaluation on Enhanced V2 model."""
        print("="*80)
        print("🎯 ENHANCED V2 MODEL - Test Set Evaluation")
        print("="*80)
        print("\n📊 This is the 84.77% validation model (more conservative)")
        print("="*80)
        
        # Load model
        print(f"\n📦 Loading Enhanced V2 model...")
        if not self.model_path.exists():
            print("❌ ERROR: Model file not found!")
            return
        
        model = lgb.Booster(model_file=str(self.model_path))
        print("   ✅ Model loaded")
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Engineer features
        print("\n🔧 Applying Enhanced feature engineering (20 interactions)...")
        X_test_enhanced = self.engineer_enhanced_features(X_test)
        print(f"   ✅ Total features: {len(X_test_enhanced.columns)} (56 base + 20 engineered)")
        
        # Predict
        print("\n🔮 Making predictions...")
        start = time.time()
        y_pred = model.predict(X_test_enhanced)
        pred_time = time.time() - start
        
        print(f"   ✅ Predicted {len(y_pred):,} trips in {pred_time:.1f}s")
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Results
        print("\n" + "="*80)
        print("🎉 ENHANCED V2 - TEST SET RESULTS")
        print("="*80)
        
        print(f"\n📈 Test Performance:")
        print(f"   R² Score:  {r2:.4f} ({r2*100:.2f}%)")
        print(f"   MAE:       {mae:.2f} minutes")
        print(f"   RMSE:      {rmse:.2f} minutes")
        
        print(f"\n📊 Complete Model Performance:")
        print(f"   Training R²:   84.68%")
        print(f"   Validation R²: 84.77%")
        print(f"   Test R²:       {r2*100:.2f}%")
        
        # Generalization analysis
        val_r2 = 0.8477
        diff = (r2 - val_r2) * 100
        
        print(f"\n🔍 Generalization Analysis:")
        print(f"   Validation: {val_r2*100:.2f}%")
        print(f"   Test:       {r2*100:.2f}%")
        print(f"   Gap:        {diff:+.2f}pp")
        
        if abs(diff) < 1.0:
            print(f"   ✅ EXCELLENT generalization!")
        elif abs(diff) < 2.0:
            print(f"   ✅ GOOD generalization!")
        elif abs(diff) < 3.0:
            print(f"   ✅ ACCEPTABLE generalization")
        else:
            print(f"   ⚠️  Some overfitting detected")
        
        # Comparison table
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON - Which generalizes better?")
        print("="*80)
        
        print(f"\n{'Model':<20} {'Val R²':<12} {'Test R²':<12} {'Gap':<10} {'Verdict'}")
        print("-"*80)
        print(f"{'Baseline LightGBM':<20} {'82.85%':<12} {'TBD':<12} {'-':<10} Original")
        print(f"{'Enhanced V2':<20} {f'{val_r2*100:.2f}%':<12} {f'{r2*100:.2f}%':<12} {f'{diff:+.2f}pp':<10} {'✅ THIS ONE' if abs(diff) < 3.0 else '⚠️'}")
        print(f"{'ULTRA':<20} {'85.58%':<12} {'82.17%':<12} {'-3.41pp':<10} ❌ Overfitted")
        
        # Improvement
        baseline_r2 = 0.8285
        improvement = (r2 - baseline_r2) * 100
        
        print(f"\n🎯 Improvement vs Baseline:")
        print(f"   Baseline:  {baseline_r2*100:.2f}%")
        print(f"   Enhanced:  {r2*100:.2f}%")
        print(f"   Gain:      {improvement:+.2f}pp")
        
        # Final verdict
        print("\n" + "="*80)
        print("🏁 FINAL VERDICT - ENHANCED V2 MODEL")
        print("="*80)
        
        if r2 >= 0.83 and abs(diff) < 2.5:
            print("\n✅✅✅ WINNER! This is THE production model! ✅✅✅")
            print("\n✅ Strong test performance")
            print("✅ Good generalization (small val-test gap)")
            print("✅ Better than ULTRA (which overfitted)")
            print("✅ Beats baseline")
            print("\n🚀 DEPLOY THIS MODEL!")
        else:
            print("\n📊 Model performance summary completed")
        
        print("\n" + "="*80)
        
        return {
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'val_r2': val_r2,
            'difference': diff,
            'improvement': improvement
        }

if __name__ == "__main__":
    evaluator = EnhancedTestEvaluator()
    results = evaluator.evaluate()
