"""
Final Test Set Evaluation - The Ultimate Validation
Evaluate ULTRA LightGBM model on held-out December test data (never seen during training)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

class FinalTestEvaluator:
    def __init__(self):
        self.data_dir = Path("Data/splits_cleaned/test")
        self.model_path = Path("models/lightgbm_ultra/duration_lightgbm_ultra.txt")
        
    def engineer_ultra_features(self, X):
        """Apply same feature engineering as training."""
        X_enhanced = X.copy()
        
        # All 51 ULTRA features (same as training)
        
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
        
        # ULTRA features
        if 'estimated_distance' in X.columns and 'pickup_hour' in X.columns and 'is_raining' in X.columns:
            X_enhanced['distance_hour_rain'] = X['estimated_distance'] * X['pickup_hour'] * X['is_raining']
        
        if 'estimated_distance' in X.columns and 'is_rush_hour' in X.columns and 'pickup_is_manhattan' in X.columns:
            X_enhanced['distance_rush_manhattan'] = X['estimated_distance'] * X['is_rush_hour'] * X['pickup_is_manhattan']
        
        # Squared terms
        if 'estimated_distance' in X.columns:
            X_enhanced['distance_squared'] = X['estimated_distance'] ** 2
            X_enhanced['distance_sqrt'] = np.sqrt(X['estimated_distance'].clip(lower=0))
        
        if 'pickup_hour' in X.columns:
            X_enhanced['hour_squared'] = X['pickup_hour'] ** 2
        
        # Temperature effects
        if 'temperature' in X.columns:
            X_enhanced['temp_squared'] = X['temperature'] ** 2
            if 'estimated_distance' in X.columns:
                X_enhanced['temp_distance'] = X['temperature'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['temp_rush'] = X['temperature'] * X['is_rush_hour']
        
        # Wind interactions
        if 'wind_speed' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['wind_distance'] = X['wind_speed'] * X['estimated_distance']
            if 'is_rush_hour' in X.columns:
                X_enhanced['wind_rush'] = X['wind_speed'] * X['is_rush_hour']
        
        # Precipitation
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
        
        # Late night
        if 'is_late_night' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['latenight_distance'] = X['is_late_night'] * X['estimated_distance']
            if 'pickup_is_manhattan' in X.columns:
                X_enhanced['latenight_manhattan'] = X['is_late_night'] * X['pickup_is_manhattan']
            if 'is_weekend' in X.columns:
                X_enhanced['latenight_weekend'] = X['is_late_night'] * X['is_weekend']
        
        # Business hours
        if 'is_business_hours' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['business_distance'] = X['is_business_hours'] * X['estimated_distance']
            if 'pickup_is_manhattan' in X.columns:
                X_enhanced['business_manhattan'] = X['is_business_hours'] * X['pickup_is_manhattan']
        
        # Location density
        if 'PULocationID' in X.columns and 'DOLocationID' in X.columns:
            X_enhanced['pickup_density'] = X['PULocationID'] / 265.0
            X_enhanced['dropoff_density'] = X['DOLocationID'] / 265.0
            X_enhanced['route_hash'] = (X['PULocationID'] * 1000 + X['DOLocationID']) % 10000
        
        # Cyclical enhancements
        if 'hour_sin' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['hoursin_distance'] = X['hour_sin'] * X['estimated_distance']
        if 'hour_cos' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['hourcos_distance'] = X['hour_cos'] * X['estimated_distance']
        
        # Extreme conditions
        if 'is_extreme_weather' in X.columns and 'estimated_distance' in X.columns:
            X_enhanced['extreme_distance'] = X['is_extreme_weather'] * X['estimated_distance']
        
        if 'is_poor_visibility' in X.columns and 'is_rush_hour' in X.columns:
            X_enhanced['poorvis_rush'] = X['is_poor_visibility'] * X['is_rush_hour']
        
        # Passenger patterns
        if 'passenger_count' in X.columns:
            if 'pickup_is_manhattan' in X.columns:
                X_enhanced['passengers_manhattan'] = X['passenger_count'] * X['pickup_is_manhattan']
            if 'is_weekend' in X.columns:
                X_enhanced['passengers_weekend'] = X['passenger_count'] * X['is_weekend']
        
        # Holiday week
        if 'is_holiday_week' in X.columns:
            if 'estimated_distance' in X.columns:
                X_enhanced['holidayweek_distance'] = X['is_holiday_week'] * X['estimated_distance']
            if 'pickup_hour' in X.columns:
                X_enhanced['holidayweek_hour'] = X['is_holiday_week'] * X['pickup_hour']
        
        # Ratios
        if 'estimated_distance' in X.columns and 'passenger_count' in X.columns:
            X_enhanced['distance_per_passenger'] = X['estimated_distance'] / (X['passenger_count'] + 1)
        
        return X_enhanced
    
    def load_test_data(self):
        """Load December test data."""
        print("\n📥 Loading TEST data (December - NEVER SEEN)...")
        
        X_files = sorted(self.data_dir.glob('features_*_X.parquet'))
        y_files = sorted(self.data_dir.glob('features_*_y_duration.parquet'))
        
        print(f"   Found {len(X_files)} test files")
        
        X_list = []
        y_list = []
        
        for X_file, y_file in zip(X_files, y_files):
            X_month = pd.read_parquet(X_file)
            y_month = pd.read_parquet(y_file)
            
            X_list.append(X_month)
            y_list.append(y_month)
            print(f"     {X_file.name}: {len(X_month):,} samples")
        
        X_test = pd.concat(X_list, ignore_index=True)
        y_test = pd.concat(y_list, ignore_index=True).values.ravel()
        
        print(f"   Total TEST samples: {len(X_test):,}")
        print(f"   Base features: {len(X_test.columns)}")
        
        return X_test, y_test
    
    def evaluate(self):
        """Run final evaluation."""
        print("="*80)
        print("🏁 FINAL TEST SET EVALUATION - THE MOMENT OF TRUTH!")
        print("="*80)
        print("\n🎯 Purpose: Validate model on UNSEEN December data")
        print("🎯 Success: Test R² ≈ 85.58% (validation performance)")
        print("🎯 This proves: NO overfitting, model generalizes well")
        print("="*80)
        
        # Load model
        print(f"\n📦 Loading ULTRA model from: {self.model_path}")
        if not self.model_path.exists():
            print("❌ ERROR: Model file not found!")
            return
        
        model = lgb.Booster(model_file=str(self.model_path))
        print("   ✅ Model loaded successfully")
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Engineer features
        print("\n🔧 Applying ULTRA feature engineering...")
        X_test_ultra = self.engineer_ultra_features(X_test)
        print(f"   ✅ Total features: {len(X_test_ultra.columns)} (56 base + 51 engineered)")
        
        # Predict
        print("\n🔮 Making predictions on test set...")
        start = time.time()
        y_pred = model.predict(X_test_ultra)
        pred_time = time.time() - start
        
        print(f"   ✅ Predicted {len(y_pred):,} trips in {pred_time:.1f}s")
        print(f"   ⚡ Speed: {len(y_pred)/pred_time:.0f} predictions/second")
        
        # Evaluate
        print("\n📊 Computing metrics...")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Results
        print("\n" + "="*80)
        print("🎉 FINAL TEST SET RESULTS")
        print("="*80)
        
        print(f"\n📈 Test Set Performance (December 2022):")
        print(f"   R² Score:  {r2:.4f} ({r2*100:.2f}%)")
        print(f"   MAE:       {mae:.2f} minutes")
        print(f"   RMSE:      {rmse:.2f} minutes")
        
        print(f"\n📊 Model Performance Summary:")
        print(f"   Training R²:   89.59%")
        print(f"   Validation R²: 85.58%")
        print(f"   Test R²:       {r2*100:.2f}%")
        
        # Analysis
        val_r2 = 0.8558
        diff = (r2 - val_r2) * 100
        
        print(f"\n🔍 Overfitting Analysis:")
        print(f"   Validation R²: {val_r2*100:.2f}%")
        print(f"   Test R²:       {r2*100:.2f}%")
        print(f"   Difference:    {diff:+.2f}pp")
        
        if abs(diff) < 0.5:
            print(f"   ✅ EXCELLENT: Test ≈ Validation (difference <0.5pp)")
            print(f"   ✅ NO overfitting detected!")
            print(f"   ✅ Model generalizes perfectly!")
        elif abs(diff) < 1.0:
            print(f"   ✅ VERY GOOD: Test ≈ Validation (difference <1pp)")
            print(f"   ✅ Minimal overfitting")
            print(f"   ✅ Model generalizes well!")
        elif abs(diff) < 2.0:
            print(f"   ✅ GOOD: Slight difference but acceptable (<2pp)")
            print(f"   ⚠️  Minor overfitting but still production-ready")
        else:
            print(f"   ⚠️  MODERATE: Noticeable difference (≥2pp)")
            print(f"   ⚠️  Some overfitting detected, investigate further")
        
        # Comparison to baseline
        baseline_r2 = 0.8285
        improvement = (r2 - baseline_r2) * 100
        
        print(f"\n🎯 Improvement vs Baseline:")
        print(f"   Baseline:  {baseline_r2*100:.2f}%")
        print(f"   ULTRA:     {r2*100:.2f}%")
        print(f"   Gain:      {improvement:+.2f}pp")
        
        if improvement >= 3.0:
            print(f"   🏆 OUTSTANDING improvement!")
        elif improvement >= 2.0:
            print(f"   🎉 EXCELLENT improvement!")
        elif improvement >= 1.0:
            print(f"   ✅ GOOD improvement!")
        else:
            print(f"   ⚠️  Modest improvement")
        
        # Real-world interpretation
        print(f"\n💡 Real-World Interpretation:")
        print(f"   • Model explains {r2*100:.2f}% of trip duration variance")
        print(f"   • Average prediction error: ±{mae:.1f} minutes")
        print(f"   • For a 20-minute trip: typically ±{mae:.1f} min ({mae/20*100:.1f}%)")
        print(f"   • For a 30-minute trip: typically ±{mae:.1f} min ({mae/30*100:.1f}%)")
        
        # Final verdict
        print("\n" + "="*80)
        print("🏁 FINAL VERDICT")
        print("="*80)
        
        if r2 >= 0.85 and abs(diff) < 1.0:
            print("\n🎉🎉🎉 SUCCESS! Model is PRODUCTION-READY! 🎉🎉🎉")
            print("\n✅ High accuracy (≥85% R²)")
            print("✅ No significant overfitting")
            print("✅ Generalizes to unseen data")
            print("✅ Validated on 3M+ test trips")
            print("\n🚀 Ready for deployment!")
        elif r2 >= 0.83:
            print("\n✅ VERY GOOD! Model is production-ready with minor caveats")
            print("\n✅ Good accuracy (≥83% R²)")
            print("✅ Acceptable generalization")
            print("⚠️  Slight performance variation")
            print("\n🚀 Ready for deployment with monitoring!")
        else:
            print("\n⚠️  NEEDS REVIEW - Further optimization recommended")
            print("\n⚠️  Performance below expectations")
            print("⚠️  Consider additional tuning")
        
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
    evaluator = FinalTestEvaluator()
    results = evaluator.evaluate()
