"""
Comprehensive Model Testing Script

Tests trained models with various scenarios to ensure prediction accuracy
and identify potential issues.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import json

def load_model(model_path):
    """Load a trained model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_test_data():
    """Load test data"""
    test_path = Path('data/processed/test.parquet')
    if test_path.exists():
        return pd.read_parquet(test_path, engine='pyarrow')
    else:
        raise FileNotFoundError(f"Test data not found at {test_path}")

def create_test_scenarios_from_data(test_df):
    """Create realistic test scenarios from actual data samples"""
    
    # Select representative samples
    scenarios = {}
    
    # Get feature columns (excluding targets)
    feature_cols = [col for col in test_df.columns if col not in ['fare_amount', 'trip_duration']]
    
    # 1. Short Manhattan rush hour trip
    mask = ((test_df['pickup_hour'] >= 17) & (test_df['pickup_hour'] <= 19) & 
            (test_df['is_airport_trip'] == 0) & (test_df['typical_distance'] < 3))
    if mask.sum() > 0:
        sample = test_df[mask].sample(1, random_state=42)
        scenarios['short_manhattan_rush_hour'] = {
            'description': 'Short trip in Manhattan during rush hour',
            'features': sample[feature_cols].iloc[0].to_dict(),
            'expected_fare': (sample['fare_amount'].values[0] * 0.8, sample['fare_amount'].values[0] * 1.2),
            'expected_duration': (sample['trip_duration'].values[0] * 0.8, sample['trip_duration'].values[0] * 1.2)
        }
    
    # 2. Airport trip
    mask = (test_df['is_airport_trip'] == 1)
    if mask.sum() > 0:
        sample = test_df[mask].sample(1, random_state=43)
        scenarios['airport_trip'] = {
            'description': 'Airport trip (JFK/LaGuardia)',
            'features': sample[feature_cols].iloc[0].to_dict(),
            'expected_fare': (sample['fare_amount'].values[0] * 0.8, sample['fare_amount'].values[0] * 1.2),
            'expected_duration': (sample['trip_duration'].values[0] * 0.8, sample['trip_duration'].values[0] * 1.2)
        }
    
    # 3. Cross-borough weekend trip
    mask = ((test_df['is_weekend'] == 1) & (test_df['is_cross_borough'] == 1))
    if mask.sum() > 0:
        sample = test_df[mask].sample(1, random_state=44)
        scenarios['cross_borough_weekend'] = {
            'description': 'Cross-borough trip on weekend',
            'features': sample[feature_cols].iloc[0].to_dict(),
            'expected_fare': (sample['fare_amount'].values[0] * 0.8, sample['fare_amount'].values[0] * 1.2),
            'expected_duration': (sample['trip_duration'].values[0] * 0.8, sample['trip_duration'].values[0] * 1.2)
        }
    
    # 4. Very short trip
    mask = (test_df['typical_distance'] < 1.5)
    if mask.sum() > 0:
        sample = test_df[mask].sample(1, random_state=45)
        scenarios['very_short_trip'] = {
            'description': 'Very short trip (< 1.5 miles)',
            'features': sample[feature_cols].iloc[0].to_dict(),
            'expected_fare': (sample['fare_amount'].values[0] * 0.8, sample['fare_amount'].values[0] * 1.2),
            'expected_duration': (sample['trip_duration'].values[0] * 0.8, sample['trip_duration'].values[0] * 1.2)
        }
    
    # 5. Late night trip
    mask = (test_df['is_night'] == 1)
    if mask.sum() > 0:
        sample = test_df[mask].sample(1, random_state=46)
        scenarios['late_night'] = {
            'description': 'Late night trip',
            'features': sample[feature_cols].iloc[0].to_dict(),
            'expected_fare': (sample['fare_amount'].values[0] * 0.8, sample['fare_amount'].values[0] * 1.2),
            'expected_duration': (sample['trip_duration'].values[0] * 0.8, sample['trip_duration'].values[0] * 1.2)
        }
    
    return scenarios

def create_test_scenarios():
    """Create realistic test scenarios (DEPRECATED - use create_test_scenarios_from_data)"""
    
    scenarios = {
        'example_scenario': {
            'description': 'Example scenario',
            'features': {
                'pickup_hour': 18,
                'pickup_day': 3,
                'pickup_month': 6,
                'is_weekend': 0,
                'is_night': 0,
                'typical_distance': 2.5,
                'is_airport_trip': 0
            },
            'expected_fare': (15, 25),
            'expected_duration': (12, 18)
        },

    }
    
    return scenarios

def test_scenario(models, scenario_name, scenario_data):
    """Test a single scenario"""
    
    print(f"\n{'='*70}")
    print(f"📍 Scenario: {scenario_name.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    print(f"Description: {scenario_data['description']}")
    
    # Create DataFrame from features dict
    X = pd.DataFrame([scenario_data['features']])
    
    print(f"\n📊 Key Input Features:")
    key_features = ['pickup_hour', 'is_weekend', 'is_airport_trip', 'typical_distance', 'typical_duration', 'is_night']
    for feat in key_features:
        if feat in scenario_data['features']:
            print(f"   • {feat}: {scenario_data['features'][feat]}")
    
    # Make predictions
    print(f"\n🎯 Predictions:")
    
    fare_pred = models['fare'].predict(X)[0]
    duration_pred = models['duration'].predict(X)[0]
    
    fare_min, fare_max = scenario_data['expected_fare']
    duration_min, duration_max = scenario_data['expected_duration']
    
    # Check if in expected range
    fare_status = "✅" if fare_min <= fare_pred <= fare_max else "⚠️"
    duration_status = "✅" if duration_min <= duration_pred <= duration_max else "⚠️"
    
    print(f"   {fare_status} Fare Amount:    ${fare_pred:.2f}")
    print(f"      Expected range: ${fare_min:.1f}-${fare_max:.1f}")
    print(f"   {duration_status} Trip Duration:  {duration_pred:.1f} minutes")
    print(f"      Expected range: {duration_min:.1f}-{duration_max:.1f} minutes")
    
    # Calculate derived metrics
    if 'typical_distance' in scenario_data['features'] and scenario_data['features']['typical_distance'] > 0:
        predicted_cost_per_mile = fare_pred / scenario_data['features']['typical_distance']
        predicted_speed = (scenario_data['features']['typical_distance'] / (duration_pred / 60)) if duration_pred > 0 else 0
        
        print(f"\n📈 Derived Metrics:")
        print(f"   • Predicted avg speed: {predicted_speed:.1f} mph")
        print(f"   • Predicted cost per mile: ${predicted_cost_per_mile:.2f}/mile")
    
    # Validation
    validation = {
        'fare_in_range': fare_min <= fare_pred <= fare_max,
        'duration_in_range': duration_min <= duration_pred <= duration_max,
        'fare_prediction': fare_pred,
        'duration_prediction': duration_pred
    }
    
    if validation['fare_in_range'] and validation['duration_in_range']:
        print(f"\n✅ PASS - Predictions within expected ranges")
    else:
        print(f"\n⚠️ WARNING - Predictions outside expected ranges")
    
    return validation

def test_edge_cases(models, test_df):
    """Test edge cases and boundary conditions using actual data"""
    
    print(f"\n{'='*70}")
    print(f"🔍 EDGE CASE TESTING")
    print(f"{'='*70}")
    
    edge_cases = []
    feature_cols = [col for col in test_df.columns if col not in ['fare_amount', 'trip_duration']]
    
    # Test 1: Minimum distance trips
    print(f"\n1️⃣ Minimum Distance Test (shortest trips)")
    min_distance_mask = test_df['typical_distance'] == test_df['typical_distance'].min()
    if min_distance_mask.sum() > 0:
        sample = test_df[min_distance_mask].sample(1, random_state=42)
        X_min = sample[feature_cols]
        
        fare_min = models['fare'].predict(X_min)[0]
        duration_min = models['duration'].predict(X_min)[0]
        
        print(f"   Distance: {sample['typical_distance'].values[0]:.2f} mi")
        print(f"   Actual fare: ${sample['fare_amount'].values[0]:.2f}")
        print(f"   Predicted fare: ${fare_min:.2f}")
        print(f"   Actual duration: {sample['trip_duration'].values[0]:.1f} min")
        print(f"   Predicted duration: {duration_min:.1f} min")
        
        min_fare_ok = 3 <= fare_min <= 25
        min_duration_ok = 1 <= duration_min <= 30
        
        print(f"   {'✅' if min_fare_ok else '⚠️'} Fare reasonableness: {min_fare_ok}")
        print(f"   {'✅' if min_duration_ok else '⚠️'} Duration reasonableness: {min_duration_ok}")
        
        edge_cases.append({
            'test': 'minimum_distance',
            'fare_ok': min_fare_ok,
            'duration_ok': min_duration_ok
        })
    
    # Test 2: Maximum distance trips
    print(f"\n2️⃣ Maximum Distance Test (longest trips)")
    max_distance_mask = test_df['typical_distance'] == test_df['typical_distance'].max()
    if max_distance_mask.sum() > 0:
        sample = test_df[max_distance_mask].sample(1, random_state=42)
        X_max = sample[feature_cols]
        
        fare_max = models['fare'].predict(X_max)[0]
        duration_max = models['duration'].predict(X_max)[0]
        
        print(f"   Distance: {sample['typical_distance'].values[0]:.2f} mi")
        print(f"   Actual fare: ${sample['fare_amount'].values[0]:.2f}")
        print(f"   Predicted fare: ${fare_max:.2f}")
        print(f"   Actual duration: {sample['trip_duration'].values[0]:.1f} min")
        print(f"   Predicted duration: {duration_max:.1f} min")
        
        # More lenient for long trips
        actual_fare = sample['fare_amount'].values[0]
        actual_duration = sample['trip_duration'].values[0]
        max_fare_ok = (0.5 * actual_fare) <= fare_max <= (1.5 * actual_fare)
        max_duration_ok = (0.5 * actual_duration) <= duration_max <= (1.5 * actual_duration)
        
        print(f"   {'✅' if max_fare_ok else '⚠️'} Fare reasonableness: {max_fare_ok}")
        print(f"   {'✅' if max_duration_ok else '⚠️'} Duration reasonableness: {max_duration_ok}")
        
        edge_cases.append({
            'test': 'maximum_distance',
            'fare_ok': max_fare_ok,
            'duration_ok': max_duration_ok
        })
    
    # Test 3: Airport trips
    print(f"\n3️⃣ Airport Trips Test")
    airport_mask = test_df['is_airport_trip'] == 1
    if airport_mask.sum() > 0:
        sample = test_df[airport_mask].sample(1, random_state=42)
        X_airport = sample[feature_cols]
        
        fare_airport = models['fare'].predict(X_airport)[0]
        duration_airport = models['duration'].predict(X_airport)[0]
        
        print(f"   Distance: {sample['typical_distance'].values[0]:.2f} mi")
        print(f"   Actual fare: ${sample['fare_amount'].values[0]:.2f}")
        print(f"   Predicted fare: ${fare_airport:.2f}")
        print(f"   Actual duration: {sample['trip_duration'].values[0]:.1f} min")
        print(f"   Predicted duration: {duration_airport:.1f} min")
        
        actual_fare = sample['fare_amount'].values[0]
        actual_duration = sample['trip_duration'].values[0]
        airport_fare_ok = (0.7 * actual_fare) <= fare_airport <= (1.3 * actual_fare)
        airport_duration_ok = (0.7 * actual_duration) <= duration_airport <= (1.3 * actual_duration)
        
        print(f"   {'✅' if airport_fare_ok else '⚠️'} Fare accuracy: {airport_fare_ok}")
        print(f"   {'✅' if airport_duration_ok else '⚠️'} Duration accuracy: {airport_duration_ok}")
        
        edge_cases.append({
            'test': 'airport_trips',
            'fare_ok': airport_fare_ok,
            'duration_ok': airport_duration_ok
        })
    
    return edge_cases

def test_edge_cases_old(models, all_features):
    """Test edge cases and boundary conditions (OLD VERSION)"""
    
    print(f"\n{'='*70}")
    print(f"🔍 EDGE CASE TESTING")
    print(f"{'='*70}")
    
    edge_cases = []
    
    # Test 1: Minimum values
    print(f"\n1️⃣ Minimum Values Test")
    min_features = {feat: 0 for feat in all_features}
    min_features.update({
        'actual_route_distance': 0.5,
        'actual_route_duration': 2.0,
        'typical_route_distance': 0.5,
        'typical_route_duration': 2.0,
        'distance_ratio': 1.0,
        'duration_ratio': 1.0
    })
    X_min = pd.DataFrame([min_features])
    
    fare_min = models['fare'].predict(X_min)[0]
    duration_min = models['duration'].predict(X_min)[0]
    
    print(f"   Distance: 0.5 mi, Duration: 2.0 min")
    print(f"   Predicted fare: ${fare_min:.2f}")
    print(f"   Predicted duration: {duration_min:.1f} min")
    
    min_fare_ok = 3 <= fare_min <= 15
    min_duration_ok = 1 <= duration_min <= 10
    
    print(f"   {'✅' if min_fare_ok else '⚠️'} Fare reasonableness: {min_fare_ok}")
    print(f"   {'✅' if min_duration_ok else '⚠️'} Duration reasonableness: {min_duration_ok}")
    
    edge_cases.append({
        'test': 'minimum_values',
        'fare_ok': min_fare_ok,
        'duration_ok': min_duration_ok
    })
    
    # Test 2: Maximum values
    print(f"\n2️⃣ Maximum Values Test")
    max_features = {feat: 0 for feat in all_features}
    max_features.update({
        'actual_route_distance': 50.0,
        'actual_route_duration': 120.0,
        'typical_route_distance': 45.0,
        'typical_route_duration': 110.0,
        'distance_ratio': 1.111,
        'duration_ratio': 1.091,
        'is_airport_trip': 1
    })
    X_max = pd.DataFrame([max_features])
    
    fare_max = models['fare'].predict(X_max)[0]
    duration_max = models['duration'].predict(X_max)[0]
    
    print(f"   Distance: 50.0 mi, Duration: 120.0 min")
    print(f"   Predicted fare: ${fare_max:.2f}")
    print(f"   Predicted duration: {duration_max:.1f} min")
    
    max_fare_ok = 100 <= fare_max <= 250
    max_duration_ok = 80 <= duration_max <= 150
    
    print(f"   {'✅' if max_fare_ok else '⚠️'} Fare reasonableness: {max_fare_ok}")
    print(f"   {'✅' if max_duration_ok else '⚠️'} Duration reasonableness: {max_duration_ok}")
    
    edge_cases.append({
        'test': 'maximum_values',
        'fare_ok': max_fare_ok,
        'duration_ok': max_duration_ok
    })
    
    # Test 3: Zero distance
    print(f"\n3️⃣ Zero/Near-Zero Distance Test")
    zero_features = {feat: 0 for feat in all_features}
    zero_features.update({
        'actual_route_distance': 0.1,
        'actual_route_duration': 1.0,
        'typical_route_distance': 0.1,
        'typical_route_duration': 1.0
    })
    X_zero = pd.DataFrame([zero_features])
    
    fare_zero = models['fare'].predict(X_zero)[0]
    duration_zero = models['duration'].predict(X_zero)[0]
    
    print(f"   Distance: 0.1 mi, Duration: 1.0 min")
    print(f"   Predicted fare: ${fare_zero:.2f}")
    print(f"   Predicted duration: {duration_zero:.1f} min")
    
    zero_fare_ok = fare_zero >= 2.5  # Minimum fare
    zero_duration_ok = duration_zero >= 0
    
    print(f"   {'✅' if zero_fare_ok else '⚠️'} Fare ≥ minimum: {zero_fare_ok}")
    print(f"   {'✅' if zero_duration_ok else '⚠️'} Duration ≥ 0: {zero_duration_ok}")
    
    edge_cases.append({
        'test': 'zero_distance',
        'fare_ok': zero_fare_ok,
        'duration_ok': zero_duration_ok
    })
    
    return edge_cases

def test_random_samples(models, test_df, n_samples=100):
    """Test random samples from test set"""
    
    print(f"\n{'='*70}")
    print(f"🎲 RANDOM SAMPLE TESTING ({n_samples} samples)")
    print(f"{'='*70}")
    
    # Select random samples
    sample_df = test_df.sample(n=n_samples, random_state=42)
    
    # Prepare features
    X = sample_df.drop(['fare_amount', 'trip_duration'], axis=1, errors='ignore')
    y_fare = sample_df['fare_amount'] if 'fare_amount' in sample_df.columns else None
    y_duration = sample_df['trip_duration'] if 'trip_duration' in sample_df.columns else None
    
    # Make predictions
    fare_pred = models['fare'].predict(X)
    duration_pred = models['duration'].predict(X)
    
    # Calculate errors
    if y_fare is not None:
        fare_errors = np.abs(fare_pred - y_fare)
        fare_pct_errors = (fare_errors / y_fare) * 100
        
        print(f"\n💰 Fare Prediction Errors:")
        print(f"   • Mean Absolute Error: ${fare_errors.mean():.2f}")
        print(f"   • Median Absolute Error: ${np.median(fare_errors):.2f}")
        print(f"   • Max Absolute Error: ${fare_errors.max():.2f}")
        print(f"   • Mean % Error: {fare_pct_errors.mean():.1f}%")
        print(f"   • Within $5: {(fare_errors <= 5).sum()}/{n_samples} ({(fare_errors <= 5).mean()*100:.1f}%)")
        print(f"   • Within $3: {(fare_errors <= 3).sum()}/{n_samples} ({(fare_errors <= 3).mean()*100:.1f}%)")
    
    if y_duration is not None:
        duration_errors = np.abs(duration_pred - y_duration)
        duration_pct_errors = (duration_errors / y_duration) * 100
        
        print(f"\n⏱️ Duration Prediction Errors:")
        print(f"   • Mean Absolute Error: {duration_errors.mean():.2f} min")
        print(f"   • Median Absolute Error: {np.median(duration_errors):.2f} min")
        print(f"   • Max Absolute Error: {duration_errors.max():.2f} min")
        print(f"   • Mean % Error: {duration_pct_errors.mean():.1f}%")
        print(f"   • Within 10 min: {(duration_errors <= 10).sum()}/{n_samples} ({(duration_errors <= 10).mean()*100:.1f}%)")
        print(f"   • Within 5 min: {(duration_errors <= 5).sum()}/{n_samples} ({(duration_errors <= 5).mean()*100:.1f}%)")
    
    # Identify problematic predictions
    if y_fare is not None and y_duration is not None:
        large_errors = (fare_errors > 10) | (duration_errors > 20)
        n_large_errors = large_errors.sum()
        
        print(f"\n⚠️ Large Errors:")
        print(f"   • Samples with fare error >$10 OR duration error >20min: {n_large_errors} ({n_large_errors/n_samples*100:.1f}%)")
        
        if n_large_errors > 0:
            print(f"\n   Top 5 worst predictions:")
            worst_idx = fare_errors.nlargest(5).index
            for i, idx in enumerate(worst_idx, 1):
                print(f"   {i}. Fare error: ${fare_errors[idx]:.2f}, "
                      f"Duration error: {duration_errors[idx]:.2f} min")
    
    return {
        'fare_mae': fare_errors.mean() if y_fare is not None else None,
        'duration_mae': duration_errors.mean() if y_duration is not None else None,
        'fare_within_5': (fare_errors <= 5).mean() if y_fare is not None else None,
        'duration_within_10': (duration_errors <= 10).mean() if y_duration is not None else None
    }

def test_consistency(models, test_df, n_repeats=5):
    """Test prediction consistency (same input should give same output)"""
    
    print(f"\n{'='*70}")
    print(f"🔁 CONSISTENCY TESTING ({n_repeats} repeats)")
    print(f"{'='*70}")
    
    # Select a single sample
    sample = test_df.sample(n=1, random_state=42)
    X = sample.drop(['fare_amount', 'trip_duration'], axis=1, errors='ignore')
    
    # Make multiple predictions
    fare_preds = []
    duration_preds = []
    
    for i in range(n_repeats):
        fare_preds.append(models['fare'].predict(X)[0])
        duration_preds.append(models['duration'].predict(X)[0])
    
    # Check consistency
    fare_std = np.std(fare_preds)
    duration_std = np.std(duration_preds)
    
    print(f"\n📊 Prediction Consistency:")
    print(f"   • Fare predictions: {fare_preds}")
    print(f"   • Fare std dev: {fare_std:.6f}")
    print(f"   • Duration predictions: {duration_preds}")
    print(f"   • Duration std dev: {duration_std:.6f}")
    
    consistent = (fare_std < 0.001) and (duration_std < 0.001)
    
    if consistent:
        print(f"\n✅ PASS - Predictions are consistent")
    else:
        print(f"\n⚠️ WARNING - Predictions vary across runs")
    
    return consistent

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("="*70)
    
    # Load models
    print("\n📥 Loading models...")
    models = {
        'fare': load_model('src/models/advanced/xgboost_fare_amount.pkl'),
        'duration': load_model('src/models/advanced/xgboost_trip_duration.pkl')
    }
    print("✅ Models loaded successfully")
    
    # Load test data
    print("\n📥 Loading test data...")
    test_df = load_test_data()
    print(f"✅ Loaded {len(test_df)} test samples")
    
    # Test realistic scenarios
    print("\n" + "="*70)
    print("🎯 TESTING REALISTIC SCENARIOS")
    print("="*70)
    
    scenarios = create_test_scenarios_from_data(test_df)
    scenario_results = {}
    
    for name, data in scenarios.items():
        result = test_scenario(models, name, data)
        scenario_results[name] = result
    
    # Calculate pass rate
    scenarios_passed = sum(1 for r in scenario_results.values() 
                          if r['fare_in_range'] and r['duration_in_range'])
    pass_rate = (scenarios_passed / len(scenarios)) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 SCENARIO TEST SUMMARY")
    print(f"{'='*70}")
    print(f"   • Total scenarios: {len(scenarios)}")
    print(f"   • Passed: {scenarios_passed}")
    print(f"   • Failed: {len(scenarios) - scenarios_passed}")
    print(f"   • Pass rate: {pass_rate:.1f}%")
    
    # Test edge cases
    edge_results = test_edge_cases(models, test_df)
    
    edge_passed = sum(1 for r in edge_results if r['fare_ok'] and r['duration_ok'])
    edge_pass_rate = (edge_passed / len(edge_results)) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 EDGE CASE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"   • Total edge cases: {len(edge_results)}")
    print(f"   • Passed: {edge_passed}")
    print(f"   • Failed: {len(edge_results) - edge_passed}")
    print(f"   • Pass rate: {edge_pass_rate:.1f}%")
    
    # Test random samples
    random_results = test_random_samples(models, test_df, n_samples=100)
    
    # Test consistency
    consistency_ok = test_consistency(models, test_df)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🏁 FINAL TEST SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n✅ Tests Passed:")
    print(f"   • Realistic scenarios: {scenarios_passed}/{len(scenarios)} ({pass_rate:.1f}%)")
    print(f"   • Edge cases: {edge_passed}/{len(edge_results)} ({edge_pass_rate:.1f}%)")
    print(f"   • Prediction consistency: {'✅ PASS' if consistency_ok else '⚠️ FAIL'}")
    
    print(f"\n📊 Accuracy Metrics (100 random samples):")
    if random_results['fare_mae']:
        print(f"   • Fare MAE: ${random_results['fare_mae']:.2f}")
        print(f"   • Predictions within $5: {random_results['fare_within_5']*100:.1f}%")
    if random_results['duration_mae']:
        print(f"   • Duration MAE: {random_results['duration_mae']:.2f} min")
        print(f"   • Predictions within 10 min: {random_results['duration_within_10']*100:.1f}%")
    
    overall_pass = (pass_rate >= 80 and edge_pass_rate >= 66 and consistency_ok)
    
    print(f"\n{'='*70}")
    if overall_pass:
        print(f"✅ OVERALL: MODELS PASS COMPREHENSIVE TESTING")
        print(f"   Models are ready for production use!")
    else:
        print(f"⚠️ OVERALL: MODELS NEED IMPROVEMENT")
        print(f"   Some tests failed. Review results above.")
    print(f"{'='*70}")
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'scenario_tests': {
            'total': len(scenarios),
            'passed': scenarios_passed,
            'pass_rate': pass_rate,
            'results': scenario_results
        },
        'edge_case_tests': {
            'total': len(edge_results),
            'passed': edge_passed,
            'pass_rate': edge_pass_rate
        },
        'random_sample_tests': random_results,
        'consistency_test': consistency_ok,
        'overall_pass': overall_pass
    }
    
    output_path = Path('src/models/advanced/test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {output_path}")

if __name__ == "__main__":
    main()
