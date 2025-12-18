"""
Comprehensive Error Analysis Script

Analyzes prediction errors to identify patterns, systematic biases,
and areas for model improvement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

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

def calculate_errors(y_true, y_pred):
    """Calculate various error metrics"""
    
    errors = {
        'raw_error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true),
        'squared_error': (y_pred - y_true) ** 2,
        'pct_error': ((y_pred - y_true) / y_true) * 100,
        'abs_pct_error': (np.abs(y_pred - y_true) / y_true) * 100
    }
    
    return pd.DataFrame(errors)

def analyze_error_distribution(errors_df, target_name):
    """Analyze error distribution statistics"""
    
    print(f"\n{'='*70}")
    print(f"📊 ERROR DISTRIBUTION ANALYSIS - {target_name.upper()}")
    print(f"{'='*70}")
    
    raw_errors = errors_df['raw_error']
    abs_errors = errors_df['abs_error']
    pct_errors = errors_df['pct_error']
    
    print(f"\n🔢 Raw Error Statistics:")
    print(f"   • Mean: {raw_errors.mean():.4f}")
    print(f"   • Median: {raw_errors.median():.4f}")
    print(f"   • Std Dev: {raw_errors.std():.4f}")
    print(f"   • Min: {raw_errors.min():.4f}")
    print(f"   • Max: {raw_errors.max():.4f}")
    print(f"   • Skewness: {raw_errors.skew():.4f}")
    print(f"   • Kurtosis: {raw_errors.kurtosis():.4f}")
    
    print(f"\n📏 Absolute Error Statistics:")
    print(f"   • Mean (MAE): {abs_errors.mean():.4f}")
    print(f"   • Median: {abs_errors.median():.4f}")
    print(f"   • 25th percentile: {abs_errors.quantile(0.25):.4f}")
    print(f"   • 75th percentile: {abs_errors.quantile(0.75):.4f}")
    print(f"   • 90th percentile: {abs_errors.quantile(0.90):.4f}")
    print(f"   • 95th percentile: {abs_errors.quantile(0.95):.4f}")
    print(f"   • 99th percentile: {abs_errors.quantile(0.99):.4f}")
    
    print(f"\n📈 Percentage Error Statistics:")
    print(f"   • Mean Absolute % Error: {pct_errors.abs().mean():.2f}%")
    print(f"   • Median Absolute % Error: {pct_errors.abs().median():.2f}%")
    print(f"   • % within 5%: {(pct_errors.abs() <= 5).mean()*100:.1f}%")
    print(f"   • % within 10%: {(pct_errors.abs() <= 10).mean()*100:.1f}%")
    print(f"   • % within 20%: {(pct_errors.abs() <= 20).mean()*100:.1f}%")
    
    # Bias analysis
    positive_errors = (raw_errors > 0).sum()
    negative_errors = (raw_errors < 0).sum()
    
    print(f"\n🎯 Bias Analysis:")
    print(f"   • Over-predictions: {positive_errors} ({positive_errors/len(raw_errors)*100:.1f}%)")
    print(f"   • Under-predictions: {negative_errors} ({negative_errors/len(raw_errors)*100:.1f}%)")
    
    if abs(raw_errors.mean()) > 0.01:
        bias_direction = "over-predicting" if raw_errors.mean() > 0 else "under-predicting"
        print(f"   • ⚠️ Model shows systematic bias: {bias_direction}")
    else:
        print(f"   • ✅ No significant systematic bias detected")
    
    # Statistical tests
    print(f"\n🔬 Statistical Tests:")
    
    # Normality test
    _, p_value_norm = stats.normaltest(raw_errors)
    print(f"   • Normality test (p-value): {p_value_norm:.6f}")
    if p_value_norm > 0.05:
        print(f"     ✅ Errors are approximately normally distributed")
    else:
        print(f"     ⚠️ Errors are NOT normally distributed")
    
    # Zero-mean test
    _, p_value_zero = stats.ttest_1samp(raw_errors, 0)
    print(f"   • Zero-mean test (p-value): {p_value_zero:.6f}")
    if p_value_zero > 0.05:
        print(f"     ✅ Errors have zero mean (no bias)")
    else:
        print(f"     ⚠️ Errors do NOT have zero mean (bias present)")

def analyze_errors_by_feature(df, errors_df, feature, target_name, bins=5):
    """Analyze errors by feature value"""
    
    if feature not in df.columns:
        return None
    
    # Create bins for continuous features
    if df[feature].dtype in ['float64', 'int64'] and df[feature].nunique() > 10:
        df[f'{feature}_bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
        group_col = f'{feature}_bin'
    else:
        group_col = feature
    
    # Group errors
    grouped = errors_df.groupby(df[group_col]).agg({
        'abs_error': ['mean', 'median', 'count'],
        'raw_error': 'mean'
    }).round(4)
    
    return grouped

def identify_problematic_predictions(df, errors_df, target_name, threshold_percentile=95):
    """Identify predictions with largest errors"""
    
    print(f"\n{'='*70}")
    print(f"🚨 PROBLEMATIC PREDICTIONS - {target_name.upper()}")
    print(f"{'='*70}")
    
    threshold = errors_df['abs_error'].quantile(threshold_percentile / 100)
    
    large_errors = errors_df['abs_error'] >= threshold
    n_large = large_errors.sum()
    
    print(f"\n⚠️ Large Errors (≥{threshold_percentile}th percentile: {threshold:.4f}):")
    print(f"   • Number of samples: {n_large} ({n_large/len(errors_df)*100:.1f}%)")
    
    # Analyze features of problematic predictions
    problematic_df = df[large_errors].copy()
    problematic_errors = errors_df[large_errors].copy()
    
    print(f"\n📊 Characteristics of Problematic Predictions:")
    
    # Numeric features
    numeric_features = ['actual_route_distance', 'actual_route_duration', 
                       'pickup_hour', 'is_airport_trip', 'is_rush_hour']
    
    for feat in numeric_features:
        if feat in problematic_df.columns:
            prob_mean = problematic_df[feat].mean()
            overall_mean = df[feat].mean()
            diff_pct = ((prob_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0
            
            print(f"   • {feat}:")
            print(f"     - Problematic avg: {prob_mean:.2f}")
            print(f"     - Overall avg: {overall_mean:.2f}")
            print(f"     - Difference: {diff_pct:+.1f}%")
    
    # Top 10 worst predictions
    print(f"\n🔴 Top 10 Worst Predictions:")
    worst_idx = errors_df['abs_error'].nlargest(10).index
    
    for i, idx in enumerate(worst_idx, 1):
        error = errors_df.loc[idx, 'abs_error']
        pct_error = errors_df.loc[idx, 'abs_pct_error']
        
        features_str = ""
        if 'actual_route_distance' in df.columns:
            features_str += f"dist={df.loc[idx, 'actual_route_distance']:.1f}mi, "
        if 'actual_route_duration' in df.columns:
            features_str += f"dur={df.loc[idx, 'actual_route_duration']:.0f}min"
        
        print(f"   {i}. Error: {error:.2f} ({pct_error:.1f}%) | {features_str}")
    
    return problematic_df, problematic_errors

def plot_error_analysis(df, errors_df, y_true, y_pred, target_name):
    """Create comprehensive error analysis plots"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.3, s=20, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax1.set_xlabel(f'Actual {target_name.title()}', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'Predicted {target_name.title()}', fontsize=11, fontweight='bold')
    ax1.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = 1 - (errors_df['squared_error'].sum() / ((y_true - y_true.mean())**2).sum())
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
            fontsize=11, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residual plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, errors_df['raw_error'], alpha=0.3, s=20, edgecolors='none')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel(f'Predicted {target_name.title()}', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(errors_df['raw_error'], bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    ax3.axvline(x=errors_df['raw_error'].mean(), color='g', linestyle='--', lw=2, 
               label=f'Mean: {errors_df["raw_error"].mean():.3f}')
    ax3.set_xlabel('Error', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Absolute error by distance
    if 'actual_route_distance' in df.columns:
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Create bins
        df['distance_bin'] = pd.qcut(df['actual_route_distance'], q=10, duplicates='drop')
        bin_errors = errors_df.groupby(df['distance_bin'])['abs_error'].agg(['mean', 'median'])
        
        x_pos = range(len(bin_errors))
        ax4.bar(x_pos, bin_errors['mean'], alpha=0.7, label='Mean', edgecolor='black')
        ax4.plot(x_pos, bin_errors['median'], 'r-o', lw=2, label='Median', markersize=8)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([str(x).split(',')[0][1:] + '-' + str(x).split(',')[1][:-1] 
                            for x in bin_errors.index], rotation=45, ha='right', fontsize=9)
        ax4.set_xlabel('Distance (miles)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
        ax4.set_title('Error by Distance', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Absolute error by duration
    if 'actual_route_duration' in df.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        
        df['duration_bin'] = pd.qcut(df['actual_route_duration'], q=10, duplicates='drop')
        bin_errors = errors_df.groupby(df['duration_bin'])['abs_error'].agg(['mean', 'median'])
        
        x_pos = range(len(bin_errors))
        ax5.bar(x_pos, bin_errors['mean'], alpha=0.7, label='Mean', edgecolor='black')
        ax5.plot(x_pos, bin_errors['median'], 'r-o', lw=2, label='Median', markersize=8)
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([str(x).split(',')[0][1:] + '-' + str(x).split(',')[1][:-1] 
                            for x in bin_errors.index], rotation=45, ha='right', fontsize=9)
        ax5.set_xlabel('Duration (minutes)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
        ax5.set_title('Error by Duration', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Error by hour
    if 'pickup_hour' in df.columns:
        ax6 = fig.add_subplot(gs[1, 2])
        
        hour_errors = errors_df.groupby(df['pickup_hour'])['abs_error'].agg(['mean', 'median'])
        
        ax6.bar(hour_errors.index, hour_errors['mean'], alpha=0.7, label='Mean', edgecolor='black')
        ax6.plot(hour_errors.index, hour_errors['median'], 'r-o', lw=2, label='Median', markersize=6)
        
        ax6.set_xlabel('Pickup Hour', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
        ax6.set_title('Error by Hour of Day', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(0, 24, 3))
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Q-Q plot
    ax7 = fig.add_subplot(gs[2, 0])
    stats.probplot(errors_df['raw_error'], dist="norm", plot=ax7)
    ax7.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Percentage error distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(errors_df['pct_error'].clip(-100, 100), bins=50, edgecolor='black', alpha=0.7)
    ax8.axvline(x=0, color='r', linestyle='--', lw=2)
    ax8.set_xlabel('Percentage Error (%)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Error by trip type
    if 'is_airport_trip' in df.columns:
        ax9 = fig.add_subplot(gs[2, 2])
        
        trip_types = {
            'Airport': df['is_airport_trip'] == 1,
            'Non-Airport': df['is_airport_trip'] == 0
        }
        
        trip_errors = []
        labels = []
        for name, mask in trip_types.items():
            if mask.sum() > 0:
                trip_errors.append(errors_df[mask]['abs_error'])
                labels.append(f"{name}\n(n={mask.sum()})")
        
        bp = ax9.boxplot(trip_errors, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax9.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
        ax9.set_title('Error by Trip Type', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Comprehensive Error Analysis - {target_name.title()}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    output_dir = Path('docs/presentation_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'error_analysis_{target_name.lower()}.png'
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Error analysis plot saved to: {output_path}")
    
    plt.show()

def generate_error_summary_report(df, errors_df, y_true, y_pred, target_name):
    """Generate a comprehensive error summary report"""
    
    print(f"\n{'='*70}")
    print(f"📄 ERROR ANALYSIS SUMMARY REPORT - {target_name.upper()}")
    print(f"{'='*70}")
    
    # Overall metrics
    mae = errors_df['abs_error'].mean()
    rmse = np.sqrt(errors_df['squared_error'].mean())
    mape = errors_df['abs_pct_error'].mean()
    r2 = 1 - (errors_df['squared_error'].sum() / ((y_true - y_true.mean())**2).sum())
    
    print(f"\n📊 Overall Performance Metrics:")
    print(f"   • R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   • Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   • Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"   • Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Error ranges
    print(f"\n📏 Error Ranges:")
    
    if target_name == 'fare':
        thresholds = [1, 3, 5, 10]
        for t in thresholds:
            pct = (errors_df['abs_error'] <= t).mean() * 100
            print(f"   • Within ${t}: {pct:.1f}%")
    else:  # duration
        thresholds = [2, 5, 10, 15]
        for t in thresholds:
            pct = (errors_df['abs_error'] <= t).mean() * 100
            print(f"   • Within {t} min: {pct:.1f}%")
    
    # Bias analysis
    mean_error = errors_df['raw_error'].mean()
    bias_pct = (errors_df['raw_error'] > 0).mean() * 100
    
    print(f"\n🎯 Bias Analysis:")
    print(f"   • Mean error: {mean_error:.4f}")
    print(f"   • Over-predictions: {bias_pct:.1f}%")
    print(f"   • Under-predictions: {100-bias_pct:.1f}%")
    
    if abs(mean_error) < 0.1:
        print(f"   • ✅ No significant bias")
    else:
        print(f"   • ⚠️ Bias detected: model tends to {'over' if mean_error > 0 else 'under'}-predict")
    
    # Worst performing segments
    print(f"\n🚨 Segments with Highest Errors:")
    
    if 'actual_route_distance' in df.columns:
        df['distance_quartile'] = pd.qcut(df['actual_route_distance'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_errors = errors_df.groupby(df['distance_quartile'])['abs_error'].mean().sort_values(ascending=False)
        
        print(f"   • By distance:")
        for q, err in quartile_errors.head(2).items():
            print(f"     - {q}: MAE = {err:.4f}")
    
    if 'pickup_hour' in df.columns:
        hour_errors = errors_df.groupby(df['pickup_hour'])['abs_error'].mean().sort_values(ascending=False)
        
        print(f"   • By hour:")
        for hour, err in hour_errors.head(3).items():
            print(f"     - Hour {hour}: MAE = {err:.4f}")
    
    # Best performing segments
    print(f"\n✅ Segments with Lowest Errors:")
    
    if 'actual_route_distance' in df.columns:
        print(f"   • By distance:")
        for q, err in quartile_errors.tail(2).items():
            print(f"     - {q}: MAE = {err:.4f}")
    
    if 'pickup_hour' in df.columns:
        print(f"   • By hour:")
        for hour, err in hour_errors.tail(3).items():
            print(f"     - Hour {hour}: MAE = {err:.4f}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if r2 < 0.90:
        print(f"   • ⚠️ R² below 90% - consider additional feature engineering")
    
    if mape > 15:
        print(f"   • ⚠️ MAPE > 15% - model struggles with percentage accuracy")
    
    if abs(mean_error) > 0.1:
        print(f"   • ⚠️ Systematic bias detected - review training data distribution")
    
    # Identify outliers
    outlier_threshold = errors_df['abs_error'].quantile(0.95)
    n_outliers = (errors_df['abs_error'] > outlier_threshold).sum()
    
    if n_outliers > len(errors_df) * 0.05:
        print(f"   • ⚠️ {n_outliers} outliers ({n_outliers/len(errors_df)*100:.1f}%) - investigate edge cases")
    
    if r2 > 0.93 and mape < 10 and abs(mean_error) < 0.1:
        print(f"   • ✅ Model performance is excellent!")
        print(f"   • ✅ Ready for production deployment")

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("🔍 COMPREHENSIVE ERROR ANALYSIS")
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
    
    # Analyze each target
    for target_name in ['fare', 'duration']:
        
        print(f"\n{'='*70}")
        print(f"ANALYZING {target_name.upper()} PREDICTIONS")
        print(f"{'='*70}")
        
        # Get actual target name
        target_col = 'fare_amount' if target_name == 'fare' else 'trip_duration'
        
        if target_col not in test_df.columns:
            print(f"⚠️ Target column '{target_col}' not found. Skipping...")
            continue
        
        # Prepare data
        X = test_df.drop(['fare_amount', 'trip_duration'], axis=1, errors='ignore')
        y_true = test_df[target_col]
        
        # Make predictions
        print(f"🔮 Making predictions...")
        y_pred = models[target_name].predict(X)
        
        # Calculate errors
        print(f"📊 Calculating errors...")
        errors_df = calculate_errors(y_true, y_pred)
        
        # Analyze error distribution
        analyze_error_distribution(errors_df, target_name)
        
        # Identify problematic predictions
        problematic_df, problematic_errors = identify_problematic_predictions(
            test_df, errors_df, target_name, threshold_percentile=95
        )
        
        # Create visualizations
        print(f"\n📈 Creating error analysis plots...")
        plot_error_analysis(test_df, errors_df, y_true, y_pred, target_name)
        
        # Generate summary report
        generate_error_summary_report(test_df, errors_df, y_true, y_pred, target_name)
        
        # Save detailed errors
        output_dir = Path('src/models/advanced')
        errors_output = test_df.copy()
        errors_output['predicted'] = y_pred
        errors_output['actual'] = y_true
        errors_output['error'] = errors_df['raw_error']
        errors_output['abs_error'] = errors_df['abs_error']
        errors_output['pct_error'] = errors_df['pct_error']
        
        csv_path = output_dir / f'error_analysis_{target_name}.csv'
        errors_output.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed errors saved to: {csv_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ ERROR ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Outputs generated:")
    print(f"   • Error distribution statistics")
    print(f"   • Problematic prediction identification")
    print(f"   • Comprehensive error plots (PNG)")
    print(f"   • Summary reports with recommendations")
    print(f"   • Detailed error data (CSV)")
    print(f"\n📂 Location: docs/presentation_plots/ and src/models/advanced/")

if __name__ == "__main__":
    main()
