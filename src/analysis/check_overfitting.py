# -*- coding: utf-8 -*-
"""
Overfitting Detection Script
Analyzes baseline models for signs of overfitting by:
1. Comparing train vs test performance
2. Performing k-fold cross-validation
3. Creating learning curves
4. Analyzing residual distributions
"""

import sys
import io
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.modules.preprocessing_utils import load_processed_data, prepare_features_and_target
from src.modules.model_evaluation import calculate_metrics

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def analyze_overfitting(model_path, scaler_path, target, model_name):
    """
    Comprehensive overfitting analysis for a trained model.
    
    Args:
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
        target (str): Target variable ('fare_amount' or 'trip_duration')
        model_name (str): Name of the model for display
    """
    print("="*70)
    print(f"OVERFITTING ANALYSIS: {model_name.upper()} - {target.upper()}")
    print("="*70)
    
    # Load model and data
    print("\nLoading model and data...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and 'linear' in model_name.lower() else None
    
    train_df = load_processed_data(dataset='train')
    test_df = load_processed_data(dataset='test')
    
    X_train, y_train, _ = prepare_features_and_target(train_df, target)
    X_test, y_test, _ = prepare_features_and_target(test_df, target)
    
    if scaler:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # 1. Train vs Test Performance
    print("\n" + "="*70)
    print("1. TRAIN VS TEST PERFORMANCE")
    print("="*70)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {metric:15s}: {value:.4f}")
        else:
            print(f"   {metric:15s}: {value}")
    
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {metric:15s}: {value:.4f}")
        else:
            print(f"   {metric:15s}: {value}")
    
    # Calculate performance gaps
    print("\n" + "-"*70)
    print("OVERFITTING INDICATORS:")
    print("-"*70)
    
    rmse_gap = test_metrics['RMSE'] - train_metrics['RMSE']
    rmse_gap_pct = (rmse_gap / train_metrics['RMSE']) * 100
    
    mae_gap = test_metrics['MAE'] - train_metrics['MAE']
    mae_gap_pct = (mae_gap / train_metrics['MAE']) * 100
    
    r2_gap = train_metrics['R²'] - test_metrics['R²']
    r2_gap_pct = (r2_gap / train_metrics['R²']) * 100
    
    print(f"\nRMSE Gap: {rmse_gap:+.4f} ({rmse_gap_pct:+.2f}%)")
    print(f"MAE Gap:  {mae_gap:+.4f} ({mae_gap_pct:+.2f}%)")
    print(f"R² Gap:   {r2_gap:+.4f} ({r2_gap_pct:+.2f}%)")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    overfitting_score = 0
    
    if rmse_gap_pct > 10:
        print("   ⚠️  RMSE: Test is >10% worse than train - MODERATE OVERFITTING")
        overfitting_score += 1
    elif rmse_gap_pct > 5:
        print("   ⚠️  RMSE: Test is 5-10% worse than train - SLIGHT OVERFITTING")
        overfitting_score += 0.5
    else:
        print("   ✅ RMSE: Test within 5% of train - GOOD GENERALIZATION")
    
    if mae_gap_pct > 10:
        print("   ⚠️  MAE: Test is >10% worse than train - MODERATE OVERFITTING")
        overfitting_score += 1
    elif mae_gap_pct > 5:
        print("   ⚠️  MAE: Test is 5-10% worse than train - SLIGHT OVERFITTING")
        overfitting_score += 0.5
    else:
        print("   ✅ MAE: Test within 5% of train - GOOD GENERALIZATION")
    
    if r2_gap_pct > 2:
        print("   ⚠️  R²: Test is >2% worse than train - MODERATE OVERFITTING")
        overfitting_score += 1
    elif r2_gap_pct > 1:
        print("   ⚠️  R²: Test is 1-2% worse than train - SLIGHT OVERFITTING")
        overfitting_score += 0.5
    else:
        print("   ✅ R²: Test within 1% of train - EXCELLENT GENERALIZATION")
    
    print("\n" + "-"*70)
    if overfitting_score == 0:
        print("🎉 VERDICT: NO OVERFITTING - Model generalizes well!")
    elif overfitting_score < 1.5:
        print("✅ VERDICT: SLIGHT OVERFITTING - Acceptable for production")
    elif overfitting_score < 2.5:
        print("⚠️  VERDICT: MODERATE OVERFITTING - Consider regularization")
    else:
        print("❌ VERDICT: SEVERE OVERFITTING - Model needs improvement")
    print("-"*70)
    
    # 2. Cross-Validation Analysis
    print("\n" + "="*70)
    print("2. CROSS-VALIDATION ANALYSIS (5-Fold)")
    print("="*70)
    
    # Use original (unscaled) data for CV
    X_train_orig, y_train_orig, _ = prepare_features_and_target(train_df, target)
    
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_orig, y_train_orig, 
                                 cv=5, scoring='r2', n_jobs=-1)
    
    cv_rmse_scores = -cross_val_score(model, X_train_orig, y_train_orig,
                                       cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    print(f"\nR² Scores per Fold:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    
    print(f"\nCross-Validation Summary:")
    print(f"   Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Mean RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
    print(f"   Min R²: {cv_scores.min():.4f}")
    print(f"   Max R²: {cv_scores.max():.4f}")
    
    # Check CV stability
    cv_stability = cv_scores.std() / cv_scores.mean() * 100
    print(f"\n   CV Stability (CoV): {cv_stability:.2f}%")
    
    if cv_stability < 1:
        print("   ✅ Very stable across folds - Excellent!")
    elif cv_stability < 3:
        print("   ✅ Stable across folds - Good")
    elif cv_stability < 5:
        print("   ⚠️  Moderate variation - Acceptable")
    else:
        print("   ❌ High variation - Model may be unstable")
    
    # Compare CV mean with test R²
    cv_test_gap = abs(cv_scores.mean() - test_metrics['R²'])
    print(f"\n   CV Mean vs Test R²: {cv_test_gap:.4f} difference")
    
    if cv_test_gap < 0.01:
        print("   ✅ CV matches test performance - No overfitting!")
    elif cv_test_gap < 0.02:
        print("   ✅ CV close to test performance - Good generalization")
    else:
        print("   ⚠️  CV differs from test - Check for data issues")
    
    # 3. Learning Curves
    print("\n" + "="*70)
    print("3. LEARNING CURVE ANALYSIS")
    print("="*70)
    
    print("\nGenerating learning curves (this may take a minute)...")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_orig, y_train_orig,
        train_sizes=train_sizes,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)
    
    print(f"\nLearning Curve Summary:")
    print(f"   With 10% data: Train R²={train_scores_mean[0]:.4f}, Test R²={test_scores_mean[0]:.4f}")
    print(f"   With 50% data: Train R²={train_scores_mean[4]:.4f}, Test R²={test_scores_mean[4]:.4f}")
    print(f"   With 100% data: Train R²={train_scores_mean[-1]:.4f}, Test R²={test_scores_mean[-1]:.4f}")
    
    # Check convergence
    gap_at_end = train_scores_mean[-1] - test_scores_mean[-1]
    gap_at_mid = train_scores_mean[4] - test_scores_mean[4]
    
    print(f"\n   Train-Test Gap at 50%: {gap_at_mid:.4f}")
    print(f"   Train-Test Gap at 100%: {gap_at_end:.4f}")
    
    if gap_at_end < gap_at_mid:
        print("   ✅ Gap is decreasing - More data helps!")
    else:
        print("   ⚠️  Gap is stable/increasing - More data may not help")
    
    # 4. Create Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Train vs Test Metrics Comparison
    metrics_names = ['RMSE', 'MAE', 'R²']
    train_vals = [train_metrics['RMSE'], train_metrics['MAE'], train_metrics['R²']]
    test_vals = [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['R²']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_vals, width, label='Train', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_vals, width, label='Test', color='coral', alpha=0.8)
    axes[0, 0].set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'{model_name} - Train vs Test Performance', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (train_val, test_val) in enumerate(zip(train_vals, test_vals)):
        axes[0, 0].text(i - width/2, train_val + 0.01, f'{train_val:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[0, 0].text(i + width/2, test_val + 0.01, f'{test_val:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Cross-Validation Scores
    axes[0, 1].boxplot([cv_scores], labels=['5-Fold CV'])
    axes[0, 1].axhline(y=test_metrics['R²'], color='red', linestyle='--', 
                      linewidth=2, label=f'Test R² ({test_metrics["R²"]:.4f})')
    axes[0, 1].axhline(y=train_metrics['R²'], color='blue', linestyle='--', 
                      linewidth=2, label=f'Train R² ({train_metrics["R²"]:.4f})')
    axes[0, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Cross-Validation Stability', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Learning Curves
    axes[1, 0].plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', 
                    linewidth=2, markersize=8, label='Training score')
    axes[1, 0].fill_between(train_sizes_abs, 
                            train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, 
                            alpha=0.2, color='blue')
    
    axes[1, 0].plot(train_sizes_abs, test_scores_mean, 'o-', color='red', 
                    linewidth=2, markersize=8, label='Cross-validation score')
    axes[1, 0].fill_between(train_sizes_abs, 
                            test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, 
                            alpha=0.2, color='red')
    
    axes[1, 0].set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Learning Curves', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='lower right', fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Residual Distribution Comparison
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    axes[1, 1].hist(train_residuals, bins=50, alpha=0.6, label='Train Residuals', 
                   color='blue', density=True)
    axes[1, 1].hist(test_residuals, bins=50, alpha=0.6, label='Test Residuals', 
                   color='red', density=True)
    axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residual Value', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Residual Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = f'../models/plots/overfitting_analysis_{target}_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved overfitting analysis plot: {output_path}")
    plt.close()
    
    return {
        'model_name': model_name,
        'target': target,
        'train_r2': train_metrics['R²'],
        'test_r2': test_metrics['R²'],
        'cv_mean_r2': cv_scores.mean(),
        'cv_std_r2': cv_scores.std(),
        'overfitting_score': overfitting_score,
        'rmse_gap_pct': rmse_gap_pct,
        'r2_gap': r2_gap
    }


def main():
    """Main function to analyze all models."""
    print("\n" + "="*70)
    print("NYC TAXI MODELS - OVERFITTING ANALYSIS")
    print("="*70)
    print("\nThis script analyzes baseline models for overfitting by:")
    print("  1. Comparing train vs test performance")
    print("  2. Running k-fold cross-validation")
    print("  3. Generating learning curves")
    print("  4. Analyzing residual distributions")
    print("\n" + "="*70)
    
    results = []
    
    # Get absolute paths to models
    models_dir = project_root / 'src' / 'models' / 'baseline'
    
    # Analyze Fare Amount Models
    print("\n\n")
    results.append(analyze_overfitting(
        model_path=str(models_dir / 'linear_regression_fare_amount.pkl'),
        scaler_path=str(models_dir / 'scaler_fare_amount.pkl'),
        target='fare_amount',
        model_name='Linear Regression'
    ))
    
    print("\n\n")
    results.append(analyze_overfitting(
        model_path=str(models_dir / 'decision_tree_fare_amount.pkl'),
        scaler_path=None,
        target='fare_amount',
        model_name='Decision Tree'
    ))
    
    # Analyze Trip Duration Models
    print("\n\n")
    results.append(analyze_overfitting(
        model_path=str(models_dir / 'linear_regression_trip_duration.pkl'),
        scaler_path=str(models_dir / 'scaler_trip_duration.pkl'),
        target='trip_duration',
        model_name='Linear Regression'
    ))
    
    print("\n\n")
    results.append(analyze_overfitting(
        model_path=str(models_dir / 'decision_tree_trip_duration.pkl'),
        scaler_path=None,
        target='trip_duration',
        model_name='Decision Tree'
    ))
    
    # Final Summary
    print("\n\n")
    print("="*70)
    print("FINAL SUMMARY - ALL MODELS")
    print("="*70)
    
    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    for result in results:
        model_target = f"{result['model_name']} ({result['target']})"
        print(f"\n{model_target}:")
        
        if result['overfitting_score'] == 0:
            print("  ✅ No overfitting detected - Ready for production!")
        elif result['overfitting_score'] < 1.5:
            print("  ✅ Slight overfitting - Acceptable, but monitor in production")
        elif result['overfitting_score'] < 2.5:
            print("  ⚠️  Moderate overfitting - Consider:")
            print("     • Regularization (L1/L2 for Linear, max_depth for Tree)")
            print("     • More training data")
            print("     • Feature selection")
        else:
            print("  ❌ Severe overfitting - Action needed:")
            print("     • Strong regularization")
            print("     • Reduce model complexity")
            print("     • Check for data leakage")
    
    print("\n" + "="*70)
    print("✅ OVERFITTING ANALYSIS COMPLETE!")
    print("="*70)
    print("\nVisualization plots saved to: models/plots/")


if __name__ == "__main__":
    main()
