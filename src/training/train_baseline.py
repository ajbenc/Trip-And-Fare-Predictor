# -*- coding: utf-8 -*-
"""
Baseline Model Training Script
==============================

Train baseline models (Linear Regression and Decision Tree) for:
1. Fare Amount Prediction
2. Trip Duration Prediction

This script establishes baseline performance metrics that advanced models should beat.

Author: Julian
Date: October 2025
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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.modules.preprocessing_utils import load_processed_data, prepare_features_and_target
from src.modules.model_evaluation import calculate_metrics

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def train_baseline_models(target='fare_amount'):
    """
    Train Linear Regression and Decision Tree baseline models.
    
    Args:
        target (str): 'fare_amount' or 'trip_duration'
        
    Returns:
        dict: Dictionary containing trained models and metrics
    """
    print("="*70)
    print(f"BASELINE MODEL TRAINING: {target.upper().replace('_', ' ')}")
    print("="*70)
    
    # Load data
    print("\n📥 Loading preprocessed data...")
    train_df = load_processed_data(dataset='train')
    test_df = load_processed_data(dataset='test')
    
    # Prepare features and target
    target_type = 'fare' if 'fare' in target else 'duration'
    X_train, y_train, features = prepare_features_and_target(train_df, target=target)
    X_test, y_test, _ = prepare_features_and_target(test_df, target=target)
    
    print(f"\n✅ Data prepared:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing: {len(X_test):,} samples")
    print(f"   Features: {len(features)} columns")
    
    results = {}
    
    # ========================================================================
    # MODEL 1: LINEAR REGRESSION
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 1: LINEAR REGRESSION")
    print("="*70)
    
    print("\n⚙️  Training Linear Regression...")
    
    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train_lr = lr_model.predict(X_train_scaled)
    y_pred_test_lr = lr_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_metrics_lr = calculate_metrics(y_train, y_pred_train_lr, "Linear Regression (Train)")
    test_metrics_lr = calculate_metrics(y_test, y_pred_test_lr, "Linear Regression (Test)")
    
    print(f"\n📊 Linear Regression Results:")
    print(f"   Training Set:")
    print(f"      RMSE: {train_metrics_lr['RMSE']:.4f}")
    print(f"      MAE:  {train_metrics_lr['MAE']:.4f}")
    print(f"      R²:   {train_metrics_lr['R²']:.4f}")
    
    print(f"\n   Test Set:")
    print(f"      RMSE: {test_metrics_lr['RMSE']:.4f}")
    print(f"      MAE:  {test_metrics_lr['MAE']:.4f}")
    print(f"      R²:   {test_metrics_lr['R²']:.4f}")
    
    results['linear_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'train_metrics': train_metrics_lr,
        'test_metrics': test_metrics_lr,
        'predictions': y_pred_test_lr
    }
    
    # ========================================================================
    # MODEL 2: DECISION TREE
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 2: DECISION TREE REGRESSOR")
    print("="*70)
    
    print("\n⚙️  Training Decision Tree (with regularization to prevent overfitting)...")
    
    # Train model (no scaling needed for tree-based models)
    # Added stronger regularization to reduce overfitting:
    # - max_depth: 8 (was 10) - shallower tree
    # - min_samples_split: 200 (was 100) - require more samples to split
    # - min_samples_leaf: 100 (was 50) - require more samples per leaf
    # - max_features: 0.8 - use only 80% of features for each split
    # - min_impurity_decrease: require minimum improvement to split
    dt_model = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=200,
        min_samples_leaf=100,
        max_features=0.8,
        min_impurity_decrease=0.001,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train_dt = dt_model.predict(X_train)
    y_pred_test_dt = dt_model.predict(X_test)
    
    # Calculate metrics
    train_metrics_dt = calculate_metrics(y_train, y_pred_train_dt, "Decision Tree (Train)")
    test_metrics_dt = calculate_metrics(y_test, y_pred_test_dt, "Decision Tree (Test)")
    
    print(f"\n📊 Decision Tree Results:")
    print(f"   Training Set:")
    print(f"      RMSE: {train_metrics_dt['RMSE']:.4f}")
    print(f"      MAE:  {train_metrics_dt['MAE']:.4f}")
    print(f"      R²:   {train_metrics_dt['R²']:.4f}")
    
    print(f"\n   Test Set:")
    print(f"      RMSE: {test_metrics_dt['RMSE']:.4f}")
    print(f"      MAE:  {test_metrics_dt['MAE']:.4f}")
    print(f"      R²:   {test_metrics_dt['R²']:.4f}")
    
    results['decision_tree'] = {
        'model': dt_model,
        'train_metrics': train_metrics_dt,
        'test_metrics': test_metrics_dt,
        'predictions': y_pred_test_dt
    }
    
    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("BASELINE MODEL COMPARISON")
    print("="*70)
    
    comparison_data = {
        'Model': ['Linear Regression', 'Decision Tree'],
        'Train RMSE': [train_metrics_lr['RMSE'], train_metrics_dt['RMSE']],
        'Test RMSE': [test_metrics_lr['RMSE'], test_metrics_dt['RMSE']],
        'Train MAE': [train_metrics_lr['MAE'], train_metrics_dt['MAE']],
        'Test MAE': [test_metrics_lr['MAE'], test_metrics_dt['MAE']],
        'Train R²': [train_metrics_lr['R²'], train_metrics_dt['R²']],
        'Test R²': [test_metrics_lr['R²'], test_metrics_dt['R²']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Identify best model based on test RMSE
    best_idx = comparison_df['Test RMSE'].idxmin()
    best_model_name = comparison_df.loc[best_idx, 'Model']
    
    print(f"\n🏆 Best Baseline Model: {best_model_name}")
    print(f"   Test RMSE: {comparison_df.loc[best_idx, 'Test RMSE']:.4f}")
    print(f"   Test R²: {comparison_df.loc[best_idx, 'Test R²']:.4f}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Set consistent axis limits for better comparison (0-60 range)
    # Note: trip_duration is already in minutes in the new dataset
    if target == 'trip_duration':
        y_test_plot = y_test  # Already in minutes
        y_pred_lr_plot = y_pred_test_lr  # Already in minutes
        y_pred_dt_plot = y_pred_test_dt  # Already in minutes
        unit = 'minutes'
        target_label = 'Trip Duration'
        axis_limit = 60  # 0-60 minutes
        residual_limit = 30  # ±30 minutes for residuals
    else:
        y_test_plot = y_test
        y_pred_lr_plot = y_pred_test_lr
        y_pred_dt_plot = y_pred_test_dt
        unit = 'dollars'
        target_label = 'Fare Amount'
        axis_limit = 60  # 0-$60 for consistency
        residual_limit = 30  # ±$30 for residuals
    
    # Plot 1: Linear Regression - Actual vs Predicted
    sample_size = min(5000, len(y_test))
    sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
    
    axes[0, 0].scatter(y_test_plot.iloc[sample_idx], y_pred_lr_plot[sample_idx], 
                       alpha=0.3, s=1, label='Predictions')
    axes[0, 0].plot([0, axis_limit], [0, axis_limit], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel(f'Actual {target_label} ({unit})')
    axes[0, 0].set_ylabel(f'Predicted {target_label} ({unit})')
    axes[0, 0].set_title(f'Linear Regression: Actual vs Predicted\nRMSE: {test_metrics_lr["RMSE"]:.2f}, R²: {test_metrics_lr["R²"]:.3f}')
    axes[0, 0].set_xlim(0, axis_limit)
    axes[0, 0].set_ylim(0, axis_limit)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Decision Tree - Actual vs Predicted
    axes[0, 1].scatter(y_test_plot.iloc[sample_idx], y_pred_dt_plot[sample_idx], 
                       alpha=0.3, s=1, label='Predictions', color='green')
    axes[0, 1].plot([0, axis_limit], [0, axis_limit], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel(f'Actual {target_label} ({unit})')
    axes[0, 1].set_ylabel(f'Predicted {target_label} ({unit})')
    axes[0, 1].set_title(f'Decision Tree: Actual vs Predicted\nRMSE: {test_metrics_dt["RMSE"]:.2f}, R²: {test_metrics_dt["R²"]:.3f}')
    axes[0, 1].set_xlim(0, axis_limit)
    axes[0, 1].set_ylim(0, axis_limit)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals - Linear Regression
    residuals_lr = y_test_plot - y_pred_lr_plot
    axes[1, 0].scatter(y_pred_lr_plot[sample_idx], residuals_lr.iloc[sample_idx], 
                       alpha=0.3, s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel(f'Predicted {target_label} ({unit})')
    axes[1, 0].set_ylabel(f'Residuals ({unit})')
    axes[1, 0].set_title('Linear Regression: Residual Plot')
    axes[1, 0].set_xlim(0, axis_limit)
    axes[1, 0].set_ylim(-residual_limit, residual_limit)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals - Decision Tree
    residuals_dt = y_test_plot - y_pred_dt_plot
    axes[1, 1].scatter(y_pred_dt_plot[sample_idx], residuals_dt.iloc[sample_idx], 
                       alpha=0.3, s=1, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel(f'Predicted {target_label} ({unit})')
    axes[1, 1].set_ylabel(f'Residuals ({unit})')
    axes[1, 1].set_title('Decision Tree: Residual Plot')
    axes[1, 1].set_xlim(0, axis_limit)
    axes[1, 1].set_ylim(-residual_limit, residual_limit)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    project_root = Path(__file__).parent.parent
    plot_dir = project_root / 'models' / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f'baseline_models_{target}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved plot: {plot_path}")
    plt.close()  # Close instead of show
    
    # Save comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison_df['Test RMSE'], width, label='Test RMSE', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, comparison_df['Test R²'], width, label='Test R²', alpha=0.8, color='orange')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE', color='b')
    ax2.set_ylabel('R²', color='orange')
    ax.set_title(f'Baseline Model Comparison: {target_label}')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plot_dir / f'baseline_comparison_{target}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved comparison: {plot_path}")
    plt.close()  # Close instead of show
    
    # ========================================================================
    # SAVE MODELS
    # ========================================================================
    print("\n💾 Saving models...")
    
    model_dir = project_root / 'models' / 'baseline'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Linear Regression
    lr_path = model_dir / f'linear_regression_{target}.pkl'
    scaler_path = model_dir / f'scaler_{target}.pkl'
    joblib.dump(lr_model, lr_path)
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Saved: {lr_path}")
    print(f"   ✅ Saved: {scaler_path}")
    
    # Save Decision Tree
    dt_path = model_dir / f'decision_tree_{target}.pkl'
    joblib.dump(dt_model, dt_path)
    print(f"   ✅ Saved: {dt_path}")
    
    # Save metrics
    metrics_path = model_dir / f'baseline_metrics_{target}.csv'
    comparison_df.to_csv(metrics_path, index=False)
    print(f"   ✅ Saved metrics: {metrics_path}")
    
    # Store additional data
    results['comparison_df'] = comparison_df
    results['best_model'] = best_model_name
    results['features'] = features
    results['y_test'] = y_test
    
    return results


def main():
    """
    Main function to train both fare and duration baseline models.
    """
    print("\n")
    print("="*70)
    print("NYC TAXI BASELINE MODEL TRAINING")
    print("="*70)
    print("\nThis script trains baseline models for fare and duration prediction.")
    print("Models: Linear Regression, Decision Tree")
    print("="*70)
    
    # Train fare prediction models
    print("\n\n")
    fare_results = train_baseline_models(target='fare_amount')
    
    # Train duration prediction models
    print("\n\n")
    duration_results = train_baseline_models(target='trip_duration')
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    print("\n📊 FARE PREDICTION:")
    print(f"   Best Model: {fare_results['best_model']}")
    fare_best = fare_results['comparison_df'].loc[
        fare_results['comparison_df']['Test RMSE'].idxmin()
    ]
    print(f"   Test RMSE: ${fare_best['Test RMSE']:.2f}")
    print(f"   Test MAE: ${fare_best['Test MAE']:.2f}")
    print(f"   Test R²: {fare_best['Test R²']:.4f}")
    
    print("\n📊 TRIP DURATION PREDICTION:")
    print(f"   Best Model: {duration_results['best_model']}")
    duration_best = duration_results['comparison_df'].loc[
        duration_results['comparison_df']['Test RMSE'].idxmin()
    ]
    print(f"   Test RMSE: {duration_best['Test RMSE']/60:.2f} minutes")
    print(f"   Test MAE: {duration_best['Test MAE']/60:.2f} minutes")
    print(f"   Test R²: {duration_best['Test R²']:.4f}")
    
    print("\n✅ All models saved to: models/baseline/")
    print("✅ Visualizations saved to: models/plots/")
    print("="*70)
    
    return fare_results, duration_results


if __name__ == "__main__":
    fare_results, duration_results = main()
