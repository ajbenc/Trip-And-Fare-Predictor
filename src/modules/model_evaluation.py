import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate and return comprehensive evaluation metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    median_ae = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Median_AE': median_ae
    }
    
    return metrics

def plot_prediction_analysis(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create comprehensive plots for prediction analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Trip Duration (seconds)')
    axes[0, 0].set_ylabel('Predicted Trip Duration (seconds)')
    axes[0, 0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Trip Duration (seconds)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{model_name}: Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{model_name}: Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error distribution (absolute errors)
    abs_errors = np.abs(residuals)
    axes[1, 1].hist(abs_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Error (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'{model_name}: Absolute Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_learning_curves(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    Plot learning curves to analyze bias-variance tradeoff.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=42
    )
    
    # Convert to positive RMSE
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_rmse, axis=1), 'o-', label='Training RMSE', linewidth=2)
    plt.plot(train_sizes, np.mean(val_rmse, axis=1), 'o-', label='Validation RMSE', linewidth=2)
    plt.fill_between(train_sizes, 
                     np.mean(train_rmse, axis=1) - np.std(train_rmse, axis=1),
                     np.mean(train_rmse, axis=1) + np.std(train_rmse, axis=1), 
                     alpha=0.1)
    plt.fill_between(train_sizes, 
                     np.mean(val_rmse, axis=1) - np.std(val_rmse, axis=1),
                     np.mean(val_rmse, axis=1) + np.std(val_rmse, axis=1), 
                     alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_models(results_list):
    """
    Compare multiple models and create a summary table.
    """
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.round(2)
    
    # Sort by RMSE (best first)
    comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None