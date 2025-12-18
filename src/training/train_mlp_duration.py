"""
Train MLP (Multi-Layer Perceptron) for Trip Duration Prediction
================================================================
Neural network approach for predicting taxi trip durations on cleaned data.

Architecture:
- Input: 56 features
- Hidden Layers: [256, 128, 64] with BatchNorm + Dropout
- Output: 1 (trip_duration in minutes)

Training Strategy:
- Adam optimizer with learning rate scheduling
- Early stopping on validation set
- Batch size: 2048 for efficient GPU/CPU usage
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
import time
from datetime import datetime
import json

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MLPDurationModel:
    """
    Multi-Layer Perceptron for trip duration prediction.
    """
    
    def __init__(self, input_dim: int, architecture: list = [256, 128, 64]):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Number of input features
            architecture: List of hidden layer sizes
        """
        self.input_dim = input_dim
        self.architecture = architecture
        self.model = None
        self.history = None
        self.feature_stats = None
        
    def build_model(self):
        """Build the neural network architecture."""
        model = keras.Sequential(name='TripDurationMLP')
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,), name='input'))
        
        # Hidden layers with BatchNorm and Dropout
        for i, units in enumerate(self.architecture):
            model.add(layers.Dense(
                units, 
                activation=None,
                kernel_initializer='he_normal',  # Better initialization
                name=f'dense_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            model.add(layers.Activation('relu', name=f'relu_{i+1}'))
            model.add(layers.Dropout(0.2, name=f'dropout_{i+1}'))
        
        # Output layer (regression)
        model.add(layers.Dense(
            1, 
            activation='linear',
            kernel_initializer='he_normal',
            name='output'
        ))
        
        # Compile with Adam optimizer with LOWER learning rate and gradient clipping
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0001,  # Much lower to prevent exploding gradients
                clipnorm=1.0  # Gradient clipping to prevent explosion
            ),
            loss='mse',
            metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """Print model architecture."""
        if self.model is None:
            self.build_model()
        return self.model.summary()


def load_data(split: str = 'train') -> tuple:
    """
    Load features and targets from cleaned splits.
    Data is stored as monthly parquet files.
    
    Args:
        split: 'train', 'val', or 'test'
    
    Returns:
        X, y as numpy arrays
    """
    base_path = Path('Data/splits_cleaned')
    split_path = base_path / split
    
    print(f"\n📂 Loading {split} data from {split_path}/...")
    
    # Find all monthly feature files
    feature_files = sorted(list(split_path.glob('features_*_X.parquet')))
    duration_files = sorted(list(split_path.glob('features_*_y_duration.parquet')))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {split_path}")
    
    print(f"   • Found {len(feature_files)} monthly files")
    
    # Load and concatenate all monthly files
    X_list = []
    y_list = []
    
    for feat_file, dur_file in zip(feature_files, duration_files):
        X_month = pd.read_parquet(feat_file)
        y_month = pd.read_parquet(dur_file)['trip_duration'].values
        
        X_list.append(X_month)
        y_list.append(y_month)
    
    # Concatenate all months
    X = pd.concat(X_list, axis=0, ignore_index=True)
    y = np.concatenate(y_list)
    
    print(f"   ✅ Loaded {len(X):,} samples with {X.shape[1]} features")
    
    return X, y


def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using training set statistics.
    Important for neural networks!
    Memory-efficient version with proper validation.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
    
    Returns:
        Normalized arrays and statistics
    """
    print("\n🔄 Normalizing features...")
    
    # Calculate statistics from training set only
    print("   • Computing mean and std from training data...")
    mean = X_train.mean()
    std = X_train.std()
    
    # Replace zero std with 1 to avoid division by zero
    std = std.replace(0, 1)
    
    # Save stats
    stats = {'mean': mean.to_dict(), 'std': std.to_dict()}
    
    # Convert to proper numpy arrays (ensure no nullable types)
    print("   • Converting to numpy arrays...")
    mean_vals = np.array(mean.values, dtype=np.float32)
    std_vals = np.array(std.values, dtype=np.float32)
    
    print("   • Converting training data to numpy...")
    X_train_np = np.array(X_train.values, dtype=np.float32)
    
    print("   • Converting validation data to numpy...")
    X_val_np = np.array(X_val.values, dtype=np.float32)
    
    print("   • Converting test data to numpy...")
    X_test_np = np.array(X_test.values, dtype=np.float32)
    
    # Clear DataFrames to free memory
    del X_train, X_val, X_test
    
    # Normalize using numpy (more memory efficient)
    print("   • Normalizing training set...")
    X_train_norm = (X_train_np - mean_vals) / std_vals
    
    print("   • Normalizing validation set...")
    X_val_norm = (X_val_np - mean_vals) / std_vals
    
    print("   • Normalizing test set...")
    X_test_norm = (X_test_np - mean_vals) / std_vals
    
    # Check for NaN or Inf values
    print("   • Checking for invalid values...")
    train_nan = np.isnan(X_train_norm).sum()
    train_inf = np.isinf(X_train_norm).sum()
    val_nan = np.isnan(X_val_norm).sum()
    val_inf = np.isinf(X_val_norm).sum()
    
    if train_nan > 0 or train_inf > 0:
        print(f"   ⚠️  Found {train_nan} NaN and {train_inf} Inf values in training data")
        print("   • Replacing NaN/Inf with 0...")
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    if val_nan > 0 or val_inf > 0:
        print(f"   ⚠️  Found {val_nan} NaN and {val_inf} Inf values in validation data")
        print("   • Replacing NaN/Inf with 0...")
        X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("   ✅ Features normalized and validated (zero mean, unit variance)")
    
    return X_train_norm, X_val_norm, X_test_norm, stats


def train_mlp_model(X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 2048):
    """
    Train MLP model with early stopping and learning rate scheduling.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum number of epochs
        batch_size: Training batch size
    
    Returns:
        Trained model and training history
    """
    print("\n🧠 Building MLP model...")
    
    # Initialize model
    mlp = MLPDurationModel(input_dim=X_train.shape[1])
    model = mlp.build_model()
    
    print("\n📋 Model Architecture:")
    mlp.get_model_summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    print(f"\n🚀 Training MLP model...")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size:,}")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Validation samples: {len(X_val):,}")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )
    
    train_time = time.time() - start_time
    print(f"\n✅ Training completed in {train_time/60:.1f} minutes")
    
    return model, history, train_time


def evaluate_model(model, X, y, set_name: str = 'Test'):
    """
    Evaluate model performance.
    
    Args:
        model: Trained Keras model
        X, y: Data to evaluate
        set_name: Name of the dataset
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 Evaluating on {set_name} set...")
    
    # Predict
    y_pred = model.predict(X, batch_size=4096, verbose=0).flatten()
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(y)
    }
    
    print(f"\n   {set_name} Results:")
    print(f"   • R² Score:  {r2*100:.2f}%")
    print(f"   • RMSE:      {rmse:.2f} minutes")
    print(f"   • MAE:       {mae:.2f} minutes")
    
    return metrics, y_pred


def save_model_and_results(model, history, metrics, feature_stats):
    """
    Save trained model, training history, and results.
    
    Args:
        model: Trained Keras model
        history: Training history
        metrics: Evaluation metrics
        feature_stats: Feature normalization statistics
    """
    output_dir = Path('models/mlp')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'mlp_duration_model.keras'
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save feature statistics
    stats_path = output_dir / 'feature_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"💾 Feature stats saved to: {stats_path}")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'rmse': [float(x) for x in history.history['rmse']],
        'val_rmse': [float(x) for x in history.history['val_rmse']]
    }
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"💾 Training history saved to: {history_path}")
    
    # Save metrics
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_path}")


def compare_with_baseline():
    """Compare MLP results with baseline models."""
    print("\n" + "="*80)
    print("📊 COMPARISON WITH BASELINE MODELS")
    print("="*80)
    print("\nTrip Duration Prediction:")
    print("\nBASELINE (LightGBM, cleaned data):")
    print("  • R² Score:  86.09%")
    print("  • Training:  27M samples")
    print("\nNEW (MLP Neural Network):")
    print("  • Check results above ⬆️")
    print("\n" + "="*80)


def main():
    """Main training pipeline."""
    print("="*80)
    print("🧠 MLP NEURAL NETWORK - TRIP DURATION PREDICTION")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print("  • Architecture: [256, 128, 64]")
    print("  • Activation: ReLU + BatchNorm")
    print("  • Dropout: 0.2")
    print("  • Optimizer: Adam (lr=0.0001, gradient clipping)")
    print("  • Initialization: He Normal")
    print("  • Batch Size: 4096 (memory efficient)")
    print("="*80)
    
    # Load data
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    X_test, y_test = load_data('test')
    
    print(f"\n📊 Data Summary:")
    print(f"   • Train: {len(X_train):,} samples")
    print(f"   • Val:   {len(X_val):,} samples")
    print(f"   • Test:  {len(X_test):,} samples")
    print(f"   • Features: {X_train.shape[1]}")
    
    # Validate target values
    print(f"\n🔍 Checking target values...")
    print(f"   • Train target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   • Train target mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    
    # Check for NaN or negative values in targets
    train_nan = np.isnan(y_train).sum()
    train_neg = (y_train < 0).sum()
    
    if train_nan > 0:
        print(f"   ⚠️  Found {train_nan} NaN values in training targets - removing...")
        valid_idx = ~np.isnan(y_train)
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
    
    if train_neg > 0:
        print(f"   ⚠️  Found {train_neg} negative values in training targets - removing...")
        valid_idx = y_train >= 0
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
    
    print(f"   ✅ Final train size: {len(y_train):,} samples")
    
    # Normalize features (critical for neural networks!)
    X_train_norm, X_val_norm, X_test_norm, feature_stats = normalize_features(
        X_train, X_val, X_test
    )
    
    # Train model with larger batch size for memory efficiency
    model, history, train_time = train_mlp_model(
        X_train_norm, y_train,
        X_val_norm, y_val,
        epochs=50,
        batch_size=4096  # Larger batches = less memory overhead
    )
    
    # Evaluate on all sets
    print("\n" + "="*80)
    print("📈 MODEL EVALUATION")
    print("="*80)
    
    train_metrics, _ = evaluate_model(model, X_train_norm, y_train, 'Train')
    val_metrics, _ = evaluate_model(model, X_val_norm, y_val, 'Validation')
    test_metrics, y_pred = evaluate_model(model, X_test_norm, y_test, 'Test')
    
    # Check for overfitting
    train_test_gap = train_metrics['r2'] - test_metrics['r2']
    print(f"\n🔍 Overfitting Check:")
    print(f"   • Train-Test R² Gap: {train_test_gap*100:.2f}%")
    if train_test_gap < 0.02:
        print("   ✅ No overfitting detected (gap < 2%)")
    else:
        print("   ⚠️  Potential overfitting detected")
    
    # Combine all metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'train_time_minutes': train_time / 60,
        'architecture': [256, 128, 64],
        'total_parameters': model.count_params()
    }
    
    # Save everything
    save_model_and_results(model, history, all_metrics, feature_stats)
    
    # Compare with baseline
    compare_with_baseline()
    
    print("\n" + "="*80)
    print("✅ MLP TRAINING COMPLETE")
    print("="*80)
    print(f"\n📁 Model saved in: models/mlp/")
    print(f"🎯 Test R²: {test_metrics['r2']*100:.2f}%")
    print(f"⏱️  Training time: {train_time/60:.1f} minutes")
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
