"""
Improved MLP Training Script for Trip Duration Prediction
Version 2: Enhanced architecture targeting 90%+ R² validation score
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def load_data(splits_dir, split='train'):
    """Load all feature and target files for a given split"""
    print(f"\n📂 Loading {split} data...")
    splits_path = Path(splits_dir) / split
    
    # Find all feature and target files for this split
    feature_files = sorted(splits_path.glob("features_*_X.parquet"))
    target_files = sorted(splits_path.glob("features_*_y_duration.parquet"))
    
    print(f"   • Found {len(feature_files)} feature files")
    print(f"   • Found {len(target_files)} target files")
    
    if len(feature_files) == 0:
        raise ValueError(f"No feature files found for split: {split}")
    
    # Load features
    X_dfs = []
    for file in feature_files:
        df = pd.read_parquet(file)
        X_dfs.append(df)
    X = pd.concat(X_dfs, ignore_index=True)
    
    # Load targets
    y_dfs = []
    for file in target_files:
        df = pd.read_parquet(file)
        y_dfs.append(df)
    y = pd.concat(y_dfs, ignore_index=True)
    
    print(f"   ✓ Loaded {len(X):,} samples with {X.shape[1]} features")
    
    return X.values, y.values.ravel()


def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using training statistics
    Handles NaN and Inf values robustly
    Uses optimized computation for large datasets
    """
    print("\n🔄 Normalizing features...")
    print(f"   • Training shape: {X_train.shape}")
    print(f"   • Validation shape: {X_val.shape}")
    print(f"   • Test shape: {X_test.shape}")
    
    # Compute statistics from training data only (optimized for large datasets)
    print("   • Computing mean and std from training data (optimized)...")
    mean = X_train.mean(axis=0, dtype=np.float64)
    std = X_train.std(axis=0, dtype=np.float64)
    print("   ✓ Statistics computed")
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    # Normalize (in-place for memory efficiency)
    print("   • Normalizing training data...")
    X_train_norm = (X_train - mean) / std
    print("   • Normalizing validation data...")
    X_val_norm = (X_val - mean) / std
    print("   • Normalizing test data...")
    X_test_norm = (X_test - mean) / std
    
    # Check for NaN or Inf values and replace
    for i, (X, name) in enumerate([(X_train_norm, 'training'), 
                                     (X_val_norm, 'validation'), 
                                     (X_test_norm, 'test')]):
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️  Found {nan_count} NaN and {inf_count} Inf values in {name} data")
            print("   • Replacing with 0...")
            
            if i == 0:
                X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
            elif i == 1:
                X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                X_test_norm = np.nan_to_num(X_test_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("   ✓ Normalization complete")
    
    return X_train_norm, X_val_norm, X_test_norm


def build_improved_model(input_dim, hidden_layers=[512, 256, 128, 64, 32], 
                        dropout_rate=0.1, learning_rate=0.001):
    """
    Build an improved deeper MLP model
    
    Architecture:
    - 5 hidden layers with decreasing neurons
    - BatchNormalization for training stability
    - Moderate dropout (0.1) to prevent overfitting
    - ReLU activation for non-linearity
    - He Normal initialization
    - Adam optimizer with gradient clipping
    """
    print("\n🏗️  Building Improved MLP Model...")
    print(f"   • Architecture: {hidden_layers}")
    print(f"   • Dropout rate: {dropout_rate}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Optimizer: Adam with gradient clipping (clipnorm=1.0)")
    print(f"   • Mixed precision: enabled (float16/float32)")
    
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Build hidden layers
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units, 
            activation='relu',
            kernel_initializer='he_normal',
            name=f'dense_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer (regression)
    model.add(layers.Dense(1, activation='linear', dtype='float32', name='output'))
    
    # Compile model with gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Prevent gradient explosion
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print("\n📊 Model Summary:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\n   ✓ Total parameters: {total_params:,}")
    
    return model


def train_mlp_model(X_train, y_train, X_val, y_val, 
                   hidden_layers=[512, 256, 128, 64, 32],
                   dropout_rate=0.1,
                   learning_rate=0.001,
                   epochs=100,
                   batch_size=8192,
                   model_save_path='models/mlp_duration_v2.keras'):
    """
    Train the improved MLP model with advanced callbacks
    """
    print("\n🚀 Starting Training...")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Validation samples: {len(X_val):,}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    
    input_dim = X_train.shape[1]
    model = build_improved_model(
        input_dim, 
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Setup callbacks for better training
    callback_list = [
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by 50%
            patience=5,  # Wait 5 epochs
            min_lr=1e-6,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Stop if no improvement for 15 epochs
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging (optional)
        callbacks.TensorBoard(
            log_dir=f'logs/mlp_v2_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=0
        )
    ]
    
    print("\n⏱️  Training in progress...")
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )
    
    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time/60:.2f} minutes")
    
    return model, history, train_time


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
    print("\n📈 Evaluating on Test Set...")
    
    # Make predictions
    y_pred = model.predict(X_test, batch_size=8192, verbose=0).flatten()
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n   📊 Test Set Results:")
    print(f"   • R² Score:  {r2*100:.2f}%")
    print(f"   • MAE:       {mae:.4f} minutes")
    print(f"   • RMSE:      {rmse:.4f} minutes")
    print(f"   • MSE:       {mse:.4f}")
    print(f"   • MAPE:      {mape:.2f}%")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'mape': mape
    }


def main():
    """Main execution function"""
    print("="*70)
    print("🚖 IMPROVED MLP TRAINING FOR TRIP DURATION PREDICTION (V2)")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    SPLITS_DIR = "Data/splits_cleaned"
    MODEL_SAVE_PATH = "models/mlp_duration_v2.keras"
    
    # Model hyperparameters - IMPROVED
    HIDDEN_LAYERS = [512, 256, 128, 64, 32]  # Deeper network
    DROPOUT_RATE = 0.1  # Less aggressive dropout
    LEARNING_RATE = 0.001  # Higher initial learning rate
    EPOCHS = 100  # More epochs with early stopping
    BATCH_SIZE = 8192  # Larger batches
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Data directory: {SPLITS_DIR}")
    print(f"   • Model architecture: {HIDDEN_LAYERS}")
    print(f"   • Dropout rate: {DROPOUT_RATE}")
    print(f"   • Initial learning rate: {LEARNING_RATE}")
    print(f"   • Max epochs: {EPOCHS}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Target: 90%+ validation R²")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    X_train, y_train = load_data(SPLITS_DIR, 'train')
    X_val, y_val = load_data(SPLITS_DIR, 'val')
    X_test, y_test = load_data(SPLITS_DIR, 'test')
    
    print(f"\n📊 Data Summary:")
    print(f"   • Training:   {len(X_train):,} samples")
    print(f"   • Validation: {len(X_val):,} samples")
    print(f"   • Test:       {len(X_test):,} samples")
    print(f"   • Features:   {X_train.shape[1]}")
    
    # Validate targets
    print(f"\n🎯 Target Statistics (Duration in minutes):")
    print(f"   • Training:   min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"   • Validation: min={y_val.min():.2f}, max={y_val.max():.2f}, mean={y_val.mean():.2f}, std={y_val.std():.2f}")
    print(f"   • Test:       min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}, std={y_test.std():.2f}")
    
    # Check for NaN in targets
    train_nan = np.isnan(y_train).sum()
    val_nan = np.isnan(y_val).sum()
    test_nan = np.isnan(y_test).sum()
    
    if train_nan > 0 or val_nan > 0 or test_nan > 0:
        print(f"\n   ⚠️  WARNING: Found NaN values in targets!")
        print(f"      Train: {train_nan}, Val: {val_nan}, Test: {test_nan}")
        raise ValueError("NaN values found in targets - data preprocessing issue!")
    else:
        print(f"   ✓ No NaN values in targets")
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm = normalize_features(X_train, X_val, X_test)
    
    # Train model
    model, history, train_time = train_mlp_model(
        X_train_norm, y_train,
        X_val_norm, y_val,
        hidden_layers=HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test_norm, y_test)
    
    # Get validation metrics from history
    best_epoch = np.argmin(history.history['val_loss'])
    val_loss = history.history['val_loss'][best_epoch]
    val_mae = history.history['val_mae'][best_epoch]
    val_rmse = history.history['val_rmse'][best_epoch]
    
    # Calculate validation R²
    y_val_pred = model.predict(X_val_norm, batch_size=BATCH_SIZE, verbose=0).flatten()
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Print final summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n🎯 Validation Performance:")
    print(f"   • R² Score:  {val_r2*100:.2f}% {'🎉 EXCEEDED TARGET!' if val_r2 >= 0.90 else '⚠️  Below 90% target'}")
    print(f"   • MAE:       {val_mae:.4f} minutes")
    print(f"   • RMSE:      {val_rmse:.4f} minutes")
    print(f"   • Best Epoch: {best_epoch + 1}/{EPOCHS}")
    
    print(f"\n🧪 Test Performance:")
    print(f"   • R² Score:  {test_metrics['r2']*100:.2f}%")
    print(f"   • MAE:       {test_metrics['mae']:.4f} minutes")
    print(f"   • RMSE:      {test_metrics['rmse']:.4f} minutes")
    print(f"   • MAPE:      {test_metrics['mape']:.2f}%")
    
    print(f"\n⏱️  Training Time: {train_time/60:.2f} minutes")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    
    print(f"\n✅ Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return model, history, test_metrics


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
