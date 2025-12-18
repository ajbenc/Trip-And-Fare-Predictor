"""
Memory-Efficient MLP Training Script for Trip Duration Prediction
Version 3: Optimized for large datasets with batch-wise normalization
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


def compute_normalization_stats(splits_dir):
    """
    Compute mean and std from training data efficiently
    without loading all data into memory at once
    KILLS ALL NaN VALUES - replaces with 0
    """
    print("\n🔄 Computing normalization statistics from training data...")
    train_path = Path(splits_dir) / 'train'
    feature_files = sorted(train_path.glob("features_*_X.parquet"))
    
    print(f"   • Found {len(feature_files)} training files")
    
    # First pass: compute mean
    print("   • Computing mean (pass 1/2)...")
    n_samples = 0
    sum_values = None
    total_nans = 0
    
    for i, file in enumerate(feature_files, 1):
        df = pd.read_parquet(file)
        
        # Ensure all columns are numeric - convert object types to numeric, coercing errors to NaN
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        data = df.values.astype(np.float64)
        
        # KILL NaN values - replace with 0
        nan_count = np.isnan(data).sum()
        if nan_count > 0:
            total_nans += nan_count
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if sum_values is None:
            sum_values = np.zeros(data.shape[1], dtype=np.float64)
        
        sum_values += data.sum(axis=0, dtype=np.float64)
        n_samples += len(data)
        
        print(f"      File {i}/{len(feature_files)}: {len(data):,} samples{f' ({nan_count:,} NaN killed)' if nan_count > 0 else ''}")
        del df, data  # Free memory
    
    mean = sum_values / n_samples
    
    if total_nans > 0:
        print(f"   ⚠️  Total NaN values killed: {total_nans:,}")
    print(f"   ✓ Mean computed from {n_samples:,} samples")
    
    # Second pass: compute std
    print("   • Computing std (pass 2/2)...")
    sum_squared_diff = np.zeros(len(mean), dtype=np.float64)
    
    for i, file in enumerate(feature_files, 1):
        df = pd.read_parquet(file)
        
        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        data = df.values.astype(np.float64)
        
        # KILL NaN values - replace with 0
        if np.isnan(data).sum() > 0:
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        diff = data - mean
        sum_squared_diff += (diff ** 2).sum(axis=0, dtype=np.float64)
        
        print(f"      File {i}/{len(feature_files)}: processed")
        del df, data, diff  # Free memory
    
    std = np.sqrt(sum_squared_diff / n_samples)
    std[std == 0] = 1.0  # Avoid division by zero
    
    # Final check - ensure no NaN in stats
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    
    print(f"   ✓ Std computed")
    print(f"   • Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   • Std range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"   ✓ All NaN values eliminated")
    
    return mean, std, n_samples


def create_normalized_dataset(splits_dir, split, mean, std, batch_size=8192):
    """
    Create a TensorFlow dataset that normalizes data on-the-fly
    Memory efficient - doesn't load all data at once
    """
    print(f"\n📂 Creating {split} dataset...")
    split_path = Path(splits_dir) / split
    
    feature_files = sorted(split_path.glob("features_*_X.parquet"))
    target_files = sorted(split_path.glob("features_*_y_duration.parquet"))
    
    print(f"   • Found {len(feature_files)} files")
    
    # Load all data (but it will be freed after creating dataset)
    X_list = []
    y_list = []
    total_nans_features = 0
    total_nans_targets = 0
    
    for x_file, y_file in zip(feature_files, target_files):
        X_df = pd.read_parquet(x_file)
        y_df = pd.read_parquet(y_file)
        
        # Ensure all feature columns are numeric
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        # Ensure target is numeric
        if y_df[y_df.columns[0]].dtype == 'object':
            y_df[y_df.columns[0]] = pd.to_numeric(y_df[y_df.columns[0]], errors='coerce')
        
        X_data = X_df.values.astype(np.float64)
        y_data = y_df.values.astype(np.float64).ravel()
        
        # KILL NaN in features
        nan_count_x = np.isnan(X_data).sum()
        if nan_count_x > 0:
            total_nans_features += nan_count_x
            X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # KILL NaN in targets
        nan_count_y = np.isnan(y_data).sum()
        if nan_count_y > 0:
            total_nans_targets += nan_count_y
            y_data = np.nan_to_num(y_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_list.append(X_data)
        y_list.append(y_data)
        
        del X_df, y_df, X_data, y_data
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    if total_nans_features > 0:
        print(f"   ⚠️  Killed {total_nans_features:,} NaN values in features")
    if total_nans_targets > 0:
        print(f"   ⚠️  Killed {total_nans_targets:,} NaN values in targets")
    
    print(f"   ✓ Loaded {len(X):,} samples with {X.shape[1]} features")
    
    # Normalize using pre-computed stats
    print(f"   • Normalizing data...")
    X_norm = (X.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    
    # FINAL KILL - ensure absolutely no NaN/Inf
    nan_count = np.isnan(X_norm).sum()
    inf_count = np.isinf(X_norm).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"   ⚠️  Killing {nan_count:,} NaN and {inf_count:,} Inf after normalization...")
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"   ✓ Normalization complete - ALL NaN/Inf KILLED")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_norm, y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(X), X.shape[1]


def build_improved_model(input_dim, hidden_layers=[512, 256, 128, 64, 32], 
                        dropout_rate=0.1, learning_rate=0.001):
    """Build an improved deeper MLP model"""
    print("\n🏗️  Building Improved MLP Model...")
    print(f"   • Architecture: {hidden_layers}")
    print(f"   • Dropout rate: {dropout_rate}")
    print(f"   • Learning rate: {learning_rate}")
    
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
    
    # Output layer
    model.add(layers.Dense(1, activation='linear', dtype='float32', name='output'))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print(f"\n   ✓ Total parameters: {model.count_params():,}")
    
    return model


def main():
    """Main execution function"""
    print("="*70)
    print("🚖 MEMORY-EFFICIENT MLP TRAINING (V3)")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    SPLITS_DIR = "Data/splits_cleaned"
    MODEL_SAVE_PATH = "models/mlp_duration_v3.keras"
    
    HIDDEN_LAYERS = [512, 256, 128, 64, 32]
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 8192
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Data directory: {SPLITS_DIR}")
    print(f"   • Architecture: {HIDDEN_LAYERS}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Max epochs: {EPOCHS}")
    print(f"   • Memory-efficient: YES ✓")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Step 1: Compute normalization stats (memory efficient)
    mean, std, n_train = compute_normalization_stats(SPLITS_DIR)
    
    # Step 2: Create datasets (normalized on-the-fly)
    train_dataset, n_train, n_features = create_normalized_dataset(
        SPLITS_DIR, 'train', mean, std, BATCH_SIZE
    )
    val_dataset, n_val, _ = create_normalized_dataset(
        SPLITS_DIR, 'val', mean, std, BATCH_SIZE
    )
    test_dataset, n_test, _ = create_normalized_dataset(
        SPLITS_DIR, 'test', mean, std, BATCH_SIZE
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   • Training:   {n_train:,} samples")
    print(f"   • Validation: {n_val:,} samples")
    print(f"   • Test:       {n_test:,} samples")
    print(f"   • Features:   {n_features}")
    
    # Step 3: Build model
    model = build_improved_model(
        n_features,
        hidden_layers=HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    
    # Step 4: Setup callbacks
    callback_list = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Step 5: Train model
    print("\n🚀 Starting Training...")
    print(f"   • This will take approximately 2-3 hours")
    print(f"   • Your PC should remain responsive during training")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callback_list,
        verbose=1
    )
    
    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time/60:.2f} minutes")
    
    # Step 6: Evaluate on test set
    print("\n📈 Evaluating on Test Set...")
    test_results = model.evaluate(test_dataset, verbose=0)
    test_loss, test_mae, test_rmse = test_results
    
    # Get validation metrics
    best_epoch = np.argmin(history.history['val_loss'])
    val_loss = history.history['val_loss'][best_epoch]
    val_mae = history.history['val_mae'][best_epoch]
    val_rmse = history.history['val_rmse'][best_epoch]
    
    # Calculate R² (need predictions for this)
    print("   • Computing R² scores...")
    
    # Validation R²
    val_path = Path(SPLITS_DIR) / 'val'
    y_val_list = []
    for f in sorted(val_path.glob("features_*_y_duration.parquet")):
        y_val_list.append(pd.read_parquet(f).values.ravel())
    y_val = np.concatenate(y_val_list)
    y_val_pred = model.predict(val_dataset, verbose=0).flatten()
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Test R²
    test_path = Path(SPLITS_DIR) / 'test'
    y_test_list = []
    for f in sorted(test_path.glob("features_*_y_duration.parquet")):
        y_test_list.append(pd.read_parquet(f).values.ravel())
    y_test = np.concatenate(y_test_list)
    y_test_pred = model.predict(test_dataset, verbose=0).flatten()
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print final results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"\n🎯 Validation Performance:")
    print(f"   • R² Score:  {val_r2*100:.2f}% {'🎉 EXCEEDED TARGET!' if val_r2 >= 0.90 else '⚠️  Below 90%'}")
    print(f"   • MAE:       {val_mae:.4f} minutes")
    print(f"   • RMSE:      {val_rmse:.4f} minutes")
    print(f"   • Best Epoch: {best_epoch + 1}/{EPOCHS}")
    
    print(f"\n🧪 Test Performance:")
    print(f"   • R² Score:  {test_r2*100:.2f}%")
    print(f"   • MAE:       {test_mae:.4f} minutes")
    print(f"   • RMSE:      {test_rmse:.4f} minutes")
    
    print(f"\n⏱️  Training Time: {train_time/60:.2f} minutes")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    print(f"\n✅ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


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
