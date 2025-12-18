# 🚕 NYC Taxi MLP Experiment

Comprehensive notes for the MLP_full_33M run: inputs, targets, preprocessing, training setup, timing, and how to use the outputs.

## Overview
- Task: Dual-output regression (fare_amount, trip_duration)
- Model: Multilayer Perceptron (Keras 3)
- Dataset size: 33,288,572 rows (Parquet)
- Training style: Streamed batches from Parquet (no full RAM load)

## Inputs (Features)
- All numeric columns from the dataset except targets.
- Exact ordering is defined by the feature discovery step used during training (derived from Parquet schema minus targets).
- Scaling: StandardScaler fitted via streaming (partial_fit) on the feature set.

## Targets
- fare_amount (float, non-negative)
- trip_duration (float, non-negative)

## Preprocessing Applied
- Replace ±Inf with NaN; drop/impute as needed.
- Impute feature NaNs with per-column mean learned by StandardScaler.
- Standardize features with the fitted scaler (saved as `scaler.pkl`).
- Clip standardized features softly to the range [-10, 10] to stabilize gradients.
- Target clipping (cap) at the 99.5th percentile estimated from a ~1M-row sample:
- Caps computed at runtime: `CAP_FARE`, `CAP_DUR` (both forced non-negative).

## Training Setup
- Framework: TensorFlow/Keras 3
- Architecture (example): Dense(64, relu) → Dense(32, relu) → two linear heads
- Optimizer: Adam (learning_rate = 5e-5, clipnorm = 1.0)
- Loss: MSE for both outputs
- Metrics: MAE for both outputs
- Batch size: 512
- Steps per epoch: ceil(33,288,572 / 512) ≈ 65,017
- Callbacks: ModelCheckpoint (best/last), CSVLogger, EarlyStopping, TerminateOnNaN
- Learning rate adjusted via `model.optimizer.learning_rate.assign(5e-5)` (Keras 3 compatible)

## Training Duration (Observed)
- Example Colab T4 logs show ~60–62 ms/step.
- With ~65,017 steps/epoch, that’s ~65–70 minutes per epoch.
- Total wall-clock depends on epoch count and early stopping. Resuming from checkpoints is recommended for long runs.

## Files in This Folder
- `model.keras` — Full model artifact (latest saved model)
- `checkpoints/best.keras` — Best checkpoint by monitored metric
- `checkpoints/last.keras` — Last checkpoint at most recent epoch
- `logs/history.csv` — CSV logs with epoch metrics
- `scaler.pkl` — Fitted StandardScaler for features
- `mlp_dual_scaler.pkl` — Alternate scaler artifact (if present in pipeline)
- `hyperparams_extracted.json` — Hyperparameters used/resolved
- `analysis_summary.{json,txt}` — Optional summaries/notes

## Inference (How to Use)
```python
from tensorflow import keras
import joblib
import numpy as np

# Load best model and scaler
model = keras.models.load_model('checkpoints/best.keras')
scaler = joblib.load('scaler.pkl')

# X_raw: numpy array with the same feature order used in training
X_raw = np.array([...], dtype=np.float32)  # shape: (n_samples, n_features)
X = scaler.transform(X_raw).astype('float32')
X = np.clip(X, -10.0, 10.0)

pred = model.predict(X)
# pred is a dict-like or list depending on your Keras model; in this run:
# {'fare_output': (n,1), 'duration_output': (n,1)}
```

## Resume Training (Optional)
If training is interrupted, resume from the last checkpoint and epoch stored in `logs/history.csv`.
```python
from tensorflow import keras
import pandas as pd

model = keras.models.load_model('checkpoints/best.keras')
hist = pd.read_csv('logs/history.csv')
last_epoch = int(hist['epoch'].max()) if 'epoch' in hist.columns else 0

history = model.fit(
    train_ds,
    steps_per_epoch=65017,   # or recompute from dataset size
    epochs=50,
    initial_epoch=last_epoch + 1,
    callbacks=callbacks,
)
```

## Reproducibility
- Use the provided `scaler.pkl` and the same feature ordering.
- Keep preprocessing (imputation, clipping) identical to the training pipeline.
- Hyperparameters/config are recorded in `hyperparams_extracted.json` and training notebooks/scripts.

## Environment Notes
- Verified with TensorFlow/Keras 3 on Colab T4 GPU.
- Python 3.11 (exact version may vary between environments).

