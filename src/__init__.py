# -*- coding: utf-8 -*-
"""
NYC Taxi Trip Prediction - Source Package

Lightweight package initializer. Avoids importing heavy plotting/evaluation
modules at import time so runtime apps (FastAPI/Streamlit) don’t pull optional
dependencies like seaborn/matplotlib unless explicitly requested.

Usage guidance:
- Import evaluation utilities directly from their modules when needed, e.g.:
    from src.modules.model_evaluation import calculate_metrics
- Import preprocessing helpers explicitly, e.g.:
    from src.modules.preprocessing_utils import load_processed_data
"""

__version__ = '2.0.1'

# Intentionally do not import submodules here to keep import side-effects minimal.
# This prevents optional dependencies (e.g., seaborn) from being required by the
# API/UI apps. Tests and notebooks should import from submodules directly.

__all__ = ['__version__']
