# credit_card_fraud_detection/src/utils.py
"""
Utility functions for the fraud detection system
"""

import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
import json
from pathlib import Path

from .config import MODELS_DIR, RESULTS_DIR


def save_model(model: Any, filename: str) -> None:
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        filename: Name of the file to save
    """
    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def load_model(filename: str) -> Any:
    """
    Load trained model from disk
    
    Args:
        filename: Name of the file to load
        
    Returns:
        Loaded model object
    """
    model_path = MODELS_DIR / filename
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def save_results(metrics: Dict[str, Any], filename: str) -> None:
    """
    Save evaluation metrics to disk
    
    Args:
        metrics: Dictionary containing evaluation metrics
        filename: Name of the file to save
    """
    results_path = RESULTS_DIR / filename
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to: {results_path}")


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load evaluation metrics from disk
    
    Args:
        filename: Name of the file to load
        
    Returns:
        Dictionary containing evaluation metrics
    """
    results_path = RESULTS_DIR / filename
    
    with open(results_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Results loaded from: {results_path}")
    return metrics


def calculate_class_weights(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost based on class imbalance
    
    Args:
        y: Target series
        
    Returns:
        scale_pos_weight value
    """
    fraud_count = len(y[y == 1])
    legit_count = len(y[y == 0])
    
    if fraud_count == 0:
        return 1.0
    
    return legit_count / fraud_count


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division to handle division by zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value when denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator