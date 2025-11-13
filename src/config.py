# credit_card_fraud_detection/src/config.py
"""
Configuration settings for the fraud detection system - UPDATED
"""

import os
from pathlib import Path

# Path configurations
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Dataset configuration
DATASET_FILENAME = "PS_20174392719_1491204439457_log.csv"
TARGET_COLUMN = "isFraud"

# Columns to exclude from modeling (identifier columns)
EXCLUDED_COLUMNS = ['nameOrig', 'nameDest']

# Feature engineering configuration
FEATURE_ENGINEERING_FEATURES = [
    'is_transfer', 
    'is_cash_out',
    'amount_balance_ratio_orig', 
    'amount_balance_ratio_dest', 
    'balance_change_orig', 
    'balance_change_dest',
    'orig_zero_balance', 
    'dest_zero_balance',
    'account_emptied', 
    'log_amount', 
    'hour_of_day'
]

ORIGINAL_NUMERICAL_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'step'
]

ORIGINAL_CATEGORICAL_FEATURES = ['type']

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # Of training data

# Hyperparameter search space for Bayesian Optimization
XGBOOST_PARAM_SPACE = {
    'model__n_estimators': (100, 500), 
    'model__max_depth': (3, 8),
    'model__learning_rate': (0.01, 0.3, 'log-uniform'),
    'model__subsample': (0.6, 1.0),
    'model__colsample_bytree': (0.6, 1.0),
    'model__gamma': (0, 5),
    'model__reg_alpha': (0, 5),  
    'model__reg_lambda': (1, 5)   
}

# SMOTE configuration
SMOTE_SAMPLING_STRATEGY = 0.1  # 10% fraud rate in training

# Threshold optimization

THRESHOLD_RANGE = (0.01, 0.99, 0.01)
