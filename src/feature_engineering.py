# credit_card_fraud_detection/src/feature_engineering.py
"""
Custom feature engineering transformer - FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings
warnings.filterwarnings('ignore')

from .config import FEATURE_ENGINEERING_FEATURES
from .utils import safe_divide


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating fraud detection features
    """
    
    def __init__(self, features_to_create: list = None):
        """
        Initialize the feature engineer
        
        Args:
            features_to_create: List of feature names to create. 
                              If None, uses default from config
        """
        self.features_to_create = features_to_create or FEATURE_ENGINEERING_FEATURES
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FraudFeatureEngineer':
        """
        Fit the transformer (no fitting needed for this feature engineering)
        """
        # Store feature names for reference
        self.feature_names_ = self._get_feature_names()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data by creating engineered features
        """
        X_transformed = X.copy()
        
        # Create all specified features
        feature_creation_methods = {
            'is_transfer': self._create_is_transfer,
            'is_cash_out': self._create_is_cash_out,
            'amount_balance_ratio_orig': self._create_amount_balance_ratio_orig,
            'amount_balance_ratio_dest': self._create_amount_balance_ratio_dest,
            'balance_change_orig': self._create_balance_change_orig,
            'balance_change_dest': self._create_balance_change_dest,
            'orig_zero_balance': self._create_orig_zero_balance,
            'dest_zero_balance': self._create_dest_zero_balance,
            'account_emptied': self._create_account_emptied,
            'log_amount': self._create_log_amount,
            'hour_of_day': self._create_hour_of_day
        }
        
        # Only create features that are requested and don't already exist
        for feature_name in self.features_to_create:
            if feature_name not in X_transformed.columns and feature_name in feature_creation_methods:
                X_transformed[feature_name] = feature_creation_methods[feature_name](X)
        
        return X_transformed
    
    def get_feature_names(self) -> list:
        """
        Get names of engineered features
        """
        if self.feature_names_ is None:
            self.feature_names_ = self._get_feature_names()
        return self.feature_names_
    
    def _get_feature_names(self) -> list:
        """Generate feature names based on features to create"""
        return self.features_to_create
    
    # Feature creation methods (UNCHANGED)
    def _create_is_transfer(self, X: pd.DataFrame) -> pd.Series:
        return (X['type'] == 'TRANSFER').astype(int)
    
    def _create_is_cash_out(self, X: pd.DataFrame) -> pd.Series:
        return (X['type'] == 'CASH_OUT').astype(int)
    
    def _create_amount_balance_ratio_orig(self, X: pd.DataFrame) -> pd.Series:
        return X['amount'] / (X['oldbalanceOrg'] + 1)
    
    def _create_amount_balance_ratio_dest(self, X: pd.DataFrame) -> pd.Series:
        return X['amount'] / (X['oldbalanceDest'] + 1)
    
    def _create_balance_change_orig(self, X: pd.DataFrame) -> pd.Series:
        return X['newbalanceOrig'] - X['oldbalanceOrg']
    
    def _create_balance_change_dest(self, X: pd.DataFrame) -> pd.Series:
        return X['newbalanceDest'] - X['oldbalanceDest']
    
    def _create_orig_zero_balance(self, X: pd.DataFrame) -> pd.Series:
        return (X['oldbalanceOrg'] == 0).astype(int)
    
    def _create_dest_zero_balance(self, X: pd.DataFrame) -> pd.Series:
        return (X['oldbalanceDest'] == 0).astype(int)
    
    def _create_account_emptied(self, X: pd.DataFrame) -> pd.Series:
        ratio = self._create_amount_balance_ratio_orig(X)
        return ((ratio >= 0.99) & (ratio <= 1.01)).astype(int)
    
    def _create_log_amount(self, X: pd.DataFrame) -> pd.Series:
        return np.log1p(X['amount'])
    
    def _create_hour_of_day(self, X: pd.DataFrame) -> pd.Series:
        return X['step'] % 24
    
    def add_custom_feature(self, feature_name: str, creation_function: callable) -> None:
        """
        Add a custom feature creation method
        """
        if not hasattr(self, '_custom_features'):
            self._custom_features = {}
        
        self._custom_features[feature_name] = creation_function
        self.features_to_create.append(feature_name)
        self.feature_names_ = self._get_feature_names()