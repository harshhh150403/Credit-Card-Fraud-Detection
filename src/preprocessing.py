# credit_card_fraud_detection/src/preprocessing.py
"""
Data preprocessing module with custom transformers - FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import ORIGINAL_NUMERICAL_FEATURES, ORIGINAL_CATEGORICAL_FEATURES


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Main preprocessor that handles both original and engineered features - FIXED
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.column_transformer_ = None
        self.feature_engineer_ = None
        self.final_feature_names_ = None
        self.columns_to_drop_ = ['nameOrig', 'nameDest']  # Exclude identifier columns
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor - FIXED to exclude identifier columns
        """
        # First, apply feature engineering
        from .feature_engineering import FraudFeatureEngineer
        self.feature_engineer_ = FraudFeatureEngineer()
        X_engineered = self.feature_engineer_.fit_transform(X)
        
        # Remove identifier columns that shouldn't be used in modeling
        X_engineered = X_engineered.drop(columns=self.columns_to_drop_, errors='ignore')
        
        # Identify all feature columns after engineering
        all_numeric_features = []
        all_categorical_features = []
        
        for col in X_engineered.columns:
            if col in ORIGINAL_CATEGORICAL_FEATURES:
                all_categorical_features.append(col)
            elif X_engineered[col].dtype in ['object', 'category']:
                # Auto-detect other categorical columns
                all_categorical_features.append(col)
            else:
                # Everything else is treated as numeric
                all_numeric_features.append(col)
        
        print(f"Numeric features: {len(all_numeric_features)}")
        print(f"Categorical features: {len(all_categorical_features)}")
        print(f"Numeric features: {all_numeric_features}")
        print(f"Categorical features: {all_categorical_features}")
        
        # Create numeric transformer pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create categorical transformer pipeline  
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, all_numeric_features),
                ('cat', categorical_transformer, all_categorical_features)
            ],
            remainder='drop',  # This will drop any columns not explicitly handled
            n_jobs=-1
        )
        
        # Fit the column transformer
        self.column_transformer_.fit(X_engineered)
        
        # Get feature names after transformation
        self._set_feature_names(X_engineered)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input data - FIXED to exclude identifier columns
        """
        # Apply feature engineering
        X_engineered = self.feature_engineer_.transform(X)
        
        # Remove identifier columns
        X_engineered = X_engineered.drop(columns=self.columns_to_drop_, errors='ignore')
        
        # Apply column transformations
        X_transformed = self.column_transformer_.transform(X_engineered)
        
        return X_transformed
    
    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing
        """
        if self.final_feature_names_ is None:
            raise ValueError("Preprocessor not fitted yet")
        return self.final_feature_names_
    
    def _set_feature_names(self, X_engineered: pd.DataFrame) -> None:
        """Set final feature names after one-hot encoding"""
        # Remove identifier columns for feature naming
        X_engineered = X_engineered.drop(columns=self.columns_to_drop_, errors='ignore')
        
        # Get numeric feature names
        numeric_features = [col for col in X_engineered.columns 
                          if col not in ORIGINAL_CATEGORICAL_FEATURES and 
                          X_engineered[col].dtype not in ['object', 'category']]
        
        # Get categorical feature names after one-hot encoding
        categorical_features = []
        categorical_cols = [col for col in X_engineered.columns 
                           if col in ORIGINAL_CATEGORICAL_FEATURES or 
                           X_engineered[col].dtype in ['object', 'category']]
        
        for col in categorical_cols:
            if col in X_engineered.columns:
                # Get unique values for this column in training data
                unique_vals = X_engineered[col].unique()
                for val in sorted(unique_vals):
                    categorical_features.append(f"{col}_{val}")
        
        self.final_feature_names_ = numeric_features + categorical_features
        print(f"Total features after preprocessing: {len(self.final_feature_names_)}")


# Rest of the file remains the same...
class FraudThresholdOptimizer:
    """
    Optimizes classification threshold for fraud detection
    """
    
    def __init__(self, metric: str = 'f1'):
        """
        Initialize threshold optimizer
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall')
        """
        self.metric = metric
        self.best_threshold_ = None
        self.best_score_ = None
        
    def fit(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> 'FraudThresholdOptimizer':
        """
        Find optimal threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            self: Fitted optimizer
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if self.metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif self.metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif self.metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError("Metric must be 'f1', 'precision', or 'recall'")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.best_threshold_ = best_threshold
        self.best_score_ = best_score
        
        print(f"Best threshold: {best_threshold:.3f}, Best {self.metric}: {best_score:.4f}")
        
        return self
    
    def predict(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Predict using optimized threshold
        
        Args:
            y_pred_proba: Predicted probabilities
            
        Returns:
            Binary predictions
        """
        if self.best_threshold_ is None:
            raise ValueError("Threshold optimizer not fitted")
        
        return (y_pred_proba >= self.best_threshold_).astype(int)