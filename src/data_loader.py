# credit_card_fraud_detection/src/data_loader.py
"""
Data loading and splitting module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from .utils import save_model, load_model


class DataLoader:
    """
    Handles data loading, cleaning, and splitting
    """
    
    def __init__(self, dataset_filename: str):
        """
        Initialize DataLoader
        
        Args:
            dataset_filename: Name of the dataset file
        """
        self.dataset_path = DATA_RAW_DIR / dataset_filename
        self.processed_path = DATA_PROCESSED_DIR / "cleaned_dataset.csv"
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset from CSV file
        
        Returns:
            Raw DataFrame
        """
        print("Loading raw dataset...")
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data cleaning operations
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
        
        # Filter only relevant transaction types (where fraud occurs)
        df_clean = df_clean[df_clean['type'].isin(['TRANSFER', 'CASH_OUT'])]
        print(f"After filtering TRANSFER and CASH_OUT: {df_clean.shape}")
        
        # Check for infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df_clean[col]).any():
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        
        # Check for missing values (though PaySim has none)
        missing_values = df_clean.isnull().sum()
        if missing_values.any():
            print(f"Missing values found: {missing_values[missing_values > 0]}")
            # For this dataset, we'll drop rows with missing values
            df_clean = df_clean.dropna()
        
        print(f"Final cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def save_cleaned_data(self, df: pd.DataFrame) -> None:
        """
        Save cleaned dataset to processed directory
        
        Args:
            df: Cleaned DataFrame
        """
        df.to_csv(self.processed_path, index=False)
        print(f"Cleaned dataset saved to: {self.processed_path}")
    
    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Load cleaned dataset from processed directory
        
        Returns:
            Cleaned DataFrame
        """
        print("Loading cleaned dataset...")
        try:
            df = pd.read_csv(self.processed_path)
            print(f"Cleaned dataset loaded. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Cleaned dataset not found at {self.processed_path}")
    
    def split_data(self, df: pd.DataFrame, validation: bool = False) -> Tuple:
        """
        Split data into features and target, then into train/test sets
        
        Args:
            df: Input DataFrame
            validation: Whether to create validation set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or 
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Splitting data into features and target...")
        
        # Separate features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Fraud rate in full dataset: {y.mean():.4f}")
        
        if not validation:
            # Single split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=TEST_SIZE, 
                stratify=y,
                random_state=RANDOM_STATE
            )
            
            print(f"Train set shape: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
            print(f"Test set shape: {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
            
            return X_train, X_test, y_train, y_test
        
        else:
            # Split into train+val and test, then split train+val into train and val
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=TEST_SIZE, 
                stratify=y,
                random_state=RANDOM_STATE
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=VALIDATION_SIZE,
                stratify=y_temp,
                random_state=RANDOM_STATE
            )
            
            print(f"Train set shape: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
            print(f"Validation set shape: {X_val.shape}, Fraud rate: {y_val.mean():.4f}")
            print(f"Test set shape: {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test