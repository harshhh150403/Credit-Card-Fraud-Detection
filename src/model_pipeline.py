# credit_card_fraud_detection/src/model_pipeline.py
"""
Model pipeline creation and training module
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
warnings.filterwarnings('ignore')

from .config import XGBOOST_PARAM_SPACE, SMOTE_SAMPLING_STRATEGY, RANDOM_STATE
from .preprocessing import DataPreprocessor
from .utils import calculate_class_weights


class FraudDetectionPipeline:
    """
    Main fraud detection pipeline with SMOTE and XGBoost
    """
    
    def __init__(self, use_smote: bool = True, optimize_threshold: bool = True):
        """
        Initialize the pipeline
        
        Args:
            use_smote: Whether to use SMOTE for oversampling
            optimize_threshold: Whether to optimize classification threshold
        """
        self.use_smote = use_smote
        self.optimize_threshold = optimize_threshold
        self.pipeline_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.threshold_optimizer_ = None
        
    def create_pipeline(self, scale_pos_weight: float = 1.0) -> Pipeline:
        """
        Create the full pipeline with preprocessing, SMOTE, and XGBoost
        
        Args:
            scale_pos_weight: Class weight for XGBoost
            
        Returns:
            Configured pipeline
        """
        # Create preprocessing step
        preprocessor = DataPreprocessor()
        
        # Create pipeline steps
        steps = [
            ('preprocessor', preprocessor)
        ]
        
        # Add SMOTE if enabled
        if self.use_smote:
            steps.append(('smote', SMOTE(
                sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                random_state=RANDOM_STATE
            )))
        
        # Add XGBoost model
        steps.append(('model', XGBClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False
        )))
        
        self.pipeline_ = Pipeline(steps)
        return self.pipeline_
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            cv: int = 5, n_iter: int = 50, n_jobs: int = -1) -> BayesSearchCV:
        """
        Perform Bayesian hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            n_iter: Number of optimization iterations
            n_jobs: Number of parallel jobs
            
        Returns:
            Fitted BayesSearchCV object
        """
        print("Starting Bayesian hyperparameter optimization...")
        
        # Calculate class weight for imbalance
        scale_pos_weight = calculate_class_weights(y_train)
        print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Create pipeline with calculated weight
        pipeline = self.create_pipeline(scale_pos_weight)
        
        # Convert param space to skopt format
        skopt_param_space = {}
        for param, space in XGBOOST_PARAM_SPACE.items():
            if isinstance(space, tuple):
                if len(space) == 2 and all(isinstance(x, (int, float)) for x in space):
                    if all(isinstance(x, int) for x in space):
                        skopt_param_space[param] = Integer(space[0], space[1])
                    else:
                        skopt_param_space[param] = Real(space[0], space[1], prior='uniform')
                elif len(space) == 3 and space[2] == 'log-uniform':
                    skopt_param_space[param] = Real(space[0], space[1], prior='log-uniform')
        
        # Perform Bayesian optimization
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=skopt_param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit the search
        bayes_search.fit(X_train, y_train)
        
        # Store results
        self.best_estimator_ = bayes_search.best_estimator_
        self.best_params_ = bayes_search.best_params_
        self.cv_results_ = bayes_search.cv_results_
        
        print(f"Best CV F1-score: {bayes_search.best_score_:.4f}")
        print("Best parameters:")
        for param, value in self.best_params_.items():
            print(f"  {param}: {value}")
        
        return bayes_search
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
           use_optimization: bool = True, **kwargs) -> 'FraudDetectionPipeline':
        """
        Fit the pipeline with optional hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training target
            use_optimization: Whether to use hyperparameter optimization
            **kwargs: Additional arguments for hyperparameter_tuning
            
        Returns:
            self: Fitted pipeline
        """
        if use_optimization:
            self.hyperparameter_tuning(X_train, y_train, **kwargs)
        else:
            # Fit with default parameters
            scale_pos_weight = calculate_class_weights(y_train)
            pipeline = self.create_pipeline(scale_pos_weight)
            self.best_estimator_ = pipeline.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: pd.DataFrame, use_optimized_threshold: bool = True) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            use_optimized_threshold: Whether to use optimized threshold
            
        Returns:
            Binary predictions
        """
        if self.best_estimator_ is None:
            raise ValueError("Pipeline not fitted yet")
        
        if use_optimized_threshold and self.threshold_optimizer_ is not None:
            y_pred_proba = self.predict_proba(X)
            return self.threshold_optimizer_.predict(y_pred_proba)
        else:
            return self.best_estimator_.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for class 1 (fraud)
        """
        if self.best_estimator_ is None:
            raise ValueError("Pipeline not fitted yet")
        
        return self.best_estimator_.predict_proba(X)[:, 1]
    
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, 
                         metric: str = 'f1') -> 'FraudDetectionPipeline':
        """
        Optimize classification threshold on validation set
        
        Args:
            X_val: Validation features
            y_val: Validation target
            metric: Metric to optimize
            
        Returns:
            self: Pipeline with optimized threshold
        """
        from .preprocessing import FraudThresholdOptimizer
        
        y_pred_proba = self.predict_proba(X_val)
        
        self.threshold_optimizer_ = FraudThresholdOptimizer(metric=metric)
        self.threshold_optimizer_.fit(y_val, y_pred_proba)
        
        return self