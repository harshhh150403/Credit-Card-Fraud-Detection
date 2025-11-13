# credit_card_fraud_detection/src/threshold_optimization.py
"""
Threshold optimization for fraud classification
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import THRESHOLD_RANGE


class ThresholdOptimizer:
    """
    Optimizes classification threshold for imbalanced fraud detection
    """
    
    def __init__(self, metric: str = 'f1', threshold_range: tuple = None):
        """
        Initialize threshold optimizer
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'custom')
            threshold_range: Range of thresholds to search (start, stop, step)
        """
        self.metric = metric
        self.threshold_range = threshold_range or THRESHOLD_RANGE
        self.best_threshold_ = None
        self.best_score_ = None
        self.threshold_scores_ = None
        
    def find_optimal_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                             custom_scorer: callable = None) -> 'ThresholdOptimizer':
        """
        Find optimal threshold for classification
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            custom_scorer: Custom scoring function (if metric='custom')
            
        Returns:
            self: Fitted optimizer
        """
        start, stop, step = self.threshold_range
        thresholds = np.arange(start, stop, step)
        
        best_score = -1
        best_threshold = 0.5
        scores = []
        
        print("Searching for optimal threshold...")
        for threshold in tqdm(thresholds):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if self.metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif self.metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif self.metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif self.metric == 'custom' and custom_scorer is not None:
                score = custom_scorer(y_true, y_pred)
            else:
                raise ValueError("Invalid metric or missing custom scorer")
            
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.best_threshold_ = best_threshold
        self.best_score_ = best_score
        self.threshold_scores_ = pd.DataFrame({
            'threshold': thresholds,
            'score': scores
        })
        
        print(f"Optimal threshold: {best_threshold:.3f}")
        print(f"Best {self.metric} score: {best_score:.4f}")
        
        return self
    
    def plot_threshold_analysis(self, save_path: str = None) -> None:
        """
        Plot threshold analysis (text-based for non-visual environment)
        
        Args:
            save_path: Path to save the plot data
        """
        if self.threshold_scores_ is None:
            raise ValueError("Must run find_optimal_threshold first")
        
        df = self.threshold_scores_
        
        print("\nTHRESHOLD ANALYSIS SUMMARY:")
        print("=" * 50)
        print(f"Best threshold: {self.best_threshold_:.3f}")
        print(f"Best {self.metric} score: {self.best_score_:.4f}")
        
        # Show top 10 thresholds
        top_thresholds = df.nlargest(10, 'score')
        print(f"\nTop 10 thresholds by {self.metric}:")
        for _, row in top_thresholds.iterrows():
            marker = " *" if row['threshold'] == self.best_threshold_ else ""
            print(f"  Threshold: {row['threshold']:.3f} -> Score: {row['score']:.4f}{marker}")
        
        # Save to file if requested
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nThreshold analysis saved to: {save_path}")
    
    def apply_threshold(self, y_pred_proba: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Apply threshold to probabilities to get binary predictions
        
        Args:
            y_pred_proba: Predicted probabilities
            threshold: Threshold to use (defaults to optimal threshold)
            
        Returns:
            Binary predictions
        """
        if threshold is None:
            if self.best_threshold_ is None:
                raise ValueError("No optimal threshold found. Run find_optimal_threshold first.")
            threshold = self.best_threshold_
        
        return (y_pred_proba >= threshold).astype(int)
    
    def get_threshold_stats(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                          threshold: float = None) -> dict:
        """
        Get comprehensive statistics for a given threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Threshold to evaluate
            
        Returns:
            Dictionary with comprehensive metrics
        """
        if threshold is None:
            threshold = self.best_threshold_
        
        y_pred = self.apply_threshold(y_pred_proba, threshold)
        
        from sklearn.metrics import (confusion_matrix, precision_score, 
                                   recall_score, f1_score, roc_auc_score)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        stats = {
            'threshold': threshold,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
        
        return stats


class BusinessDrivenThresholdOptimizer(ThresholdOptimizer):
    """
    Threshold optimizer that considers business costs
    """
    
    def __init__(self, false_positive_cost: float = 10, 
                 false_negative_cost: float = 100,
                 true_positive_benefit: float = 50):
        """
        Initialize business cost optimizer
        
        Args:
            false_positive_cost: Cost of false positive (investigating legitimate transaction)
            false_negative_cost: Cost of false negative (missing fraud)
            true_positive_benefit: Benefit of true positive (catching fraud)
        """
        super().__init__(metric='custom')
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        self.tp_benefit = true_positive_benefit
    
    def _business_scorer(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Custom business cost scoring function
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Business score (higher is better)
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fp * self.fp_cost) + (fn * self.fn_cost) - (tp * self.tp_benefit)
        
        # Convert to score (higher is better)
        return -total_cost  # Negative because we want to minimize cost
    
    def find_optimal_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> 'BusinessDrivenThresholdOptimizer':
        """
        Find optimal threshold based on business costs
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            self: Fitted optimizer
        """
        return super().find_optimal_threshold(y_true, y_pred_proba, self._business_scorer)