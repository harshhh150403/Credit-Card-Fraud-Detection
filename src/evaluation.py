# credit_card_fraud_detection/src/evaluation.py
"""
Model evaluation and metrics calculation module
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           precision_score, recall_score, f1_score)
from typing import Dict, Tuple, Any
import json

from .utils import safe_divide


class ModelEvaluator:
    """
    Comprehensive model evaluation for fraud detection
    """
    
    def __init__(self, model, threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            model: Trained model or pipeline
            threshold: Classification threshold
        """
        self.model = model
        self.threshold = threshold
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                threshold: float = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test target
            threshold: Optional custom threshold
            
        Returns:
            Dictionary with all evaluation metrics
        """
        threshold = threshold or self.threshold
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Add confusion matrix details
        cm = confusion_matrix(y_test, y_pred)
        metrics.update(self._confusion_matrix_metrics(cm))
        
        # Add classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Rates
        metrics['false_positive_rate'] = safe_divide(fp, fp + tn)
        metrics['false_negative_rate'] = safe_divide(fn, fn + tp)
        metrics['true_positive_rate'] = metrics['recall']  # Same as recall
        metrics['true_negative_rate'] = safe_divide(tn, tn + fp)
        
        # Business-oriented metrics
        metrics['fraud_capture_rate'] = metrics['recall']
        metrics['false_alarm_rate'] = metrics['false_positive_rate']
        metrics['accuracy'] = safe_divide(tp + tn, tp + tn + fp + fn)
        
        return metrics
    
    def _confusion_matrix_metrics(self, cm: np.ndarray) -> Dict[str, Any]:
        """
        Extract detailed metrics from confusion matrix
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Dictionary with confusion matrix details
        """
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'total_samples': int(tn + fp + fn + tp),
            'fraud_prevalence': safe_divide(tp + fn, tn + fp + fn + tp)
        }
    
    def compare_thresholds(self, X_test: pd.DataFrame, y_test: pd.Series, 
                         thresholds: list = None) -> pd.DataFrame:
        """
        Compare performance across different thresholds
        
        Args:
            X_test: Test features
            y_test: Test target
            thresholds: List of thresholds to try
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.05)
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model must support predict_proba for threshold comparison")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def print_detailed_report(self, metrics: Dict[str, Any], 
                            title: str = "Model Evaluation Results") -> None:
        """
        Print comprehensive evaluation report
        
        Args:
            metrics: Dictionary of evaluation metrics
            title: Report title
        """
        print("=" * 70)
        print(f"{title}")
        print("=" * 70)
        
        # Key metrics
        print("\nKEY METRICS:")
        print(f"F1-Score:        {metrics['f1_score']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC:         {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"PR AUC:          {metrics['pr_auc']:.4f}")
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print(f"\nCONFUSION MATRIX:")
        print(f"True Negatives:  {cm['true_negatives']:>8}")
        print(f"False Positives: {cm['false_positives']:>8}")
        print(f"False Negatives: {cm['false_negatives']:>8}")
        print(f"True Positives:  {cm['true_positives']:>8}")
        
        # Rates
        print(f"\nRATES:")
        print(f"True Positive Rate (Recall): {metrics['true_positive_rate']:.4f}")
        print(f"True Negative Rate:          {metrics['true_negative_rate']:.4f}")
        print(f"False Positive Rate:         {metrics['false_positive_rate']:.4f}")
        print(f"False Negative Rate:         {metrics['false_negative_rate']:.4f}")
        
        # Business metrics
        print(f"\nBUSINESS METRICS:")
        print(f"Fraud Capture Rate: {metrics['fraud_capture_rate']:.4f}")
        print(f"False Alarm Rate:   {metrics['false_alarm_rate']:.4f}")
        
        # Classification report
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        class_report = metrics['classification_report']
        for label, scores in class_report.items():
            if label in ['0', '1']:
                label_name = 'Legitimate' if label == '0' else 'Fraud'
                print(f"\n{label_name}:")
                print(f"  Precision: {scores['precision']:.4f}")
                print(f"  Recall:    {scores['recall']:.4f}")
                print(f"  F1-Score:  {scores['f1-score']:.4f}")
                print(f"  Support:   {scores['support']:>8}")
    
    def save_evaluation_report(self, metrics: Dict[str, Any], 
                            filename: str = "evaluation_report.json") -> None:
        """
        Save evaluation results to file
        
        Args:
            metrics: Dictionary of evaluation metrics
            filename: Output filename
        """
        from .utils import save_results
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):  # numpy types
                serializable_metrics[key] = value.item() if hasattr(value, 'item') else value
            elif isinstance(value, dict):
                serializable_metrics[key] = {
                    k: v.item() if hasattr(v, 'item') else v 
                    for k, v in value.items()
                }
            else:
                serializable_metrics[key] = value
        
        save_results(serializable_metrics, filename)