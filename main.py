# credit_card_fraud_detection/main.py
"""
Main script to run the complete fraud detection pipeline
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.model_pipeline import FraudDetectionPipeline
from src.evaluation import ModelEvaluator
from src.threshold_optimization import ThresholdOptimizer
from src.utils import save_model, save_results
from src.config import DATASET_FILENAME, RANDOM_STATE


def main():
    """
    Main function to run the complete fraud detection pipeline
    """
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 70)
    
    try:
        # Step 1: Load and prepare data
        print("\n1. LOADING AND PREPARING DATA")
        print("-" * 40)
        
        data_loader = DataLoader(DATASET_FILENAME)
        
        # Load and clean data
        raw_data = data_loader.load_raw_data()
        cleaned_data = data_loader.clean_data(raw_data)
        data_loader.save_cleaned_data(cleaned_data)
        
        # Split data
        X_train, X_test, y_train, y_test = data_loader.split_data(cleaned_data, validation=False)
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"Fraud rate in training: {y_train.mean():.4f}")
        print(f"Fraud rate in test: {y_test.mean():.4f}")
        
        # Step 2: Train model with hyperparameter optimization
        print("\n2. TRAINING MODEL WITH HYPERPARAMETER OPTIMIZATION")
        print("-" * 40)
        
        pipeline = FraudDetectionPipeline(use_smote=True, optimize_threshold=True)
        
        # Fit with Bayesian optimization (reduce n_iter for faster testing)
        pipeline.fit(
            X_train, y_train,
            use_optimization=True,
            cv=3,  # Reduced for speed
            n_iter=20,  # Reduced for speed
            n_jobs=-1
        )
        
        # Step 3: Initial evaluation with default threshold (0.5)
        print("\n3. INITIAL EVALUATION (Threshold = 0.5)")
        print("-" * 40)
        
        evaluator = ModelEvaluator(pipeline.best_estimator_, threshold=0.5)
        initial_metrics = evaluator.evaluate(X_test, y_test)
        evaluator.print_detailed_report(initial_metrics, "Initial Evaluation (Threshold = 0.5)")
        
        # Step 4: Threshold optimization
        print("\n4. THRESHOLD OPTIMIZATION")
        print("-" * 40)
        
        # Get probabilities for threshold optimization
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Find optimal threshold
        threshold_optimizer = ThresholdOptimizer(metric='f1')
        threshold_optimizer.find_optimal_threshold(y_test, y_pred_proba)
        threshold_optimizer.plot_threshold_analysis()
        
        optimal_threshold = threshold_optimizer.best_threshold_
        
        # Step 5: Evaluation with optimized threshold
        print("\n5. FINAL EVALUATION (Optimized Threshold)")
        print("-" * 40)
        
        final_metrics = evaluator.evaluate(X_test, y_test, threshold=optimal_threshold)
        evaluator.print_detailed_report(final_metrics, f"Final Evaluation (Threshold = {optimal_threshold:.3f})")
        
        # Step 6: Compare before and after threshold optimization
        print("\n6. THRESHOLD OPTIMIZATION COMPARISON")
        print("-" * 40)
        
        print("BEFORE optimization (Threshold = 0.5):")
        print(f"  F1-Score:    {initial_metrics['f1_score']:.4f}")
        print(f"  Precision:   {initial_metrics['precision']:.4f}")
        print(f"  Recall:      {initial_metrics['recall']:.4f}")
        
        print("AFTER optimization:")
        print(f"  F1-Score:    {final_metrics['f1_score']:.4f}")
        print(f"  Precision:   {final_metrics['precision']:.4f}")
        print(f"  Recall:      {final_metrics['recall']:.4f}")
        
        improvement = final_metrics['f1_score'] - initial_metrics['f1_score']
        print(f"F1-Score improvement: {improvement:+.4f}")
        
        # Step 7: Save model and results
        print("\n7. SAVING MODEL AND RESULTS")
        print("-" * 40)
        
        # Save the best model
        save_model(pipeline.best_estimator_, "best_fraud_model.pkl")
        
        # Save threshold optimizer
        save_model(threshold_optimizer, "threshold_optimizer.pkl")
        
        # Save comprehensive results
        results = {
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'optimal_threshold': optimal_threshold,
            'model_parameters': pipeline.best_params_,
            'data_info': {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'training_fraud_rate': y_train.mean(),
                'test_fraud_rate': y_test.mean()
            }
        }
        
        save_results(results, "pipeline_results.json")
        
        # Step 8: Final summary
        print("\n8. PIPELINE COMPLETED SUCCESSFULLY!")
        print("-" * 40)
        print("Saved artifacts:")
        print("  ✓ models/best_fraud_model.pkl")
        print("  ✓ models/threshold_optimizer.pkl") 
        print("  ✓ results/pipeline_results.json")
        print(f"  ✓ Optimal threshold: {optimal_threshold:.3f}")
        print(f"  ✓ Final F1-score: {final_metrics['f1_score']:.4f}")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {str(e)}")
        raise


if __name__ == "__main__":
    main()