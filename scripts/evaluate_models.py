"""
Script to evaluate models on the test set (Step 9).

This script loads tuned models, evaluates them on the held-out test set,
calculates metrics, generates visualizations, and compares model performance.
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from models.evaluation import evaluate_all_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Evaluate all tuned models on the test set."""
    
    # Set up paths
    test_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "engineered_features_test.parquet"
    models_dir = project_root / "models"
    output_dir = project_root / "docs" / "evaluation"
    
    # Configuration
    primary_metric = "roc_auc"  # Primary metric for model comparison
    
    print("="*80)
    print("MODEL EVALUATION - STEP 9")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load tuned models from disk")
    print("  2. Evaluate models on the held-out test set")
    print("  3. Calculate metrics (ROC-AUC, Precision, Recall, F1)")
    print("  4. Generate visualizations (ROC curves, PR curves, confusion matrices)")
    print("  5. Compare models and identify the best performer")
    print(f"\nPrimary metric: {primary_metric}")
    print("="*80)
    
    # Evaluate all models
    evaluation_results = evaluate_all_models(
        models_dir=models_dir,
        test_data_path=test_data_path,
        output_dir=output_dir,
        primary_metric=primary_metric
    )
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for model_name, results in evaluation_results.items():
        metrics = results["metrics"]
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  F1-Score:       {metrics['f1_score']:.4f}")
        print(f"  Avg Precision:  {metrics['average_precision']:.4f}")
    
    # Find best model
    best_model = max(
        evaluation_results.items(),
        key=lambda x: x[1]["metrics"][primary_metric]
    )
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model[0].upper().replace('_', ' ')}")
    print(f"  {primary_metric.upper()}: {best_model[1]['metrics'][primary_metric]:.4f}")
    print(f"{'='*80}")
    
    print(f"\nEvaluation results and plots saved to: {output_dir}")
    print("="*80)
    
    return evaluation_results

if __name__ == "__main__":
    evaluation_results = main()
