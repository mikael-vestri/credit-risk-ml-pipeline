"""
Script to perform SHAP-based model interpretability analysis (Step 10).

This script loads tuned models and performs SHAP analysis to understand
feature importance and individual predictions.
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from models.interpretability import explain_all_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Perform SHAP analysis on all tuned models."""
    
    # Set up paths
    train_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "engineered_features_train.parquet"
    test_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "engineered_features_test.parquet"
    models_dir = project_root / "models"
    output_dir = project_root / "docs" / "interpretability"
    
    # Configuration
    background_sample_size = 500  # Sample size for background data (for faster computation)
    explanation_sample_size = 200  # Sample size for explanation data
    
    print("="*80)
    print("MODEL INTERPRETABILITY - STEP 10")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load tuned models from disk")
    print("  2. Perform SHAP analysis on each model")
    print("  3. Generate feature importance rankings")
    print("  4. Create visualizations:")
    print("     - SHAP summary plots (beeswarm)")
    print("     - SHAP bar plots (feature importance)")
    print("     - SHAP waterfall plots (individual predictions)")
    print(f"\nConfiguration:")
    print(f"  Background sample size: {background_sample_size}")
    print(f"  Explanation sample size: {explanation_sample_size}")
    print("\nNote: SHAP analysis may take several minutes...")
    print("="*80)
    
    # Perform SHAP analysis
    shap_results = explain_all_models(
        models_dir=models_dir,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        background_sample_size=background_sample_size,
        explanation_sample_size=explanation_sample_size
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SHAP ANALYSIS SUMMARY")
    print("="*80)
    
    for model_name, results in shap_results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("  Top 5 Most Important Features:")
        top_features = results["feature_importance"].head(5)
        for idx, row in top_features.iterrows():
            print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"SHAP visualizations and results saved to: {output_dir}")
    print("="*80)
    
    return shap_results

if __name__ == "__main__":
    shap_results = main()
