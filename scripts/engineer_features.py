"""
Script to engineer features (Step 6).

Supports DEV and FULL execution modes:
- DEV: Fast iteration, relaxed validation
- FULL: Production-ready, strict validation
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config, feature_config
from features.pipeline import engineer_features_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run feature engineering pipeline."""
    
    # Set up paths
    cleaned_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "cleaned_data.parquet"
    output_path = project_root / dataset_config.PROCESSED_DATA_PATH / "engineered_features.parquet"
    
    # Execution mode
    # Change to "full" for production processing
    mode = "dev"  # Options: "dev" or "full"
    
    # Split date (None = compute from data)
    split_date = None  # Will be computed dynamically from data
    
    print("="*80)
    print("FEATURE ENGINEERING - STEP 6")
    print(f"EXECUTION MODE: {mode.upper()}")
    print("="*80)
    
    if mode == "dev":
        print("\nDEV MODE:")
        print("  - Fast iteration")
        print("  - Relaxed validation (warnings instead of errors)")
        print("  - Split date computed from data")
    else:
        print("\nFULL MODE:")
        print("  - Production-ready")
        print("  - Strict validation")
        print("  - Split date computed from data")
    
    print(f"\nSplit date: {split_date if split_date else 'Will be computed from data'}")
    print("="*80)
    
    # Run pipeline
    train_df, test_df, metadata = engineer_features_pipeline(
        cleaned_data_path=cleaned_data_path,
        date_column=dataset_config.DATE_COLUMN,
        output_path=output_path,
        excluded_features=feature_config.EXCLUDED_FEATURES,
        mode=mode,
        split_date=split_date,
        train_ratio=0.8
    )
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"Mode: {metadata['mode'].upper()}")
    print(f"Split date: {metadata['split_date']}")
    print(f"Date range: {metadata['date_range']['min']} to {metadata['date_range']['max']}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Features created: {metadata['features_created']}")
    print(f"Train target distribution:")
    print(train_df['target'].value_counts().to_dict())
    print("="*80)
    
    return train_df, test_df, metadata

if __name__ == "__main__":
    train_df, test_df, metadata = main()
