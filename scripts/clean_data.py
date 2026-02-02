"""
Script to clean and transform raw data (Step 5).

Supports DEV and FULL execution modes:
- DEV: Temporally diverse sampling, fast iteration
- FULL: Chunked processing, production-ready
"""

import logging
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from data.pipeline import process_data_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Run data cleaning and transformation pipeline."""

    # Set up paths
    raw_data_path = project_root / dataset_config.RAW_DATA_PATH
    processed_data_path = project_root / dataset_config.PROCESSED_DATA_PATH
    validation_report_path = project_root / "docs" / "validation_report.json"

    # Execution mode
    # Change to "full" for production processing
    mode = "dev"  # Options: "dev" or "full"

    print("=" * 80)
    print("DATA CLEANING & TRANSFORMATION - STEP 5")
    print(f"EXECUTION MODE: {mode.upper()}")
    print("=" * 80)

    if mode == "dev":
        print("\nDEV MODE:")
        print("  - Temporally diverse sampling (10,000 rows)")
        print("  - Fast iteration for development")
        print("  - Relaxed validation")
    else:
        print("\nFULL MODE:")
        print("  - Chunked processing (full dataset)")
        print("  - Production-ready")
        print("  - Strict validation")

    print("=" * 80)

    # Run pipeline
    metadata = process_data_pipeline(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        validation_report_path=validation_report_path,
        mode=mode,
        sample_size=10000,  # For DEV mode
        chunk_size=10000,  # For FULL mode
        split_date=None,  # Will be computed from data
    )

    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"Mode: {metadata['mode'].upper()}")
    print(f"Date range: {metadata['date_range']['min']} to {metadata['date_range']['max']}")
    if "split_date" in metadata:
        print(
            f"Split date: {metadata['split_date']} ({metadata.get('split_date_source', 'unknown')})"
        )
    print("=" * 80)

    return metadata


if __name__ == "__main__":
    metadata = main()
