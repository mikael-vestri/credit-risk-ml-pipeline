"""
Script to ingest raw data (Step 4).

This script demonstrates the data ingestion pipeline.
"""

import logging
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from data.ingestion import ingest_raw_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Run raw data ingestion."""

    # Set up paths
    raw_data_path = project_root / dataset_config.RAW_DATA_PATH
    metadata_path = project_root / "docs" / "ingestion_metadata.json"

    print("=" * 80)
    print("RAW DATA INGESTION - STEP 4")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Load raw data (unchanged)")
    print("  2. Extract schema and statistics")
    print("  3. Log metadata")
    print("  4. Save ingestion log")
    print("\nNote: For large datasets, consider using nrows parameter for testing")
    print("=" * 80)

    # For demonstration, we'll load a sample first
    # In production, you'd load the full dataset
    print("\nLoading sample (10,000 rows) for demonstration...")
    print("(Remove nrows parameter to load full dataset)")

    df, metadata = ingest_raw_data(
        raw_data_path=raw_data_path,
        output_metadata_path=metadata_path,
        nrows=10000,  # Remove this for full dataset
    )

    print("\n" + "=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {metadata['basic_stats']['memory_usage_mb']} MB")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 80)

    return df, metadata


if __name__ == "__main__":
    df, metadata = main()
