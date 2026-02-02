"""Check date range in raw dataset (without loading full data)."""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from data.ingestion import get_dataset_file_path

# Get file path
raw_data_path = project_root / dataset_config.RAW_DATA_PATH
file_path = get_dataset_file_path(raw_data_path)

# Determine compression
compression = "gzip" if str(file_path).endswith(".gz") else None

print("=" * 80)
print("CHECKING DATE RANGE IN RAW DATASET")
print("=" * 80)
print(f"\nFile: {file_path.name}")
print("Loading sample to check date range...")

# Load a larger sample to see date distribution
df_sample = pd.read_csv(
    file_path,
    nrows=100000,  # Larger sample to see date diversity
    compression=compression,
    low_memory=False,
)

date_col = dataset_config.DATE_COLUMN
if date_col in df_sample.columns:
    dates = pd.to_datetime(df_sample[date_col], errors="coerce")

    print(f"\nSample size: {len(df_sample):,} rows")
    print(f"Date column: {date_col}")
    print("\nDate range in sample:")
    print(f"  Min: {dates.min()}")
    print(f"  Max: {dates.max()}")

    # Show date distribution
    print("\nDate distribution (top 10 dates):")
    date_counts = dates.value_counts().head(10)
    for date, count in date_counts.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {count:,} rows")

    # Suggest split date
    sorted_dates = dates.sort_values().dropna()
    if len(sorted_dates) > 0:
        split_idx_80 = int(len(sorted_dates) * 0.8)
        split_date_80 = sorted_dates.iloc[split_idx_80]
        print(f"\nSuggested 80/20 split date: {split_date_80.strftime('%Y-%m-%d')}")
        print("  (Based on sample - actual may vary with full dataset)")

    print("=" * 80)
    print("\nNote: This is based on a 100k row sample.")
    print("For the full dataset, we should calculate split date from all data.")
else:
    print(f"ERROR: Date column '{date_col}' not found!")
