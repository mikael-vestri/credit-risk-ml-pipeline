"""Quick script to check date range in cleaned data."""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config

# Load cleaned data
cleaned_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "cleaned_data.parquet"
df = pd.read_parquet(cleaned_data_path)

# Check date column
date_col = dataset_config.DATE_COLUMN
if date_col in df.columns:
    dates = pd.to_datetime(df[date_col], errors='coerce')
    
    print("="*80)
    print("DATE RANGE ANALYSIS")
    print("="*80)
    print(f"\nDate column: {date_col}")
    print(f"Total rows: {len(df):,}")
    print(f"Rows with valid dates: {dates.notna().sum():,}")
    print(f"\nDate range:")
    print(f"  Min: {dates.min()}")
    print(f"  Max: {dates.max()}")
    
    # Suggest split dates
    print(f"\nSuggested split dates (for 80/20 split):")
    sorted_dates = dates.sort_values().dropna()
    if len(sorted_dates) > 0:
        split_idx_80 = int(len(sorted_dates) * 0.8)
        split_date_80 = sorted_dates.iloc[split_idx_80]
        print(f"  80/20 split: {split_date_80.strftime('%Y-%m-%d')}")
        print(f"    - Train: {len(sorted_dates[sorted_dates < split_date_80]):,} rows")
        print(f"    - Test: {len(sorted_dates[sorted_dates >= split_date_80]):,} rows")
        
        split_idx_70 = int(len(sorted_dates) * 0.7)
        split_date_70 = sorted_dates.iloc[split_idx_70]
        print(f"\n  70/30 split: {split_date_70.strftime('%Y-%m-%d')}")
        print(f"    - Train: {len(sorted_dates[sorted_dates < split_date_70]):,} rows")
        print(f"    - Test: {len(sorted_dates[sorted_dates >= split_date_70]):,} rows")
    
    print("="*80)
else:
    print(f"ERROR: Date column '{date_col}' not found in data!")

