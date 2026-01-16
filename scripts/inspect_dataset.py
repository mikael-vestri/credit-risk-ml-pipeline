"""
Quick dataset inspection script for Step 3 validation.

This script loads and inspects the Lending Club dataset to:
- Understand the schema
- Identify columns
- Check for target variable
- Begin leakage identification
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config

def inspect_dataset():
    """Inspect the raw dataset."""
    
    # Find the dataset file
    raw_data_path = project_root / dataset_config.RAW_DATA_PATH
    
    # Check for gzipped file first
    gz_file = raw_data_path / "accepted_2007_to_2018Q4.csv.gz"
    csv_file = raw_data_path / "accepted_2007_to_2018Q4.csv" / "accepted_2007_to_2018Q4.csv"
    
    if gz_file.exists():
        print(f"Loading dataset from: {gz_file}")
        print("Note: This is a large file, loading may take a moment...")
        df = pd.read_csv(gz_file, compression='gzip', low_memory=False, nrows=10000)  # Sample first
    elif csv_file.exists():
        print(f"Loading dataset from: {csv_file}")
        df = pd.read_csv(csv_file, low_memory=False, nrows=10000)  # Sample first
    else:
        print(f"Error: Dataset file not found in {raw_data_path}")
        return
    
    print("\n" + "="*80)
    print("DATASET INSPECTION REPORT")
    print("="*80)
    
    # Basic info
    print(f"\n1. DATASET SHAPE (sample of 10k rows):")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Column info
    print(f"\n2. COLUMN INFORMATION:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"\n   Column names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:3d}. {col}")
    
    # Data types
    print(f"\n3. DATA TYPES:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Check for target column
    print(f"\n4. TARGET VARIABLE CHECK:")
    target_col = dataset_config.TARGET_COLUMN
    if target_col in df.columns:
        print(f"   [OK] Found target column: '{target_col}'")
        print(f"\n   Target value distribution:")
        value_counts = df[target_col].value_counts()
        for value, count in value_counts.items():
            print(f"   - {value}: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"   [ERROR] Target column '{target_col}' not found!")
        print(f"   Available columns containing 'status' or 'loan':")
        status_cols = [c for c in df.columns if 'status' in c.lower() or 'loan' in c.lower()]
        for col in status_cols:
            print(f"   - {col}")
    
    # Check for date column
    print(f"\n5. DATE COLUMN CHECK:")
    date_col = dataset_config.DATE_COLUMN
    if date_col in df.columns:
        print(f"   [OK] Found date column: '{date_col}'")
        print(f"   Date range (sample):")
        date_series = pd.to_datetime(df[date_col], errors='coerce')
        print(f"   Min: {date_series.min()}")
        print(f"   Max: {date_series.max()}")
        print(f"   Missing dates: {date_series.isna().sum()}")
    else:
        print(f"   [ERROR] Date column '{date_col}' not found!")
        print(f"   Available columns containing 'date' or 'd':")
        date_cols = [c for c in df.columns if 'date' in c.lower() or c.endswith('_d')]
        for col in date_cols[:10]:  # Show first 10
            print(f"   - {col}")
    
    # Check for known leakage columns
    print(f"\n6. LEAKAGE COLUMN CHECK:")
    known_leakage = dataset_config.KNOWN_LEAKAGE_COLUMNS
    found_leakage = [col for col in known_leakage if col in df.columns]
    missing_leakage = [col for col in known_leakage if col not in df.columns]
    
    print(f"   Found {len(found_leakage)}/{len(known_leakage)} suspected leakage columns:")
    for col in found_leakage:
        print(f"   [FOUND] {col}")
    
    if missing_leakage:
        print(f"\n   Not found (may have different names):")
        for col in missing_leakage[:5]:  # Show first 5
            print(f"   [NOT FOUND] {col}")
    
    # Look for columns with payment/recovery keywords
    print(f"\n7. POTENTIAL LEAKAGE KEYWORDS:")
    leakage_keywords = ['pymnt', 'payment', 'recover', 'collection', 'out_prncp', 'total_rec']
    for keyword in leakage_keywords:
        matching_cols = [c for c in df.columns if keyword.lower() in c.lower()]
        if matching_cols:
            print(f"   '{keyword}': {len(matching_cols)} columns")
            for col in matching_cols[:3]:  # Show first 3
                print(f"     - {col}")
    
    # Missing values overview
    print(f"\n8. MISSING VALUES OVERVIEW:")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        print(f"   Columns with >50% missing values: {len(high_missing)}")
        for col, pct in high_missing.head(10).items():
            print(f"   - {col}: {pct:.1f}% missing")
    else:
        print("   No columns with >50% missing values")
    
    print("\n" + "="*80)
    print("Inspection complete!")
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = inspect_dataset()

