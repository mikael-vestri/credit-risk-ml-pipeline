"""
Comprehensive dataset validation script for Step 3.

This script performs scalable validation using a layered approach:
- Header-only validation (exact, fast) - no data loading
- Sample-based validation (approximate, fast) - limited rows loaded
- Chunked analysis (when needed) - streaming for large datasets

Dataset validation is performed using a layered approach (header-only and 
sample-based) to ensure scalability to multi-GB datasets without requiring 
distributed infrastructure.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config

# Validation configuration
VALIDATION_CONFIG = {
    "sample_size": 50000,  # Rows to sample for approximate stats
    "chunk_size": 10000,   # For chunked operations (if needed)
    "skip_expensive_ops": True  # Skip duplicate detection, etc.
}


def find_dataset_file() -> Path:
    """Find the dataset file."""
    raw_data_path = project_root / dataset_config.RAW_DATA_PATH
    
    gz_file = raw_data_path / "accepted_2007_to_2018Q4.csv.gz"
    csv_file = raw_data_path / "accepted_2007_to_2018Q4.csv" / "accepted_2007_to_2018Q4.csv"
    
    if gz_file.exists():
        return gz_file
    elif csv_file.exists():
        return csv_file
    else:
        raise FileNotFoundError(f"Dataset file not found in {raw_data_path}")


def validate_dataset_header_only(file_path: Path) -> Dict:
    """
    Header-only validation - exact, fast, no data loading.
    
    Uses pd.read_csv(..., nrows=0) to get schema without loading data.
    
    Returns
    -------
    Dict with schema information (exact)
    """
    print("\n[HEADER] Validating dataset schema (header-only, exact)...")
    print("-" * 80)
    
    # Determine compression
    compression = 'gzip' if str(file_path).endswith('.gz') else None
    
    # Read header only
    df_header = pd.read_csv(
        file_path, 
        nrows=0, 
        compression=compression,
        low_memory=False
    )
    
    schema_info = {
        "total_columns": len(df_header.columns),
        "column_names": list(df_header.columns),
        "data_types": {col: str(dtype) for col, dtype in df_header.dtypes.items()},
        "validation_type": "exact",
        "note": "Header-only validation - exact column names and types"
    }
    
    # Data type summary
    dtype_counts = df_header.dtypes.value_counts().to_dict()
    schema_info["dtype_summary"] = {str(k): int(v) for k, v in dtype_counts.items()}
    
    print(f"   [HEADER] Total columns: {schema_info['total_columns']}")
    print(f"   [HEADER] Data types: {dtype_counts}")
    print(f"   [HEADER] Validation type: EXACT (header-only)")
    
    return schema_info


def identify_leakage_columns(column_names: List[str]) -> Dict[str, List[str]]:
    """
    Identify leakage columns based on column names only (no data loading).
    
    Parameters
    ----------
    column_names : List[str]
        List of column names from header
        
    Returns
    -------
    Dict with leakage categories
    """
    print("\n[HEADER] Identifying leakage columns (header-based, exact)...")
    print("-" * 80)
    
    high_risk = []
    medium_risk = []
    safe = []
    
    # Known high-risk leakage columns (post-loan information)
    high_risk_keywords = [
        'total_pymnt', 'total_rec', 'recoveries', 'collection_recovery',
        'last_pymnt', 'next_pymnt', 'out_prncp', 'hardship_', 'settlement_',
        'payment_plan', 'deferral', 'debt_settlement'
    ]
    
    # Medium-risk (need review)
    medium_risk_keywords = ['last_credit_pull', 'last_fico', 'issue_d']
    
    for col in column_names:
        col_lower = col.lower()
        
        # Check against known leakage list
        if col in dataset_config.KNOWN_LEAKAGE_COLUMNS:
            high_risk.append(col)
        # Check keywords
        elif any(keyword in col_lower for keyword in high_risk_keywords):
            high_risk.append(col)
        # Check medium-risk keywords
        elif any(keyword in col_lower for keyword in medium_risk_keywords):
            if col not in high_risk:
                medium_risk.append(col)
        # Safe columns (application-time information)
        else:
            safe.append(col)
    
    # Remove target column from safe (it's the target!)
    if dataset_config.TARGET_COLUMN in safe:
        safe.remove(dataset_config.TARGET_COLUMN)
    
    leakage_report = {
        "high_risk": sorted(high_risk),
        "medium_risk": sorted(medium_risk),
        "safe": sorted(safe),
        "validation_type": "exact",
        "note": "Leakage identification based on column names only"
    }
    
    print(f"   [HEADER] High-risk leakage columns: {len(leakage_report['high_risk'])}")
    for col in leakage_report['high_risk'][:10]:  # Show first 10
        print(f"     - {col}")
    if len(leakage_report['high_risk']) > 10:
        print(f"     ... and {len(leakage_report['high_risk']) - 10} more")
    
    print(f"   [HEADER] Medium-risk columns: {len(leakage_report['medium_risk'])}")
    print(f"   [HEADER] Safe columns: {len(leakage_report['safe'])}")
    print(f"   [HEADER] Validation type: EXACT (header-based)")
    
    return leakage_report


def validate_dataset_sample(
    file_path: Path, 
    sample_size: int = VALIDATION_CONFIG["sample_size"]
) -> Dict:
    """
    Sample-based validation - approximate, fast.
    
    Uses pd.read_csv(..., nrows=sample_size) to get approximate statistics.
    
    Parameters
    ----------
    file_path : Path
        Path to dataset file
    sample_size : int
        Number of rows to sample
        
    Returns
    -------
    Dict with approximate statistics
    """
    print(f"\n[SAMPLE: {sample_size} rows] Validating dataset (sample-based, approximate)...")
    print("-" * 80)
    
    # Determine compression
    compression = 'gzip' if str(file_path).endswith('.gz') else None
    
    # Read sample
    df_sample = pd.read_csv(
        file_path,
        nrows=sample_size,
        compression=compression,
        low_memory=False
    )
    
    print(f"   [SAMPLE: {sample_size} rows] Loaded {len(df_sample):,} rows for analysis")
    print(f"   [SAMPLE: {sample_size} rows] Validation type: APPROXIMATE (sample-based)")
    
    sample_info = {
        "sample_size": len(df_sample),
        "validation_type": "approximate",
        "note": f"Statistics computed on sample of {len(df_sample):,} rows"
    }
    
    # Missing values (approximate)
    print(f"\n   [SAMPLE: {sample_size} rows] Computing missing value statistics...")
    missing_pct = (df_sample.isnull().sum() / len(df_sample) * 100).sort_values(ascending=False)
    sample_info["missing_values"] = {
        col: float(pct) for col, pct in missing_pct.head(20).items()
    }
    sample_info["high_missing_columns"] = [
        col for col, pct in missing_pct.items() if pct > 50
    ]
    
    print(f"   [SAMPLE: {sample_size} rows] Columns with >50% missing: {len(sample_info['high_missing_columns'])}")
    print(f"   [SAMPLE: {sample_size} rows] Top 10 columns by missing %:")
    for col, pct in list(missing_pct.head(10).items()):
        print(f"     - {col}: {pct:.1f}% (approximate)")
    
    # Target distribution (approximate)
    if dataset_config.TARGET_COLUMN in df_sample.columns:
        print(f"\n   [SAMPLE: {sample_size} rows] Analyzing target variable...")
        target_dist = df_sample[dataset_config.TARGET_COLUMN].value_counts()
        sample_info["target_distribution"] = {
            str(k): {
                "count": int(v),
                "percentage": float(v / len(df_sample) * 100)
            }
            for k, v in target_dist.items()
        }
        
        # Ongoing loans (approximate)
        ongoing_statuses = ["Current", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"]
        ongoing_count = df_sample[df_sample[dataset_config.TARGET_COLUMN].isin(ongoing_statuses)].shape[0]
        sample_info["ongoing_loans"] = {
            "count": int(ongoing_count),
            "percentage": float(ongoing_count / len(df_sample) * 100)
        }
        
        print(f"   [SAMPLE: {sample_size} rows] Target distribution (approximate):")
        for value, info in sample_info["target_distribution"].items():
            print(f"     - {value}: {info['count']:,} ({info['percentage']:.2f}%) [APPROXIMATE]")
        print(f"   [SAMPLE: {sample_size} rows] Ongoing loans: {ongoing_count:,} ({ongoing_count/len(df_sample)*100:.2f}%) [APPROXIMATE]")
    else:
        print(f"   [SAMPLE: {sample_size} rows] [WARNING] Target column '{dataset_config.TARGET_COLUMN}' not found!")
        sample_info["target_distribution"] = {}
    
    # Date analysis (approximate)
    if dataset_config.DATE_COLUMN in df_sample.columns:
        print(f"\n   [SAMPLE: {sample_size} rows] Analyzing date column...")
        date_series = pd.to_datetime(df_sample[dataset_config.DATE_COLUMN], errors='coerce')
        
        sample_info["date_analysis"] = {
            "min_date": str(date_series.min()) if not pd.isna(date_series.min()) else None,
            "max_date": str(date_series.max()) if not pd.isna(date_series.max()) else None,
            "missing_count": int(date_series.isna().sum()),
            "missing_percentage": float(date_series.isna().sum() / len(df_sample) * 100)
        }
        
        # Suggest split date (approximate, based on sample)
        date_sorted = date_series.sort_values().dropna()
        if len(date_sorted) > 0:
            split_idx = int(len(date_sorted) * 0.8)
            suggested_split = date_sorted.iloc[split_idx]
            sample_info["suggested_split_date"] = suggested_split.strftime('%Y-%m-%d')
            
            train_count = (date_series < suggested_split).sum()
            test_count = (date_series >= suggested_split).sum()
            sample_info["suggested_split"] = {
                "split_date": suggested_split.strftime('%Y-%m-%d'),
                "train_count": int(train_count),
                "train_percentage": float(train_count / len(df_sample) * 100),
                "test_count": int(test_count),
                "test_percentage": float(test_count / len(df_sample) * 100),
                "note": "Based on sample - actual split may differ"
            }
            
            print(f"   [SAMPLE: {sample_size} rows] Date range: {date_series.min()} to {date_series.max()} [APPROXIMATE]")
            print(f"   [SAMPLE: {sample_size} rows] Missing dates: {date_series.isna().sum():,} ({date_series.isna().sum()/len(df_sample)*100:.2f}%) [APPROXIMATE]")
            print(f"   [SAMPLE: {sample_size} rows] Suggested split date: {suggested_split.strftime('%Y-%m-%d')} [APPROXIMATE]")
            print(f"   [SAMPLE: {sample_size} rows] Train: {train_count:,} ({train_count/len(df_sample)*100:.2f}%) [APPROXIMATE]")
            print(f"   [SAMPLE: {sample_size} rows] Test: {test_count:,} ({test_count/len(df_sample)*100:.2f}%) [APPROXIMATE]")
    else:
        print(f"   [SAMPLE: {sample_size} rows] [WARNING] Date column '{dataset_config.DATE_COLUMN}' not found!")
        sample_info["date_analysis"] = {}
    
    # Skip expensive operations
    if VALIDATION_CONFIG["skip_expensive_ops"]:
        print(f"\n   [SAMPLE: {sample_size} rows] Skipping expensive operations (duplicate detection, etc.)")
        sample_info["duplicate_rows"] = "skipped"
        sample_info["note_expensive_ops"] = "Expensive operations skipped for scalability"
    
    return sample_info


def generate_validation_report(
    header_info: Dict,
    leakage_report: Dict,
    sample_info: Dict,
    file_path: Path
) -> Dict:
    """
    Generate comprehensive validation report combining header and sample results.
    
    Parameters
    ----------
    header_info : Dict
        Header-only validation results (exact)
    leakage_report : Dict
        Leakage identification results (exact)
    sample_info : Dict
        Sample-based validation results (approximate)
    file_path : Path
        Path to dataset file
        
    Returns
    -------
    Dict with complete validation report
    """
    print("\n" + "="*80)
    print("GENERATING VALIDATION REPORT")
    print("="*80)
    
    # Calculate columns to drop
    dropped_columns = {
        "leakage_columns": leakage_report['high_risk'],
        "high_missing_columns": sample_info.get('high_missing_columns', [])[:20],  # Top 20
        "rationale": "Leakage columns contain post-loan information. High missing columns (>50%) provide little information.",
        "validation_type": "mixed",
        "note": "Leakage columns: exact. High missing columns: approximate (sample-based)."
    }
    
    remaining_columns = (
        header_info['total_columns'] 
        - len(dropped_columns['leakage_columns']) 
        - len(dropped_columns['high_missing_columns'])
    )
    
    final_report = {
        "validation_date": datetime.now().isoformat(),
        "validation_approach": {
            "method": "layered_validation",
            "description": "Dataset validation is performed using a layered approach (header-only and sample-based) to ensure scalability to multi-GB datasets without requiring distributed infrastructure.",
            "header_validation": {
                "type": "exact",
                "operations": ["schema", "leakage_identification"],
                "rows_loaded": 0
            },
            "sample_validation": {
                "type": "approximate",
                "operations": ["missing_values", "target_distribution", "date_analysis"],
                "rows_loaded": sample_info.get('sample_size', 0)
            }
        },
        "dataset_info": {
            "source": "Lending Club Loan Data (Kaggle)",
            "file": str(file_path.name),
            "file_path": str(file_path),
            "total_columns": header_info['total_columns'],
            "total_rows": "unknown (not loaded for scalability)",
            "note": "Row count not computed to avoid loading full dataset"
        },
        "schema_validation": header_info,
        "leakage_analysis": leakage_report,
        "data_quality": sample_info,
        "dropped_columns": dropped_columns,
        "summary": {
            "total_columns": header_info['total_columns'],
            "columns_to_drop_leakage": len(dropped_columns['leakage_columns']),
            "columns_to_drop_missing": len(dropped_columns['high_missing_columns']),
            "remaining_columns": remaining_columns,
            "validation_notes": [
                "Schema and leakage identification are EXACT (header-based)",
                f"Data quality metrics are APPROXIMATE (based on {sample_info.get('sample_size', 0):,} row sample)",
                "Full dataset statistics not computed for scalability"
            ]
        }
    }
    
    return final_report


def main():
    """Run comprehensive dataset validation using layered approach."""
    
    print("="*80)
    print("COMPREHENSIVE DATASET VALIDATION - STEP 3")
    print("LAYERED VALIDATION APPROACH (Scalable for Large Datasets)")
    print("="*80)
    print("\nValidation Strategy:")
    print("  [HEADER] - Header-only validation (exact, no data loading)")
    print(f"  [SAMPLE: {VALIDATION_CONFIG['sample_size']} rows] - Sample-based validation (approximate)")
    print("="*80)
    
    # Find dataset file
    file_path = find_dataset_file()
    print(f"\nDataset file: {file_path}")
    
    # Layer 1: Header-only validation (exact)
    header_info = validate_dataset_header_only(file_path)
    
    # Leakage identification (header-based, exact)
    leakage_report = identify_leakage_columns(header_info['column_names'])
    
    # Layer 2: Sample-based validation (approximate)
    sample_info = validate_dataset_sample(file_path, VALIDATION_CONFIG['sample_size'])
    
    # Generate final report
    final_report = generate_validation_report(
        header_info, leakage_report, sample_info, file_path
    )
    
    # Save report
    report_path = project_root / "docs" / "validation_report.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n   Report saved to: {report_path}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total columns: {header_info['total_columns']} [EXACT]")
    print(f"Columns to drop (leakage): {len(leakage_report['high_risk'])} [EXACT]")
    print(f"Columns to drop (high missing): {len(sample_info.get('high_missing_columns', []))} [APPROXIMATE]")
    print(f"Remaining columns: {final_report['summary']['remaining_columns']}")
    print("\nValidation Types:")
    print("  [EXACT] - Schema, column names, leakage identification")
    print(f"  [APPROXIMATE] - Missing values, distributions (based on {VALIDATION_CONFIG['sample_size']:,} row sample)")
    print("="*80)
    
    return final_report


if __name__ == "__main__":
    report = main()
