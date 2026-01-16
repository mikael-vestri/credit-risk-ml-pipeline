"""
Utilities for temporal split date computation and validation.
"""

import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def compute_split_date(
    df: pd.DataFrame,
    date_column: str,
    train_ratio: float = 0.8
) -> pd.Timestamp:
    """
    Compute split date from data based on train ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of date column
    train_ratio : float
        Proportion of data for training (default 0.8 = 80/20 split)
        
    Returns
    -------
    pd.Timestamp
        Computed split date
    """
    dates = pd.to_datetime(df[date_column], errors='coerce')
    dates_sorted = dates.sort_values().dropna()
    
    if len(dates_sorted) == 0:
        raise ValueError(f"No valid dates found in column '{date_column}'")
    
    split_idx = int(len(dates_sorted) * train_ratio)
    split_date = dates_sorted.iloc[split_idx]
    
    logger.info(f"Computed split date: {split_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Train ratio: {train_ratio:.1%}")
    logger.info(f"  Train rows: {(dates < split_date).sum():,}")
    logger.info(f"  Test rows: {(dates >= split_date).sum():,}")
    
    return split_date


def validate_split_date(
    df: pd.DataFrame,
    date_column: str,
    split_date: str,
    mode: str = "full"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that split date is appropriate for the data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of date column
    split_date : str
        Split date to validate (format: 'YYYY-MM-DD')
    mode : str
        Execution mode ('dev' or 'full')
        
    Returns
    -------
    Tuple of (is_valid, validation_report)
    """
    dates = pd.to_datetime(df[date_column], errors='coerce')
    date_min = dates.min()
    date_max = dates.max()
    split_date_dt = pd.to_datetime(split_date)
    
    validation_report = {
        "date_range": {
            "min": str(date_min) if not pd.isna(date_min) else None,
            "max": str(date_max) if not pd.isna(date_max) else None
        },
        "split_date": split_date,
        "warnings": [],
        "errors": []
    }
    
    # Check if split date is within data range
    if pd.isna(date_min) or pd.isna(date_max):
        validation_report["errors"].append("Invalid date range in data")
        return False, validation_report
    
    if split_date_dt < date_min:
        validation_report["errors"].append(
            f"Split date {split_date} is before data minimum {date_min.strftime('%Y-%m-%d')}"
        )
        return False, validation_report
    
    if split_date_dt > date_max:
        validation_report["errors"].append(
            f"Split date {split_date} is after data maximum {date_max.strftime('%Y-%m-%d')}"
        )
        return False, validation_report
    
    # Check train/test sizes
    train_count = (dates < split_date_dt).sum()
    test_count = (dates >= split_date_dt).sum()
    total_count = len(dates.dropna())
    
    validation_report["train_count"] = int(train_count)
    validation_report["test_count"] = int(test_count)
    validation_report["train_percentage"] = float(train_count / total_count * 100) if total_count > 0 else 0
    validation_report["test_percentage"] = float(test_count / total_count * 100) if total_count > 0 else 0
    
    # Warnings (non-fatal)
    if test_count == 0:
        if mode == "full":
            validation_report["errors"].append("Test set is empty - cannot proceed in FULL mode")
            return False, validation_report
        else:
            validation_report["warnings"].append(
                "Test set is empty. Temporal split not possible. "
                "Consider adjusting split date or using larger sample."
            )
    
    if train_count == 0:
        if mode == "full":
            validation_report["errors"].append("Train set is empty - cannot proceed in FULL mode")
            return False, validation_report
        else:
            validation_report["warnings"].append(
                "Train set is empty. Temporal split not possible. "
                "Consider adjusting split date or using larger sample."
            )
    
    if test_count < 100 and mode == "full":
        validation_report["warnings"].append(
            f"Test set is very small ({test_count} rows). Consider adjusting split date."
        )
    
    if train_count < 100 and mode == "full":
        validation_report["warnings"].append(
            f"Train set is very small ({train_count} rows). Consider adjusting split date."
        )
    
    # In DEV mode, warnings don't fail validation
    is_valid = len(validation_report["errors"]) == 0
    
    return is_valid, validation_report

