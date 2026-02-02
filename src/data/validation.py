"""
Dataset validation module.

This module contains functions to validate the Lending Club dataset:
- Schema consistency across vintages
- Target leakage identification
- Data quality checks
- Temporal split correctness validation

Note: This module validates data and splits but does NOT perform splits.
Use splitting.py for performing temporal splits.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_schema_consistency(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """
    Validate schema consistency across different vintage files.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary mapping vintage/file names to DataFrames

    Returns
    -------
    Dict[str, any]
        Validation report with:
        - common_columns: List of columns present in all files
        - unique_columns: Dict mapping file names to unique columns
        - type_inconsistencies: Dict of columns with type mismatches
        - column_counts: Dict mapping file names to column counts
    """
    logger.info("Validating schema consistency across vintages...")

    # TODO: Implement schema validation logic
    # This will be implemented once we have the actual dataset

    return {
        "common_columns": [],
        "unique_columns": {},
        "type_inconsistencies": {},
        "column_counts": {},
    }


def identify_target_leakage(
    df: pd.DataFrame, target_column: str = "loan_status"
) -> Dict[str, List[str]]:
    """
    Identify columns that may contain target leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with leakage categories:
        - high_risk: Columns that definitely contain leakage
        - medium_risk: Columns that need careful review
        - safe: Columns that are safe to use
    """
    logger.info("Identifying potential target leakage columns...")

    # TODO: Implement leakage detection logic
    # This will be implemented once we have the actual dataset

    return {"high_risk": [], "medium_risk": [], "safe": []}


def validate_temporal_split_correctness(
    train_df: pd.DataFrame, test_df: pd.DataFrame, date_column: str, split_date: str
) -> Dict[str, any]:
    """
    Validate that a temporal split is correct (no temporal leakage).

    This function validates an existing split but does not perform it.
    Use splitting.perform_temporal_split() to create the split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    date_column : str
        Name of the date column used for splitting
    split_date : str
        Date threshold that was used for the split (format: 'YYYY-MM-DD')

    Returns
    -------
    Dict[str, any]
        Validation report with:
        - is_valid: bool - Whether the split is temporally correct
        - errors: List[str] - List of validation errors found
        - warnings: List[str] - List of warnings
        - train_date_range: Dict with min/max dates in train
        - test_date_range: Dict with min/max dates in test
        - overlap_count: int - Number of records that violate temporal ordering
    """
    logger.info(f"Validating temporal split correctness (split_date={split_date})...")

    errors = []
    warnings = []

    # Check date_column exists in both dataframes
    if date_column not in train_df.columns:
        errors.append(f"Date column '{date_column}' not found in train_df")
    if date_column not in test_df.columns:
        errors.append(f"Date column '{date_column}' not found in test_df")

    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "train_date_range": {},
            "test_date_range": {},
            "overlap_count": 0,
        }

    # Convert date columns to datetime
    train_dates = pd.to_datetime(train_df[date_column], errors="coerce")
    test_dates = pd.to_datetime(test_df[date_column], errors="coerce")
    split_date_dt = pd.to_datetime(split_date)

    # Get date ranges
    train_min = train_dates.min()
    train_max = train_dates.max()
    test_min = test_dates.min()
    test_max = test_dates.max()

    # Validate: No dates in train should be >= split_date
    train_violations = (train_dates >= split_date_dt).sum()
    if train_violations > 0:
        errors.append(
            f"Found {train_violations} records in train_df with date >= split_date ({split_date})"
        )

    # Validate: No dates in test should be < split_date
    test_violations = (test_dates < split_date_dt).sum()
    if test_violations > 0:
        errors.append(
            f"Found {test_violations} records in test_df with date < split_date ({split_date})"
        )

    # Check for overlap (train_max should be < test_min ideally)
    if not pd.isna(train_max) and not pd.isna(test_min):
        if train_max >= test_min:
            warnings.append(
                f"Potential temporal overlap: train_max ({train_max}) >= test_min ({test_min})"
            )

    # Check for empty splits
    if len(train_df) == 0:
        errors.append("train_df is empty")
    if len(test_df) == 0:
        errors.append("test_df is empty")

    # Check for missing dates
    train_missing = train_dates.isna().sum()
    test_missing = test_dates.isna().sum()
    if train_missing > 0:
        warnings.append(f"Found {train_missing} missing dates in train_df")
    if test_missing > 0:
        warnings.append(f"Found {test_missing} missing dates in test_df")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "train_date_range": {
            "min": str(train_min) if not pd.isna(train_min) else None,
            "max": str(train_max) if not pd.isna(train_max) else None,
        },
        "test_date_range": {
            "min": str(test_min) if not pd.isna(test_min) else None,
            "max": str(test_max) if not pd.isna(test_max) else None,
        },
        "overlap_count": train_violations + test_violations,
    }


def generate_validation_report(
    schema_report: Dict, leakage_report: Dict, dropped_columns: List[str]
) -> str:
    """
    Generate a comprehensive validation report.

    Parameters
    ----------
    schema_report : Dict
        Schema validation results
    leakage_report : Dict
        Leakage identification results
    dropped_columns : List[str]
        List of columns to be dropped

    Returns
    -------
    str
        Formatted validation report
    """
    # TODO: Implement report generation
    return "Validation report will be generated here."
