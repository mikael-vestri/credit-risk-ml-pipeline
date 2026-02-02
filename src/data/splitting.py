"""
Data splitting module.

This module handles temporal splitting of datasets for train/test separation.
It is responsible for performing the split, not validating it.
"""

from typing import Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def perform_temporal_split(
    df: pd.DataFrame, date_column: str, split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform temporal split of the dataset based on a date threshold.

    This function creates the split but does not validate it.
    Use validation.validate_temporal_split_correctness() to validate the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with temporal data
    date_column : str
        Name of the date column for temporal ordering
    split_date : str
        Date threshold for train/test split (format: 'YYYY-MM-DD')
        Records with date < split_date go to train, >= split_date go to test

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) - Train and test dataframes

    Raises
    ------
    ValueError
        If date_column is not found in dataframe
        If split_date format is invalid
    """
    logger.info(f"Performing temporal split at {split_date} using column '{date_column}'...")

    # Validate date_column exists
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in dataframe")

    # Convert date_column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Convert split_date to datetime
    try:
        split_date_dt = pd.to_datetime(split_date)
    except ValueError as e:
        raise ValueError(
            f"Invalid split_date format '{split_date}'. Expected 'YYYY-MM-DD' format."
        ) from e

    # Perform the split
    train_df = df[df[date_column] < split_date_dt].copy()
    test_df = df[df[date_column] >= split_date_dt].copy()

    logger.info(f"Split complete: Train={len(train_df)} records, Test={len(test_df)} records")

    return train_df, test_df
