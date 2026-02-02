"""
Temporally diverse sampling for DEV mode.

This module provides functions to sample data while preserving temporal diversity,
ensuring that samples contain multiple dates for proper temporal split testing.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def sample_temporally_diverse(
    file_path: Path, date_column: str, sample_size: int = 10000, compression: Optional[str] = None
) -> pd.DataFrame:
    """
    Sample data with temporal diversity.

    Strategy:
    1. Read date column first to identify date range
    2. Sample proportionally across date periods (months)
    3. Ensures multiple dates for temporal split testing

    Parameters
    ----------
    file_path : Path
        Path to dataset file
    date_column : str
        Name of date column
    sample_size : int
        Target sample size
    compression : Optional[str]
        Compression type ('gzip' or None)

    Returns
    -------
    pd.DataFrame with temporally diverse sample
    """
    logger.info(f"Sampling {sample_size:,} rows with temporal diversity...")

    # Step 1: Read date column in chunks to identify date range and distribution
    logger.info("Step 1: Reading date column in chunks to identify date range...")

    date_periods = {}  # Track row indices by period
    all_dates = []
    total_rows = 0
    chunk_size_read = 50000

    # First, check if file has header by reading first row
    first_chunk = pd.read_csv(
        file_path, usecols=[date_column], compression=compression, low_memory=False, nrows=1
    )
    has_header = True  # Assume header exists (standard CSV)

    # Reset file pointer by creating new iterator
    for chunk_num, chunk in enumerate(
        pd.read_csv(
            file_path,
            usecols=[date_column],
            compression=compression,
            low_memory=False,
            chunksize=chunk_size_read,
        ),
        start=1,
    ):
        # Suppress date parsing warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dates = pd.to_datetime(chunk[date_column], errors="coerce", format="mixed")
        dates_period = dates.dt.to_period("M")

        # Track which rows belong to which period
        # Use position-based indexing (not DataFrame index) to ensure correctness
        chunk_length = len(chunk)

        # For first chunk, check if first row looks like a header
        if chunk_num == 1 and has_header:
            # Skip header row in index calculation
            # But pandas already handles this, so indices should be 0-based for data rows
            pass

        for period in dates_period.unique():
            if pd.notna(period):
                period_str = str(period)
                if period_str not in date_periods:
                    date_periods[period_str] = []
                # Get positions within chunk (0-based, using np.where)
                period_mask = dates_period == period
                period_positions_in_chunk = np.where(period_mask.values)[0]
                # Convert to global indices (file row numbers, 0-based for data rows)
                # Note: pandas chunks exclude header, so first data row is index 0
                global_indices = period_positions_in_chunk + total_rows
                date_periods[period_str].extend(global_indices.tolist())

        all_dates.extend(dates.dropna().tolist())
        total_rows += len(chunk)

        if chunk_num % 10 == 0:
            logger.info(f"  Processed {chunk_num} chunks, {total_rows:,} rows so far...")

    # Get date range
    if all_dates:
        date_min = min(all_dates)
        date_max = max(all_dates)
        logger.info(f"Date range: {date_min} to {date_max}")
        logger.info(f"Total rows processed: {total_rows:,}")
    else:
        raise ValueError("No valid dates found in dataset")

    # Step 2: Identify unique date periods
    unique_periods = {k: len(v) for k, v in date_periods.items()}
    unique_periods_sorted = dict(sorted(unique_periods.items()))

    logger.info(f"Found {len(unique_periods_sorted)} unique month periods")
    logger.info(f"Top 10 periods by count:")
    for period, count in list(unique_periods_sorted.items())[:10]:
        logger.info(f"  {period}: {count:,} rows")

    # Step 3: Sample proportionally across periods
    logger.info("Step 2: Sampling proportionally across date periods...")

    sampled_indices_by_period = {}

    for period_str, period_indices in date_periods.items():
        period_count = len(period_indices)
        # Calculate proportional sample size for this period
        period_proportion = period_count / total_rows
        period_sample_size = max(1, int(sample_size * period_proportion))

        # Sample from this period
        n_sample = min(period_sample_size, period_count)
        sampled = np.random.choice(period_indices, size=n_sample, replace=False)
        sampled_indices_by_period[period_str] = sampled.tolist()

    # Combine sampled indices
    all_sampled_indices = []
    for indices in sampled_indices_by_period.values():
        all_sampled_indices.extend(indices)

    np.random.shuffle(all_sampled_indices)  # Shuffle for randomness

    # Limit to target sample size
    if len(all_sampled_indices) > sample_size:
        all_sampled_indices = all_sampled_indices[:sample_size]

    all_sampled_indices = np.array(sorted(all_sampled_indices))  # Sort for efficient reading

    logger.info(f"Step 3: Reading {len(all_sampled_indices):,} sampled rows...")
    logger.info(f"Sample index range: {all_sampled_indices.min()} to {all_sampled_indices.max()}")

    # Read only the sampled rows
    # Since indices are sorted, we can read efficiently
    sampled_rows = []
    rows_read = 0
    indices_to_read = set(all_sampled_indices)
    chunk_size_read = 50000

    logger.info(f"Reading full dataset to extract {len(all_sampled_indices):,} sampled rows...")

    for chunk_num, chunk in enumerate(
        pd.read_csv(
            file_path, compression=compression, low_memory=False, chunksize=chunk_size_read
        ),
        start=1,
    ):
        # Reset index to ensure sequential numbering
        chunk = chunk.reset_index(drop=True)

        # Calculate actual row indices in original file
        # Note: rows_read starts at 0, so first chunk is 0 to len(chunk)-1
        chunk_start = rows_read
        chunk_end = rows_read + len(chunk)
        chunk_actual_indices = np.arange(chunk_start, chunk_end)

        # Find which rows in this chunk are in our sample
        # Convert to set for faster lookup
        chunk_indices_set = set(chunk_actual_indices)
        matching_indices = chunk_indices_set.intersection(indices_to_read)

        if matching_indices:
            # Create boolean mask for matching rows
            mask = np.isin(chunk_actual_indices, list(matching_indices))
            sampled_chunk = chunk[mask].copy()
            sampled_rows.append(sampled_chunk)
            found_count = len(matching_indices)
            total_found = sum(len(r) for r in sampled_rows)
            logger.info(
                f"  Chunk {chunk_num}: Range [{chunk_start}, {chunk_end}), Extracted {found_count} rows (total so far: {total_found}/{len(all_sampled_indices)})"
            )

        rows_read += len(chunk)

        # Early exit if we've processed all needed rows
        if len(sampled_rows) > 0 and sum(len(r) for r in sampled_rows) >= len(all_sampled_indices):
            logger.info(f"  All sampled rows found. Stopping at chunk {chunk_num}.")
            break

        # Safety check: if we've passed the max index, stop
        if rows_read > max(all_sampled_indices) + chunk_size_read:
            logger.warning(
                f"  Passed max index ({max(all_sampled_indices)}), stopping. Found {sum(len(r) for r in sampled_rows)}/{len(all_sampled_indices)} rows."
            )
            break

    if sampled_rows:
        df_sample = pd.concat(sampled_rows, ignore_index=True)
        total_extracted = len(df_sample)
        logger.info(f"Successfully extracted {total_extracted:,} rows")

        # Verify we got the right number
        if total_extracted != len(all_sampled_indices):
            logger.warning(
                f"Extracted {total_extracted} rows but expected {len(all_sampled_indices)}. "
                f"This may be due to duplicate indices or missing rows."
            )
    else:
        # Debug: show what went wrong
        logger.error(f"Failed to sample rows. Expected {len(all_sampled_indices)} rows.")
        logger.error(f"Index range: {all_sampled_indices.min()} to {all_sampled_indices.max()}")
        logger.error(f"Total rows read: {rows_read}")
        logger.error(f"First 10 expected indices: {all_sampled_indices[:10]}")
        raise ValueError(
            f"Failed to sample rows - no rows matched sampled indices. "
            f"Expected {len(all_sampled_indices)} rows with indices in range "
            f"[{all_sampled_indices.min()}, {all_sampled_indices.max()}], "
            f"but read {rows_read} total rows."
        )

    # Verify temporal diversity
    sample_dates = pd.to_datetime(df_sample[date_column], errors="coerce")
    unique_dates_in_sample = sample_dates.nunique()

    logger.info(f"Sampling complete: {len(df_sample):,} rows")
    logger.info(f"Temporal diversity: {unique_dates_in_sample} unique dates in sample")
    logger.info(f"Sample date range: {sample_dates.min()} to {sample_dates.max()}")

    if unique_dates_in_sample < 2:
        logger.warning(
            f"WARNING: Sample has only {unique_dates_in_sample} unique date(s). "
            "Temporal split may not be possible."
        )

    return df_sample
