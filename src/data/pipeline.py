"""
Data processing pipeline with DEV/FULL mode support.

This module orchestrates the complete data processing pipeline:
- DEV mode: Temporally diverse sampling, fast iteration
- FULL mode: Chunked processing, production-ready
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def process_data_pipeline(
    raw_data_path: Path,
    processed_data_path: Path,
    validation_report_path: Optional[Path],
    mode: str = "dev",
    sample_size: int = 10000,
    chunk_size: int = 10000,
    split_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete data processing pipeline with DEV/FULL mode support.

    DEV Mode:
    - Samples data with temporal diversity
    - Processes sample in memory
    - Fast iteration for development

    FULL Mode:
    - Processes full dataset in chunks
    - Writes cleaned data incrementally
    - Production-ready, handles large datasets

    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory
    processed_data_path : Path
        Path to processed data directory
    validation_report_path : Optional[Path]
        Path to validation report
    mode : str
        Execution mode: 'dev' or 'full'
    sample_size : int
        Sample size for DEV mode
    chunk_size : int
        Chunk size for FULL mode
    split_date : Optional[str]
        Split date (if None, will be computed from data)

    Returns
    -------
    Dict with processing metadata
    """
    from config import dataset_config
    from data.ingestion import get_dataset_file_path, get_file_metadata
    from data.sampling import sample_temporally_diverse
    from data.chunked_processing import process_chunked_cleaning
    from data.cleaning import clean_and_transform
    from data.split_utils import compute_split_date, validate_split_date

    logger.info("=" * 80)
    logger.info(f"DATA PROCESSING PIPELINE - {mode.upper()} MODE")
    logger.info("=" * 80)

    # Validate mode
    if mode not in ["dev", "full"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'dev' or 'full'")

    # Get dataset file
    file_path = get_dataset_file_path(raw_data_path)
    compression = "gzip" if str(file_path).endswith(".gz") else None

    # Output path
    cleaned_data_path = processed_data_path / "cleaned_data.parquet"

    metadata = {
        "mode": mode,
        "file_path": str(file_path),
        "file_metadata": get_file_metadata(file_path),
    }

    if mode == "dev":
        # DEV MODE: Temporally diverse sampling
        logger.info("\n[MODE: DEV] Using temporally diverse sampling")
        logger.info(f"Sample size: {sample_size:,} rows")
        logger.info("Sampling strategy: Proportional across date periods")

        df_sample = sample_temporally_diverse(
            file_path,
            date_column=dataset_config.DATE_COLUMN,
            sample_size=sample_size,
            compression=compression,
        )

        metadata["sample_size"] = len(df_sample)
        metadata["sampling_strategy"] = "temporally_diverse"

        # Clean sample
        logger.info("\n[MODE: DEV] Cleaning sampled data...")
        df_cleaned, transformation_log = clean_and_transform(
            df_sample,
            validation_report_path=validation_report_path,
            save_path=cleaned_data_path,
            mode="dev",
        )

        metadata["cleaning"] = transformation_log

    else:  # FULL mode
        # FULL MODE: Chunked processing
        logger.info("\n[MODE: FULL] Using chunked processing")
        logger.info(f"Chunk size: {chunk_size:,} rows")
        logger.info("Processing strategy: Incremental chunked cleaning")

        processing_metadata = process_chunked_cleaning(
            file_path,
            cleaned_data_path,
            validation_report_path,
            chunk_size=chunk_size,
            compression=compression,
        )

        metadata["chunked_processing"] = processing_metadata
        metadata["chunk_size"] = chunk_size

    # Load cleaned data to check date range and compute split date
    logger.info("\nAnalyzing cleaned data for temporal split...")
    df_cleaned_check = pd.read_parquet(cleaned_data_path)

    dates = pd.to_datetime(df_cleaned_check[dataset_config.DATE_COLUMN], errors="coerce")
    date_min = dates.min()
    date_max = dates.max()

    logger.info(f"Date range in cleaned data: {date_min} to {date_max}")
    metadata["date_range"] = {
        "min": str(date_min) if not pd.isna(date_min) else None,
        "max": str(date_max) if not pd.isna(date_max) else None,
    }

    # Compute or validate split date
    if split_date is None:
        logger.info("Computing split date from data (80/20 split)...")
        split_date_dt = compute_split_date(
            df_cleaned_check, dataset_config.DATE_COLUMN, train_ratio=0.8
        )
        split_date = split_date_dt.strftime("%Y-%m-%d")
        metadata["split_date"] = split_date
        metadata["split_date_source"] = "computed"
    else:
        logger.info(f"Validating provided split date: {split_date}")
        is_valid, validation_report = validate_split_date(
            df_cleaned_check, dataset_config.DATE_COLUMN, split_date, mode=mode
        )

        metadata["split_date"] = split_date
        metadata["split_date_source"] = "provided"
        metadata["split_validation"] = validation_report

        if not is_valid:
            if mode == "full":
                logger.error("Split date validation failed in FULL mode. Cannot proceed.")
                raise ValueError(f"Invalid split date: {validation_report['errors']}")
            else:
                logger.warning("Split date validation failed in DEV mode.")
                for error in validation_report.get("errors", []):
                    logger.warning(f"  ERROR: {error}")
                for warning in validation_report.get("warnings", []):
                    logger.warning(f"  WARNING: {warning}")

    logger.info(f"\nFinal split date: {split_date}")
    logger.info("=" * 80)
    logger.info("DATA PROCESSING PIPELINE COMPLETE")
    logger.info("=" * 80)

    return metadata
