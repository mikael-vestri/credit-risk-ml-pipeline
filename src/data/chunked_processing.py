"""
Chunked processing for FULL mode.

This module provides functions to process large datasets in chunks,
writing intermediate results incrementally to avoid memory issues.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)


def process_chunked_cleaning(
    file_path: Path,
    output_path: Path,
    validation_report_path: Optional[Path],
    chunk_size: int = 10000,
    compression: Optional[str] = None,
    cleaning_function: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Process dataset in chunks for cleaning (FULL mode).

    Strategy:
    1. Read CSV in chunks
    2. Clean each chunk independently (no global statistics)
    3. Append cleaned chunks to parquet incrementally
    4. Log progress

    Parameters
    ----------
    file_path : Path
        Path to raw dataset file
    output_path : Path
        Path to save cleaned parquet file
    validation_report_path : Optional[Path]
        Path to validation report
    chunk_size : int
        Number of rows per chunk
    compression : Optional[str]
        Compression type
    cleaning_function : Optional[Callable]
        Function to clean each chunk. If None, uses default clean_and_transform

    Returns
    -------
    Dict with processing metadata
    """
    logger.info("=" * 80)
    logger.info("CHUNKED CLEANING PROCESSING - FULL MODE")
    logger.info("=" * 80)
    logger.info(f"Chunk size: {chunk_size:,} rows")
    logger.info(f"Output: {output_path}")

    from data.cleaning import clean_and_transform

    if cleaning_function is None:
        cleaning_function = clean_and_transform

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track metadata
    total_rows_processed = 0
    total_chunks = 0
    schema_written = False
    parquet_writer = None

    try:
        # Process in chunks
        for chunk_num, chunk_df in enumerate(
            pd.read_csv(file_path, compression=compression, low_memory=False, chunksize=chunk_size),
            start=1,
        ):
            logger.info(f"\nProcessing chunk {chunk_num} ({len(chunk_df):,} rows)...")

            # Clean chunk
            chunk_cleaned, _ = cleaning_function(
                chunk_df,
                validation_report_path=validation_report_path,
                save_path=None,  # Don't save individual chunks
            )

            # Write to parquet (append mode)
            if not schema_written:
                # First chunk: write with schema
                chunk_cleaned.to_parquet(output_path, index=False, engine="pyarrow")
                schema_written = True
                logger.info(f"Initialized parquet file with schema")
            else:
                # Subsequent chunks: append
                table = pa.Table.from_pandas(chunk_cleaned)
                parquet_file = pq.ParquetFile(output_path)
                existing_table = parquet_file.read()
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, output_path)

            total_rows_processed += len(chunk_cleaned)
            total_chunks = chunk_num

            logger.info(
                f"Chunk {chunk_num} complete. Total processed: {total_rows_processed:,} rows"
            )

            # Progress update every 10 chunks
            if chunk_num % 10 == 0:
                logger.info(f"Progress: {chunk_num} chunks, {total_rows_processed:,} rows")

        logger.info("\n" + "=" * 80)
        logger.info("CHUNKED PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total chunks processed: {total_chunks}")
        logger.info(f"Total rows: {total_rows_processed:,}")
        logger.info(f"Output file: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info("=" * 80)

        return {
            "total_chunks": total_chunks,
            "total_rows": total_rows_processed,
            "output_path": str(output_path),
            "file_size_mb": output_path.stat().st_size / (1024 * 1024),
        }

    except Exception as e:
        logger.error(f"Error during chunked processing: {e}")
        raise
