"""
Data ingestion module (ETL - Raw Layer).

This module handles loading raw data from source files.
Raw data is NEVER modified - it's loaded as-is for downstream processing.

Key principles:
- Reproducible: Same input always produces same output
- Scalable: Handles large datasets efficiently
- Logged: All ingestion operations are logged with metadata
- Immutable: Raw data files are never modified
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


def get_dataset_file_path(raw_data_path: Path) -> Path:
    """
    Find and return the dataset file path.
    
    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory
        
    Returns
    -------
    Path
        Path to dataset file
        
    Raises
    ------
    FileNotFoundError
        If dataset file is not found
    """
    gz_file = raw_data_path / "accepted_2007_to_2018Q4.csv.gz"
    csv_file = raw_data_path / "accepted_2007_to_2018Q4.csv" / "accepted_2007_to_2018Q4.csv"
    
    if gz_file.exists():
        return gz_file
    elif csv_file.exists():
        return csv_file
    else:
        raise FileNotFoundError(
            f"Dataset file not found in {raw_data_path}. "
            f"Expected: accepted_2007_to_2018Q4.csv.gz or accepted_2007_to_2018Q4.csv"
        )


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of file for version tracking.
    
    Parameters
    ----------
    file_path : Path
        Path to file
        
    Returns
    -------
    str
        SHA256 hash of file
    """
    sha256_hash = hashlib.sha256()
    
    # Handle gzipped files
    if str(file_path).endswith('.gz'):
        import gzip
        with gzip.open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    else:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Get metadata about the dataset file.
    
    Parameters
    ----------
    file_path : Path
        Path to dataset file
        
    Returns
    -------
    Dict with file metadata
    """
    stat = file_path.stat()
    
    metadata = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "is_compressed": str(file_path).endswith('.gz'),
        "file_hash": compute_file_hash(file_path)
    }
    
    return metadata


def load_raw_data(
    raw_data_path: Path,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """
    Load raw data from source file.
    
    This function loads raw data AS-IS without any modifications.
    Raw data is immutable - all transformations happen downstream.
    
    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory
    nrows : Optional[int]
        Number of rows to load (for testing/sampling)
        If None, loads full dataset
    chunksize : Optional[int]
        If provided, returns iterator over chunks
        If None, loads into single DataFrame
        
    Returns
    -------
    pd.DataFrame or Iterator[pd.DataFrame]
        Raw data as DataFrame(s)
        
    Raises
    ------
    FileNotFoundError
        If dataset file is not found
    ValueError
        If both nrows and chunksize are provided
    """
    if nrows is not None and chunksize is not None:
        raise ValueError("Cannot specify both nrows and chunksize")
    
    file_path = get_dataset_file_path(raw_data_path)
    
    # Determine compression
    compression = 'gzip' if str(file_path).endswith('.gz') else None
    
    logger.info(f"Loading raw data from: {file_path}")
    logger.info(f"Compression: {compression}")
    
    if chunksize is not None:
        logger.info(f"Loading in chunks of {chunksize} rows")
        return pd.read_csv(
            file_path,
            compression=compression,
            low_memory=False,
            chunksize=chunksize
        )
    elif nrows is not None:
        logger.info(f"Loading sample of {nrows} rows")
        df = pd.read_csv(
            file_path,
            compression=compression,
            low_memory=False,
            nrows=nrows
        )
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    else:
        logger.info("Loading full dataset (this may take a while for large files)")
        df = pd.read_csv(
            file_path,
            compression=compression,
            low_memory=False
        )
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df


def get_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract schema information from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    Dict with schema information
    """
    schema = {
        "total_columns": len(df.columns),
        "total_rows": len(df),
        "column_names": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "dtype_summary": {
            str(dtype): int(count) 
            for dtype, count in df.dtypes.value_counts().items()
        }
    }
    
    return schema


def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic statistics about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    Dict with basic statistics
    """
    stats = {
        "shape": {
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        },
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_percentage": round(
            (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percentage": round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0
    }
    
    return stats


def log_ingestion_metadata(
    file_metadata: Dict[str, Any],
    schema: Dict[str, Any],
    stats: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create and optionally save ingestion metadata log.
    
    Parameters
    ----------
    file_metadata : Dict
        File metadata from get_file_metadata()
    schema : Dict
        Schema information from get_dataset_schema()
    stats : Dict
        Basic statistics from get_basic_stats()
    output_path : Optional[Path]
        If provided, save metadata to this path
        
    Returns
    -------
    Dict with complete ingestion metadata
    """
    ingestion_log = {
        "ingestion_timestamp": datetime.now().isoformat(),
        "file_metadata": file_metadata,
        "schema": schema,
        "basic_stats": stats,
        "ingestion_notes": [
            "Raw data loaded without modifications",
            "All transformations happen downstream",
            "This log serves as version control for raw data"
        ]
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ingestion_log, f, indent=2, default=str)
        logger.info(f"Ingestion metadata saved to: {output_path}")
    
    return ingestion_log


def ingest_raw_data(
    raw_data_path: Path,
    output_metadata_path: Optional[Path] = None,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete raw data ingestion pipeline.
    
    This is the main entry point for data ingestion. It:
    1. Loads raw data (unchanged)
    2. Extracts schema and statistics
    3. Logs metadata
    4. Returns data and metadata
    
    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory
    output_metadata_path : Optional[Path]
        Path to save ingestion metadata JSON
    nrows : Optional[int]
        Number of rows to load (for testing)
    chunksize : Optional[int]
        Chunk size for streaming (not supported in this function)
        
    Returns
    -------
    Tuple of (DataFrame, ingestion_metadata)
    """
    logger.info("="*80)
    logger.info("RAW DATA INGESTION - STEP 4")
    logger.info("="*80)
    
    # Get file metadata
    file_path = get_dataset_file_path(raw_data_path)
    file_metadata = get_file_metadata(file_path)
    
    logger.info(f"Dataset file: {file_metadata['file_name']}")
    logger.info(f"File size: {file_metadata['file_size_mb']} MB")
    logger.info(f"File hash: {file_metadata['file_hash'][:16]}...")
    
    # Load raw data
    df = load_raw_data(raw_data_path, nrows=nrows, chunksize=chunksize)
    
    # Extract schema
    logger.info("Extracting schema information...")
    schema = get_dataset_schema(df)
    
    # Get basic stats
    logger.info("Computing basic statistics...")
    stats = get_basic_stats(df)
    
    # Log metadata
    ingestion_metadata = log_ingestion_metadata(
        file_metadata, schema, stats, output_metadata_path
    )
    
    logger.info("="*80)
    logger.info("INGESTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Rows loaded: {stats['shape']['rows']:,}")
    logger.info(f"Columns: {stats['shape']['columns']}")
    logger.info(f"Memory usage: {stats['memory_usage_mb']} MB")
    logger.info("="*80)
    
    return df, ingestion_metadata



