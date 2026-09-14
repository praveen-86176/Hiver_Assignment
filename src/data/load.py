"""Data ingestion and validation module for the Customer Support on Twitter dataset.

Loads raw CSV data, validates schema integrity, deduplicates records,
parses timestamp fields, and reports null metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: List[str] = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


class SchemaValidationError(Exception):
    """Raised when the loaded dataset schema does not match expected columns."""
    pass


def validate_schema(df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> None:
    """Validate that the DataFrame contains exactly the expected columns.

    Args:
        df: Pandas DataFrame to validate.
        expected_columns: List of expected column names (defaults to EXPECTED_COLUMNS).

    Raises:
        SchemaValidationError: If columns do not match expected schema.
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    actual_columns = list(df.columns)
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    unexpected_columns = [col for col in actual_columns if col not in expected_columns]

    if missing_columns or unexpected_columns:
        error_msg = (
            f"Schema validation failed!\n"
            f"  Expected columns ({len(expected_columns)}): {expected_columns}\n"
            f"  Actual columns ({len(actual_columns)}): {actual_columns}\n"
            f"  Missing columns: {missing_columns}\n"
            f"  Unexpected columns: {unexpected_columns}"
        )
        logger.error(error_msg)
        raise SchemaValidationError(error_msg)

    logger.info("Schema validation passed: All %d expected columns present.", len(expected_columns))


def compute_null_report(df: pd.DataFrame) -> Dict[str, Dict[str, Union[int, float]]]:
    """Compute null-rates and counts for all columns.

    Args:
        df: Input DataFrame.

    Returns:
        Dict mapping column name to null count and percentage.
    """
    total_rows = len(df)
    report = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = float((null_count / total_rows) * 100) if total_rows > 0 else 0.0
        report[col] = {
            "null_count": null_count,
            "null_pct": round(null_pct, 2),
        }
    return report


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/twcs.csv",
    dedupe: bool = True,
    parse_dates: bool = True,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, Dict]]]:
    """Load and validate raw Customer Support on Twitter dataset.

    Args:
        filepath: Path to raw CSV file.
        dedupe: Whether to deduplicate on tweet_id.
        parse_dates: Whether to parse created_at into datetime objects.
        nrows: Optional number of rows to load (useful for testing/sampling).

    Returns:
        Tuple of (cleaned DataFrame, metadata dictionary with null and dedupe stats).

    Raises:
        FileNotFoundError: If filepath does not exist.
        SchemaValidationError: If columns do not match expected schema.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    logger.info("Loading raw dataset from %s (nrows=%s)...", path, nrows)
    # Read raw CSV
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    raw_count = len(df)
    logger.info("Loaded %s raw records.", f"{raw_count:,}")

    # Validate schema loudly if mismatch
    validate_schema(df)

    # Compute null rates before dropping/handling
    null_report = compute_null_report(df)
    logger.info("Null rates before handling:")
    for col, stats in null_report.items():
        logger.info(
            "  - %s: %s nulls (%.2f%%)",
            col,
            f"{stats['null_count']:,}",
            stats["null_pct"],
        )

    # Deduplicate on tweet_id
    dedupe_count = 0
    if dedupe:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["tweet_id"], keep="first").copy()
        dedupe_count = initial_len - len(df)
        if dedupe_count > 0:
            logger.warning("Dropped %d duplicate tweet_id records.", dedupe_count)
        else:
            logger.info("Deduplication complete: No duplicate tweet_ids found.")

    # Convert inbound to boolean if not already
    if not pd.api.types.is_bool_dtype(df["inbound"]):
        df["inbound"] = df["inbound"].astype(bool)

    # Parse created_at timestamps
    if parse_dates:
        logger.info("Parsing 'created_at' timestamp column...")
        # TWCS format: 'Tue Oct 31 22:10:47 +0000 2017'
        try:
            df["created_at"] = pd.to_datetime(
                df["created_at"],
                format="%a %b %d %H:%M:%S %z %Y",
                errors="coerce",
            )
        except Exception:
            logger.warning("Fast datetime parsing failed; falling back to generic parser.")
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    metadata = {
        "raw_count": raw_count,
        "processed_count": len(df),
        "duplicates_dropped": dedupe_count,
        "null_report": null_report,
    }

    return df, metadata


if __name__ == "__main__":
    df, meta = load_raw_data()
    print("\nSummary Metadata:")
    print(f"Total Rows: {meta['processed_count']:,}")
    print(f"Duplicates Dropped: {meta['duplicates_dropped']}")
    print("Null Report:")
    for col, st in meta["null_report"].items():
        print(f"  {col}: {st['null_count']:,} ({st['null_pct']}%)")
