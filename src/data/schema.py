"""
Data schema definitions and validation contracts for market time series.
Enforces point-in-time tracking (observation_date vs publication_date)
to prevent look-ahead bias in historical arbitrage analysis.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


REQUIRED_MARKET_COLUMNS: List[str] = [
    "observation_date",
    "publication_date",
    "retrieval_timestamp",
    "benchmark",
    "instrument_type",
    "contract_month",
    "raw_price",
    "raw_currency",
    "raw_unit",
    "price_usd_mmbtu",
    "source",
    "dataset_id",
    "timezone",
    "methodology",
    "qc_flag",
]


@dataclass(frozen=True)
class SourceMetadata:
    """Immutable metadata descriptor for a market data series."""
    benchmark: str
    instrument_type: str
    contract_month: str
    source_name: str
    dataset_id: str
    url: str
    raw_currency: str
    raw_unit: str
    timezone: str
    publication_lag_business_days: int
    methodology: str


def validate_market_dataframe_schema(df: pd.DataFrame) -> None:
    """
    Validates that a standardized market DataFrame contains all required
    columns and satisfies point-in-time temporal ordering constraints.

    Raises:
        ValueError: If required columns are missing or publication_date precedes observation_date.
    """
    missing_cols = [col for col in REQUIRED_MARKET_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame failed schema validation. Missing columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame failed schema validation: dataset is empty.")

    obs_dates = pd.to_datetime(df["observation_date"])
    pub_dates = pd.to_datetime(df["publication_date"])

    if (pub_dates < obs_dates).any():
        invalid_count = int((pub_dates < obs_dates).sum())
        raise ValueError(
            f"Look-ahead bias violation: {invalid_count} records have "
            "publication_date earlier than observation_date."
        )