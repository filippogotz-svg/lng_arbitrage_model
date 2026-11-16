"""
Automated unit tests for Phase 1A Henry Hub Data Pipeline and Quality Control Engine.
Verifies schema enforcement, look-ahead bias protection, non-destructive outlier flagging,
and missing value preservation.
"""

import pandas as pd
import pytest

from src.data.schema import SourceMetadata, validate_market_dataframe_schema
from src.data.quality_control import QualityControlConfig, apply_quality_control_checks
from src.data.henry_hub_pipeline import standardize_henry_hub_dataframe


@pytest.fixture
def sample_metadata() -> SourceMetadata:
    return SourceMetadata(
        benchmark="HENRY_HUB",
        instrument_type="SPOT",
        contract_month="SPOT",
        source_name="EIA_VIA_FRED",
        dataset_id="DHHNGSP",
        url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP",
        raw_currency="USD",
        raw_unit="MMBtu",
        timezone="America/Chicago",
        publication_lag_business_days=1,
        methodology="EIA official daily Henry Hub natural gas spot price assessment.",
    )


def test_standardize_and_preserve_missing_values(sample_metadata: SourceMetadata) -> None:
    """Verifies that '.' missing markers from FRED are preserved as NaN, not dropped."""
    raw_csv = (
        "DATE,DHHNGSP\n"
        "2024-01-02,2.55\n"
        "2024-01-03,.\n"
        "2024-01-04,2.68\n"
    )
    df = standardize_henry_hub_dataframe(raw_csv, sample_metadata, "2026-11-02T12:00:00Z")
    assert len(df) == 3
    assert df["price_usd_mmbtu"].isna().sum() == 1
    # Verify T+1 business day publication lag (2024-01-02 Tuesday -> 2024-01-03 Wednesday)
    assert df.loc[0, "publication_date"] == "2024-01-03"


def test_non_destructive_qc_flags_outliers_and_invalid(sample_metadata: SourceMetadata) -> None:
    """
    Verifies that negative prices, missing values, and extreme spikes (e.g. Uri $23.86)
    are accurately flagged without deleting rows from the dataset.
    """
    dates = pd.date_range("2021-01-01", periods=70, freq="B")
    lines = ["DATE,DHHNGSP"]
    for i, d in enumerate(dates):
        if i == 30:
            lines.append(f"{d.strftime('%Y-%m-%d')},.")        # Missing observation
        elif i == 40:
            lines.append(f"{d.strftime('%Y-%m-%d')},-1.50")    # Invalid negative price
        elif i == 55:
            lines.append(f"{d.strftime('%Y-%m-%d')},23.86")    # Winter Storm Uri spike
        else:
            lines.append(f"{d.strftime('%Y-%m-%d')},{2.70 + (i % 5) * 0.03:.2f}")

    raw_csv = "\n".join(lines)
    std_df = standardize_henry_hub_dataframe(raw_csv, sample_metadata, "2026-11-02T12:00:00Z")
    qc_cfg = QualityControlConfig(rolling_window_days=30, mad_outlier_threshold=5.0)

    checked_df, report = apply_quality_control_checks(std_df, qc_cfg)

    # Row count must remain strictly unchanged (Principle 10)
    assert len(checked_df) == 70
    assert report.missing_value_count == 1
    assert report.invalid_price_count == 1
    assert report.outlier_high_count >= 1
    assert checked_df.loc[30, "qc_flag"] == "MISSING_VALUE"
    assert checked_df.loc[40, "qc_flag"] == "INVALID_PRICE"
    assert checked_df.loc[55, "qc_flag"] == "OUTLIER_HIGH"

    # Schema validation must pass
    validate_market_dataframe_schema(checked_df)


def test_look_ahead_bias_detection(sample_metadata: SourceMetadata) -> None:
    """Verifies that schema validation rejects publication_date earlier than observation_date."""
    raw_csv = "DATE,DHHNGSP\n2024-05-10,2.20\n"
    df = standardize_henry_hub_dataframe(raw_csv, sample_metadata, "2026-11-02T12:00:00Z")
    df.loc[0, "publication_date"] = "2024-05-09"  # Corrupt publication date

    with pytest.raises(ValueError, match="Look-ahead bias violation"):
        validate_market_dataframe_schema(df)
        