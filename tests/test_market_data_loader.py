"""
Automated unit tests for Phase 1B Multi-Benchmark Market Data Loader.
Verifies Bloomberg Excel parsing, chronological sorting, EUR/MWh to USD/MMBtu
conversion, missing FX protection, and cross-basin spread alignment.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schema import SourceMetadata, validate_market_dataframe_schema
from src.data.market_data_loader import (
    parse_bloomberg_excel,
    standardize_bloomberg_series,
    build_cross_basin_master_panel,
)


def test_parse_bloomberg_excel_descending_and_multiblock(tmp_path: Path) -> None:
    """
    Verifies parsing of Bloomberg HP Excel exports containing Date, Settlement Price,
    Volume, and SMAVG (15) ordered newest-to-oldest.
    """
    excel_file = tmp_path / "ttf_test.xlsx"
    raw_df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-11-16", "2026-11-13", "2026-11-12"]),
        "Settlement Price": [np.nan, 41.25, 40.80],
        "Volume": [np.nan, 185000, 192000],
        "SMAVG (15)": [np.nan, 228005, 230124],
    })
    raw_df.to_excel(excel_file, index=False)

    parsed = parse_bloomberg_excel(excel_file)
    assert len(parsed) == 3
    # Must be sorted ascending (oldest first)
    assert parsed.loc[0, "observation_date"].strftime("%Y-%m-%d") == "2026-11-12"
    assert parsed.loc[0, "raw_price"] == pytest.approx(40.80)
    assert pd.isna(parsed.loc[2, "raw_price"])


def test_ttf_eur_mwh_to_usd_mmbtu_conversion() -> None:
    """
    Verifies exact conversion: USD/MMBtu = (EUR/MWh * EURUSD) / 3.41214
    and proper flagging when same-day EURUSD is missing.
    """
    meta = SourceMetadata(
        benchmark="TTF",
        instrument_type="FUTURES",
        contract_month="FRONT_MONTH_GENERIC",
        source_name="BLOOMBERG_EXPORT",
        dataset_id="TZT1_COMDTY",
        url="local://ttf.xlsx",
        raw_currency="EUR",
        raw_unit="MWh",
        timezone="Europe/Amsterdam",
        publication_lag_business_days=0,
        methodology="Test TTF",
    )
    parsed = pd.DataFrame({
        "observation_date": pd.to_datetime(["2026-11-12", "2026-11-13"]),
        "raw_price": [40.00, 42.00],
    })
    # Provide FX only for 2026-11-12 (1.1000 USD/EUR); omit 2026-11-13
    fx_lookup = pd.Series({"2026-11-12": 1.1000})

    std = standardize_bloomberg_series(
        parsed, meta, "2026-11-16T16:00:00Z", fx_lookup=fx_lookup, mwh_to_mmbtu=3.41214
    )
    expected_usd_mmbtu = (40.00 * 1.1000) / 3.41214
    assert std.loc[0, "price_usd_mmbtu"] == pytest.approx(expected_usd_mmbtu, rel=1e-6)
    assert pd.isna(std.loc[1, "price_usd_mmbtu"])
    assert bool(std.attrs["missing_fx_mask"].iloc[1]) is True
    validate_market_dataframe_schema(std)


def test_cross_basin_master_panel_no_forward_fill() -> None:
    """
    Verifies that inter-basin spreads are computed accurately and remain NaN
    on holidays without artificial forward-filling.
    """
    base = {
        "publication_date": ["2026-11-12", "2026-11-13"],
        "retrieval_timestamp": ["2026-11-16T16:00:00Z"] * 2,
        "instrument_type": ["FUTURES"] * 2,
        "contract_month": ["FRONT_MONTH_GENERIC"] * 2,
        "raw_currency": ["USD"] * 2,
        "raw_unit": ["MMBtu"] * 2,
        "source": ["TEST"] * 2,
        "dataset_id": ["TEST"] * 2,
        "timezone": ["UTC"] * 2,
        "methodology": ["TEST"] * 2,
        "qc_flag": ["PASS", "PASS"],
    }

    ttf_df = pd.DataFrame({
        **base,
        "observation_date": ["2026-11-12", "2026-11-13"],
        "benchmark": ["TTF"] * 2,
        "raw_price": [40.0, 41.0],
        "price_usd_mmbtu": [13.00, 13.50],
    })
    # JKM closed on 2026-11-13 (NaN)
    jkm_df = pd.DataFrame({
        **base,
        "observation_date": ["2026-11-12", "2026-11-13"],
        "benchmark": ["JKM"] * 2,
        "raw_price": [14.50, np.nan],
        "price_usd_mmbtu": [14.50, np.nan],
        "qc_flag": ["PASS", "MISSING_VALUE"],
    })
    hh_df = pd.DataFrame({
        **base,
        "observation_date": ["2026-11-12", "2026-11-13"],
        "benchmark": ["HENRY_HUB_FUTURES"] * 2,
        "raw_price": [2.80, 2.85],
        "price_usd_mmbtu": [2.80, 2.85],
    })
    fx_df = pd.DataFrame({
        **base,
        "observation_date": ["2026-11-12", "2026-11-13"],
        "benchmark": ["EURUSD"] * 2,
        "raw_price": [1.10, 1.10],
        "price_usd_mmbtu": [np.nan, np.nan],
    })

    panel = build_cross_basin_master_panel(
        {"TTF": ttf_df, "JKM": jkm_df, "HENRY_HUB_FUTURES": hh_df}, fx_df
    )
    assert len(panel) == 2
    assert panel.loc[0, "spread_jkm_minus_ttf"] == pytest.approx(1.50)
    assert panel.loc[0, "spread_ttf_minus_hh_fut"] == pytest.approx(10.20)
    assert bool(panel.loc[0, "is_complete_arb_day"]) is True

    # On 2026-11-13 JKM was closed -> JKM-TTF spread must remain NaN
    assert pd.isna(panel.loc[1, "spread_jkm_minus_ttf"])
    assert bool(panel.loc[1, "is_complete_arb_day"]) is False