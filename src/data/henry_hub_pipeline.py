"""
Phase 1A Pipeline: Historical Henry Hub Natural Gas Spot Price (EIA / FRED DHHNGSP).
Responsibilities:
1. Retrieve raw historical Henry Hub spot prices from public FRED endpoint (or local fallback).
2. Preserve immutable raw CSV copy with retrieval timestamp.
3. Standardize column names, dates, T+1 publication dates, and metadata.
4. Execute non-destructive Quality Control checks.
5. Export clean Parquet and CSV datasets ready for TTF, JKM, and EUR/USD date-join.
6. Generate institutional diagnostic price and volatility chart.
"""

import io
import json
import logging
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

from src.data.schema import (
    REQUIRED_MARKET_COLUMNS,
    SourceMetadata,
    validate_market_dataframe_schema,
)
from src.data.quality_control import (
    QualityControlConfig,
    QualityControlReport,
    apply_quality_control_checks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("HenryHubPipeline")


def load_pipeline_config(config_path: Path) -> dict:
    """Loads TOML configuration file using Python standard library tomllib."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def fetch_raw_henry_hub_fred(url: str, timeout_seconds: int = 30) -> str:
    """
    Retrieves raw CSV text from St. Louis FRED public endpoint for series DHHNGSP.

    Raises:
        RuntimeError: If HTTP request fails or returns empty payload.
    """
    logger.info("Requesting historical Henry Hub spot data from %s", url)
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    if not response.text or "DATE" not in response.text.upper():
        raise RuntimeError("Invalid or empty CSV payload received from FRED endpoint.")
    return response.text


def standardize_henry_hub_dataframe(
    raw_csv_text: str,
    meta: SourceMetadata,
    retrieval_ts_utc: str,
) -> pd.DataFrame:
    """
    Transforms raw FRED CSV text into the canonical 15-column point-in-time schema.
    Preserves missing values (represented as '.' in FRED) as NaN without dropping rows.
    Computes publication_date using the configured business-day publication lag.
    """
    raw_df = pd.read_csv(io.StringIO(raw_csv_text))
    date_col = raw_df.columns[0]
    val_col = raw_df.columns[1]

    obs_dates = pd.to_datetime(raw_df[date_col], errors="raise")
    pub_dates = obs_dates + pd.offsets.BDay(meta.publication_lag_business_days)
    raw_prices = pd.to_numeric(raw_df[val_col].replace(".", np.nan), errors="coerce")

    standardized = pd.DataFrame({
        "observation_date": obs_dates.dt.strftime("%Y-%m-%d"),
        "publication_date": pub_dates.dt.strftime("%Y-%m-%d"),
        "retrieval_timestamp": retrieval_ts_utc,
        "benchmark": meta.benchmark,
        "instrument_type": meta.instrument_type,
        "contract_month": meta.contract_month,
        "raw_price": raw_prices,
        "raw_currency": meta.raw_currency,
        "raw_unit": meta.raw_unit,
        "price_usd_mmbtu": raw_prices,
        "source": meta.source_name,
        "dataset_id": meta.dataset_id,
        "timezone": meta.timezone,
        "methodology": meta.methodology,
        "qc_flag": "UNCHECKED",
    })

    return standardized[REQUIRED_MARKET_COLUMNS]


def generate_henry_hub_diagnostic_chart(
    df: pd.DataFrame,
    output_chart_path: Path,
) -> None:
    """
    Generates a two-panel diagnostic plot:
    Panel 1: Historical Henry Hub Spot Price (USD/MMBtu) with QC-flagged regime spikes.
    Panel 2: 30-Trading-Day Rolling Annualized Log-Return Volatility (%).
    """
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["obs_dt"] = pd.to_datetime(plot_df["observation_date"])
    plot_df = plot_df.sort_values("obs_dt")

    valid_mask = plot_df["price_usd_mmbtu"].notna() & (plot_df["price_usd_mmbtu"] > 0)
    clean_series = plot_df.loc[valid_mask].copy()

    clean_series["log_ret"] = np.log(
        clean_series["price_usd_mmbtu"] / clean_series["price_usd_mmbtu"].shift(1)
    )
    clean_series["ann_vol_pct"] = (
        clean_series["log_ret"].rolling(window=30, min_periods=15).std() * np.sqrt(252) * 100.0
    )

    outliers = plot_df[plot_df["qc_flag"].isin(["OUTLIER_HIGH", "OUTLIER_LOW"])]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(
        clean_series["obs_dt"],
        clean_series["price_usd_mmbtu"],
        color="#1f4e78",
        linewidth=1.1,
        label="Henry Hub Spot (USD/MMBtu)",
    )
    if not outliers.empty:
        ax1.scatter(
            outliers["obs_dt"],
            outliers["price_usd_mmbtu"],
            color="#d9534f",
            s=18,
            zorder=5,
            label="QC Flagged Regime Spike (Retained)",
        )
    ax1.set_title(
        "Henry Hub Natural Gas Spot Price (EIA / FRED DHHNGSP) & Quality Control Diagnostics",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USD / MMBtu)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True)

    ax2.plot(
        clean_series["obs_dt"],
        clean_series["ann_vol_pct"],
        color="#2e75b6",
        linewidth=1.0,
        label="30-Day Rolling Annualized Volatility (%)",
    )
    ax2.set_ylabel("Annualized Vol (%)", fontsize=10)
    ax2.set_xlabel("Observation Date", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved diagnostic chart to %s", output_chart_path)


def run_henry_hub_pipeline(
    config_path: Path = Path("config/data_config.toml"),
    local_raw_csv_override: Optional[Path] = None,
) -> tuple[pd.DataFrame, QualityControlReport]:
    """
    Executes the end-to-end Phase 1A Henry Hub data pipeline.
    """
    cfg = load_pipeline_config(config_path)
    src_cfg = cfg["sources"]["henry_hub_spot"]
    qc_cfg_dict = cfg["quality_control"]
    paths_cfg = cfg["paths"]

    meta = SourceMetadata(
        benchmark=src_cfg["benchmark"],
        instrument_type=src_cfg["instrument_type"],
        contract_month=src_cfg["contract_month"],
        source_name=src_cfg["source_name"],
        dataset_id=src_cfg["dataset_id"],
        url=src_cfg["url"],
        raw_currency=src_cfg["raw_currency"],
        raw_unit=src_cfg["raw_unit"],
        timezone=src_cfg["timezone"],
        publication_lag_business_days=int(src_cfg["publication_lag_business_days"]),
        methodology=src_cfg["methodology"],
    )
    qc_config = QualityControlConfig(**qc_cfg_dict)

    retrieval_ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if local_raw_csv_override is not None and local_raw_csv_override.exists():
        logger.info("Loading raw Henry Hub data from local file: %s", local_raw_csv_override)
        raw_csv_text = local_raw_csv_override.read_text(encoding="utf-8")
    else:
        raw_csv_text = fetch_raw_henry_hub_fred(meta.url)

    raw_dir = Path(paths_cfg["raw_public_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = raw_dir / f"henry_hub_raw_{meta.dataset_id}.csv"
    raw_file_path.write_text(raw_csv_text, encoding="utf-8")
    logger.info("Preserved immutable raw dataset at %s", raw_file_path)

    std_df = standardize_henry_hub_dataframe(raw_csv_text, meta, retrieval_ts_utc)
    checked_df, qc_report = apply_quality_control_checks(std_df, qc_config)
    validate_market_dataframe_schema(checked_df)

    proc_dir = Path(paths_cfg["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    csv_out = proc_dir / "henry_hub_spot_clean.csv"
    parquet_out = proc_dir / "henry_hub_spot_clean.parquet"

    checked_df.to_csv(csv_out, index=False)
    checked_df.to_parquet(parquet_out, index=False)
    logger.info("Exported processed datasets to %s and %s", csv_out, parquet_out)

    meta_dir = Path(paths_cfg["metadata_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    qc_json_path = meta_dir / "henry_hub_qc_report.json"
    with open(qc_json_path, "w", encoding="utf-8") as f:
        json.dump(qc_report.to_dict(), f, indent=2)
    logger.info(
        "QC Complete | Total Rows: %d | PASS: %d | Missing: %d | Outliers: %d",
        qc_report.total_rows,
        qc_report.valid_pass_count,
        qc_report.missing_value_count,
        qc_report.outlier_high_count + qc_report.outlier_low_count,
    )

    chart_path = Path(paths_cfg["charts_dir"]) / "phase1_henry_hub_spot_history.png"
    generate_henry_hub_diagnostic_chart(checked_df, chart_path)

    return checked_df, qc_report


if __name__ == "__main__":
    run_henry_hub_pipeline()