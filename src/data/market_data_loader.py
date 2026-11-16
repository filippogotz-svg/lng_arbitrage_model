"""
Phase 1B Pipeline: Multi-Benchmark Market Data Loader & Cross-Basin Spread Builder.
Responsibilities:
1. Parse Bloomberg Excel (.xlsx) exports for EUR/USD, Henry Hub Futures (NG1),
   TTF Front-Month (TZT1), and JKM Front-Month (JKL1).
2. Standardize all series to ascending chronological order and point-in-time schema.
3. Convert TTF from EUR/MWh to USD/MMBtu using same-day EUR/USD spot rates:
   USD/MMBtu = (EUR/MWh * EURUSD) / 3.41214.
4. Apply non-destructive Quality Control checks to each benchmark.
5. Merge all benchmarks (including Phase 1A Henry Hub Spot) into a unified
   cross-basin analytical panel without forward-filling closed holiday observations.
6. Generate multi-panel historical price and arbitrage spread diagnostics.
"""

import json
import logging
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.quality_control import (
    QualityControlConfig,
    QualityControlReport,
    apply_quality_control_checks,
)
from src.data.schema import (
    REQUIRED_MARKET_COLUMNS,
    SourceMetadata,
    validate_market_dataframe_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("MarketDataLoader")

PREFERRED_PRICE_COLUMNS: List[str] = [
    "Settlement Price",
    "PX_SETTLE",
    "Last Price",
    "PX_LAST",
    "Close",
    "Price",
]


def load_toml_config(config_path: Path) -> dict:
    """Loads TOML configuration safely, stripping UTF-8 BOM if present on Windows."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    raw_text = config_path.read_text(encoding="utf-8-sig")
    return tomllib.loads(raw_text)


def parse_bloomberg_excel(excel_path: Path) -> pd.DataFrame:
    """
    Parses a Bloomberg HP Excel export into a clean two-column DataFrame:
    ['observation_date', 'raw_price'], sorted ascending by date.

    Handles Bloomberg multi-column exports (e.g., Date, Settlement Price, Volume, SMAVG)
    and optional metadata header rows above the 'Date' row.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Bloomberg Excel export not found at {excel_path}")

    preview_df = pd.read_excel(excel_path, header=None, nrows=15)
    header_row_idx: Optional[int] = None
    for idx, row in preview_df.iterrows():
        row_strings = [str(val).strip().lower() for val in row.values if pd.notna(val)]
        if "date" in row_strings:
            header_row_idx = int(idx)
            break

    if header_row_idx is None:
        raise ValueError(f"Could not locate 'Date' header row in {excel_path.name}")

    df = pd.read_excel(excel_path, header=header_row_idx)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"Missing 'Date' column in {excel_path.name}")

    # Select price column by priority, ignoring Volume and SMAVG columns
    price_col: Optional[str] = None
    lower_col_map = {c.lower(): c for c in df.columns}
    for candidate in PREFERRED_PRICE_COLUMNS:
        if candidate.lower() in lower_col_map:
            price_col = lower_col_map[candidate.lower()]
            break

    if price_col is None:
        non_date_cols = [
            c for c in df.columns
            if c != date_col and "volume" not in c.lower() and "smavg" not in c.lower()
        ]
        if not non_date_cols:
            raise ValueError(f"Could not identify price column in {excel_path.name}: {df.columns.tolist()}")
        price_col = non_date_cols[0]

    logger.info(
        "Parsed %s | Detected columns: Date='%s', Price='%s' (ignored: %s)",
        excel_path.name,
        date_col,
        price_col,
        [c for c in df.columns if c not in (date_col, price_col)],
    )

    clean = pd.DataFrame({
        "observation_date": pd.to_datetime(df[date_col], errors="coerce"),
        "raw_price": pd.to_numeric(df[price_col], errors="coerce"),
    })

    # Drop footer/note rows where Date could not be parsed
    clean = clean.dropna(subset=["observation_date"]).copy()
    clean = clean.sort_values("observation_date").reset_index(drop=True)
    return clean


def standardize_bloomberg_series(
    parsed_df: pd.DataFrame,
    meta: SourceMetadata,
    retrieval_ts_utc: str,
    fx_lookup: Optional[pd.Series] = None,
    mwh_to_mmbtu: float = 3.41214,
) -> pd.DataFrame:
    """
    Converts a parsed Bloomberg series into the canonical 15-column point-in-time schema.
    If raw_currency is EUR and raw_unit is MWh, converts to USD/MMBtu using same-day EUR/USD.
    """
    obs_dates = pd.to_datetime(parsed_df["observation_date"])
    pub_dates = obs_dates + pd.offsets.BDay(meta.publication_lag_business_days)
    obs_str = obs_dates.dt.strftime("%Y-%m-%d")
    pub_str = pub_dates.dt.strftime("%Y-%m-%d")

    raw_prices = parsed_df["raw_price"].astype(float)

    if meta.raw_currency == "USD" and meta.raw_unit == "MMBtu":
        price_usd_mmbtu = raw_prices.copy()
        missing_fx_mask = pd.Series(False, index=parsed_df.index)
    elif meta.raw_currency == "EUR" and meta.raw_unit == "MWh":
        if fx_lookup is None:
            raise ValueError(f"EUR/USD lookup series is required to convert {meta.benchmark} from EUR/MWh.")
        aligned_fx = obs_str.map(fx_lookup)
        price_usd_mmbtu = (raw_prices * aligned_fx) / mwh_to_mmbtu
        missing_fx_mask = raw_prices.notna() & aligned_fx.isna()
    elif meta.benchmark == "EURUSD":
        # Non-energy FX series: store rate in raw_price; price_usd_mmbtu remains NaN
        price_usd_mmbtu = pd.Series(np.nan, index=parsed_df.index, dtype=float)
        missing_fx_mask = pd.Series(False, index=parsed_df.index)
    else:
        raise ValueError(f"Unsupported currency/unit combination: {meta.raw_currency}/{meta.raw_unit}")

    standardized = pd.DataFrame({
        "observation_date": obs_str,
        "publication_date": pub_str,
        "retrieval_timestamp": retrieval_ts_utc,
        "benchmark": meta.benchmark,
        "instrument_type": meta.instrument_type,
        "contract_month": meta.contract_month,
        "raw_price": raw_prices,
        "raw_currency": meta.raw_currency,
        "raw_unit": meta.raw_unit,
        "price_usd_mmbtu": price_usd_mmbtu,
        "source": meta.source_name,
        "dataset_id": meta.dataset_id,
        "timezone": meta.timezone,
        "methodology": meta.methodology,
        "qc_flag": "UNCHECKED",
    })

    standardized.attrs["missing_fx_mask"] = missing_fx_mask
    return standardized[REQUIRED_MARKET_COLUMNS]


def run_fx_quality_control(
    fx_df: pd.DataFrame,
    min_fx: float = 0.70,
    max_fx: float = 1.80,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Runs dedicated non-destructive QC checks on EUR/USD exchange rates.
    Verifies quotation convention (USD per 1.00 EUR, expected within [0.70, 1.80]).
    """
    checked = fx_df.copy().sort_values("observation_date").reset_index(drop=True)
    rates = pd.to_numeric(checked["raw_price"], errors="coerce")

    flags = pd.Series("PASS", index=checked.index, dtype="object")
    is_dup = checked.duplicated(subset=["observation_date"], keep="first")
    is_missing = rates.isna()
    is_invalid = ((rates < min_fx) | (rates > max_fx)) & (~is_missing)

    flags.loc[is_missing] = "MISSING_VALUE"
    flags.loc[is_invalid] = "INVALID_FX_CONVENTION"
    flags.loc[is_dup] = "DUPLICATE_DATE"

    checked["qc_flag"] = flags
    # Clear non-serializable attrs before Parquet export in pandas 3.x
    checked.attrs.clear()

    summary = {
        "total_rows": int(len(checked)),
        "valid_pass_count": int((flags == "PASS").sum()),
        "missing_value_count": int(is_missing.sum()),
        "invalid_fx_count": int(is_invalid.sum()),
        "duplicate_date_count": int(is_dup.sum()),
    }
    return checked, summary


def build_cross_basin_master_panel(
    series_dict: Dict[str, pd.DataFrame],
    fx_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges Henry Hub Spot, Henry Hub Futures, TTF (USD/MMBtu), JKM (USD/MMBtu),
    and EUR/USD on observation_date without forward-filling closed holiday days.

    Computes historical inter-basin spreads only on dates where both legs have
    valid non-missing settlements.
    """
    def extract_valid_col(df: pd.DataFrame, val_col: str, out_name: str, flag_name: str) -> pd.DataFrame:
        sub = df[["observation_date", val_col, "qc_flag"]].copy()
        sub = sub.drop_duplicates(subset=["observation_date"], keep="first")
        invalid_mask = sub["qc_flag"].isin(["INVALID_PRICE", "INVALID_FX_CONVENTION"])
        sub.loc[invalid_mask, val_col] = np.nan
        return sub.rename(columns={val_col: out_name, "qc_flag": flag_name})

    ttf_sub = extract_valid_col(series_dict["TTF"], "price_usd_mmbtu", "ttf_usd_mmbtu", "ttf_qc")
    ttf_raw = series_dict["TTF"][["observation_date", "raw_price"]].drop_duplicates("observation_date").rename(
        columns={"raw_price": "ttf_eur_mwh"}
    )
    jkm_sub = extract_valid_col(series_dict["JKM"], "price_usd_mmbtu", "jkm_usd_mmbtu", "jkm_qc")
    hh_fut_sub = extract_valid_col(
        series_dict["HENRY_HUB_FUTURES"], "price_usd_mmbtu", "hh_futures_usd_mmbtu", "hh_fut_qc"
    )
    fx_sub = extract_valid_col(fx_df, "raw_price", "eurusd", "fx_qc")

    common_start = max(
        ttf_sub["observation_date"].min(),
        jkm_sub["observation_date"].min(),
        hh_fut_sub["observation_date"].min(),
    )
    common_end = max(
        ttf_sub["observation_date"].max(),
        jkm_sub["observation_date"].max(),
        hh_fut_sub["observation_date"].max(),
    )

    panel = pd.DataFrame({
        "observation_date": pd.date_range(common_start, common_end, freq="B").strftime("%Y-%m-%d")
    })

    panel = (
        panel.merge(fx_sub, on="observation_date", how="left")
        .merge(ttf_raw, on="observation_date", how="left")
        .merge(ttf_sub, on="observation_date", how="left")
        .merge(jkm_sub, on="observation_date", how="left")
        .merge(hh_fut_sub, on="observation_date", how="left")
    )

    if "HENRY_HUB_SPOT" in series_dict:
        hh_spot_sub = extract_valid_col(
            series_dict["HENRY_HUB_SPOT"], "price_usd_mmbtu", "hh_spot_usd_mmbtu", "hh_spot_qc"
        )
        panel = panel.merge(hh_spot_sub, on="observation_date", how="left")
    else:
        panel["hh_spot_usd_mmbtu"] = np.nan
        panel["hh_spot_qc"] = "UNAVAILABLE"

    panel["spread_jkm_minus_ttf"] = panel["jkm_usd_mmbtu"] - panel["ttf_usd_mmbtu"]
    panel["spread_ttf_minus_hh_fut"] = panel["ttf_usd_mmbtu"] - panel["hh_futures_usd_mmbtu"]
    panel["spread_jkm_minus_hh_fut"] = panel["jkm_usd_mmbtu"] - panel["hh_futures_usd_mmbtu"]
    panel["spread_hh_fut_minus_spot"] = panel["hh_futures_usd_mmbtu"] - panel["hh_spot_usd_mmbtu"]

    panel["is_complete_arb_day"] = (
        panel["ttf_usd_mmbtu"].notna()
        & panel["jkm_usd_mmbtu"].notna()
        & panel["hh_futures_usd_mmbtu"].notna()
    )

    return panel


def generate_cross_basin_diagnostic_chart(
    panel: pd.DataFrame,
    output_chart_path: Path,
) -> None:
    """
    Generates a 3-panel institutional chart:
    Panel 1: Global Gas Benchmarks (JKM, TTF, Henry Hub Futures) in USD/MMBtu.
    Panel 2: Inter-Basin Arbitrage Spreads (JKM - TTF, TTF - HH, JKM - HH) in USD/MMBtu.
    Panel 3: EUR/USD Spot Exchange Rate.
    """
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = panel.copy()
    plot_df["obs_dt"] = pd.to_datetime(plot_df["observation_date"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.8, 1.0]},
    )

    ax1.plot(
        plot_df["obs_dt"],
        plot_df["jkm_usd_mmbtu"],
        color="#c0392b",
        linewidth=1.1,
        label="JKM Front-Month (USD/MMBtu)",
    )
    ax1.plot(
        plot_df["obs_dt"],
        plot_df["ttf_usd_mmbtu"],
        color="#1f4e78",
        linewidth=1.1,
        label="TTF Front-Month (USD/MMBtu)",
    )
    ax1.plot(
        plot_df["obs_dt"],
        plot_df["hh_futures_usd_mmbtu"],
        color="#2e8b57",
        linewidth=1.1,
        label="Henry Hub Futures NG1 (USD/MMBtu)",
    )
    ax1.set_title(
        "Global LNG & Natural Gas Benchmarks, Inter-Basin Spreads & EUR/USD FX (USD/MMBtu)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USD / MMBtu)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True)

    ax2.axhline(0.0, color="#333333", linewidth=0.8, linestyle="-")
    ax2.plot(
        plot_df["obs_dt"],
        plot_df["spread_jkm_minus_ttf"],
        color="#8e44ad",
        linewidth=1.1,
        label="JKM - TTF Spread (Asian Premium over Europe)",
    )
    ax2.plot(
        plot_df["obs_dt"],
        plot_df["spread_ttf_minus_hh_fut"],
        color="#2980b9",
        linewidth=0.9,
        alpha=0.85,
        label="TTF - Henry Hub Spread (Gross Transatlantic Margin)",
    )
    ax2.set_ylabel("Spread (USD / MMBtu)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True)

    ax3.plot(
        plot_df["obs_dt"],
        plot_df["eurusd"],
        color="#555555",
        linewidth=1.0,
        label="EUR/USD Spot (USD per 1 EUR)",
    )
    ax3.set_ylabel("EUR / USD", fontsize=10)
    ax3.set_xlabel("Observation Date", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", frameon=True)
    ax3.xaxis.set_major_locator(mdates.YearLocator(1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved cross-basin diagnostic chart to %s", output_chart_path)


def run_market_data_pipeline(
    config_path: Path = Path("config/data_config.toml"),
) -> Tuple[pd.DataFrame, Dict[str, QualityControlReport]]:
    """
    Executes the full Phase 1B multi-benchmark ingestion, FX conversion, QC,
    and cross-basin master panel construction pipeline.
    """
    cfg = load_toml_config(config_path)
    paths_cfg = cfg["paths"]
    sources_cfg = cfg["sources"]
    qc_cfg = QualityControlConfig(**cfg["quality_control"])
    mwh_to_mmbtu = float(cfg["constants"]["mwh_to_mmbtu_hhv"])

    comm_dir = Path(paths_cfg["raw_commercial_dir"])
    proc_dir = Path(paths_cfg["processed_dir"])
    meta_dir = Path(paths_cfg["metadata_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    retrieval_ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def build_meta(key: str) -> SourceMetadata:
        s = sources_cfg[key]
        return SourceMetadata(
            benchmark=s["benchmark"],
            instrument_type=s["instrument_type"],
            contract_month=s["contract_month"],
            source_name=s["source_name"],
            dataset_id=s["dataset_id"],
            url=s.get("url", f"local://{s.get('filename', '')}"),
            raw_currency=s["raw_currency"],
            raw_unit=s["raw_unit"],
            timezone=s["timezone"],
            publication_lag_business_days=int(s["publication_lag_business_days"]),
            methodology=s["methodology"],
        )

    # 1. Process EUR/USD first (required for TTF EUR/MWh -> USD/MMBtu conversion)
    fx_meta = build_meta("eurusd_spot")
    fx_raw_df = parse_bloomberg_excel(comm_dir / sources_cfg["eurusd_spot"]["filename"])
    fx_std_df = standardize_bloomberg_series(fx_raw_df, fx_meta, retrieval_ts_utc)
    fx_checked_df, fx_qc_summary = run_fx_quality_control(fx_std_df)
    validate_market_dataframe_schema(fx_checked_df)

    fx_checked_df.to_csv(proc_dir / "eurusd_spot_clean.csv", index=False)
    fx_checked_df.to_parquet(proc_dir / "eurusd_spot_clean.parquet", index=False)
    logger.info("EUR/USD QC Summary: %s", fx_qc_summary)

    valid_fx = fx_checked_df.loc[fx_checked_df["qc_flag"] == "PASS"].set_index("observation_date")["raw_price"]

    # 2. Process Energy Benchmarks (Henry Hub Futures, TTF, JKM)
    benchmark_keys = [
        ("henry_hub_futures", "henry_hub_futures_clean", None),
        ("ttf_front_month", "ttf_front_month_clean", valid_fx),
        ("jkm_front_month", "jkm_front_month_clean", None),
    ]

    processed_series: Dict[str, pd.DataFrame] = {}
    qc_reports: Dict[str, QualityControlReport] = {}

    for cfg_key, out_stem, fx_series in benchmark_keys:
        meta = build_meta(cfg_key)
        raw_df = parse_bloomberg_excel(comm_dir / sources_cfg[cfg_key]["filename"])
        std_df = standardize_bloomberg_series(
            raw_df,
            meta,
            retrieval_ts_utc,
            fx_lookup=fx_series,
            mwh_to_mmbtu=mwh_to_mmbtu,
        )
        missing_fx_mask = std_df.attrs.get("missing_fx_mask", pd.Series(False, index=std_df.index))

        checked_df, report = apply_quality_control_checks(std_df, qc_cfg)
        checked_df.loc[missing_fx_mask, "qc_flag"] = "MISSING_FX_RATE"
        # Clear non-serializable attrs before Parquet export in pandas 3.x
        checked_df.attrs.clear()

        validate_market_dataframe_schema(checked_df)

        checked_df.to_csv(proc_dir / f"{out_stem}.csv", index=False)
        checked_df.to_parquet(proc_dir / f"{out_stem}.parquet", index=False)

        processed_series[meta.benchmark] = checked_df
        qc_reports[meta.benchmark] = report

        logger.info(
            "%s QC Complete | Range: %s to %s | Total: %d | PASS: %d | Missing: %d | Outliers: %d",
            meta.benchmark,
            report.start_date,
            report.end_date,
            report.total_rows,
            report.valid_pass_count,
            report.missing_value_count,
            report.outlier_high_count + report.outlier_low_count,
        )

    # 3. Load Phase 1A Henry Hub Spot dataset if already generated
    hh_spot_parquet = proc_dir / "henry_hub_spot_clean.parquet"
    if hh_spot_parquet.exists():
        processed_series["HENRY_HUB_SPOT"] = pd.read_parquet(hh_spot_parquet)
        logger.info("Loaded Phase 1A Henry Hub Spot dataset (%d rows)", len(processed_series["HENRY_HUB_SPOT"]))

    # 4. Build Cross-Basin Master Panel
    master_panel = build_cross_basin_master_panel(processed_series, fx_checked_df)
    master_panel.to_csv(proc_dir / "master_market_panel.csv", index=False)
    master_panel.to_parquet(proc_dir / "master_market_panel.parquet", index=False)

    complete_days = int(master_panel["is_complete_arb_day"].sum())
    logger.info(
        "Master Cross-Basin Panel exported | Window: %s to %s | Business Days: %d | Complete 3-Basin Days: %d",
        master_panel["observation_date"].min(),
        master_panel["observation_date"].max(),
        len(master_panel),
        complete_days,
    )

    # 5. Export Consolidated Phase 1B QC Audit Report
    audit_payload = {
        "generated_at_utc": retrieval_ts_utc,
        "eurusd_qc": fx_qc_summary,
        "benchmarks_qc": {k: v.to_dict() for k, v in qc_reports.items()},
        "master_panel_summary": {
            "start_date": str(master_panel["observation_date"].min()),
            "end_date": str(master_panel["observation_date"].max()),
            "total_business_days": int(len(master_panel)),
            "complete_three_basin_days": complete_days,
        },
    }
    with open(meta_dir / "phase1b_market_qc_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    # 6. Generate Diagnostic Chart
    chart_path = Path(paths_cfg["charts_dir"]) / "phase1b_cross_basin_prices_and_spreads.png"
    generate_cross_basin_diagnostic_chart(master_panel, chart_path)

    return master_panel, qc_reports


if __name__ == "__main__":
    run_market_data_pipeline()