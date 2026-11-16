"""
Phase 5 Module: Historical Backtesting & Cross-Basin Arbitrage Engine.
Simulates daily LNG cargo economics across the entire multi-year market panel (2017-2026),
identifying historical arbitrage windows, netback spreads, and cargo profitability.
"""

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import tomllib
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physical.cargo_specs import load_physical_catalog
from src.physical.routing import load_routes_catalog
from src.physical.energy_balance import compute_cargo_energy_balance
from src.economics.cargo_pricer import evaluate_cargo_economics
from src.economics.scenario_engine import compare_nwe_vs_nea_cargo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("HistoricalEngine")


def load_backtest_config(config_path: Path = Path("config/backtesting_params.toml")) -> dict:
    """Loads backtesting TOML configuration safely."""
    if not config_path.exists():
        raise FileNotFoundError(f"Backtesting configuration file not found at {config_path}")
    return tomllib.loads(config_path.read_text(encoding="utf-8-sig"))


def run_historical_backtest(
    panel_path: Path = Path("data/processed/master_market_panel.parquet"),
    custom_bt_config_path: Path = Path("config/backtesting_params.toml"),
) -> pd.DataFrame:
    """
    Runs day-by-day cargo economics simulation across all complete trading days
    in master_market_panel.parquet.
    """
    if not panel_path.exists():
        raise FileNotFoundError(f"Master market panel not found at {panel_path}. Run Phase 1B first.")
    
    bt_cfg = load_backtest_config(custom_bt_config_path)["backtest_parameters"]
    freight_rate = float(bt_cfg["default_freight_rate_usd_day"])
    vessel_id = str(bt_cfg["vessel_id"])
    grade_id = str(bt_cfg["grade_id"])

    df = pd.read_parquet(panel_path)
    complete_df = df[df["is_complete_arb_day"]].copy().sort_values("observation_date").reset_index(drop=True)

    logger.info("Starting historical backtest across %d complete trading days...", len(complete_df))

    records = []
    for _, row in complete_df.iterrows():
        obs_date_str = str(row["observation_date"])
        hh = float(row["hh_futures_usd_mmbtu"])
        ttf = float(row["ttf_usd_mmbtu"])
        jkm = float(row["jkm_usd_mmbtu"])

        try:
            comp = compare_nwe_vs_nea_cargo(
                scenario_name=obs_date_str,
                as_of_date=obs_date_str,
                hh_price=hh,
                ttf_price=ttf,
                jkm_price=jkm,
                freight_rate_usd_day=freight_rate,
                asia_route_id="NEA_PANAMA",
                vessel_id=vessel_id,
                grade_id=grade_id,
            )

            records.append({
                "observation_date": obs_date_str,
                "hh_usd_mmbtu": hh,
                "ttf_usd_mmbtu": ttf,
                "jkm_usd_mmbtu": jkm,
                "nwe_netback_fob_usd_mmbtu": round(comp.nwe_econ.netback_at_fob_flange_usd_mmbtu, 2),
                "nea_netback_fob_usd_mmbtu": round(comp.nea_econ.netback_at_fob_flange_usd_mmbtu, 2),
                "nwe_net_profit_m_usd": round(comp.nwe_econ.net_profit_cargo_usd / 1e6, 2),
                "nea_net_profit_m_usd": round(comp.nea_econ.net_profit_cargo_usd / 1e6, 2),
                "netback_spread_nea_minus_nwe": round(comp.netback_spread_nea_minus_nwe_usd_mmbtu, 2),
                "preferred_destination": comp.preferred_destination,
                "is_arbitrage_open_nwe": comp.nwe_econ.net_profit_cargo_usd > 0,
                "is_arbitrage_open_nea": comp.nea_econ.net_profit_cargo_usd > 0,
            })
        except Exception as e:
            continue

    backtest_df = pd.DataFrame(records)
    
    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    csv_out = out_tables / "phase5_historical_backtest_summary.csv"
    backtest_df.to_csv(csv_out, index=False)
    logger.info("Exported historical backtest summary (%d rows) to %s", len(backtest_df), csv_out)

    generate_historical_backtest_chart(backtest_df, Path("outputs/charts/phase5_historical_arbitrage_backtest.png"))

    return backtest_df


def generate_historical_backtest_chart(
    backtest_df: pd.DataFrame,
    output_chart_path: Path,
) -> None:
    """
    Generates a 3-panel historical institutional chart:
    Panel 1: Outright Benchmark Prices (HH, TTF, JKM) 2017-2026.
    Panel 2: Daily Net Profit per Cargo ($ Million) for Europe vs Asia.
    Panel 3: Historical Netback Spread (NEA - NWE) in USD/MMBtu.
    """
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = backtest_df.copy()
    plot_df["obs_dt"] = pd.to_datetime(plot_df["observation_date"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 2.0, 1.5]},
    )

    # Panel 1: Prices
    ax1.plot(plot_df["obs_dt"], plot_df["jkm_usd_mmbtu"], color="#c0392b", linewidth=1.0, label="JKM Front-Month")
    ax1.plot(plot_df["obs_dt"], plot_df["ttf_usd_mmbtu"], color="#1f4e78", linewidth=1.0, label="TTF Front-Month")
    ax1.plot(plot_df["obs_dt"], plot_df["hh_usd_mmbtu"], color="#2e8b57", linewidth=1.0, label="Henry Hub Futures")
    ax1.set_title("Global LNG Benchmarks, Historical Cargo Net Profit & Arbitrage Spreads (2017-2026)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price (USD / MMBtu)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Net Profit per Cargo ($M)
    ax2.axhline(0.0, color="#333333", linewidth=0.8, linestyle="-")
    ax2.plot(plot_df["obs_dt"], plot_df["nwe_net_profit_m_usd"], color="#1f4e78", linewidth=0.9, alpha=0.85, label="NWE Direct Net Profit ($M)")
    ax2.plot(plot_df["obs_dt"], plot_df["nea_net_profit_m_usd"], color="#c0392b", linewidth=0.9, alpha=0.85, label="NEA Panama Net Profit ($M)")
    ax2.set_ylabel("Full-Cycle Net Profit (USD Millions)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True)

    # Panel 3: Netback Spread (NEA - NWE)
    ax3.axhline(0.0, color="#333333", linewidth=0.8, linestyle="-")
    ax3.plot(plot_df["obs_dt"], plot_df["netback_spread_nea_minus_nwe"], color="#8e44ad", linewidth=1.0, label="Netback Spread: NEA minus NWE (USD/MMBtu)")
    ax3.set_ylabel("Spread (USD / MMBtu)", fontsize=10)
    ax3.set_xlabel("Observation Date", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", frameon=True)
    ax3.xaxis.set_major_locator(mdates.YearLocator(1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved historical backtest chart to %s", output_chart_path)


if __name__ == "__main__":
    run_historical_backtest()