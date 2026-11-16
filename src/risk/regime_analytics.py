"""
Phase 6 Module: Market Regime Decomposition, Tail Risk Metrics (VaR / CVaR),
Diversion Option Valuation & Cross-Sensitivity Matrix (JKM-TTF Spread vs Freight Rate).
"""

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.economics.scenario_engine import compare_nwe_vs_nea_cargo, get_latest_complete_market_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("RiskAnalytics")


def assign_market_regime(obs_date_str: str) -> str:
    """Classifies each trading day into structural global LNG market regimes."""
    if obs_date_str < "2020-01-01":
        return "1. Pre-Crisis Baseline (2017-2019)"
    elif obs_date_str < "2021-01-01":
        return "2. COVID Shock & US Shut-Ins (2020)"
    elif obs_date_str < "2023-01-01":
        return "3. European Energy Crisis (2021-2022)"
    else:
        return "4. Rebalancing & Supply Wave (2023-2026)"


def compute_tail_risk_metrics(profit_series_m_usd: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Computes historical Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall)
    at confidence level (1 - alpha) on cargo net profit ($ Millions).
    """
    clean = profit_series_m_usd.dropna()
    if clean.empty:
        return np.nan, np.nan
    var_95 = float(np.quantile(clean, alpha))
    tail = clean[clean <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95
    return var_95, cvar_95


def analyze_historical_regimes_and_risk(
    backtest_csv_path: Path = Path("outputs/tables/phase5_historical_backtest_summary.csv"),
) -> pd.DataFrame:
    """
    Decomposes historical backtest performance across structural LNG regimes,
    computing arbitrage frequency, net profit statistics, VaR/CVaR, and
    the embedded geographic diversion option value.
    """
    if not backtest_csv_path.exists():
        raise FileNotFoundError(
            f"Backtest summary not found at {backtest_csv_path}. Run Phase 5 first."
        )

    df = pd.read_csv(backtest_csv_path)
    df["regime"] = df["observation_date"].astype(str).apply(assign_market_regime)

    # Optimal dynamic routing profit per day (choosing the higher-netback basin)
    df["best_basin_profit_m_usd"] = np.maximum(
        df["nwe_net_profit_m_usd"], df["nea_net_profit_m_usd"]
    )
    df["diversion_option_gain_m_usd"] = (
        df["best_basin_profit_m_usd"] - df["nwe_net_profit_m_usd"]
    )

    groups = [("ALL_PERIOD (2017-2026)", df)] + [
        (reg_name, grp) for reg_name, grp in df.groupby("regime", sort=True)
    ]

    summary_rows = []
    for label, sub in groups:
        n_days = len(sub)
        nwe_var95, nwe_cvar95 = compute_tail_risk_metrics(sub["nwe_net_profit_m_usd"], alpha=0.05)
        nea_var95, nea_cvar95 = compute_tail_risk_metrics(sub["nea_net_profit_m_usd"], alpha=0.05)

        summary_rows.append({
            "market_regime": label,
            "trading_days": int(n_days),
            "avg_hh_usd_mmbtu": round(float(sub["hh_usd_mmbtu"].mean()), 2),
            "avg_ttf_usd_mmbtu": round(float(sub["ttf_usd_mmbtu"].mean()), 2),
            "avg_jkm_usd_mmbtu": round(float(sub["jkm_usd_mmbtu"].mean()), 2),
            "pct_days_arb_open_nwe": round(float(sub["is_arbitrage_open_nwe"].mean() * 100.0), 1),
            "pct_days_arb_open_nea": round(float(sub["is_arbitrage_open_nea"].mean() * 100.0), 1),
            "pct_days_asia_beats_europe": round(
                float((sub["nea_net_profit_m_usd"] > sub["nwe_net_profit_m_usd"]).mean() * 100.0), 1
            ),
            "mean_nwe_profit_m_usd": round(float(sub["nwe_net_profit_m_usd"].mean()), 2),
            "mean_nea_profit_m_usd": round(float(sub["nea_net_profit_m_usd"].mean()), 2),
            "mean_diversion_option_value_m_usd": round(
                float(sub["diversion_option_gain_m_usd"].mean()), 2
            ),
            "nwe_profit_var95_m_usd": round(nwe_var95, 2),
            "nwe_profit_cvar95_m_usd": round(nwe_cvar95, 2),
            "nea_profit_var95_m_usd": round(nea_var95, 2),
            "nea_profit_cvar95_m_usd": round(nea_cvar95, 2),
        })

    regime_df = pd.DataFrame(summary_rows)
    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    regime_csv = out_tables / "phase6_regime_risk_summary.csv"
    regime_df.to_csv(regime_csv, index=False)
    logger.info("Exported regime & tail-risk summary to %s", regime_csv)

    for _, r in regime_df.iterrows():
        logger.info(
            "%-38s | Days: %4d | Open NWE: %5.1f%% | Open NEA: %5.1f%% | "
            "Avg NWE: $%6.2fM | Avg NEA: $%6.2fM | Diversion Opt: +$%4.2fM",
            r["market_regime"],
            r["trading_days"],
            r["pct_days_arb_open_nwe"],
            r["pct_days_arb_open_nea"],
            r["mean_nwe_profit_m_usd"],
            r["mean_nea_profit_m_usd"],
            r["mean_diversion_option_value_m_usd"],
        )

    return regime_df


def compute_spread_freight_sensitivity_matrix(
    as_of_date: str,
    hh_price: float,
    ttf_price: float,
    jkm_minus_ttf_spreads: List[float],
    freight_rates_usd_day: List[float],
    asia_route_id: str = "NEA_PANAMA",
) -> pd.DataFrame:
    """
    Computes a 2D matrix of Net Profit Advantage (NEA Profit - NWE Profit, in $ Millions)
    across a grid of (JKM - TTF Spread) x (Spot Freight Rate $/day).
    """
    matrix = np.zeros((len(freight_rates_usd_day), len(jkm_minus_ttf_spreads)))

    for i, freight in enumerate(freight_rates_usd_day):
        for j, spread in enumerate(jkm_minus_ttf_spreads):
            jkm_sim = max(0.50, ttf_price + spread)
            comp = compare_nwe_vs_nea_cargo(
                scenario_name="SENSITIVITY",
                as_of_date=as_of_date,
                hh_price=hh_price,
                ttf_price=ttf_price,
                jkm_price=jkm_sim,
                freight_rate_usd_day=freight,
                asia_route_id=asia_route_id,
            )
            diff_m_usd = (
                comp.nea_econ.net_profit_cargo_usd - comp.nwe_econ.net_profit_cargo_usd
            ) / 1e6
            matrix[i, j] = round(diff_m_usd, 2)

    sens_df = pd.DataFrame(
        matrix,
        index=[f"${int(f/1000)}k/d" for f in freight_rates_usd_day],
        columns=[f"{s:+.2f}" for s in jkm_minus_ttf_spreads],
    )
    sens_df.index.name = "Freight_Rate"
    return sens_df


def generate_phase6_risk_and_sensitivity_chart(
    regime_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    as_of_date: str,
    ttf_price: float,
    output_chart_path: Path,
) -> None:
    """
    Generates a two-panel institutional risk & sensitivity chart:
    Panel 1: Average Cargo Profit & 95% VaR across Market Regimes ($M).
    Panel 2: 2D Arbitrage Indifference Heatmap (NEA - NWE Profit Advantage in $M)
             across JKM-TTF Spread vs Daily Time-Charter Freight Rate.
    """
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

    reg_only = regime_df[~regime_df["market_regime"].str.startswith("ALL_PERIOD")].copy()
    short_labels = ["2017-2019\nPre-Crisis", "2020\nCOVID Shock", "2021-2022\nEU Crisis", "2023-2026\nRebalancing"]
    x = np.arange(len(reg_only))
    w = 0.35

    ax1.axhline(0.0, color="#333333", linewidth=0.8)
    ax1.bar(x - w / 2, reg_only["mean_nwe_profit_m_usd"], w, label="Mean NWE Profit ($M)", color="#1f4e78")
    ax1.bar(x + w / 2, reg_only["mean_nea_profit_m_usd"], w, label="Mean NEA Profit ($M)", color="#c0392b")
    ax1.plot(x - w / 2, reg_only["nwe_profit_var95_m_usd"], "o--", color="#0d2b45", label="NWE 95% VaR ($M)")
    ax1.plot(x + w / 2, reg_only["nea_profit_var95_m_usd"], "s--", color="#781d14", label="NEA 95% VaR ($M)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels)
    ax1.set_ylabel("Cargo Net Profit / VaR (USD Millions)", fontsize=10)
    ax1.set_title("Historical Cargo Profitability & 95% VaR by Market Regime", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    data_vals = sens_df.values
    vmax = max(abs(data_vals.min()), abs(data_vals.max()), 1.0)
    im = ax2.imshow(data_vals, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax2.set_xticks(np.arange(len(sens_df.columns)))
    ax2.set_xticklabels(sens_df.columns)
    ax2.set_yticks(np.arange(len(sens_df.index)))
    ax2.set_yticklabels(sens_df.index)
    ax2.set_xlabel("JKM - TTF Price Spread (USD / MMBtu)", fontsize=10)
    ax2.set_ylabel("Spot LNG Time-Charter Freight Rate", fontsize=10)
    ax2.set_title(
        f"Diversion Profit Advantage: NEA minus NWE ($M)\n(Base: {as_of_date}, TTF=${ttf_price:.2f}/MMBtu, Panama)",
        fontsize=10.5,
        fontweight="bold",
    )

    for i in range(data_vals.shape[0]):
        for j in range(data_vals.shape[1]):
            val = data_vals[i, j]
            text_color = "white" if abs(val) > 0.65 * vmax else "black"
            ax2.text(j, i, f"{val:+.1f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("NEA - NWE Profit ($M) [Red = Asia, Blue = Europe]", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved Phase 6 regime & sensitivity chart to %s", output_chart_path)


def run_phase6_risk_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes the full Phase 6 regime decomposition, risk metrics, and sensitivity suite."""
    regime_df = analyze_historical_regimes_and_risk()

    as_of_dt, base_hh, base_ttf, _ = get_latest_complete_market_prices()
    spreads = [-1.00, 0.00, 0.50, 1.00, 1.50, 2.50, 4.00]
    freights = [30000.0, 50000.0, 70000.0, 100000.0, 140000.0, 180000.0]

    sens_df = compute_spread_freight_sensitivity_matrix(
        as_of_date=as_of_dt,
        hh_price=base_hh,
        ttf_price=base_ttf,
        jkm_minus_ttf_spreads=spreads,
        freight_rates_usd_day=freights,
        asia_route_id="NEA_PANAMA",
    )
    out_tables = Path("outputs/tables")
    sens_df.to_csv(out_tables / "phase6_spread_freight_sensitivity_matrix.csv")
    logger.info("Exported 2D Spread x Freight sensitivity matrix to %s", out_tables / "phase6_spread_freight_sensitivity_matrix.csv")

    generate_phase6_risk_and_sensitivity_chart(
        regime_df=regime_df,
        sens_df=sens_df,
        as_of_date=as_of_dt,
        ttf_price=base_ttf,
        output_chart_path=Path("outputs/charts/phase6_regime_and_sensitivity_analysis.png"),
    )

    return regime_df, sens_df


if __name__ == "__main__":
    run_phase6_risk_pipeline()