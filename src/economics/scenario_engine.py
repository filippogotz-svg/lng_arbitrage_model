"""
Phase 3B & Section 8.1/11: Destination Comparison (NWE vs NEA), Implied Breakeven Freight,
and Deterministic Scenario Engine.
"""

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import tomllib
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physical.cargo_specs import load_physical_catalog
from src.physical.routing import load_routes_catalog
from src.physical.energy_balance import compute_cargo_energy_balance
from src.economics.cargo_pricer import CargoEconomicResult, evaluate_cargo_economics
from src.economics.breakeven import BreakEvenAnalysisResult, calculate_cargo_breakeven_thresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ScenarioEngine")


@dataclass(frozen=True)
class DestinationComparisonResult:
    """Head-to-head commercial comparison between Northwest Europe (TTF) and Northeast Asia (JKM)."""
    scenario_name: str
    as_of_date: str
    hh_price_usd_mmbtu: float
    ttf_price_usd_mmbtu: float
    jkm_price_usd_mmbtu: float
    freight_rate_usd_day: float
    asia_route_id: str

    nwe_econ: CargoEconomicResult
    nea_econ: CargoEconomicResult
    nwe_breakeven: BreakEvenAnalysisResult
    nea_breakeven: BreakEvenAnalysisResult

    preferred_destination: str
    profit_advantage_usd: float
    netback_spread_nea_minus_nwe_usd_mmbtu: float
    implied_breakeven_freight_usd_day: float


def load_scenarios_config(config_path: Path = Path("config/scenarios.toml")) -> dict:
    """Loads scenarios TOML configuration safely."""
    if not config_path.exists():
        raise FileNotFoundError(f"Scenarios configuration file not found at {config_path}")
    return tomllib.loads(config_path.read_text(encoding="utf-8-sig"))


def get_latest_complete_market_prices(
    panel_path: Path = Path("data/processed/master_market_panel.parquet"),
) -> Tuple[str, float, float, float]:
    """
    Retrieves the most recent complete trading day from master_market_panel.parquet
    where Henry Hub Futures, TTF (USD/MMBtu), and JKM (USD/MMBtu) all settled.
    """
    if not panel_path.exists():
        raise FileNotFoundError(f"Master market panel not found at {panel_path}. Run Phase 1B first.")
    df = pd.read_parquet(panel_path)
    complete = df[df["is_complete_arb_day"]].sort_values("observation_date")
    if complete.empty:
        raise ValueError("No complete 3-basin observations found in master_market_panel.")
    latest = complete.iloc[-1]
    return (
        str(latest["observation_date"]),
        float(latest["hh_futures_usd_mmbtu"]),
        float(latest["ttf_usd_mmbtu"]),
        float(latest["jkm_usd_mmbtu"]),
    )


def compare_nwe_vs_nea_cargo(
    scenario_name: str,
    as_of_date: str,
    hh_price: float,
    ttf_price: float,
    jkm_price: float,
    freight_rate_usd_day: float,
    asia_route_id: str = "NEA_PANAMA",
    vessel_id: str = "two_stroke_174k",
    grade_id: str = "usgc_lean",
) -> DestinationComparisonResult:
    """
    Evaluates the exact same USGC FOB cargo delivered to NWE (TTF) vs NEA (JKM),
    computing netback differentials, break-evens, and the implied breakeven freight rate.
    """
    grades, vessels, terminals, cfg = load_physical_catalog()
    routes = load_routes_catalog()
    mgo_lhv = float(cfg["constants"]["mgo_lhv_mmbtu_per_mt"])

    vessel = vessels[vessel_id]
    grade = grades[grade_id]
    load_dt = date.fromisoformat(as_of_date)

    nwe_energy = compute_cargo_energy_balance(
        vessel=vessel,
        grade=grade,
        route=routes["NWE_DIRECT"],
        terminals=terminals,
        mgo_lhv_mmbtu_per_mt=mgo_lhv,
        loading_start_date=load_dt,
    )
    nea_energy = compute_cargo_energy_balance(
        vessel=vessel,
        grade=grade,
        route=routes[asia_route_id],
        terminals=terminals,
        mgo_lhv_mmbtu_per_mt=mgo_lhv,
        loading_start_date=load_dt,
    )

    nwe_econ = evaluate_cargo_economics(nwe_energy, ttf_price, hh_price, freight_rate_usd_day)
    nea_econ = evaluate_cargo_economics(nea_energy, jkm_price, hh_price, freight_rate_usd_day)

    nwe_be = calculate_cargo_breakeven_thresholds(nwe_energy, ttf_price, hh_price, freight_rate_usd_day)
    nea_be = calculate_cargo_breakeven_thresholds(nea_energy, jkm_price, hh_price, freight_rate_usd_day)

    profit_diff = nea_econ.net_profit_cargo_usd - nwe_econ.net_profit_cargo_usd
    netback_diff = nea_econ.netback_at_fob_flange_usd_mmbtu - nwe_econ.netback_at_fob_flange_usd_mmbtu

    # Implied Breakeven Freight Rate (USD/day) where Net Profit(NEA) == Net Profit(NWE):
    # Profit_NEA(F) = Profit_NEA(F0) - (F - F0)*Days_NEA
    # Profit_NWE(F) = Profit_NWE(F0) - (F - F0)*Days_NWE
    # Setting equal => F_BE = F0 + (Profit_NEA(F0) - Profit_NWE(F0)) / (Days_NEA - Days_NWE)
    delta_days = nea_energy.schedule.total_round_trip_days - nwe_energy.schedule.total_round_trip_days
    implied_be_freight = freight_rate_usd_day + (profit_diff / delta_days) if delta_days > 0 else np.nan

    if max(nwe_econ.net_profit_cargo_usd, nea_econ.net_profit_cargo_usd) < 0 and (
        max(nwe_econ.gross_revenue_usd - nwe_econ.total_avoidable_cost_usd,
            nea_econ.gross_revenue_usd - nea_econ.total_avoidable_cost_usd) < 0
    ):
        preferred = "CANCEL_LIFT (SHUT-IN)"
    elif nea_econ.net_profit_cargo_usd > nwe_econ.net_profit_cargo_usd:
        preferred = f"ASIA ({asia_route_id})"
    else:
        preferred = "EUROPE (NWE_DIRECT)"

    return DestinationComparisonResult(
        scenario_name=scenario_name,
        as_of_date=as_of_date,
        hh_price_usd_mmbtu=hh_price,
        ttf_price_usd_mmbtu=ttf_price,
        jkm_price_usd_mmbtu=jkm_price,
        freight_rate_usd_day=freight_rate_usd_day,
        asia_route_id=asia_route_id,
        nwe_econ=nwe_econ,
        nea_econ=nea_econ,
        nwe_breakeven=nwe_be,
        nea_breakeven=nea_be,
        preferred_destination=preferred,
        profit_advantage_usd=abs(profit_diff),
        netback_spread_nea_minus_nwe_usd_mmbtu=netback_diff,
        implied_breakeven_freight_usd_day=implied_be_freight,
    )


def generate_phase3_charts(
    base_comp: DestinationComparisonResult,
    scenario_df: pd.DataFrame,
    output_chart_path: Path,
) -> None:
    """
    Generates a two-panel executive chart:
    Panel 1: Full Cost Stack & Net Profit Comparison (NWE vs NEA) on Latest Market Prices ($ Million).
    Panel 2: Net Cargo Profit across Deterministic Market Scenarios ($ Million).
    """
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel 1: Cost Component Comparison (NWE vs NEA)
    categories = [
        "Feedgas (1.15x HH)",
        "Liquefaction (Sunk)",
        "Charter Hire",
        "Bunker & Canal",
        "Terminal & Regas",
        "Financing",
    ]

    def group_costs(econ: CargoEconomicResult) -> List[float]:
        c_map = {item.name: item.amount_usd / 1e6 for item in econ.cost_items}
        return [
            c_map["Feedgas Procurement (HH 1.15x)"],
            c_map["Liquefaction Tolling Fee"],
            c_map["Vessel Time Charter Hire"],
            c_map["Marine Bunker MGO Fuel"] + c_map["Canal Transit Fees"],
            c_map["Loading Terminal Fees"] + c_map["Destination Unloading Fees"] + c_map["Regasification & Grid Tariff"],
            c_map["Working Capital Financing"],
        ]

    nwe_vals = group_costs(base_comp.nwe_econ)
    nea_vals = group_costs(base_comp.nea_econ)

    x = np.arange(len(categories))
    w = 0.36
    ax1.bar(x - w / 2, nwe_vals, w, label=f"NWE Direct (${base_comp.ttf_price_usd_mmbtu:.2f}/MMBtu)", color="#1f4e78")
    ax1.bar(x + w / 2, nea_vals, w, label=f"NEA Panama (${base_comp.jkm_price_usd_mmbtu:.2f}/MMBtu)", color="#c0392b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=22, ha="right")
    ax1.set_ylabel("Cost per Cargo (USD Millions)", fontsize=10)
    ax1.set_title(f"Cargo Cost Stack Breakdown (As of {base_comp.as_of_date})", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True)

    # Panel 2: Scenario Net Profit Comparison
    x2 = np.arange(len(scenario_df))
    ax2.axhline(0.0, color="#333333", linewidth=0.9)
    ax2.bar(x2 - w / 2, scenario_df["nwe_net_profit_m_usd"], w, label="NWE Net Profit ($M)", color="#1f4e78")
    ax2.bar(x2 + w / 2, scenario_df["nea_net_profit_m_usd"], w, label="NEA Net Profit ($M)", color="#c0392b")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenario_df["scenario_name"], rotation=22, ha="right")
    ax2.set_ylabel("Full-Cycle Net Profit per Cargo (USD Millions)", fontsize=10)
    ax2.set_title("Deterministic Stress Scenarios: Cargo Net Profit ($M)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved Phase 3 economics & scenario chart to %s", output_chart_path)


def run_phase3_economics_pipeline() -> pd.DataFrame:
    """
    Runs the Phase 3 destination comparison and deterministic scenario suite using
    the latest observed market settlement prices from Phase 1B.
    """
    as_of_dt, base_hh, base_ttf, base_jkm = get_latest_complete_market_prices()
    scenarios_cfg = load_scenarios_config()["scenarios"]

    logger.info(
        "Latest Complete Market Date: %s | HH NG1: $%.2f | TTF: $%.2f | JKM: $%.2f (USD/MMBtu)",
        as_of_dt,
        base_hh,
        base_ttf,
        base_jkm,
    )

    records = []
    base_comp: Optional[DestinationComparisonResult] = None

    for scen_name, s_params in scenarios_cfg.items():
        hh = max(0.50, base_hh + float(s_params["hh_shock_usd_mmbtu"]))
        ttf = max(0.50, base_ttf + float(s_params["ttf_shock_usd_mmbtu"]))
        jkm = max(0.50, base_jkm + float(s_params["jkm_shock_usd_mmbtu"]))
        freight = float(s_params["freight_rate_override_usd_day"])
        asia_route = str(s_params["asia_route_id"])

        comp = compare_nwe_vs_nea_cargo(
            scenario_name=scen_name,
            as_of_date=as_of_dt,
            hh_price=hh,
            ttf_price=ttf,
            jkm_price=jkm,
            freight_rate_usd_day=freight,
            asia_route_id=asia_route,
        )
        if scen_name == "BASE_LATEST_MARKET":
            base_comp = comp

        records.append({
            "scenario_name": scen_name,
            "as_of_date": as_of_dt,
            "hh_usd_mmbtu": round(hh, 2),
            "ttf_usd_mmbtu": round(ttf, 2),
            "jkm_usd_mmbtu": round(jkm, 2),
            "freight_usd_day": round(freight, 0),
            "asia_route": asia_route,
            "nwe_netback_fob_usd_mmbtu": round(comp.nwe_econ.netback_at_fob_flange_usd_mmbtu, 2),
            "nea_netback_fob_usd_mmbtu": round(comp.nea_econ.netback_at_fob_flange_usd_mmbtu, 2),
            "netback_diff_nea_minus_nwe": round(comp.netback_spread_nea_minus_nwe_usd_mmbtu, 2),
            "nwe_net_profit_m_usd": round(comp.nwe_econ.net_profit_cargo_usd / 1e6, 2),
            "nea_net_profit_m_usd": round(comp.nea_econ.net_profit_cargo_usd / 1e6, 2),
            "nwe_breakeven_ttf": round(comp.nwe_breakeven.breakeven_destination_price_usd_mmbtu, 2),
            "nea_breakeven_jkm": round(comp.nea_breakeven.breakeven_destination_price_usd_mmbtu, 2),
            "implied_breakeven_freight_usd_day": round(comp.implied_breakeven_freight_usd_day, 0),
            "preferred_destination": comp.preferred_destination,
        })

        logger.info(
            "Scenario %-26s | NWE Netback: $%5.2f | NEA Netback: $%5.2f | NWE Profit: $%6.2fM | NEA Profit: $%6.2fM | BE Freight: $%7.0f/d -> %s",
            scen_name,
            comp.nwe_econ.netback_at_fob_flange_usd_mmbtu,
            comp.nea_econ.netback_at_fob_flange_usd_mmbtu,
            comp.nwe_econ.net_profit_cargo_usd / 1e6,
            comp.nea_econ.net_profit_cargo_usd / 1e6,
            comp.implied_breakeven_freight_usd_day,
            comp.preferred_destination,
        )

    summary_df = pd.DataFrame(records)
    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_tables / "phase3_cargo_economics_comparison.csv", index=False)

    if base_comp is not None:
        generate_phase3_charts(
            base_comp,
            summary_df,
            Path("outputs/charts/phase3_cargo_economics_waterfall.png"),
        )

    return summary_df


if __name__ == "__main__":
    run_phase3_economics_pipeline()