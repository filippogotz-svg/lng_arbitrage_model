"""
Phase 3 Module 3: Analytical Break-Even Solver.
Computes break-even destination prices (TTF/JKM), break-even Henry Hub,
maximum viable freight rate, and maximum viable liquefaction fee.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Dict, Optional

from src.physical.energy_balance import CargoEnergyBalanceResult
from src.economics.cargo_pricer import evaluate_cargo_economics, load_commercial_config


@dataclass(frozen=True)
class BreakEvenAnalysisResult:
    """Break-even threshold prices and rates for commercial viability."""
    destination_region: str
    route_id: str
    breakeven_destination_price_usd_mmbtu: float
    breakeven_henry_hub_usd_mmbtu: float
    maximum_viable_freight_rate_usd_per_day: float
    maximum_viable_tolling_fee_usd_mmbtu: float


def calculate_cargo_breakeven_thresholds(
    energy_res: CargoEnergyBalanceResult,
    baseline_destination_price_usd_mmbtu: float,
    baseline_henry_hub_usd_mmbtu: float,
    freight_daily_rate_usd: Optional[float] = None,
    custom_commercial_config_path: Path = Path("config/commercial_terms.toml"),
) -> BreakEvenAnalysisResult:
    """
    Computes analytical break-even thresholds where net cargo profit equals zero (Net Profit = 0).
    """
    cfg = load_commercial_config(custom_commercial_config_path)["commercial_terms"]
    freight_rate = freight_daily_rate_usd if freight_daily_rate_usd is not None else float(cfg["spot_time_charter_rate_usd_per_day"])

    # Base evaluation
    base_econ = evaluate_cargo_economics(
        energy_res=energy_res,
        destination_price_usd_mmbtu=baseline_destination_price_usd_mmbtu,
        henry_hub_price_usd_mmbtu=baseline_henry_hub_usd_mmbtu,
        freight_daily_rate_usd=freight_rate,
        custom_commercial_config_path=custom_commercial_config_path,
    )

    e_del = energy_res.delivered_energy_des_mmbtu
    e_load = energy_res.gross_loaded_energy_fob_mmbtu

    # 1. Break-even Destination Price: Revenue needed to cover all full-cycle costs
    # Net Profit = (P_dest * E_del) - Total_Cost = 0 => P_dest = Total_Cost / E_del
    cost_without_revenue_scale = base_econ.total_full_cycle_cost_usd
    be_dest_price = cost_without_revenue_scale / e_del if e_del > 0 else 0.0

    # 2. Break-even Henry Hub: Maximum Henry Hub price before profit hits zero
    # Revenue - [ (1.15 * HH * E_load) + Other_Costs ] = 0
    other_costs = base_econ.total_full_cycle_cost_usd - (1.15 * baseline_henry_hub_usd_mmbtu * e_load)
    be_hh = (base_econ.gross_revenue_usd - other_costs) / (1.15 * e_load) if (1.15 * e_load) > 0 else 0.0

    # 3. Maximum Viable Freight Rate (USD/day): Daily charter rate where net profit hits zero
    # Profit = Base_Profit - (Freight_Rate - Base_Freight) * Voyage_Days = 0
    voyage_days = energy_res.schedule.total_round_trip_days
    max_freight = freight_rate + (base_econ.net_profit_cargo_usd / voyage_days) if voyage_days > 0 else 0.0

    # 4. Maximum Viable Tolling Fee: Tolling fee where net profit hits zero
    current_tolling = float(cfg["fixed_liquefaction_tolling_fee_usd_mmbtu"])
    max_tolling = current_tolling + (base_econ.net_profit_cargo_usd / e_load) if e_load > 0 else 0.0

    return BreakEvenAnalysisResult(
        destination_region=energy_res.destination_region,
        route_id=energy_res.route_id,
        breakeven_destination_price_usd_mmbtu=be_dest_price,
        breakeven_henry_hub_usd_mmbtu=be_hh,
        maximum_viable_freight_rate_usd_per_day=max_freight,
        maximum_viable_tolling_fee_usd_mmbtu=max_tolling,
    )
