"""
Phase 4 Module: Dynamic Speed Optimization & Slow-Steaming Engine.
Solves for the optimal laden and ballast steaming speeds (in knots) that
maximize daily net profit or minimize delivered unit costs, accounting for the
cubic power law and round-trip charter hire trade-offs.
"""

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import tomllib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.physical.cargo_specs import load_physical_catalog, VesselProfileSpec, LNGGradeSpec, TerminalPhysicalSpec
from src.physical.routing import load_routes_catalog, RouteSpec
from src.physical.energy_balance import compute_cargo_energy_balance, CargoEnergyBalanceResult
from src.economics.cargo_pricer import evaluate_cargo_economics, CargoEconomicResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("SpeedOptimizer")


@dataclass(frozen=True)
class SpeedOptimizationResult:
    """Optimal speed profile and comparative economics for an LNG cargo voyage."""
    vessel_id: str
    route_id: str
    destination_region: str
    optimal_laden_speed_knots: float
    optimal_ballast_speed_knots: float
    baseline_laden_speed_knots: float
    baseline_ballast_speed_knots: float
    
    optimized_energy_res: CargoEnergyBalanceResult
    baseline_energy_res: CargoEnergyBalanceResult
    optimized_econ: CargoEconomicResult
    baseline_econ: CargoEconomicResult

    daily_profit_optimized_usd: float
    daily_profit_baseline_usd: float
    profit_gain_usd: float
    bunker_savings_mt: float
    charter_cost_delta_usd: float


def load_optimization_config(config_path: Path = Path("config/optimization_params.toml")) -> dict:
    """Loads optimization TOML configuration safely."""
    if not config_path.exists():
        raise FileNotFoundError(f"Optimization configuration file not found at {config_path}")
    return tomllib.loads(config_path.read_text(encoding="utf-8-sig"))


def optimize_voyage_speed(
    vessel: VesselProfileSpec,
    grade: LNGGradeSpec,
    route: RouteSpec,
    terminals: TerminalPhysicalSpec,
    destination_price_usd_mmbtu: float,
    henry_hub_price_usd_mmbtu: float,
    freight_daily_rate_usd: float,
    mgo_lhv_mmbtu_per_mt: float = 40.19,
    loading_start_date: Optional[date] = None,
    custom_opt_config_path: Path = Path("config/optimization_params.toml"),
) -> SpeedOptimizationResult:
    """
    Performs a grid search across laden and ballast speeds [min_speed, max_speed]
    to find the combination that maximizes daily net profit (Net Profit / Total Round-Trip Days).
    """
    opt_cfg = load_optimization_config(custom_opt_config_path)
    min_v = float(opt_cfg["optimization_bounds"]["min_speed_knots"])
    max_v = float(opt_cfg["optimization_bounds"]["max_speed_knots"])
    step_v = float(opt_cfg["optimization_bounds"]["speed_step_knots"])

    speeds = np.arange(min_v, max_v + step_v / 2, step_v)

    baseline_laden = vessel.design_speed_laden_knots
    baseline_ballast = vessel.design_speed_ballast_knots

    baseline_energy = compute_cargo_energy_balance(
        vessel=vessel,
        grade=grade,
        route=route,
        terminals=terminals,
        mgo_lhv_mmbtu_per_mt=mgo_lhv_mmbtu_per_mt,
        fuel_deficit_mode="BURN_FUEL_OIL",
        loading_start_date=loading_start_date,
        speed_laden_override_knots=baseline_laden,
        speed_ballast_override_knots=baseline_ballast,
    )
    baseline_econ = evaluate_cargo_economics(
        energy_res=baseline_energy,
        destination_price_usd_mmbtu=destination_price_usd_mmbtu,
        henry_hub_price_usd_mmbtu=henry_hub_price_usd_mmbtu,
        freight_daily_rate_usd=freight_daily_rate_usd,
    )
    baseline_daily_profit = baseline_econ.net_profit_cargo_usd / baseline_energy.schedule.total_round_trip_days

    best_laden = baseline_laden
    best_ballast = baseline_ballast
    best_daily_profit = baseline_daily_profit
    best_energy = baseline_energy
    best_econ = baseline_econ

    # Grid search over laden and ballast speeds
    for v_l in speeds:
        for v_b in speeds:
            try:
                en_res = compute_cargo_energy_balance(
                    vessel=vessel,
                    grade=grade,
                    route=route,
                    terminals=terminals,
                    mgo_lhv_mmbtu_per_mt=mgo_lhv_mmbtu_per_mt,
                    fuel_deficit_mode="BURN_FUEL_OIL",
                    loading_start_date=loading_start_date,
                    speed_laden_override_knots=float(v_l),
                    speed_ballast_override_knots=float(v_b),
                )
                ec_res = evaluate_cargo_economics(
                    energy_res=en_res,
                    destination_price_usd_mmbtu=destination_price_usd_mmbtu,
                    henry_hub_price_usd_mmbtu=henry_hub_price_usd_mmbtu,
                    freight_daily_rate_usd=freight_daily_rate_usd,
                )
                rt_days = en_res.schedule.total_round_trip_days
                if rt_days <= 0:
                    continue
                daily_prof = ec_res.net_profit_cargo_usd / rt_days

                if daily_prof > best_daily_profit:
                    best_daily_profit = daily_prof
                    best_laden = float(v_l)
                    best_ballast = float(v_b)
                    best_energy = en_res
                    best_econ = ec_res
            except Exception:
                continue

    profit_gain = best_econ.net_profit_cargo_usd - baseline_econ.net_profit_cargo_usd
    bunker_savings = baseline_energy.total_round_trip_bunker_mgo_mt - best_energy.total_round_trip_bunker_mgo_mt
    charter_delta = best_econ.cost_items[3].amount_usd - baseline_econ.cost_items[3].amount_usd

    return SpeedOptimizationResult(
        vessel_id=vessel.vessel_id,
        route_id=route.route_id,
        destination_region=route.destination_region,
        optimal_laden_speed_knots=best_laden,
        optimal_ballast_speed_knots=best_ballast,
        baseline_laden_speed_knots=baseline_laden,
        baseline_ballast_speed_knots=baseline_ballast,
        optimized_energy_res=best_energy,
        baseline_energy_res=baseline_energy,
        optimized_econ=best_econ,
        baseline_econ=baseline_econ,
        daily_profit_optimized_usd=best_daily_profit,
        daily_profit_baseline_usd=baseline_daily_profit,
        profit_gain_usd=profit_gain,
        bunker_savings_mt=bunker_savings,
        charter_cost_delta_usd=charter_delta,
    )