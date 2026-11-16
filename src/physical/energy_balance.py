"""
Phase 2 Module 3: 11-Stage LNG Physical Value Chain & Thermodynamic Energy Balance.
Enforces strict energy conservation and prevents double-counting between
natural/forced Boil-Off Gas (BOG) and liquid marine bunker fuel (MGO/VLSFO).
"""

from dataclasses import dataclass, asdict
from datetime import date
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physical.cargo_specs import (
    LNGGradeSpec,
    TerminalPhysicalSpec,
    VesselProfileSpec,
    load_physical_catalog,
)
from src.physical.routing import (
    RouteSpec,
    VoyageSchedule,
    calculate_voyage_schedule,
    load_routes_catalog,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("EnergyBalanceEngine")


@dataclass(frozen=True)
class LegFuelBalance:
    """Detailed propulsion energy, boil-off, reliquefaction, and bunker MGO balance for a voyage leg."""
    leg_name: str
    sea_days: float
    initial_energy_mmbtu: float
    natural_bog_mmbtu: float
    total_propulsion_demand_mmbtu: float
    mandatory_pilot_fuel_mmbtu: float
    gas_eligible_propulsion_demand_mmbtu: float
    natural_bog_burned_in_engine_mmbtu: float
    excess_natural_bog_mmbtu: float
    reliquefied_bog_mmbtu: float
    gcu_flared_bog_mmbtu: float
    energy_deficit_after_natural_bog_mmbtu: float
    forced_bog_mmbtu: float
    supplementary_fuel_oil_mmbtu: float
    reliquefaction_aux_mgo_mt: float
    total_bunker_mgo_burned_mt: float
    net_cargo_energy_lost_mmbtu: float


@dataclass(frozen=True)
class CargoEnergyBalanceResult:
    """Complete 11-stage physical volume and energy balance for an LNG cargo."""
    vessel_id: str
    vessel_name: str
    grade_id: str
    route_id: str
    destination_region: str
    fuel_deficit_mode: str

    # Volumetric metrics (m3)
    vessel_capacity_m3: float
    arrival_heel_m3: float
    net_loaded_volume_m3: float
    departure_volume_m3: float
    volumetric_hhv_mmbtu_m3: float
    delivered_volume_des_m3: float

    # Stage 1-5: Upstream & Liquefaction Energy (MMBtu)
    physical_feedgas_required_mmbtu: float
    physical_liquefaction_loss_mmbtu: float
    contractual_feedgas_billed_mmbtu: float
    arrival_heel_in_mmbtu: float
    gross_loaded_energy_fob_mmbtu: float
    total_departure_energy_mmbtu: float

    # Stage 6-7: Marine Transport & Boil-Off (Laden & Ballast)
    laden_leg: LegFuelBalance
    ballast_leg: LegFuelBalance
    energy_onboard_Upon_arrival_mmbtu: float

    # Stage 8-11: Unloading, Retained Ballast Heel, Regasification & Grid Injection
    retained_departure_heel_out_mmbtu: float
    net_heel_change_mmbtu: float
    delivered_energy_des_mmbtu: float
    delivered_fraction_of_loaded_pct: float
    regasification_loss_mmbtu: float
    net_grid_injected_energy_mmbtu: float

    # Aggregate Voyage Bunker Summary & Schedule
    total_round_trip_bunker_mgo_mt: float
    laden_bunker_mgo_mt: float
    ballast_bunker_mgo_mt: float
    schedule: VoyageSchedule

    # Conservation Verification
    energy_conservation_residual_mmbtu: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def solve_voyage_leg_fuel_balance(
    leg_name: str,
    initial_energy_mmbtu: float,
    sea_days: float,
    daily_bor_fraction: float,
    daily_propulsion_mmbtu: float,
    pilot_fuel_fraction: float,
    reliquefaction_fraction: float,
    reliq_power_mgo_mt_per_mmbtu: float,
    mgo_lhv_mmbtu_per_mt: float,
    fuel_deficit_mode: str = "BURN_FUEL_OIL",
) -> LegFuelBalance:
    """
    Computes exact exponential boil-off decay and reconciles natural BOG against
    engine propulsion demand without double-counting BOG and liquid fuel oil.

    Equation for natural BOG over t days:
        E_BOG = E_initial * (1 - (1 - BOR)^t)
    """
    if sea_days < 0:
        raise ValueError("Voyage sea_days cannot be negative.")

    natural_bog = initial_energy_mmbtu * (1.0 - ((1.0 - daily_bor_fraction) ** sea_days))
    total_prop_demand = daily_propulsion_mmbtu * sea_days

    # Mandatory liquid pilot fuel (e.g., 2% on two-stroke ME-GI)
    pilot_fuel_mmbtu = total_prop_demand * pilot_fuel_fraction
    gas_eligible_demand = total_prop_demand - pilot_fuel_mmbtu

    if natural_bog >= gas_eligible_demand:
        # Case B: Natural BOG exceeds gas propulsion requirement
        bog_burned = gas_eligible_demand
        excess_bog = natural_bog - gas_eligible_demand
        reliquefied = excess_bog * reliquefaction_fraction
        gcu_flared = excess_bog - reliquefied
        deficit = 0.0
        forced_bog = 0.0
        supp_fuel_oil_mmbtu = 0.0
    else:
        # Case A: Propulsion demand exceeds natural BOG
        bog_burned = natural_bog
        excess_bog = 0.0
        reliquefied = 0.0
        gcu_flared = 0.0
        deficit = gas_eligible_demand - natural_bog

        if fuel_deficit_mode.upper() == "FORCE_BOG":
            forced_bog = deficit
            supp_fuel_oil_mmbtu = 0.0
        elif fuel_deficit_mode.upper() == "BURN_FUEL_OIL":
            forced_bog = 0.0
            supp_fuel_oil_mmbtu = deficit
        else:
            raise ValueError(f"Unsupported fuel_deficit_mode: {fuel_deficit_mode}")

    # Net cargo thermal energy lost from tanks during this leg:
    # Natural BOG + Forced BOG - Reliquefied BOG
    net_cargo_lost = natural_bog + forced_bog - reliquefied

    # Liquid MGO consumed (metric tons) = Pilot + Supplementary Fuel Oil + Reliq Compressor Power
    propulsion_mgo_mt = (pilot_fuel_mmbtu + supp_fuel_oil_mmbtu) / mgo_lhv_mmbtu_per_mt
    reliq_aux_mgo_mt = reliquefied * reliq_power_mgo_mt_per_mmbtu
    total_mgo_mt = propulsion_mgo_mt + reliq_aux_mgo_mt

    return LegFuelBalance(
        leg_name=leg_name,
        sea_days=sea_days,
        initial_energy_mmbtu=initial_energy_mmbtu,
        natural_bog_mmbtu=natural_bog,
        total_propulsion_demand_mmbtu=total_prop_demand,
        mandatory_pilot_fuel_mmbtu=pilot_fuel_mmbtu,
        gas_eligible_propulsion_demand_mmbtu=gas_eligible_demand,
        natural_bog_burned_in_engine_mmbtu=bog_burned,
        excess_natural_bog_mmbtu=excess_bog,
        reliquefied_bog_mmbtu=reliquefied,
        gcu_flared_bog_mmbtu=gcu_flared,
        energy_deficit_after_natural_bog_mmbtu=deficit,
        forced_bog_mmbtu=forced_bog,
        supplementary_fuel_oil_mmbtu=supp_fuel_oil_mmbtu,
        reliquefaction_aux_mgo_mt=reliq_aux_mgo_mt,
        total_bunker_mgo_burned_mt=total_mgo_mt,
        net_cargo_energy_lost_mmbtu=net_cargo_lost,
    )


def compute_cargo_energy_balance(
    vessel: VesselProfileSpec,
    grade: LNGGradeSpec,
    route: RouteSpec,
    terminals: TerminalPhysicalSpec,
    mgo_lhv_mmbtu_per_mt: float = 40.19,
    fuel_deficit_mode: str = "BURN_FUEL_OIL",
    loading_start_date: Optional[date] = None,
    speed_laden_override_knots: Optional[float] = None,
    speed_ballast_override_knots: Optional[float] = None,
) -> CargoEnergyBalanceResult:
    """
    Computes the complete 11-stage physical LNG value chain energy balance.

    Enforces closed-cycle heel management:
        The vessel retains sufficient departure heel after unloading (E_heel_out)
        to supply natural ballast boil-off during the return voyage and arrive back
        at the loading terminal with exactly E_heel_in:
            E_heel_out * (1 - BOR_ballast)^t_ballast = E_heel_in
            => E_heel_out = E_heel_in / (1 - BOR_ballast)^t_ballast
    """
    schedule = calculate_voyage_schedule(
        route=route,
        vessel=vessel,
        terminals=terminals,
        loading_start_date=loading_start_date,
        speed_laden_override_knots=speed_laden_override_knots,
        speed_ballast_override_knots=speed_ballast_override_knots,
    )

    k_hhv = grade.volumetric_hhv_mmbtu_m3

    # Stages 1-5: Loading & Upstream Feedgas
    v_heel_in = vessel.arrival_heel_volume_m3
    v_loaded = vessel.net_loaded_volume_m3
    v_dep = vessel.departure_volume_m3

    e_heel_in = v_heel_in * k_hhv
    e_loaded_fob = v_loaded * k_hhv
    e_dep = v_dep * k_hhv

    e_feed_phys = e_loaded_fob * (1.0 + terminals.physical_liquefaction_loss_fraction)
    e_liq_loss = e_feed_phys - e_loaded_fob
    e_feed_contract = e_loaded_fob * terminals.contractual_feedgas_multiplier

    # Stage 6-7 (Laden Leg): Boil-off and propulsion fuel balance
    # Note: Daily BOR in LNG carriers is quoted as % of total liquid cargo onboard at departure
    laden_leg = solve_voyage_leg_fuel_balance(
        leg_name="LADEN",
        initial_energy_mmbtu=e_dep,
        sea_days=schedule.total_laden_sea_days,
        daily_bor_fraction=vessel.natural_bor_laden_pct_per_day,
        daily_propulsion_mmbtu=vessel.propulsion_energy_laden_mmbtu_per_day,
        pilot_fuel_fraction=vessel.pilot_fuel_fraction,
        reliquefaction_fraction=vessel.reliquefaction_fraction,
        reliq_power_mgo_mt_per_mmbtu=vessel.reliq_power_mgo_mt_per_mmbtu,
        mgo_lhv_mmbtu_per_mt=mgo_lhv_mmbtu_per_mt,
        fuel_deficit_mode=fuel_deficit_mode,
    )

    e_arrival = e_dep - laden_leg.net_cargo_energy_lost_mmbtu

    # Ballast Leg Heel Requirement:
    # On the ballast leg, daily BOR % applies to nominal loaded capacity (or retained heel).
    # In industry practice, ballast BOR (e.g., 0.06%/day of nominal departure energy) is
    # evaporated from the retained departure heel.
    ballast_natural_bog = e_dep * (
        1.0 - ((1.0 - vessel.natural_bor_ballast_pct_per_day) ** schedule.total_ballast_sea_days)
    )
    # Solve ballast fuel balance where natural BOG equals ballast_natural_bog
    # Effective BOR relative to e_dep is used; on ballast, deficit is always met with fuel oil
    # so the vessel preserves its minimum cryogenic arrival heel (e_heel_in).
    ballast_leg = solve_voyage_leg_fuel_balance(
        leg_name="BALLAST",
        initial_energy_mmbtu=e_dep,
        sea_days=schedule.total_ballast_sea_days,
        daily_bor_fraction=vessel.natural_bor_ballast_pct_per_day,
        daily_propulsion_mmbtu=vessel.propulsion_energy_ballast_mmbtu_per_day,
        pilot_fuel_fraction=vessel.pilot_fuel_fraction,
        reliquefaction_fraction=vessel.reliquefaction_fraction,
        reliq_power_mgo_mt_per_mmbtu=vessel.reliq_power_mgo_mt_per_mmbtu,
        mgo_lhv_mmbtu_per_mt=mgo_lhv_mmbtu_per_mt,
        fuel_deficit_mode="BURN_FUEL_OIL",
    )

    # Retained heel at unloading port to cover net ballast BOG loss + arrival heel for next cargo
    e_heel_out = e_heel_in + ballast_leg.net_cargo_energy_lost_mmbtu
    net_heel_change = e_heel_out - e_heel_in

    # Stage 8: Unloading at Destination Flange (DES Delivered Energy)
    e_delivered_des = e_arrival - e_heel_out
    v_delivered_des = e_delivered_des / k_hhv
    delivered_pct = (e_delivered_des / e_loaded_fob) * 100.0

    # Stage 9-11: Regasification & Downstream Grid Injection
    regas_loss_frac = (
        terminals.nwe_regasification_loss_fraction
        if route.destination_region == "NWE"
        else terminals.nea_regasification_loss_fraction
    )
    e_regas_loss = e_delivered_des * regas_loss_frac
    e_grid_injected = e_delivered_des - e_regas_loss

    # Conservation Verification:
    # Gross Loaded FOB Energy == Delivered DES Energy + Laden Cargo Loss + Ballast Cargo Loss (Net Heel Change)
    conservation_residual = e_loaded_fob - (
        e_delivered_des + laden_leg.net_cargo_energy_lost_mmbtu + net_heel_change
    )

    if abs(conservation_residual) > 1e-6:
        raise RuntimeError(
            f"Thermodynamic energy conservation violated! Residual: {conservation_residual} MMBtu"
        )

    return CargoEnergyBalanceResult(
        vessel_id=vessel.vessel_id,
        vessel_name=vessel.name,
        grade_id=grade.grade_id,
        route_id=route.route_id,
        destination_region=route.destination_region,
        fuel_deficit_mode=fuel_deficit_mode,
        vessel_capacity_m3=vessel.nominal_capacity_m3,
        arrival_heel_m3=v_heel_in,
        net_loaded_volume_m3=v_loaded,
        departure_volume_m3=v_dep,
        volumetric_hhv_mmbtu_m3=k_hhv,
        delivered_volume_des_m3=v_delivered_des,
        physical_feedgas_required_mmbtu=e_feed_phys,
        physical_liquefaction_loss_mmbtu=e_liq_loss,
        contractual_feedgas_billed_mmbtu=e_feed_contract,
        arrival_heel_in_mmbtu=e_heel_in,
        gross_loaded_energy_fob_mmbtu=e_loaded_fob,
        total_departure_energy_mmbtu=e_dep,
        laden_leg=laden_leg,
        ballast_leg=ballast_leg,
        energy_onboard_Upon_arrival_mmbtu=e_arrival,
        retained_departure_heel_out_mmbtu=e_heel_out,
        net_heel_change_mmbtu=net_heel_change,
        delivered_energy_des_mmbtu=e_delivered_des,
        delivered_fraction_of_loaded_pct=delivered_pct,
        regasification_loss_mmbtu=e_regas_loss,
        net_grid_injected_energy_mmbtu=e_grid_injected,
        total_round_trip_bunker_mgo_mt=laden_leg.total_bunker_mgo_burned_mt + ballast_leg.total_bunker_mgo_burned_mt,
        laden_bunker_mgo_mt=laden_leg.total_bunker_mgo_burned_mt,
        ballast_bunker_mgo_mt=ballast_leg.total_bunker_mgo_burned_mt,
        schedule=schedule,
        energy_conservation_residual_mmbtu=conservation_residual,
    )


def generate_phase2_diagnostic_chart(
    summary_df: pd.DataFrame,
    output_chart_path: Path,
) -> None:
    """
    Generates a two-panel institutional physical diagnostic chart:
    Panel 1: Delivered DES Energy vs Boil-Off Losses across Routes (174k Two-Stroke Baseline).
    Panel 2: Fleet Vintage Comparison (Two-Stroke 174k vs TFDE 160k vs Steam 145k) Delivered % of Loaded.
    """
    output_chart_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: 174k Two-Stroke across the 4 routes
    base_df = summary_df[
        (summary_df["vessel_id"] == "two_stroke_174k")
        & (summary_df["fuel_deficit_mode"] == "BURN_FUEL_OIL")
    ].copy()

    x = np.arange(len(base_df))
    width = 0.55

    ax1.bar(
        x,
        base_df["delivered_energy_des_mmbtu"] / 1e6,
        width,
        label="Delivered DES Energy (Million MMBtu)",
        color="#1f4e78",
    )
    ax1.bar(
        x,
        (base_df["laden_bog_loss_mmbtu"] + base_df["ballast_bog_heel_mmbtu"]) / 1e6,
        width,
        bottom=base_df["delivered_energy_des_mmbtu"] / 1e6,
        label="Total Round-Trip BOG Loss (Million MMBtu)",
        color="#d9534f",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(base_df["route_id"], rotation=15)
    ax1.set_ylabel("Energy (Million MMBtu)", fontsize=10)
    ax1.set_title("174k Two-Stroke: Loaded vs. Delivered Energy by Route", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 4.3)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", frameon=True)

    # Panel 2: Fleet Vintage Comparison (% of Loaded Energy Delivered on NWE vs NEA Panama vs NEA COGH)
    routes_to_plot = ["NWE_DIRECT", "NEA_PANAMA", "NEA_COGH"]
    v_ids = ["two_stroke_174k", "tfde_160k", "steam_145k"]
    v_labels = ["174k Two-Stroke (0.085% BOR)", "160k TFDE (0.105% BOR)", "145k Steam (0.150% BOR)"]
    colors = ["#1f4e78", "#2e8b57", "#c0392b"]

    x2 = np.arange(len(routes_to_plot))
    bar_w = 0.25

    for idx, (v_id, v_label, col) in enumerate(zip(v_ids, v_labels, colors)):
        sub = summary_df[
            (summary_df["vessel_id"] == v_id)
            & (summary_df["fuel_deficit_mode"] == "BURN_FUEL_OIL")
            & (summary_df["route_id"].isin(routes_to_plot))
        ].set_index("route_id").reindex(routes_to_plot)

        ax2.bar(
            x2 + (idx - 1) * bar_w,
            sub["delivered_fraction_of_loaded_pct"],
            bar_w,
            label=v_label,
            color=col,
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(routes_to_plot, rotation=15)
    ax2.set_ylabel("Delivered DES Energy (% of FOB Loaded)", fontsize=10)
    ax2.set_title("Fleet Vintage Efficiency: % of Loaded Cargo Delivered", fontsize=11, fontweight="bold")
    ax2.set_ylim(88.0, 100.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved Phase 2 physical diagnostic chart to %s", output_chart_path)


def run_phase2_physical_pipeline(
    config_path: Path = Path("config/physical_assumptions.toml"),
) -> pd.DataFrame:
    """
    Runs the Phase 2 physical engine across all vessel profiles, routes, and fuel modes.
    Exports summary tables to outputs/tables/ and diagnostic charts to outputs/charts/.
    """
    grades, vessels, terminals, cfg = load_physical_catalog(config_path)
    routes = load_routes_catalog(config_path)

    default_grade = grades[cfg["project"]["default_lng_grade"]]
    mgo_lhv = float(cfg["constants"]["mgo_lhv_mmbtu_per_mt"])

    records = []
    for v_id, vessel in vessels.items():
        for r_id, route in routes.items():
            for mode in ["BURN_FUEL_OIL", "FORCE_BOG"]:
                res = compute_cargo_energy_balance(
                    vessel=vessel,
                    grade=default_grade,
                    route=route,
                    terminals=terminals,
                    mgo_lhv_mmbtu_per_mt=mgo_lhv,
                    fuel_deficit_mode=mode,
                    loading_start_date=date(2026, 11, 16),
                )
                records.append({
                    "vessel_id": v_id,
                    "vessel_name": vessel.name,
                    "route_id": r_id,
                    "destination_region": route.destination_region,
                    "fuel_deficit_mode": mode,
                    "laden_sea_days": round(res.schedule.total_laden_sea_days, 2),
                    "ballast_sea_days": round(res.schedule.total_ballast_sea_days, 2),
                    "round_trip_days": round(res.schedule.total_round_trip_days, 2),
                    "net_loaded_m3": round(res.net_loaded_volume_m3, 1),
                    "gross_loaded_fob_mmbtu": round(res.gross_loaded_energy_fob_mmbtu, 1),
                    "contractual_feedgas_mmbtu": round(res.contractual_feedgas_billed_mmbtu, 1),
                    "laden_bog_loss_mmbtu": round(res.laden_leg.net_cargo_energy_lost_mmbtu, 1),
                    "ballast_bog_heel_mmbtu": round(res.net_heel_change_mmbtu, 1),
                    "delivered_energy_des_mmbtu": round(res.delivered_energy_des_mmbtu, 1),
                    "delivered_fraction_of_loaded_pct": round(res.delivered_fraction_of_loaded_pct, 2),
                    "net_grid_injected_mmbtu": round(res.net_grid_injected_energy_mmbtu, 1),
                    "laden_bunker_mgo_mt": round(res.laden_bunker_mgo_mt, 1),
                    "ballast_bunker_mgo_mt": round(res.ballast_bunker_mgo_mt, 1),
                    "total_round_trip_bunker_mgo_mt": round(res.total_round_trip_bunker_mgo_mt, 1),
                    "conservation_residual_mmbtu": res.energy_conservation_residual_mmbtu,
                })

    summary_df = pd.DataFrame(records)

    out_tables_dir = Path("outputs/tables")
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_tables_dir / "phase2_physical_cargo_matrix.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("Exported Phase 2 physical cargo matrix (%d configurations) to %s", len(summary_df), csv_path)

    # Log key baseline comparisons for 174k Two-Stroke
    base_sub = summary_df[
        (summary_df["vessel_id"] == "two_stroke_174k")
        & (summary_df["fuel_deficit_mode"] == "BURN_FUEL_OIL")
    ]
    for _, row in base_sub.iterrows():
        logger.info(
            "Baseline 174k Two-Stroke | Route: %-10s | RT Days: %5.2f | Loaded: %10.0f MMBtu | "
            "Delivered DES: %10.0f MMBtu (%5.2f%%) | RT Bunker MGO: %6.1f mt",
            row["route_id"],
            row["round_trip_days"],
            row["gross_loaded_fob_mmbtu"],
            row["delivered_energy_des_mmbtu"],
            row["delivered_fraction_of_loaded_pct"],
            row["total_round_trip_bunker_mgo_mt"],
        )

    chart_path = Path("outputs/charts/phase2_physical_energy_balance.png")
    generate_phase2_diagnostic_chart(summary_df, chart_path)

    return summary_df


if __name__ == "__main__":
    run_phase2_physical_pipeline()
