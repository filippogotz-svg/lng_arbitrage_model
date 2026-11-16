"""
Phase 3 Module 2: Cargo Pricer, Revenue, Gross Margin & Netback Engine.
Calculates full-cycle P&L, marginal routing economics, netback per MMBtu,
and profit per cargo for Northwest Europe (TTF) and Northeast Asia (JKM).
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Dict, List, Optional

from src.physical.energy_balance import CargoEnergyBalanceResult
from src.economics.cost_classification import CostCategory, CostItem


@dataclass(frozen=True)
class CargoEconomicResult:
    """Complete economic valuation and profitability metrics for a single LNG cargo."""
    destination_region: str
    benchmark_name: str
    benchmark_price_usd_mmbtu: float
    delivered_energy_mmbtu: float
    gross_revenue_usd: float

    # Cost stack breakdown
    cost_items: List[CostItem]
    total_contract_cost_usd: float
    total_full_cycle_cost_usd: float
    total_marginal_routing_cost_usd: float
    total_avoidable_cost_usd: float

    # Profitability metrics
    gross_margin_usd: float
    net_profit_cargo_usd: float
    profit_per_delivered_mmbtu_usd: float
    profit_per_loaded_mmbtu_usd: float
    netback_at_fob_flange_usd_mmbtu: float
    netback_at_wellhead_usd_mmbtu: float


def load_commercial_config(config_path: Path = Path("config/commercial_terms.toml")) -> dict:
    """Loads commercial terms configuration safely (strips UTF-8 BOM if present)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Commercial configuration file not found at {config_path}")
    raw_text = config_path.read_text(encoding="utf-8-sig")
    return tomllib.loads(raw_text)


def evaluate_cargo_economics(
    energy_res: CargoEnergyBalanceResult,
    destination_price_usd_mmbtu: float,
    henry_hub_price_usd_mmbtu: float,
    freight_daily_rate_usd: Optional[float] = None,
    custom_commercial_config_path: Path = Path("config/commercial_terms.toml"),
) -> CargoEconomicResult:
    """
    Evaluates the complete cost stack, revenue, and netback for an LNG cargo
    delivered to either Northwest Europe (TTF) or Northeast Asia (JKM).
    """
    cfg = load_commercial_config(custom_commercial_config_path)["commercial_terms"]
    
    freight_rate = freight_daily_rate_usd if freight_daily_rate_usd is not None else float(cfg["spot_time_charter_rate_usd_per_day"])
    mgo_price = float(cfg["bunker_mgo_price_usd_per_mt"])
    cost_of_capital = float(cfg["annual_cost_of_capital_pct"])
    payment_days = float(cfg["payment_term_days"])

    e_loaded = energy_res.gross_loaded_energy_fob_mmbtu
    e_delivered = energy_res.delivered_energy_des_mmbtu
    sched = energy_res.schedule

    # 1. Revenue
    gross_revenue = e_delivered * destination_price_usd_mmbtu

    # 2. Cost Stack Components
    # A. Feedgas Cost (Variable Upstream)
    feedgas_qty = e_loaded * float(cfg["spa_feedgas_multiplier"])
    feedgas_cost = feedgas_qty * henry_hub_price_usd_mmbtu

    # B. Liquefaction Tolling Fee (Sunk Committed)
    tolling_fee = e_loaded * float(cfg["fixed_liquefaction_tolling_fee_usd_mmbtu"])

    # C. Terminal Loading Fees (Avoidable / Variable)
    loading_fees = e_loaded * float(cfg["loading_port_charges_usd_mmbtu"])

    # D. Shipping / Charter Cost (Destination-Dependent via round-trip duration)
    total_voyage_days = sched.total_round_trip_days
    charter_cost = freight_rate * total_voyage_days

    # E. Bunker MGO Cost (Destination-Dependent)
    bunker_mgo_cost = energy_res.total_round_trip_bunker_mgo_mt * mgo_price

    # F. Canal Fees (Destination-Dependent)
    canal_cost = 0.0
    if sched.route_id == "NEA_PANAMA":
        canal_cost = float(cfg["panama_canal_transit_cost_usd"])
    elif sched.route_id == "NEA_SUEZ":
        canal_cost = float(cfg["suez_canal_transit_cost_usd"])

    # G. Destination Unloading & Regasification Fees
    if energy_res.destination_region == "NWE":
        unloading_fees = e_delivered * float(cfg["unloading_nwe_charges_usd_mmbtu"])
        regas_fees = energy_res.net_grid_injected_energy_mmbtu * float(cfg["nwe_regasification_tariff_usd_mmbtu"])
    else:
        unloading_fees = e_delivered * float(cfg["unloading_nea_charges_usd_mmbtu"])
        regas_fees = energy_res.net_grid_injected_energy_mmbtu * float(cfg["nea_regasification_tariff_usd_mmbtu"])

    # H. Financing & Working Capital Cost
    # Interest on feedstock + voyage expenses tied up for payment duration (e.g. 30 days + voyage)
    total_capital_at_risk = feedgas_cost + charter_cost + bunker_mgo_cost + canal_cost
    financing_cost = total_capital_at_risk * (cost_of_capital * (payment_days / 365.0))

    # Assemble Cost Items with Taxonomy Tagging
    cost_items = [
        CostItem("Feedgas Procurement (HH 1.15x)", feedgas_cost, CostCategory.VARIABLE_UPSTREAM, "Physical feedgas purchased from pipeline grid at 115% Henry Hub"),
        CostItem("Liquefaction Tolling Fee", tolling_fee, CostCategory.SUNK_COMMITTED, "Contractually committed take-or-pay liquefaction capacity fee"),
        CostItem("Loading Terminal Fees", loading_fees, CostCategory.AVOIDABLE, "USGC port berth and departure handling charges"),
        CostItem("Vessel Time Charter Hire", charter_cost, CostCategory.DESTINATION_DEPENDENT, "Time charter hire cost for total round-trip voyage duration"),
        CostItem("Marine Bunker MGO Fuel", bunker_mgo_cost, CostCategory.DESTINATION_DEPENDENT, "VLSFO/MGO consumed during laden and ballast legs"),
        CostItem("Canal Transit Fees", canal_cost, CostCategory.DESTINATION_DEPENDENT, "Panama or Suez canal transit and booking slot tolls"),
        CostItem("Destination Unloading Fees", unloading_fees, CostCategory.DESTINATION_DEPENDENT, "Receiving terminal port and send-out fees"),
        CostItem("Regasification & Grid Tariff", regas_fees, CostCategory.DESTINATION_DEPENDENT, "Downstream pipeline injection and regasification tariff"),
        CostItem("Working Capital Financing", financing_cost, CostCategory.FINANCING, "Opportunity cost of capital tied up during voyage & payment terms"),
    ]

    total_contract_cost = sum(item.amount_usd for item in cost_items)
    total_full_cycle = total_contract_cost
    total_marginal = sum(item.amount_usd for item in cost_items if item.is_marginal_routing_relevant)
    total_avoidable = sum(item.amount_usd for item in cost_items if item.is_lift_shut_in_relevant)

    net_profit = gross_revenue - total_full_cycle
    gross_margin = gross_revenue - (feedgas_cost + charter_cost + bunker_mgo_cost + canal_cost)

    profit_per_del_mmbtu = net_profit / e_delivered if e_delivered > 0 else 0.0
    profit_per_load_mmbtu = net_profit / e_loaded if e_loaded > 0 else 0.0

    # Netback calculations
    # FOB Netback = (Gross Revenue - Destination & Voyage Costs) / Loaded MMBtu
    destination_and_voyage_costs = charter_cost + bunker_mgo_cost + canal_cost + unloading_fees + regas_fees + financing_cost
    netback_fob = (gross_revenue - destination_and_voyage_costs) / e_loaded if e_loaded > 0 else 0.0
    
    # Wellhead Netback = FOB Netback minus Liquefaction and Upstream Transport
    netback_wellhead = netback_fob - float(cfg["fixed_liquefaction_tolling_fee_usd_mmbtu"])

    benchmark_name = "TTF Front-Month" if energy_res.destination_region == "NWE" else "JKM Front-Month"

    return CargoEconomicResult(
        destination_region=energy_res.destination_region,
        benchmark_name=benchmark_name,
        benchmark_price_usd_mmbtu=destination_price_usd_mmbtu,
        delivered_energy_mmbtu=e_delivered,
        gross_revenue_usd=gross_revenue,
        cost_items=cost_items,
        total_contract_cost_usd=total_contract_cost,
        total_full_cycle_cost_usd=total_full_cycle,
        total_marginal_routing_cost_usd=total_marginal,
        total_avoidable_cost_usd=total_avoidable,
        gross_margin_usd=gross_margin,
        net_profit_cargo_usd=net_profit,
        profit_per_delivered_mmbtu_usd=profit_per_del_mmbtu,
        profit_per_loaded_mmbtu_usd=profit_per_load_mmbtu,
        netback_at_fob_flange_usd_mmbtu=netback_fob,
        netback_at_wellhead_usd_mmbtu=netback_wellhead,
    )
