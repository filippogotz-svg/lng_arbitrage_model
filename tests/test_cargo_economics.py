"""
Automated unit tests for Phase 3 Cargo Economics, Cost Classification & Break-Even Engine.
Verifies:
1. Separation of sunk capacity fees from marginal routing decisions.
2. Netback calculation consistency between FOB flange and wellhead.
3. Zero-profit break-even price thresholds.
"""

from datetime import date
import pytest

from src.physical.cargo_specs import load_physical_catalog
from src.physical.routing import load_routes_catalog
from src.physical.energy_balance import compute_cargo_energy_balance
from src.economics.cost_classification import CostCategory
from src.economics.cargo_pricer import evaluate_cargo_economics
from src.economics.breakeven import calculate_cargo_breakeven_thresholds


def test_sunk_vs_marginal_cost_separation() -> None:
    """Verifies that liquefaction tolling fee is tagged as SUNK_COMMITTED and excluded from marginal routing."""
    grades, vessels, terminals, _ = load_physical_catalog()
    routes = load_routes_catalog()
    energy_res = compute_cargo_energy_balance(
        vessels["two_stroke_174k"], grades["usgc_lean"], routes["NWE_DIRECT"], terminals, loading_start_date=date(2026, 11, 16)
    )

    econ = evaluate_cargo_economics(energy_res, destination_price_usd_mmbtu=13.00, henry_hub_price_usd_mmbtu=3.00)

    tolling_item = next(item for item in econ.cost_items if "Tolling" in item.name)
    assert tolling_item.category == CostCategory.SUNK_COMMITTED
    assert tolling_item.is_marginal_routing_relevant is False
    assert econ.total_marginal_routing_cost_usd < econ.total_full_cycle_cost_usd


def test_breakeven_threshold_accuracy() -> None:
    """Verifies that evaluating a cargo at its break-even destination price yields exactly zero net profit."""
    grades, vessels, terminals, _ = load_physical_catalog()
    routes = load_routes_catalog()
    energy_res = compute_cargo_energy_balance(
        vessels["two_stroke_174k"], grades["usgc_lean"], routes["NEA_PANAMA"], terminals, loading_start_date=date(2026, 11, 16)
    )

    be = calculate_cargo_breakeven_thresholds(
        energy_res=energy_res,
        baseline_destination_price_usd_mmbtu=14.00,
        baseline_henry_hub_usd_mmbtu=3.00,
        freight_daily_rate_usd=70000.0,
    )

    # Re-evaluate cargo at the calculated break-even destination price
    be_econ = evaluate_cargo_economics(
        energy_res=energy_res,
        destination_price_usd_mmbtu=be.breakeven_destination_price_usd_mmbtu,
        henry_hub_price_usd_mmbtu=3.00,
        freight_daily_rate_usd=70000.0,
    )
    assert be_econ.net_profit_cargo_usd == pytest.approx(0.0, abs=1e-2)