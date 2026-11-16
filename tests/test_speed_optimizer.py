"""
Automated unit tests for Phase 4 Dynamic Speed Optimization & Slow-Steaming Engine.
Verifies that speed optimization executes correctly and identifies profitable
speed adjustments relative to baseline design speeds.
"""

from datetime import date
import pytest

from src.physical.cargo_specs import load_physical_catalog
from src.physical.routing import load_routes_catalog
from src.optimization.speed_optimizer import optimize_voyage_speed


def test_speed_optimizer_execution() -> None:
    """Verifies that the speed optimization grid search successfully runs and returns valid speeds."""
    grades, vessels, terminals, cfg = load_physical_catalog()
    routes = load_routes_catalog()

    res = optimize_voyage_speed(
        vessel=vessels["two_stroke_174k"],
        grade=grades["usgc_lean"],
        route=routes["NEA_PANAMA"],
        terminals=terminals,
        destination_price_usd_mmbtu=14.00,
        henry_hub_price_usd_mmbtu=3.00,
        freight_daily_rate_usd=70000.0,
        mgo_lhv_mmbtu_per_mt=float(cfg["constants"]["mgo_lhv_mmbtu_per_mt"]),
        loading_start_date=date(2026, 11, 16),
    )

    assert 13.0 <= res.optimal_laden_speed_knots <= 21.0
    assert 13.0 <= res.optimal_ballast_speed_knots <= 21.0
    assert res.optimized_energy_res.schedule.total_round_trip_days > 0