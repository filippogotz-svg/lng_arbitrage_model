"""
Automated unit tests for Phase 2 Physical LNG Value Chain & Energy Balance Engine.
Verifies:
1. Strict thermodynamic energy conservation (Loaded == Delivered + Laden Loss + Ballast Heel BOG).
2. Zero double-counting between Boil-Off Gas (BOG) and liquid bunker fuel oil (MGO).
3. Sensitivity to LNG composition/density (USGC Lean vs Rich LNG).
4. Partial reliquefaction activation when slow-steaming causes natural BOG to exceed propulsion demand.
"""

from datetime import date
import pytest

from src.physical.cargo_specs import load_physical_catalog
from src.physical.routing import load_routes_catalog
from src.physical.energy_balance import compute_cargo_energy_balance


def test_strict_energy_conservation_across_all_vessels_and_routes() -> None:
    """
    Verifies that for every vessel profile, route, and fuel mode,
    Gross Loaded FOB Energy equals Delivered DES Energy + Laden BOG Loss + Net Ballast Heel Change.
    """
    grades, vessels, terminals, cfg = load_physical_catalog()
    routes = load_routes_catalog()
    usgc_grade = grades["usgc_lean"]

    for v_id, vessel in vessels.items():
        for r_id, route in routes.items():
            for mode in ["BURN_FUEL_OIL", "FORCE_BOG"]:
                res = compute_cargo_energy_balance(
                    vessel=vessel,
                    grade=usgc_grade,
                    route=route,
                    terminals=terminals,
                    fuel_deficit_mode=mode,
                    loading_start_date=date(2026, 11, 16),
                )
                assert abs(res.energy_conservation_residual_mmbtu) < 1e-7
                assert res.delivered_energy_des_mmbtu < res.gross_loaded_energy_fob_mmbtu
                assert res.net_grid_injected_energy_mmbtu < res.delivered_energy_des_mmbtu


def test_no_double_counting_between_bog_and_bunker_fuel() -> None:
    """
    Verifies Principle 5: Switching from BURN_FUEL_OIL to FORCE_BOG on the laden leg
    must reduce Delivered DES MMBtu by the exact propulsion energy deficit while reducing
    laden bunker MGO consumption by the exact fuel-oil equivalent tonnage.
    """
    grades, vessels, terminals, cfg = load_physical_catalog()
    routes = load_routes_catalog()
    mgo_lhv = float(cfg["constants"]["mgo_lhv_mmbtu_per_mt"])

    vessel = vessels["two_stroke_174k"]
    route = routes["NEA_PANAMA"]
    grade = grades["usgc_lean"]

    res_oil = compute_cargo_energy_balance(
        vessel, grade, route, terminals, mgo_lhv_mmbtu_per_mt=mgo_lhv, fuel_deficit_mode="BURN_FUEL_OIL"
    )
    res_bog = compute_cargo_energy_balance(
        vessel, grade, route, terminals, mgo_lhv_mmbtu_per_mt=mgo_lhv, fuel_deficit_mode="FORCE_BOG"
    )

    # Energy difference in delivered cargo must equal the forced BOG on the laden leg
    delivered_diff_mmbtu = res_oil.delivered_energy_des_mmbtu - res_bog.delivered_energy_des_mmbtu
    assert delivered_diff_mmbtu > 0.0
    assert delivered_diff_mmbtu == pytest.approx(res_bog.laden_leg.forced_bog_mmbtu, rel=1e-9)

    # Bunker MGO difference on laden leg must equal delivered_diff_mmbtu / mgo_lhv
    mgo_diff_mt = res_oil.laden_bunker_mgo_mt - res_bog.laden_bunker_mgo_mt
    assert mgo_diff_mt == pytest.approx(delivered_diff_mmbtu / mgo_lhv, rel=1e-9)


def test_lng_composition_sensitivity_lean_vs_rich() -> None:
    """
    Verifies Principle 4.1: Rich LNG (higher density) yields materially different
    MMBtu loaded and delivered for the exact same cubic metre vessel capacity.
    """
    grades, vessels, terminals, _ = load_physical_catalog()
    routes = load_routes_catalog()
    vessel = vessels["two_stroke_174k"]
    route = routes["NWE_DIRECT"]

    res_lean = compute_cargo_energy_balance(vessel, grades["usgc_lean"], route, terminals)
    res_rich = compute_cargo_energy_balance(vessel, grades["rich_associated"], route, terminals)

    assert res_lean.net_loaded_volume_m3 == pytest.approx(res_rich.net_loaded_volume_m3)
    assert res_rich.gross_loaded_energy_fob_mmbtu > res_lean.gross_loaded_energy_fob_mmbtu


def test_partial_reliquefaction_activates_under_excess_bog() -> None:
    """
    Verifies that when propulsion demand is lower than natural boil-off (e.g. slow steaming),
    the 174k Two-Stroke reliquefies 75% of excess BOG back into the cargo tanks.
    """
    grades, vessels, terminals, _ = load_physical_catalog()
    routes = load_routes_catalog()

    # Create a custom vessel state or test solve_voyage_leg_fuel_balance directly
    from src.physical.energy_balance import solve_voyage_leg_fuel_balance

    leg = solve_voyage_leg_fuel_balance(
        leg_name="LADEN_SLOW",
        initial_energy_mmbtu=3_900_000.0,
        sea_days=15.0,
        daily_bor_fraction=0.0010,           # ~3,900 MMBtu/day natural BOG
        daily_propulsion_mmbtu=2000.0,       # Slow steaming demand < natural BOG
        pilot_fuel_fraction=0.02,
        reliquefaction_fraction=0.75,
        reliq_power_mgo_mt_per_mmbtu=0.0018,
        mgo_lhv_mmbtu_per_mt=40.19,
        fuel_deficit_mode="BURN_FUEL_OIL",
    )
    assert leg.excess_natural_bog_mmbtu > 0.0
    assert leg.reliquefied_bog_mmbtu == pytest.approx(0.75 * leg.excess_natural_bog_mmbtu)
    assert leg.gcu_flared_bog_mmbtu == pytest.approx(0.25 * leg.excess_natural_bog_mmbtu)
    assert leg.reliquefaction_aux_mgo_mt > 0.0