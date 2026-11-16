"""
Automated unit tests for Phase 6 Regime Analytics, Tail Risk & Sensitivity Engine.
Verifies:
1. Consistency between VaR (95%) and Expected Shortfall (CVaR 95%: CVaR <= VaR).
2. Sensitivity matrix monotonicity with respect to JKM-TTF spread and freight rate.
"""

import numpy as np
import pandas as pd

from src.risk.regime_analytics import (
    compute_tail_risk_metrics,
    compute_spread_freight_sensitivity_matrix,
)


def test_var_and_cvar_tail_consistency() -> None:
    """Verifies that Conditional VaR (Expected Shortfall) is always <= VaR at 95% confidence."""
    rng = np.random.default_rng(42)
    sample_profits = pd.Series(rng.normal(loc=15.0, scale=12.0, size=500))

    var_95, cvar_95 = compute_tail_risk_metrics(sample_profits, alpha=0.05)
    assert cvar_95 <= var_95


def test_sensitivity_matrix_monotonicity() -> None:
    """
    Verifies commercial economics on the 2D sensitivity grid:
    1. Higher JKM-TTF spread strictly increases Asia profit advantage over Europe.
    2. Higher daily freight rate strictly reduces Asia profit advantage over Europe.
    """
    sens_df = compute_spread_freight_sensitivity_matrix(
        as_of_date="2026-11-13",
        hh_price=3.60,
        ttf_price=29.33,
        jkm_minus_ttf_spreads=[-1.00, 1.00, 3.00],
        freight_rates_usd_day=[40000.0, 120000.0],
        asia_route_id="NEA_PANAMA",
    )

    # Across columns (increasing JKM - TTF spread): NEA advantage must increase
    assert sens_df.iloc[0, 2] > sens_df.iloc[0, 1] > sens_df.iloc[0, 0]

    # Down rows (increasing freight rate): NEA advantage must decrease
    assert sens_df.iloc[1, 1] < sens_df.iloc[0, 1]