"""
Automated unit tests for Phase 5 Historical Backtesting Engine.
Verifies that historical backtest execution successfully processes the master panel
and generates valid summary records.
"""

from pathlib import Path
import pytest

from src.backtesting.historical_engine import run_historical_backtest


def test_historical_backtest_execution() -> None:
    """Verifies that the historical backtest successfully runs on master_market_panel."""
    panel_path = Path("data/processed/master_market_panel.parquet")
    if not panel_path.exists():
        pytest.skip("Master market panel not found. Run Phase 1B first.")

    backtest_df = run_historical_backtest(panel_path=panel_path)
    assert not backtest_df.empty
    assert "observation_date" in backtest_df.columns
    assert "nwe_net_profit_m_usd" in backtest_df.columns
    assert "nea_net_profit_m_usd" in backtest_df.columns