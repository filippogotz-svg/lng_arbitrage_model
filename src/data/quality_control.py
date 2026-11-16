"""
Non-destructive Quality Control (QC) engine for commodity price series.
Flags missing values, duplicates, invalid/non-positive prices, and statistical
outliers using a robust rolling Median Absolute Deviation (MAD) filter.
Principle 10 compliance: Never silently deletes outliers or missing observations.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class QualityControlConfig:
    """Configuration thresholds for market data quality checks."""
    allow_negative_prices: bool = False
    allow_zero_prices: bool = False
    min_plausible_price_usd_mmbtu: float = 0.50
    max_plausible_price_usd_mmbtu: float = 100.00
    rolling_window_days: int = 60
    mad_outlier_threshold: float = 6.0
    max_consecutive_missing_business_days: int = 5


@dataclass
class QualityControlReport:
    """Structured audit report summarizing dataset health and anomalies."""
    benchmark: str
    dataset_id: str
    total_rows: int
    start_date: str
    end_date: str
    valid_pass_count: int
    missing_value_count: int
    duplicate_date_count: int
    invalid_price_count: int
    outlier_high_count: int
    outlier_low_count: int
    max_consecutive_missing_b_days: int
    unexpected_frequency_gaps: List[Dict[str, Any]]
    flagged_outliers_sample: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to a standard Python dictionary for JSON export."""
        return asdict(self)


def apply_quality_control_checks(
    df: pd.DataFrame,
    qc_config: QualityControlConfig,
) -> tuple[pd.DataFrame, QualityControlReport]:
    """
    Applies non-destructive quality checks to a standardized market DataFrame.
    Populates the 'qc_flag' column without dropping rows.

    Args:
        df: Standardized market DataFrame sorted by observation_date.
        qc_config: Threshold parameters for QC validation.

    Returns:
        Tuple of (QC-annotated DataFrame, QualityControlReport).
    """
    checked_df = df.copy()
    checked_df["observation_date"] = pd.to_datetime(checked_df["observation_date"])
    checked_df = checked_df.sort_values("observation_date").reset_index(drop=True)

    flags = pd.Series("PASS", index=checked_df.index, dtype="object")
    prices = pd.to_numeric(checked_df["price_usd_mmbtu"], errors="coerce")

    # 1. Check duplicate observation dates
    is_duplicate = checked_df.duplicated(subset=["observation_date", "contract_month"], keep="first")

    # 2. Check missing values
    is_missing = prices.isna()

    # 3. Check invalid physical/economic prices (negative, zero, or outside plausible bounds)
    is_invalid = pd.Series(False, index=checked_df.index)
    if not qc_config.allow_negative_prices:
        is_invalid = is_invalid | (prices < 0.0)
    if not qc_config.allow_zero_prices:
        is_invalid = is_invalid | (prices == 0.0)
    is_invalid = is_invalid | (prices < qc_config.min_plausible_price_usd_mmbtu)
    is_invalid = is_invalid | (prices > qc_config.max_plausible_price_usd_mmbtu)
    is_invalid = is_invalid & (~is_missing)

    # 4. Robust rolling Median Absolute Deviation (MAD) outlier detection on valid observations
    valid_prices = prices.where(~(is_missing | is_invalid))
    rolling_median = valid_prices.rolling(
        window=qc_config.rolling_window_days, min_periods=20
    ).median()
    abs_dev = (valid_prices - rolling_median).abs()
    rolling_mad = abs_dev.rolling(
        window=qc_config.rolling_window_days, min_periods=20
    ).median()

    # Consistent normal-equivalent MAD scale factor (1.4826)
    robust_sigma = 1.4826 * rolling_mad
    # Avoid division by zero during flat price regimes
    robust_sigma = robust_sigma.replace(0.0, np.nan)
    robust_z = (valid_prices - rolling_median) / robust_sigma

    is_outlier_high = (robust_z > qc_config.mad_outlier_threshold) & (~is_missing) & (~is_invalid)
    is_outlier_low = (robust_z < -qc_config.mad_outlier_threshold) & (~is_missing) & (~is_invalid)

    # Assign flags in order of severity (higher severity overwrites lower)
    flags.loc[is_outlier_low] = "OUTLIER_LOW"
    flags.loc[is_outlier_high] = "OUTLIER_HIGH"
    flags.loc[is_missing] = "MISSING_VALUE"
    flags.loc[is_invalid] = "INVALID_PRICE"
    flags.loc[is_duplicate] = "DUPLICATE_DATE"

    checked_df["qc_flag"] = flags

    # 5. Evaluate consecutive missing business days and unexpected calendar gaps
    is_consecutive_na = is_missing.astype(int)
    na_groups = is_consecutive_na.groupby((~is_missing).cumsum()).sum()
    max_consec_na = int(na_groups.max()) if not na_groups.empty else 0

    date_diffs = checked_df["observation_date"].diff().dt.days
    unexpected_gaps: List[Dict[str, Any]] = []
    gap_mask = date_diffs > qc_config.max_consecutive_missing_business_days
    for idx in checked_df.index[gap_mask]:
        prev_date = checked_df.loc[idx - 1, "observation_date"].strftime("%Y-%m-%d")
        curr_date = checked_df.loc[idx, "observation_date"].strftime("%Y-%m-%d")
        unexpected_gaps.append({
            "previous_date": prev_date,
            "current_date": curr_date,
            "calendar_days_gap": int(date_diffs.loc[idx]),
        })

    # Sample top outliers by absolute price for human trader review
    outliers_df = checked_df.loc[is_outlier_high | is_outlier_low, [
        "observation_date", "price_usd_mmbtu", "qc_flag"
    ]].copy()
    outliers_df["observation_date"] = outliers_df["observation_date"].dt.strftime("%Y-%m-%d")
    outliers_sample = outliers_df.sort_values(
        by="price_usd_mmbtu", ascending=False
    ).head(15).to_dict(orient="records")

    checked_df["observation_date"] = checked_df["observation_date"].dt.strftime("%Y-%m-%d")

    report = QualityControlReport(
        benchmark=str(checked_df["benchmark"].iloc[0]),
        dataset_id=str(checked_df["dataset_id"].iloc[0]),
        total_rows=int(len(checked_df)),
        start_date=str(checked_df["observation_date"].iloc[0]),
        end_date=str(checked_df["observation_date"].iloc[-1]),
        valid_pass_count=int((flags == "PASS").sum()),
        missing_value_count=int(is_missing.sum()),
        duplicate_date_count=int(is_duplicate.sum()),
        invalid_price_count=int(is_invalid.sum()),
        outlier_high_count=int(is_outlier_high.sum()),
        outlier_low_count=int(is_outlier_low.sum()),
        max_consecutive_missing_b_days=max_consec_na,
        unexpected_frequency_gaps=unexpected_gaps,
        flagged_outliers_sample=outliers_sample,
    )

    return checked_df, report