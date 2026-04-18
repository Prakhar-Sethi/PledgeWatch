"""
Phase 2A: Pledge Feature Engineering
Computes per-company-quarter pledge features from BSE shareholding data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import BSE_DIR, PROCESSED_DIR
from src.utils.helpers import logger, update_progress


def load_all_pledging_data() -> pd.DataFrame:
    """Load all BSE pledging CSVs into one DataFrame."""
    dfs = []
    for f in BSE_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def compute_pledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all pledge-based features per company-quarter."""
    if df.empty:
        return df

    # Standardize column names
    col_map = {
        "pledge_pct_promoter": "pledge_pct_promoter",
        "pledge_pct_total": "pledge_pct_total",
        "promoter_holding_pct": "promoter_holding_pct",
        "pledged_shares": "pledged_shares",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    numeric_cols = ["pledge_pct_promoter", "pledge_pct_total",
                    "promoter_holding_pct", "pledged_shares"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # Sort for rolling calculations
    df = df.sort_values(["nse_symbol", "quarter"]).reset_index(drop=True)

    # Fill missing pledge_pct_promoter from pledge_pct_total if needed
    missing_mask = df["pledge_pct_promoter"].isna() & df["pledge_pct_total"].notna()
    df.loc[missing_mask, "pledge_pct_promoter"] = (
        df.loc[missing_mask, "pledge_pct_total"] /
        (df.loc[missing_mask, "promoter_holding_pct"] / 100 + 1e-9)
    ).clip(0, 100)

    grp = df.groupby("nse_symbol")

    # Quarter-over-quarter changes
    df["pledge_change_1q"] = grp["pledge_pct_promoter"].diff(1)
    df["pledge_change_2q"] = grp["pledge_pct_promoter"].diff(2)

    # Acceleration (second derivative)
    df["pledge_acceleration"] = grp["pledge_change_1q"].diff(1)

    # High pledge flag
    df["pledge_high_flag"] = (df["pledge_pct_promoter"] > 50).astype(int)

    # Rising for 3 consecutive quarters
    df["_rising_1q"] = (df["pledge_change_1q"] > 0).astype(int)
    df["pledge_rising_3q"] = (
        grp["_rising_1q"].transform(lambda x: x.rolling(3, min_periods=3).sum()) == 3
    ).astype(int)

    # Level category
    def categorize_pledge(pct):
        if pd.isna(pct):
            return "unknown"
        if pct < 20:
            return "low"
        elif pct < 40:
            return "medium"
        elif pct < 60:
            return "high"
        else:
            return "critical"

    df["pledge_level_category"] = df["pledge_pct_promoter"].apply(categorize_pledge)

    # Rolling max pledge (momentum indicator)
    df["pledge_pct_4q_max"] = grp["pledge_pct_promoter"].transform(
        lambda x: x.rolling(4, min_periods=1).max()
    )

    # Distance from 4-quarter max
    df["pledge_pct_vs_4q_max"] = df["pledge_pct_promoter"] - df["pledge_pct_4q_max"]

    # Drop helper col
    df = df.drop(columns=["_rising_1q"], errors="ignore")

    return df


def run_phase_2a():
    logger.info("=== PHASE 2A: Pledge Features ===")
    df = load_all_pledging_data()

    if df.empty:
        logger.error("No pledging data found. Run Phase 1A first.")
        update_progress("FAILED", "Phase 2A: Pledge Features", "No input data")
        return pd.DataFrame()

    logger.info(f"Loaded {len(df)} pledging records for {df['nse_symbol'].nunique()} companies")
    df_feat = compute_pledge_features(df)

    out_path = PROCESSED_DIR / "pledge_features.parquet"
    df_feat.to_parquet(out_path, index=False)

    note = f"Computed pledge features for {df_feat['nse_symbol'].nunique()} companies, {len(df_feat)} rows"
    update_progress("DONE", "Phase 2A: Pledge Features", note)
    logger.info(f"Phase 2A complete. {note}")
    return df_feat


if __name__ == "__main__":
    run_phase_2a()
