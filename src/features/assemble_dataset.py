"""
Phase 2D: Assemble Master Dataset
Joins pledge, financial, tone features with crash labels.
Adds temporal train/val/test split column.
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, LABELS_DIR, TRAIN_END, VAL_END
from src.utils.helpers import logger, update_progress


def quarter_to_sortable(quarter: str) -> str:
    """Convert YYYYQN to sortable string YYYY-QN."""
    m = re.match(r"(\d{4})Q([1-4])", str(quarter))
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    return quarter


def assign_split(quarter: str, train_end: str, val_end: str) -> str:
    """Assign train/val/test based on quarter."""
    m = re.match(r"(\d{4})Q([1-4])", str(quarter))
    if not m:
        return "unknown"
    yr, q = int(m.group(1)), int(m.group(2))
    date_approx = pd.Timestamp(yr, q * 3, 28)

    train_ts = pd.Timestamp(train_end)
    val_ts = pd.Timestamp(val_end)

    if date_approx <= train_ts:
        return "train"
    elif date_approx <= val_ts:
        return "val"
    else:
        return "test"


def run_phase_2d():
    logger.info("=== PHASE 2D: Assemble Master Dataset ===")

    # Load all feature parquets
    def load_parquet(name):
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_parquet(path)
        logger.warning(f"Missing: {name}")
        return pd.DataFrame()

    pledge_df = load_parquet("pledge_features.parquet")
    financial_df = load_parquet("financial_features.parquet")
    tone_df = load_parquet("tone_features.parquet")

    # Load crash labels
    labels_path = LABELS_DIR / "crash_events.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
    else:
        logger.error("No crash_events.csv. Run Phase 1D first.")
        update_progress("FAILED", "Phase 2D: Assemble Dataset", "Missing crash labels")
        return

    # Standardize join keys
    for df in [pledge_df, financial_df, tone_df, labels_df]:
        if not df.empty and "quarter" in df.columns:
            df["quarter"] = df["quarter"].astype(str).str.strip()
        if not df.empty and "nse_symbol" in df.columns:
            df["nse_symbol"] = df["nse_symbol"].astype(str).str.strip().str.upper()

    if labels_df["nse_symbol"].dtype == object:
        labels_df["nse_symbol"] = labels_df["nse_symbol"].str.upper()

    # Start from pledge_features as base (all valid company-quarter combos)
    if pledge_df.empty:
        # Fall back to labels_df as base
        base_df = labels_df[["nse_symbol", "quarter"]].drop_duplicates()
        logger.warning("No pledge features found. Using labels as base.")
    else:
        base_df = pledge_df.copy()

    # Left join financial features
    if not financial_df.empty:
        base_df = base_df.merge(financial_df, on=["nse_symbol", "quarter"], how="left", suffixes=("", "_fin"))

    # Left join tone features
    if not tone_df.empty:
        base_df = base_df.merge(tone_df, on=["nse_symbol", "quarter"], how="left", suffixes=("", "_tone"))
        base_df["tone_available"] = base_df["tone_available"].fillna(0).astype(int)
    else:
        base_df["tone_available"] = 0

    # Left join crash labels
    label_cols = ["nse_symbol", "quarter", "crash_label", "max_drawdown_pct",
                  "drawdown_verified", "news_headline", "company_name", "bse_code"]
    label_cols = [c for c in label_cols if c in labels_df.columns]
    base_df = base_df.merge(labels_df[label_cols], on=["nse_symbol", "quarter"], how="left")
    base_df["crash_label"] = base_df["crash_label"].fillna(0).astype(int)

    # Add split column
    base_df["split"] = base_df["quarter"].apply(
        lambda q: assign_split(q, TRAIN_END, VAL_END)
    )

    # Add sortable quarter for ordering
    base_df["quarter_sort"] = base_df["quarter"].apply(quarter_to_sortable)
    base_df = base_df.sort_values(["nse_symbol", "quarter_sort"]).drop(columns=["quarter_sort"])

    # Fill tone score columns with -1 sentinel where tone_available=0
    tone_score_cols = [
        "evasiveness_score", "confidence_score", "debt_mention_count",
        "reassurance_score", "analyst_tension_score", "guidance_vagueness_score",
        "liquidity_stress_mentions", "tone_shift_flag", "overall_distress_score",
    ]
    for col in tone_score_cols:
        if col in base_df.columns:
            base_df.loc[base_df["tone_available"] == 0, col] = -1
            base_df[col] = base_df[col].fillna(-1)
        else:
            base_df[col] = -1

    out_path = PROCESSED_DIR / "master_dataset.parquet"
    base_df.to_parquet(out_path, index=False)

    # Print and log class distribution
    total = len(base_df)
    pos = int(base_df["crash_label"].sum())
    neg = total - pos
    train_pos = int((base_df[base_df["split"] == "train"]["crash_label"]).sum())
    val_pos = int((base_df[base_df["split"] == "val"]["crash_label"]).sum())
    test_pos = int((base_df[base_df["split"] == "test"]["crash_label"]).sum())
    tone_cov = int(base_df["tone_available"].sum())

    stats = (
        f"\n{'='*50}\n"
        f"DATASET STATISTICS\n"
        f"Total rows: {total:,}\n"
        f"Companies: {base_df['nse_symbol'].nunique()}\n"
        f"Positive (crash=1): {pos} ({pos/total*100:.1f}%)\n"
        f"Negative (crash=0): {neg} ({neg/total*100:.1f}%)\n"
        f"Train positives: {train_pos}\n"
        f"Val positives: {val_pos}\n"
        f"Test positives: {test_pos}\n"
        f"Tone coverage: {tone_cov}/{total} ({tone_cov/total*100:.1f}%)\n"
        f"{'='*50}"
    )
    logger.info(stats)
    print(stats)

    note = f"Master dataset: {total} rows, {pos} positives, {tone_cov} with tone features"
    update_progress("DONE", "Phase 2D: Assemble Dataset", note)
    logger.info(f"Phase 2D complete. {note}")
    return base_df


if __name__ == "__main__":
    run_phase_2d()
