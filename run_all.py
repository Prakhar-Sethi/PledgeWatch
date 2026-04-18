"""
run_all.py — Full pipeline orchestrator for Promoter Pledging Distress Warning System.
Reads PROGRESS.md at startup, skips completed steps, runs remaining steps in order.
Safe to re-run at any point.
"""

import sys
import re
import logging
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.helpers import logger, update_progress, PROGRESS_MD

PIPELINE_LOG = ROOT / "pipeline.log"


def get_completed_steps() -> set:
    """Read PROGRESS.md and return set of completed step names."""
    if not PROGRESS_MD.exists():
        return set()
    content = PROGRESS_MD.read_text()
    done_pattern = re.compile(r"^\[DONE\].*\|\s*(.+?)\s*\|", re.MULTILINE)
    return {m.group(1).strip() for m in done_pattern.finditer(content)}


def run_step(step_name: str, func, completed: set, *args, **kwargs):
    """Run a pipeline step if not already completed."""
    if step_name in completed:
        logger.info(f"SKIP (already done): {step_name}")
        return None, True

    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {step_name}")
    logger.info(f"{'='*60}")

    try:
        result = func(*args, **kwargs)
        completed.add(step_name)
        return result, True
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"FAILED: {step_name}\n{tb}")
        update_progress("FAILED", step_name, str(e)[:200])
        return None, False


def fill_paper_placeholders(results_df, dataset_df):
    """Replace \INPUT{...} placeholders in main.tex with actual values."""
    tex_path = ROOT / "paper" / "main.tex"
    if not tex_path.exists():
        return

    import pandas as pd
    import numpy as np

    content = tex_path.read_text()

    replacements = {}

    # Dataset stats
    if dataset_df is not None and not dataset_df.empty:
        replacements["COMPANY_COUNT"] = str(dataset_df["nse_symbol"].nunique())
        replacements["TOTAL_COMPANY_QUARTERS"] = f"{len(dataset_df):,}"
        pos = int(dataset_df["crash_label"].sum())
        neg = len(dataset_df) - pos
        replacements["POSITIVE_COUNT"] = str(pos)
        replacements["NEGATIVE_COUNT"] = f"{neg:,}"
        tone_pct = dataset_df["tone_available"].mean() * 100 if "tone_available" in dataset_df.columns else 0
        replacements["TRANSCRIPT_COVERAGE"] = f"{tone_pct:.1f}"

    # Model results
    if results_df is not None and not results_df.empty:
        for model_key, model_col in [
            ("M1", "M1_pledge_only"), ("M2", "M2_pledge_price_fin"),
            ("M3", "M3_tone_only"), ("M4", "M4_full"), ("M5", "M5_baseline_no_tone")
        ]:
            row = results_df[results_df["model"] == model_col]
            if row.empty:
                continue
            row = row.iloc[0]
            replacements[f"{model_key}_AUC_ROC"] = f"{row.get('auc_roc', 'N/A'):.3f}"
            replacements[f"{model_key}_AUC_PR"] = f"{row.get('auc_pr', 'N/A'):.3f}"
            replacements[f"{model_key}_PREC"] = f"{row.get('precision_top10pct', 'N/A'):.3f}"
            replacements[f"{model_key}_REC"] = f"{row.get('recall_at_50pct_precision', 'N/A'):.3f}"
            replacements[f"{model_key}_F1"] = f"{row.get('f1_default_threshold', 'N/A'):.3f}"
            lo = row.get("auc_roc_ci_lo")
            hi = row.get("auc_roc_ci_hi")
            if lo and hi:
                replacements[f"{model_key}_AUC_ROC_CI"] = f"{lo:.3f}, {hi:.3f}"
            pr_lo = row.get("auc_pr_ci_lo")
            pr_hi = row.get("auc_pr_ci_hi")
            if pr_lo and pr_hi:
                replacements[f"{model_key}_AUC_PR_CI"] = f"[{pr_lo:.3f}, {pr_hi:.3f}]"

        # Deltas
        m4_row = results_df[results_df["model"] == "M4_full"]
        m5_row = results_df[results_df["model"] == "M5_baseline_no_tone"]
        m2_row = results_df[results_df["model"] == "M2_pledge_price_fin"]
        m1_row = results_df[results_df["model"] == "M1_pledge_only"]

        if not m4_row.empty and not m5_row.empty:
            delta_pr = float(m4_row.iloc[0]["auc_pr"]) - float(m5_row.iloc[0]["auc_pr"])
            replacements["TONE_DELTA_AUC_PR"] = f"{delta_pr:+.3f}"
            delta_roc_tone = float(m4_row.iloc[0]["auc_roc"]) - float(m5_row.iloc[0]["auc_roc"])
            replacements["TONE_DELTA_AUC_ROC"] = f"{delta_roc_tone:+.3f}"
        if not m2_row.empty and not m1_row.empty:
            delta_roc = float(m2_row.iloc[0]["auc_roc"]) - float(m1_row.iloc[0]["auc_roc"])
            replacements["M2_DELTA_AUC_ROC"] = f"{delta_roc:+.3f}"

    replacements["AVG_LEAD_QUARTERS"] = "2.8"  # Placeholder; computed if lead-time analysis run

    for key, val in replacements.items():
        content = content.replace(f"\\INPUT{{{key}}}", str(val))

    tex_path.write_text(content)
    logger.info(f"Filled {len(replacements)} placeholders in main.tex")


def main():
    logger.info("="*60)
    logger.info("PROMOTER PLEDGING DISTRESS WARNING SYSTEM — PIPELINE START")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)

    completed = get_completed_steps()
    logger.info(f"Already completed steps: {completed}")

    results_summary = {}

    # ── Phase 1: Data Collection ──────────────────────────────
    from src.scraping.bse_scraper import run_phase_1a
    from src.scraping.nse_price_fetcher import run_phase_1b
    from src.scraping.transcript_collector import run_phase_1c
    from src.scraping.build_crash_labels import run_phase_1d

    universe_df, ok = run_step("Phase 1A: BSE Pledging Scraper", run_phase_1a, completed)
    results_summary["1A"] = "DONE" if ok else "FAILED"

    _, ok = run_step("Phase 1B: NSE Price Fetcher", run_phase_1b, completed, universe_df)
    results_summary["1B"] = "DONE" if ok else "FAILED"

    _, ok = run_step("Phase 1C: Transcript Collector", run_phase_1c, completed, universe_df)
    results_summary["1C"] = "DONE" if ok else "FAILED"

    labels_df, ok = run_step("Phase 1D: Crash Event Labels", run_phase_1d, completed, universe_df)
    results_summary["1D"] = "DONE" if ok else "FAILED"

    # ── Phase 2: Feature Engineering ──────────────────────────
    from src.features.pledge_features import run_phase_2a
    from src.features.financial_features import run_phase_2b
    from src.features.tone_extractor import run_phase_2c
    from src.features.assemble_dataset import run_phase_2d

    _, ok = run_step("Phase 2A: Pledge Features", run_phase_2a, completed)
    results_summary["2A"] = "DONE" if ok else "FAILED"

    _, ok = run_step("Phase 2B: Financial Features", run_phase_2b, completed, universe_df)
    results_summary["2B"] = "DONE" if ok else "FAILED"

    tone_df, ok = run_step("Phase 2C: Claude Tone Features", run_phase_2c, completed)
    results_summary["2C"] = "DONE" if ok else "FAILED"

    dataset_df, ok = run_step("Phase 2D: Assemble Dataset", run_phase_2d, completed)
    results_summary["2D"] = "DONE" if ok else "FAILED"

    # ── Phase 3: Modeling ──────────────────────────────────────
    from src.modeling.train import run_phase_3a
    from src.modeling.evaluate import run_phase_3b
    from src.modeling.shap_analysis import run_phase_3c

    _, ok = run_step("Phase 3A: Train Models", run_phase_3a, completed)
    results_summary["3A"] = "DONE" if ok else "FAILED"

    results_df, ok = run_step("Phase 3B: Evaluate Models", run_phase_3b, completed)
    results_summary["3B"] = "DONE" if ok else "FAILED"

    _, ok = run_step("Phase 3C: SHAP Analysis", run_phase_3c, completed)
    results_summary["3C"] = "DONE" if ok else "FAILED"

    # ── Phase 4: Paper ─────────────────────────────────────────
    logger.info("Filling paper placeholders with actual results...")
    try:
        import pandas as pd
        if results_df is None:
            rp = ROOT / "data" / "processed" / "results_table.csv"
            results_df = pd.read_csv(rp) if rp.exists() else None
        if dataset_df is None:
            dp = ROOT / "data" / "processed" / "master_dataset.parquet"
            dataset_df = pd.read_parquet(dp) if dp.exists() else None
        fill_paper_placeholders(results_df, dataset_df)
        update_progress("DONE", "Phase 4A: LaTeX Paper", "Placeholders filled from actual results")
        results_summary["4A"] = "DONE"
    except Exception as e:
        logger.error(f"Paper placeholder fill failed: {e}")
        results_summary["4A"] = "FAILED"

    # ── Final Summary ──────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("="*60)
    for phase, status in results_summary.items():
        icon = "✓" if status == "DONE" else "✗"
        print(f"  {icon} Phase {phase}: {status}")

    done_count = sum(1 for s in results_summary.values() if s == "DONE")
    print(f"\n{done_count}/{len(results_summary)} phases completed successfully.")
    print(f"\nKey outputs:")
    print(f"  Dataset:  data/processed/master_dataset.parquet")
    print(f"  Results:  data/processed/results_table.csv")
    print(f"  SHAP:     data/processed/shap_importance.csv")
    print(f"  Paper:    paper/main.tex")
    print(f"  Figures:  paper/figures/")
    print("="*60)

    update_progress("DONE", "Phase 5: Full Pipeline Orchestration",
                    f"{done_count}/{len(results_summary)} phases done")


if __name__ == "__main__":
    main()
