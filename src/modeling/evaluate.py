"""
Phase 3B: Model Evaluation
Computes AUC-ROC, AUC-PR, precision/recall metrics + bootstrap CIs on test split.
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score,
                              precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.helpers import logger, update_progress

N_BOOTSTRAP = 1000
RNG = np.random.RandomState(RANDOM_SEED)


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn, n_iter: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Return (lower_95, upper_95) bootstrap confidence interval."""
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = RNG.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        try:
            scores.append(metric_fn(yt, ys))
        except Exception:
            pass
    if not scores:
        return np.nan, np.nan
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float = 0.10) -> float:
    """Precision among top k% predicted."""
    k = max(1, int(len(y_score) * k_pct))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_idx].mean())


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float = 0.5) -> float:
    """Recall at the operating point where precision >= min_precision."""
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    mask = prec >= min_precision
    if mask.any():
        return float(rec[mask].max())
    return 0.0


def evaluate_model(model_dict: dict, test_df: pd.DataFrame, model_name: str) -> dict:
    model = model_dict["model"]
    features = model_dict["features"]

    avail = [f for f in features if f in test_df.columns]
    if not avail:
        return {}

    X_test = test_df[avail].fillna(0)
    y_test = test_df["crash_label"].values

    if y_test.sum() == 0:
        logger.warning(f"{model_name}: no positive labels in test set")
        return {}

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, y_score)
    auc_pr = average_precision_score(y_test, y_score)
    prec_top10 = precision_at_top_k(y_test, y_score, 0.10)
    rec_at_50prec = recall_at_precision(y_test, y_score, 0.50)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Bootstrap CIs
    auc_roc_lo, auc_roc_hi = bootstrap_ci(y_test, y_score, roc_auc_score)
    auc_pr_lo, auc_pr_hi = bootstrap_ci(y_test, y_score, average_precision_score)

    return {
        "model": model_name,
        "test_samples": len(y_test),
        "test_positives": int(y_test.sum()),
        "auc_roc": round(auc_roc, 4),
        "auc_roc_ci_lo": round(auc_roc_lo, 4) if not np.isnan(auc_roc_lo) else None,
        "auc_roc_ci_hi": round(auc_roc_hi, 4) if not np.isnan(auc_roc_hi) else None,
        "auc_pr": round(auc_pr, 4),
        "auc_pr_ci_lo": round(auc_pr_lo, 4) if not np.isnan(auc_pr_lo) else None,
        "auc_pr_ci_hi": round(auc_pr_hi, 4) if not np.isnan(auc_pr_hi) else None,
        "precision_top10pct": round(prec_top10, 4),
        "recall_at_50pct_precision": round(rec_at_50prec, 4),
        "f1_default_threshold": round(f1, 4),
        "precision_default": round(prec, 4),
        "recall_default": round(rec, 4),
    }


def run_phase_3b():
    logger.info("=== PHASE 3B: Model Evaluation ===")

    dataset_path = PROCESSED_DIR / "master_dataset.parquet"
    if not dataset_path.exists():
        logger.error("No master_dataset.parquet.")
        update_progress("FAILED", "Phase 3B: Evaluate Models", "No dataset")
        return

    df = pd.read_parquet(dataset_path)
    test_df = df[df["split"] == "test"].copy()
    test_tone = test_df[test_df["tone_available"] == 1].copy()

    # M4 trained on full data with tone imputed -1; evaluate same way
    TONE_FEATURES = [
        "evasiveness_score", "confidence_score", "debt_mention_count",
        "reassurance_score", "analyst_tension_score", "guidance_vagueness_score",
        "liquidity_stress_mentions", "tone_shift_flag", "overall_distress_score",
    ]
    test_full_imputed = test_df.copy()
    for feat in TONE_FEATURES:
        if feat in test_full_imputed.columns:
            test_full_imputed[feat] = test_full_imputed[feat].fillna(-1)

    model_files = list(MODELS_DIR.glob("*.pkl"))
    model_files = [f for f in model_files if "feature_lists" not in f.name]

    test_sets = {
        "M1_pledge_only": test_df,
        "M2_pledge_price_fin": test_df,
        "M3_tone_only": test_tone,
        "M4_full": test_full_imputed,
        "M5_baseline_no_tone": test_df,
    }

    results = []
    for model_file in model_files:
        model_name = model_file.stem
        try:
            with open(model_file, "rb") as f:
                model_dict = pickle.load(f)
            test_data = test_sets.get(model_name, test_df)
            metrics = evaluate_model(model_dict, test_data, model_name)
            if metrics:
                results.append(metrics)
                logger.info(f"{model_name}: AUC-ROC={metrics['auc_roc']}, AUC-PR={metrics['auc_pr']}")
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")

    if not results:
        logger.error("No models evaluated successfully.")
        update_progress("FAILED", "Phase 3B: Evaluate Models", "No results")
        return

    results_df = pd.DataFrame(results)
    # Sort by model name for consistent ordering
    model_order = ["M1_pledge_only", "M2_pledge_price_fin", "M3_tone_only",
                   "M4_full", "M5_baseline_no_tone"]
    results_df["_order"] = results_df["model"].apply(
        lambda m: model_order.index(m) if m in model_order else 99
    )
    results_df = results_df.sort_values("_order").drop(columns=["_order"])

    out_path = PROCESSED_DIR / "results_table.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    note = f"Evaluated {len(results)} models; best AUC-ROC={results_df['auc_roc'].max():.4f}"
    update_progress("DONE", "Phase 3B: Evaluate Models", note)
    logger.info(f"Phase 3B complete. {note}")
    return results_df


if __name__ == "__main__":
    run_phase_3b()
