"""
Phase 3A: Model Training
Trains 5 XGBoost model variants (M1-M5) with temporal train/val/test split.
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.helpers import logger, update_progress

PLEDGE_FEATURES = [
    "pledge_pct_promoter", "pledge_pct_total", "pledge_change_1q",
    "pledge_change_2q", "pledge_acceleration", "pledge_high_flag",
    "pledge_rising_3q", "pledge_pct_4q_max", "pledge_pct_vs_4q_max",
    "promoter_holding_pct",
]

PRICE_FIN_FEATURES = [
    "price_return_3m", "price_return_6m", "price_volatility_60d",
    "price_vs_52w_high", "volume_spike_flag", "market_cap_log",
    "debt_to_equity", "interest_coverage", "current_ratio",
    "roe_ttm", "revenue_growth_yoy",
]

TONE_FEATURES = [
    "evasiveness_score", "confidence_score", "debt_mention_count",
    "reassurance_score", "analyst_tension_score", "guidance_vagueness_score",
    "liquidity_stress_mentions", "tone_shift_flag", "overall_distress_score",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
    "tree_method": "hist",
}


def get_available_features(df: pd.DataFrame, feature_list: list) -> list:
    return [f for f in feature_list if f in df.columns]


def prepare_split(df: pd.DataFrame, features: list, split_name: str):
    subset = df[df["split"] == split_name].copy()
    avail = get_available_features(subset, features)
    X = subset[avail].copy()
    y = subset["crash_label"].values
    # Fill NaN with median
    X = X.fillna(X.median())
    return X, y, avail


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                features: list, model_name: str) -> tuple:
    """Train single XGBoost model."""
    avail = get_available_features(train_df, features)
    if not avail:
        logger.warning(f"{model_name}: No valid features found.")
        return None, []

    X_train = train_df[avail].fillna(train_df[avail].median())
    y_train = train_df["crash_label"].values
    X_val = val_df[avail].fillna(train_df[avail].median())
    y_val = val_df["crash_label"].values

    if y_train.sum() == 0:
        logger.warning(f"{model_name}: No positive samples in training set.")
        return None, avail

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(f"{model_name}: best_iteration={model.best_iteration}, "
                f"scale_pos_weight={scale_pos:.1f}, features={len(avail)}")
    return model, avail


def run_phase_3a():
    logger.info("=== PHASE 3A: Model Training ===")

    dataset_path = PROCESSED_DIR / "master_dataset.parquet"
    if not dataset_path.exists():
        logger.error("No master_dataset.parquet. Run Phase 2D first.")
        update_progress("FAILED", "Phase 3A: Train Models", "No dataset")
        return

    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset: {len(df)} rows, crash rate={df['crash_label'].mean():.3%}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    # Tone-available subsets for M3 (tone-only ablation)
    train_tone = train_df[train_df["tone_available"] == 1].copy()
    val_tone = val_df[val_df["tone_available"] == 1].copy()

    # M4/M5: train on FULL dataset; tone features imputed -1 where missing.
    # This lets the model use tone where available while keeping full training size.
    train_full_imputed = train_df.copy()
    val_full_imputed = val_df.copy()
    for feat in TONE_FEATURES:
        if feat in train_full_imputed.columns:
            train_full_imputed[feat] = train_full_imputed[feat].fillna(-1)
            val_full_imputed[feat] = val_full_imputed[feat].fillna(-1)

    model_configs = {
        "M1_pledge_only": (train_df, val_df, PLEDGE_FEATURES),
        "M2_pledge_price_fin": (train_df, val_df, PLEDGE_FEATURES + PRICE_FIN_FEATURES),
        "M3_tone_only": (train_tone, val_tone, TONE_FEATURES),
        "M4_full": (train_full_imputed, val_full_imputed, PLEDGE_FEATURES + PRICE_FIN_FEATURES + TONE_FEATURES),
        "M5_baseline_no_tone": (train_df, val_df, PLEDGE_FEATURES + PRICE_FIN_FEATURES),
    }

    trained_models = {}
    trained_features = {}

    for model_name, (tr, va, features) in model_configs.items():
        logger.info(f"Training {model_name}...")
        if len(tr) == 0 or tr["crash_label"].sum() == 0:
            logger.warning(f"{model_name}: insufficient training data, skipping.")
            continue
        model, feat_list = train_model(tr, va, features, model_name)
        if model is not None:
            trained_models[model_name] = model
            trained_features[model_name] = feat_list
            model_path = MODELS_DIR / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "features": feat_list}, f)
            logger.info(f"Saved {model_name} to {model_path}")

    # Save feature lists
    feat_path = MODELS_DIR / "feature_lists.pkl"
    with open(feat_path, "wb") as f:
        pickle.dump(trained_features, f)

    note = f"Trained {len(trained_models)}/5 models: {list(trained_models.keys())}"
    update_progress("DONE", "Phase 3A: Train Models", note)
    logger.info(f"Phase 3A complete. {note}")
    return trained_models, trained_features


if __name__ == "__main__":
    run_phase_3a()
