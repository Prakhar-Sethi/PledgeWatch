"""
Phase 2B: Price and Financial Ratio Features
Computes price-based features per company-quarter + pulls Screener.in ratios.
"""

import sys
import re
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import NSE_PRICES_DIR, PROCESSED_DIR
from src.utils.helpers import logger, update_progress

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.screener.in/",
}


def quarter_to_date_range(quarter: str) -> tuple:
    """Return (start, end) dates for a quarter."""
    m = re.match(r"(\d{4})Q([1-4])", quarter)
    if not m:
        return None, None
    yr, q = int(m.group(1)), int(m.group(2))
    starts = {1: (yr, 1, 1), 2: (yr, 4, 1), 3: (yr, 7, 1), 4: (yr, 10, 1)}
    ends = {1: (yr, 3, 31), 2: (yr, 6, 30), 3: (yr, 9, 30), 4: (yr, 12, 31)}
    start = pd.Timestamp(*starts[q])
    end = pd.Timestamp(*ends[q])
    return start, end


def load_price_data(symbol: str) -> pd.DataFrame:
    path = NSE_PRICES_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce")
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


def compute_price_features_for_quarter(prices_df: pd.DataFrame, quarter: str) -> dict:
    """Compute price features as of quarter-end date."""
    _, q_end = quarter_to_date_range(quarter)
    if q_end is None or prices_df.empty:
        return {}

    prices_df = prices_df[prices_df["date"] <= q_end].copy()
    if len(prices_df) < 21:
        return {}

    close = prices_df["close"]

    def safe_pct(series, periods):
        if len(series) < periods + 1:
            return np.nan
        return float((series.iloc[-1] / series.iloc[-(periods + 1)] - 1) * 100)

    vol60 = float(close.pct_change().iloc[-60:].std() * np.sqrt(252)) if len(close) >= 60 else np.nan
    high52 = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    pct_from_high = float((close.iloc[-1] - high52) / high52 * 100) if high52 > 0 else np.nan

    volume = prices_df.get("volume", pd.Series(dtype=float))
    if not volume.empty and not volume.isna().all():
        vol10 = float(volume.iloc[-10:].mean())
        vol60d = float(volume.iloc[-60:].mean())
        vol_spike = int(vol60d > 0 and vol10 > 2 * vol60d)
    else:
        vol_spike = 0

    close_val = float(close.iloc[-1])
    market_cap_log = np.log(close_val) if close_val > 0 else np.nan

    return {
        "price_return_3m": safe_pct(close, 63),
        "price_return_6m": safe_pct(close, 126),
        "price_volatility_60d": vol60 * 100 if not np.isnan(vol60) else np.nan,
        "price_vs_52w_high": pct_from_high,
        "volume_spike_flag": vol_spike,
        "market_cap_log": market_cap_log,
        "close_price": close_val,
    }


def fetch_screener_ratios(symbol: str, session: requests.Session) -> dict:
    """Fetch financial ratios from Screener.in JSON API."""
    url = f"https://www.screener.in/api/company/{symbol}/"
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            ratios = {}

            def extract_ratio(name_variants, data_dict):
                for key in name_variants:
                    val = data_dict.get(key)
                    if val is not None:
                        try:
                            return float(str(val).replace(",", "").replace("%", "").strip())
                        except Exception:
                            pass
                return np.nan

            # Common ratio field names in Screener API
            ratios["debt_to_equity"] = extract_ratio(
                ["Debt to equity", "D/E", "debt_to_equity"], data)
            ratios["interest_coverage"] = extract_ratio(
                ["Interest Coverage Ratio", "interest_coverage"], data)
            ratios["current_ratio"] = extract_ratio(
                ["Current ratio", "current_ratio"], data)
            ratios["promoter_holding_pct_fin"] = extract_ratio(
                ["Promoter holding", "promoter_holding"], data)
            ratios["roe_ttm"] = extract_ratio(
                ["ROE", "Return on equity", "roe"], data)
            ratios["revenue_growth_yoy"] = extract_ratio(
                ["Revenue growth", "Sales growth"], data)

            # Try from ratios list if present
            for item in data.get("ratios", []):
                name = str(item.get("name", "")).lower()
                val = item.get("value")
                try:
                    fval = float(str(val).replace(",", "").replace("%", "").strip())
                    if "debt" in name and "equity" in name:
                        ratios["debt_to_equity"] = fval
                    elif "interest coverage" in name:
                        ratios["interest_coverage"] = fval
                    elif "current ratio" in name:
                        ratios["current_ratio"] = fval
                    elif "roe" in name or "return on equity" in name:
                        ratios["roe_ttm"] = fval
                except Exception:
                    pass

            ratios["ratio_available"] = 1
            return ratios
    except Exception as e:
        logger.debug(f"Screener API failed for {symbol}: {e}")
    return {"ratio_available": 0}


def run_phase_2b(universe_df: pd.DataFrame = None):
    logger.info("=== PHASE 2B: Financial Features ===")

    if universe_df is None:
        universe_path = PROCESSED_DIR / "universe.csv"
        if universe_path.exists():
            universe_df = pd.read_csv(universe_path)
        else:
            logger.error("No universe.csv. Run Phase 1A first.")
            return

    session = requests.Session()
    session.headers.update(HEADERS)

    # Load pledge features to get all (symbol, quarter) pairs
    pledge_path = PROCESSED_DIR / "pledge_features.parquet"
    if pledge_path.exists():
        base_df = pd.read_parquet(pledge_path)[["nse_symbol", "quarter"]].drop_duplicates()
    else:
        # Build from universe x quarters
        quarters = [f"{yr}Q{q}" for yr in range(2015, 2025) for q in range(1, 5)]
        base_df = pd.DataFrame([
            {"nse_symbol": row["nse_symbol"], "quarter": q}
            for _, row in universe_df.iterrows()
            for q in quarters
        ])

    all_rows = []
    cached_ratios = {}

    for _, row in base_df.iterrows():
        symbol = row["nse_symbol"]
        quarter = row["quarter"]

        prices_df = load_price_data(symbol)
        price_feats = compute_price_features_for_quarter(prices_df, quarter)

        if symbol not in cached_ratios:
            cached_ratios[symbol] = fetch_screener_ratios(symbol, session)
            time.sleep(0.8)

        fin_ratios = cached_ratios[symbol].copy()

        combined = {"nse_symbol": symbol, "quarter": quarter}
        combined.update(price_feats)
        combined.update(fin_ratios)
        all_rows.append(combined)

    df_feat = pd.DataFrame(all_rows)

    # Fill sentinel for missing ratios
    ratio_cols = ["debt_to_equity", "interest_coverage", "current_ratio",
                  "promoter_holding_pct_fin", "roe_ttm", "revenue_growth_yoy"]
    for col in ratio_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(-1)
        else:
            df_feat[col] = -1

    df_feat["ratio_available"] = df_feat.get("ratio_available", pd.Series(0, index=df_feat.index)).fillna(0).astype(int)

    out_path = PROCESSED_DIR / "financial_features.parquet"
    df_feat.to_parquet(out_path, index=False)

    note = f"Financial features for {df_feat['nse_symbol'].nunique()} companies, {len(df_feat)} rows"
    update_progress("DONE", "Phase 2B: Financial Features", note)
    logger.info(f"Phase 2B complete. {note}")
    return df_feat


if __name__ == "__main__":
    run_phase_2b()
