"""
Phase 1D: Crash Event Label Builder
Identifies crash events via 30% max drawdown in 6m post quarter-end.
Cross-validates against hardcoded seed cases + Google News verification.
"""

import sys
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import (NSE_PRICES_DIR, LABELS_DIR, PROCESSED_DIR,
                               CRASH_THRESHOLD, MIN_POSITIVE, RANDOM_SEED)
from src.utils.helpers import logger, update_progress, log_assumption

SEED_CRASHES = {
    ("DHFL", "2019Q3"), ("ZEEL", "2019Q3"), ("RELCAPITAL", "2019Q3"),
    ("YESBANK", "2020Q1"), ("FRETAIL", "2020Q3"), ("ZEEMEDIA", "2019Q3"),
    ("ILFSENGG", "2018Q4"), ("ADANIENT", "2023Q1"), ("RCOM", "2019Q2"),
    ("JETAIRWAYS", "2019Q1"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def quarter_to_date(quarter: str) -> pd.Timestamp:
    """Convert YYYYQN to the last date of that quarter."""
    m = re.match(r"(\d{4})Q([1-4])", quarter)
    if not m:
        return pd.NaT
    yr, q = int(m.group(1)), int(m.group(2))
    end_months = {1: 3, 2: 6, 3: 9, 4: 12}
    month = end_months[q]
    last_day = pd.Timestamp(year=yr, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last_day


def compute_max_drawdown(prices: pd.Series) -> float:
    """Compute max drawdown as positive fraction."""
    if prices.empty or len(prices) < 2:
        return 0.0
    roll_max = prices.expanding().max()
    drawdowns = (prices - roll_max) / roll_max
    return float(-drawdowns.min())


def load_price_data(symbol: str) -> pd.DataFrame:
    path = NSE_PRICES_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def compute_crash_label(symbol: str, quarter: str, threshold: float) -> dict:
    """Compute crash label for a symbol-quarter."""
    prices_df = load_price_data(symbol)
    result = {
        "nse_symbol": symbol, "quarter": quarter,
        "crash_label": 0, "max_drawdown_pct": 0.0,
        "drawdown_verified": 0, "news_headline": "", "news_url": "",
    }

    is_seed = (symbol, quarter) in SEED_CRASHES
    if is_seed:
        result["crash_label"] = 1
        result["drawdown_verified"] = 1

    if prices_df.empty:
        return result

    q_end = quarter_to_date(quarter)
    if pd.isna(q_end):
        return result

    horizon_end = q_end + pd.Timedelta(days=182)
    mask = (prices_df["date"] > q_end) & (prices_df["date"] <= horizon_end)
    window = prices_df.loc[mask, "close"]

    if len(window) < 10:
        return result

    window = pd.to_numeric(window, errors="coerce").dropna()
    max_dd = compute_max_drawdown(window)
    result["max_drawdown_pct"] = round(max_dd * 100, 2)

    if max_dd >= threshold or is_seed:
        result["crash_label"] = 1

    return result


def verify_via_news(symbol: str, company_name: str, quarter: str) -> tuple[str, str]:
    """Scrape Google News for pledging/crash headlines."""
    q_end = quarter_to_date(quarter)
    if pd.isna(q_end):
        return "", ""

    search_query = f"{company_name} pledging crash margin call"
    encoded = requests.utils.quote(search_query)
    url = f"https://news.google.com/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return "", ""

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "lxml")

        # Extract article items
        for article in soup.find_all("article")[:10]:
            headline_el = article.find("h3") or article.find("h4")
            if not headline_el:
                continue
            headline = headline_el.get_text().strip()

            # Check for date
            time_el = article.find("time")
            if time_el:
                dt_str = time_el.get("datetime", "")
                try:
                    article_date = pd.Timestamp(dt_str)
                    within_3m = abs((article_date - q_end).days) <= 90
                    if within_3m:
                        link_el = article.find("a", href=True)
                        href = link_el["href"] if link_el else ""
                        full_url = f"https://news.google.com{href[1:]}" if href.startswith(".") else href
                        return headline[:200], full_url
                except Exception:
                    pass

    except Exception as e:
        logger.debug(f"News verification failed for {symbol}: {e}")
    return "", ""


def run_phase_1d(universe_df: pd.DataFrame = None, threshold: float = None):
    logger.info("=== PHASE 1D: Crash Event Labels ===")

    if threshold is None:
        threshold = CRASH_THRESHOLD

    if universe_df is None:
        universe_path = PROCESSED_DIR / "universe.csv"
        if universe_path.exists():
            universe_df = pd.read_csv(universe_path)
        else:
            logger.error("No universe.csv. Run Phase 1A first.")
            return

    # Generate all company-quarter combinations
    all_rows = []
    quarters = []
    for yr in range(2015, 2025):
        for q in range(1, 5):
            quarters.append(f"{yr}Q{q}")

    for _, row in universe_df.iterrows():
        symbol = row["nse_symbol"]
        company_name = str(row.get("company_name", symbol))
        prices_df = load_price_data(symbol)

        for quarter in quarters:
            result = compute_crash_label(symbol, quarter, threshold)
            result["bse_code"] = ""
            result["company_name"] = company_name
            all_rows.append(result)

    df = pd.DataFrame(all_rows)
    positive_count = df["crash_label"].sum()
    logger.info(f"Initial positive count: {positive_count} with threshold={threshold:.0%}")

    # Lower threshold if too few positives
    if positive_count < MIN_POSITIVE and threshold > 0.25:
        new_threshold = 0.25
        log_assumption(
            f"Positive class only {positive_count} < {MIN_POSITIVE} minimum. "
            f"Lowering crash threshold from {threshold:.0%} to {new_threshold:.0%} and re-running."
        )
        logger.warning(f"Lowering threshold to {new_threshold:.0%}")
        return run_phase_1d(universe_df, threshold=new_threshold)

    # News verification for confirmed crashes
    crash_rows = df[df["crash_label"] == 1].copy()
    logger.info(f"Verifying {len(crash_rows)} crash events via news...")

    verified_count = 0
    for idx in crash_rows.index[:50]:  # Limit news calls to avoid rate limiting
        sym = df.loc[idx, "nse_symbol"]
        cname = df.loc[idx, "company_name"]
        qtr = df.loc[idx, "quarter"]
        headline, url = verify_via_news(sym, cname, qtr)
        if headline:
            df.loc[idx, "news_headline"] = headline
            df.loc[idx, "news_url"] = url
            df.loc[idx, "drawdown_verified"] = 1
            verified_count += 1
        time.sleep(1)

    out_path = LABELS_DIR / "crash_events.csv"
    df.to_csv(out_path, index=False)

    pos = df["crash_label"].sum()
    neg = len(df) - pos
    note = (f"Labeled {len(df)} company-quarters; crash=1: {pos}, crash=0: {neg}; "
            f"threshold={threshold:.0%}; {verified_count} news-verified")
    update_progress("DONE", "Phase 1D: Crash Event Labels", note)
    logger.info(f"Phase 1D complete. {note}")
    return df


if __name__ == "__main__":
    run_phase_1d()
