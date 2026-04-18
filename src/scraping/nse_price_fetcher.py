"""
Phase 1B: NSE Price Data Fetcher
Downloads daily OHLCV + computes return/volatility features for all universe companies.
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import NSE_PRICES_DIR, PROCESSED_DIR, START_DATE, END_DATE
from src.utils.helpers import logger, update_progress

PRICE_START = "2014-01-01"  # Extra year for 12m return calc
PRICE_END = END_DATE


def fetch_prices_yfinance(symbol: str) -> pd.DataFrame:
    """Try .NS then .BO suffix."""
    for suffix in [".NS", ".BO"]:
        try:
            ticker = f"{symbol}{suffix}"
            df = yf.download(ticker, start=PRICE_START, end=PRICE_END,
                             progress=False, auto_adjust=True, actions=False)
            if df is not None and not df.empty and len(df) > 50:
                df = df.reset_index()
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df["symbol"] = symbol
                logger.debug(f"yfinance OK: {ticker} ({len(df)} days)")
                return df
        except Exception as e:
            logger.debug(f"yfinance failed {symbol}{suffix}: {e}")
    return pd.DataFrame()


def fetch_prices_nsepython(symbol: str) -> pd.DataFrame:
    """Fallback: nsepython equity history."""
    try:
        from nsepython import equity_history
        start_str = pd.Timestamp(PRICE_START).strftime("%d-%m-%Y")
        end_str = pd.Timestamp(PRICE_END).strftime("%d-%m-%Y")
        df = equity_history(symbol, "EQ", start_str, end_str)
        if df is not None and not df.empty:
            df = df.rename(columns={"CH_TIMESTAMP": "date", "CH_CLOSING_PRICE": "close",
                                     "CH_OPENING_PRICE": "open", "CH_TRADE_HIGH_PRICE": "high",
                                     "CH_TRADE_LOW_PRICE": "low", "CH_TOT_TRADED_QTY": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            logger.debug(f"nsepython OK: {symbol} ({len(df)} days)")
            return df[["date", "open", "high", "low", "close", "volume", "symbol"]]
    except Exception as e:
        logger.debug(f"nsepython failed {symbol}: {e}")
    return pd.DataFrame()


def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling returns, volatility, 52w high distance."""
    df = df.sort_values("date").copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce")

    df["return_1m"] = df["close"].pct_change(21)
    df["return_3m"] = df["close"].pct_change(63)
    df["return_6m"] = df["close"].pct_change(126)
    df["return_12m"] = df["close"].pct_change(252)
    df["volatility_60d"] = df["close"].pct_change().rolling(60).std() * np.sqrt(252)
    df["52w_high"] = df["close"].rolling(252).max()
    df["pct_from_52w_high"] = (df["close"] - df["52w_high"]) / df["52w_high"]

    if "volume" in df.columns:
        df["vol_10d_avg"] = df["volume"].rolling(10).mean()
        df["vol_60d_avg"] = df["volume"].rolling(60).mean()
        df["volume_spike_flag"] = (df["vol_10d_avg"] > 2 * df["vol_60d_avg"]).astype(int)
    else:
        df["volume_spike_flag"] = 0

    return df


def run_phase_1b(universe_df: pd.DataFrame = None):
    logger.info("=== PHASE 1B: NSE Price Fetcher ===")

    if universe_df is None:
        universe_path = PROCESSED_DIR / "universe.csv"
        if universe_path.exists():
            universe_df = pd.read_csv(universe_path)
        else:
            logger.error("No universe.csv found. Run Phase 1A first.")
            return

    symbols = universe_df["nse_symbol"].tolist()
    downloaded = 0
    failed = []

    for symbol in symbols:
        out_path = NSE_PRICES_DIR / f"{symbol}.csv"
        ret_path = NSE_PRICES_DIR / f"{symbol}_returns.csv"

        if out_path.exists() and ret_path.exists():
            downloaded += 1
            continue

        df = fetch_prices_yfinance(symbol)
        if df.empty:
            logger.warning(f"yfinance failed {symbol}, trying nsepython...")
            df = fetch_prices_nsepython(symbol)
            time.sleep(1)

        if not df.empty:
            df_feat = compute_return_features(df)
            df.to_csv(out_path, index=False)
            ret_cols = ["date", "close", "return_1m", "return_3m", "return_6m",
                        "return_12m", "volatility_60d", "pct_from_52w_high"]
            available_cols = [c for c in ret_cols if c in df_feat.columns]
            df_feat[available_cols].to_csv(ret_path, index=False)
            downloaded += 1
            logger.info(f"Downloaded {symbol}: {len(df)} days")
        else:
            failed.append(symbol)
            logger.warning(f"All sources failed for {symbol}")

        time.sleep(0.3)

    note = f"Downloaded prices for {downloaded}/{len(symbols)} companies; {len(failed)} failed"
    update_progress("DONE", "Phase 1B: NSE Price Fetcher", note)
    logger.info(f"Phase 1B complete. {note}")


if __name__ == "__main__":
    run_phase_1b()
