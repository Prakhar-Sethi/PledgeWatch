"""
Phase 1A: Pledging Data Scraper
Uses NSE corporate-share-holdings-master API + parallel XBRL parsing for pledge %.
Replaces broken BSE API (returns HTML) and broken NSE shareholding-patterns (404).
"""

import sys
import re
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import BSE_DIR, START_DATE, END_DATE
from src.utils.helpers import logger, update_progress, log_assumption

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
XBRL_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
MAX_WORKERS = 15


def get_universe() -> pd.DataFrame:
    """Fetch NSE Midcap150 + Smallcap250 via NSE API, fall back to hardcoded snapshot."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1.5)
        rows = []
        for idx_name, idx_code in [
            ("NIFTY MIDCAP 150", "NIFTY_MIDCAP_150"),
            ("NIFTY SMALLCAP 250", "NIFTY_SMLCAP_250"),
        ]:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={idx_code}"
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                for item in r.json().get("data", []):
                    rows.append({
                        "nse_symbol": item.get("symbol", ""),
                        "company_name": item.get("meta", {}).get("companyName", item.get("symbol", "")),
                        "index": idx_name,
                        "isin": item.get("meta", {}).get("isin", ""),
                    })
                logger.info(f"Got {len(rows)} from {idx_name}")
            time.sleep(1)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["nse_symbol"])
            logger.info(f"Universe from NSE API: {len(df)} companies")
            return df
    except Exception as e:
        logger.warning(f"NSE index API failed: {e}")

    log_assumption("NSE index API returned 0; using hardcoded snapshot")
    return _hardcoded_universe()


def _hardcoded_universe() -> pd.DataFrame:
    symbols = [
        "DHFL", "ZEEL", "RCOM", "YESBANK", "JETAIRWAYS", "ILFSENGG",
        "ADANIENT", "ADANIPORTS", "ADANIGREEN", "ADANITRANS",
        "ZEEMEDIA", "RELCAPITAL", "FRETAIL",
        "AUROPHARMA", "BALKRISIND", "BATAINDIA", "BERGEPAINT", "BHARATFORG",
        "BPCL", "CANBK", "CHOLAFIN", "CIPLA", "COFORGE", "COLPAL",
        "CROMPTON", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR",
        "DELTACORP", "DIVISLAB", "DMART", "EDELWEISS", "EMAMILTD",
        "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK",
        "GODREJCP", "GODREJPROP", "GRANULES", "GUJGASLTD", "HDFCAMC",
        "HDFCLIFE", "HEROMOTOCO", "HINDPETRO", "HINDUNILVR", "HONAUT",
        "IBULHSGFIN", "ICICIPRULI", "IDFCFIRSTB", "IGL", "INDHOTEL",
        "INDIGO", "INDUSTOWER", "INOXLEISUR", "IOC", "IPCALAB",
        "IRCTC", "ISEC", "ITC", "JINDALSTEL", "JSWENERGY",
        "JUBLFOOD", "JUBILANT", "KANSAINER", "KOTAKBANK", "LALPATHLAB",
        "LICHSGFIN", "LUPIN", "MANAPPURAM", "METROPOLIS",
        "MFSL", "MINDTREE", "MOTHERSUMI", "MPHASIS", "MRF",
        "MUTHOOTFIN", "NAVINFLUOR", "NAUKRI", "NBCC", "NHPC",
        "NLCINDIA", "NMDC", "OBEROIRLTY", "OFSS", "OIL",
        "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC",
        "PFIZER", "PIDILITIND", "PIIND", "POLYCAB", "POWERGRID",
        "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "REDINGTON",
        "RELAXO", "SAIL", "SBICARD", "SBILIFE", "SHREECEM",
        "SIEMENS", "SRF", "STARHEALTH", "SUNDARMFIN", "SUNTV",
        "SUPREMEIND", "SYNGENE", "TATACHEM", "TATACOMM", "TATAELXSI",
        "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM",
        "TIINDIA", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT",
        "TVSMOTOR", "UBL", "ULTRACEMCO", "UNIONBANK", "UPL",
        "VOLTAS", "WHIRLPOOL", "WIPRO", "ZYDUSLIFE",
        "APLAPOLLO", "ANGELONE", "APTUS", "ARVINDFASN", "ASAHIINDIA",
        "ASHOKLEY", "ASTERDM", "ATUL", "AWHCL", "BAJAJFINSV",
        "BAJAJHLDNG", "BALRAMCHIN", "BECTORFOOD", "BIKAJI",
        "BLUESTARCO", "BOSCHLTD", "BRIGADE", "BSE",
        "CAMPUS", "CANFINHOME", "CAPLIPOINT", "CASTROLIND", "CEATLTD",
        "CENTURYPLY", "CENTURYTEX", "CERA", "CHALET", "CHAMBLFERT",
        "CLEAN", "COCHINSHIP", "CONCORDBIO", "CRAFTSMAN", "CRISIL",
        "DCMSHRIRAM", "DELHIVERY", "DEVYANI", "DHANUKA", "DIXON",
        "DREDGECORP", "ECLERX", "EIHOTEL", "ELGIEQUIP", "EPIGRAL",
        "EQUITASBNK", "ESTER", "FINEORG", "FLUOROCHEM",
        "FORCEMOT", "GALAXYSURF", "GARFIBRES", "GARWARE",
        "GESHIP", "GHCL", "GICRE", "GILLETTE", "GLOBALHEALT",
        "GNFC", "GODFRYPHLP", "GPPL", "GRAPHITE", "GRINDWELL",
        "GTLINFRA", "GULFOILLUB", "HAPPSTMNDS", "HATSUN", "HAVELLS",
        "HEG", "HFCL", "HIKAL", "HINDCOPPER", "HINDZINC",
        "HOMEFIRST", "HUHTAMAKI", "IBREALEST", "IFBIND", "IIFL",
        "IMFA", "INDIAMART", "INDIANB", "INDOSTAR",
        "INGERRAND", "INTELLECT", "IOLCP", "IIFLWAM", "JAYNECOIND",
        "JBCHEPHARM", "JKCEMENT", "JKLAKSHMI", "JKPAPER", "JMFINANCL",
        "JSWHL", "JUSTDIAL", "JYOTHYLAB", "KAJARIACER",
        "KALPATPOWR", "KFINTECH", "KIOCL", "KIRLOSENG",
        "KNRCON", "KOPRAN", "KRBL", "KSB", "KSCL",
        "LATENTVIEW", "LAXMIMACH", "LEMONTREE", "LINDEINDIA", "LUXIND",
        "MAHINDCIE", "MAHLIFE", "MANINFRA", "MAPMYINDIA", "MARICO",
        "MARKSANS", "MASTEK", "MAXHEALTH", "MCX", "MEDANTA",
        "METROBRAND", "MHRIL", "MIDHANI", "MMTC", "MOIL",
        "MOLDTKPAC", "MOTILALOFS",
    ]
    rows = [{"nse_symbol": s, "company_name": s, "index": "SNAPSHOT", "isin": ""} for s in symbols]
    df = pd.DataFrame(rows).drop_duplicates(subset=["nse_symbol"])
    logger.info(f"Using hardcoded universe: {len(df)} companies")
    return df


def _normalize_quarter(s: str) -> str:
    """Convert date string (e.g. '30-SEP-2019', 'September 2019') to YYYYQN."""
    s = s.strip()
    if re.match(r"^\d{4}Q[1-4]$", s):
        return s

    month_to_q = {
        1: 4, 2: 4, 3: 4,   # Jan-Mar → Q4 (Indian FY ends Mar)
        4: 1, 5: 1, 6: 1,   # Apr-Jun → Q1
        7: 2, 8: 2, 9: 2,   # Jul-Sep → Q2
        10: 3, 11: 3, 12: 3, # Oct-Dec → Q3
    }
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    s_lower = s.lower()
    year_m = re.search(r"\d{4}", s)
    if not year_m:
        return s
    year = int(year_m.group())

    for abbr, num in month_map.items():
        if abbr in s_lower:
            return f"{year}Q{month_to_q[num]}"

    q_m = re.search(r"[Qq]([1-4])", s)
    if q_m:
        return f"{year}Q{q_m.group(1)}"
    return s


def _parse_xbrl_pledge(xbrl_url: str, symbol: str, quarter: str,
                        promoter_pct_from_master: str) -> dict:
    """Download XBRL and extract pledge % for promoter group."""
    base = {
        "nse_symbol": symbol,
        "quarter": quarter,
        "promoter_holding_pct": _safe_float(promoter_pct_from_master),
        "pledged_shares": None,
        "pledge_pct_promoter": None,
        "pledge_pct_total": None,
        "isin": "",
    }
    if not xbrl_url:
        return base
    try:
        r = requests.get(xbrl_url, headers=XBRL_HEADERS, timeout=20)
        if r.status_code != 200:
            return base
        text = r.text

        # Shares pledged by promoter group
        m = re.search(
            r'PledgedOrEncumberedNumberOfShares\s+contextRef="ShareholdingOfPromoterAndPromoterGroupI"[^>]*>([^<]+)<',
            text
        )
        if m:
            base["pledged_shares"] = _safe_float(m.group(1))

        # % of total shares pledged by promoter group
        m = re.search(
            r'PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares\s+contextRef="ShareholdingOfPromoterAndPromoterGroupI"[^>]*>([^<]+)<',
            text
        )
        if m:
            base["pledge_pct_total"] = _safe_float(m.group(1))

        # % of promoter shares that are pledged
        # = pledged / promoter_shares * 100; or find PercentageOfSharesPledged tag
        m2 = re.search(
            r'PercentageOfSharesPledgedOrEncumbered[^>]*contextRef="ShareholdingOfPromoterAndPromoterGroupI"[^>]*>([^<]+)<',
            text
        )
        if m2:
            base["pledge_pct_promoter"] = _safe_float(m2.group(1))

        # Derive pledge_pct_promoter if not found directly
        if base["pledge_pct_promoter"] is None and base["pledge_pct_total"] is not None:
            ph = base["promoter_holding_pct"]
            if ph and ph > 0:
                # pledge_pct_total is % of all shares; convert to % of promoter shares
                base["pledge_pct_promoter"] = round(base["pledge_pct_total"] / ph * 100, 4)

    except Exception as e:
        logger.debug(f"XBRL parse failed {symbol} {quarter}: {e}")
    return base


def _safe_float(val) -> float | None:
    try:
        return float(str(val).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def fetch_company_pledging(symbol: str, session: requests.Session) -> pd.DataFrame:
    """Fetch all quarterly pledging data for one company via NSE API + XBRL."""
    try:
        url = f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={symbol}"
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            logger.warning(f"{symbol}: master API returned {r.status_code}")
            return pd.DataFrame()

        records = r.json()
        if not records:
            logger.warning(f"{symbol}: no records from master API")
            return pd.DataFrame()

        # Filter to relevant date range
        rows_to_process = []
        for rec in records:
            q = _normalize_quarter(rec.get("date", ""))
            if not q or q < "2015":
                continue
            rows_to_process.append({
                "quarter": q,
                "pr_and_prgrp": rec.get("pr_and_prgrp", ""),
                "xbrl": rec.get("xbrl", ""),
            })

        if not rows_to_process:
            return pd.DataFrame()

        # Parallel XBRL download
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(_parse_xbrl_pledge, row["xbrl"], symbol, row["quarter"], row["pr_and_prgrp"]): row
                for row in rows_to_process
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.debug(f"{symbol} XBRL future error: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).drop_duplicates(subset=["quarter"]).sort_values("quarter")
        return df

    except Exception as e:
        logger.warning(f"fetch_company_pledging failed for {symbol}: {e}")
        return pd.DataFrame()


def run_phase_1a():
    logger.info("=== PHASE 1A: Pledging Scraper (NSE XBRL) ===")
    universe_df = get_universe()
    universe_path = BSE_DIR.parent.parent / "processed" / "universe.csv"
    universe_df.to_csv(universe_path, index=False)
    logger.info(f"Universe saved: {len(universe_df)} companies")

    # Single NSE session for master API calls (anti-bot requires session cookies)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(2)

    scraped = 0
    failed = []
    symbols = universe_df["nse_symbol"].tolist()

    for i, symbol in enumerate(symbols):
        out_path = BSE_DIR / f"{symbol}.csv"
        if out_path.exists():
            logger.debug(f"Skip {symbol}: already done")
            scraped += 1
            continue

        df = fetch_company_pledging(symbol, session)
        if not df.empty:
            df.to_csv(out_path, index=False)
            scraped += 1
            pledged_any = (df["pledge_pct_total"].fillna(0) > 0).sum()
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}: {len(df)} quarters, {pledged_any} with pledging")
        else:
            failed.append(symbol)
            logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: no data")

        # Small delay between companies to respect NSE rate limits
        time.sleep(0.8)

        # Refresh session cookies every 50 requests
        if (i + 1) % 50 == 0:
            logger.info(f"Refreshing NSE session at company {i+1}...")
            session = requests.Session()
            session.headers.update(HEADERS)
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(2)

    note = f"Scraped {scraped}/{len(symbols)} companies via NSE XBRL; {len(failed)} failed"
    if failed[:10]:
        note += f"; first failures: {failed[:10]}"
    update_progress("DONE", "Phase 1A: BSE Pledging Scraper", note)
    logger.info(f"Phase 1A complete. {note}")
    return universe_df


if __name__ == "__main__":
    run_phase_1a()
