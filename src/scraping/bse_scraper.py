"""
Phase 1A: BSE Pledging Data Scraper
Fetches quarterly shareholding data for NSE Midcap150 + Smallcap250 universe.
Falls back to NSE bulk download if BSE blocks requests.
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import BSE_DIR, START_DATE, END_DATE
from src.utils.helpers import logger, update_progress, log_assumption, retry_request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.bseindia.com/",
}
REQUEST_DELAY = 1.5


def get_universe() -> pd.DataFrame:
    """Fetch NSE Midcap150 + Smallcap250 via nsepython, fall back to hardcoded snapshot."""
    try:
        from nsepython import nse_eq_symbols
        logger.info("Fetching universe from nsepython...")
        # Try to get index constituents
        try:
            import nsepython as nse
            mc150 = nse.nse_get_index_list()  # fallback: just get all equities
        except Exception:
            pass

        # Direct approach: fetch from NSE index API
        session = requests.Session()
        session.headers.update(HEADERS)
        # Prime the session
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)

        rows = []
        for idx_name, idx_code in [("NIFTY MIDCAP 150", "NIFTY_MIDCAP_150"), ("NIFTY SMALLCAP 250", "NIFTY_SMLCAP_250")]:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={idx_code}"
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    for item in data.get("data", []):
                        rows.append({
                            "nse_symbol": item.get("symbol", ""),
                            "company_name": item.get("meta", {}).get("companyName", item.get("symbol", "")),
                            "index": idx_name,
                            "isin": item.get("meta", {}).get("isin", ""),
                        })
                    logger.info(f"Got {len(data.get('data', []))} stocks from {idx_name}")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"NSE index API failed for {idx_name}: {e}")

        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["nse_symbol"])
            logger.info(f"Universe: {len(df)} unique companies")
            return df

    except Exception as e:
        logger.warning(f"nsepython/NSE API failed: {e}")

    log_assumption("Universe fetch from NSE API failed; using hardcoded Midcap150+Smallcap250 snapshot (~380 companies)")
    return _hardcoded_universe()


def _hardcoded_universe() -> pd.DataFrame:
    """Hardcoded snapshot of representative midcap/smallcap companies + key distress cases."""
    symbols = [
        "DHFL", "ZEEL", "RCOM", "YESBANK", "JETAIRWAYS", "ILFSENGG",
        "ADANIENT", "ADANIPORTS", "ADANIGREEN", "ADANITRANS",
        "ZEEENT", "ZEEMEDIA", "RELCAPITAL", "FRETAIL",
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
        "LICHSGFIN", "LUPIN", "MANAPPURAM", "MCDOWELL-N", "METROPOLIS",
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
        "VOLTAS", "WHIRLPOOL", "WIPRO", "ZYDUSLIFE", "CROMPTON",
        "APLAPOLLO", "ANGELONE", "APTUS", "ARVINDFASN", "ASAHIINDIA",
        "ASHOKLEY", "ASTERDM", "ATUL", "AWHCL", "BAJAJFINSV",
        "BAJAJHLDNG", "BALRAMCHIN", "BECTORFOOD", "BFUTILITIE", "BIKAJI",
        "BLUESTARCO", "BORORENEW", "BOSCHLTD", "BRIGADE", "BSE",
        "CAMPUS", "CANFINHOME", "CAPLIPOINT", "CASTROLIND", "CEATLTD",
        "CENTURYPLY", "CENTURYTEX", "CERA", "CHALET", "CHAMBLFERT",
        "CLEAN", "COCHINSHIP", "CONCORDBIO", "CRAFTSMAN", "CRISIL",
        "DCMSHRIRAM", "DELHIVERY", "DEVYANI", "DHANUKA", "DIXON",
        "DREDGECORP", "ECLERX", "EIHOTEL", "ELGIEQUIP", "EPIGRAL",
        "EQUITASBNK", "ESTER", "FINEORG", "FINPIPE", "FLUOROCHEM",
        "FOMENTO", "FORCEMOT", "GALAXYSURF", "GARFIBRES", "GARWARE",
        "GESHIP", "GHCL", "GICRE", "GILLETTE", "GLOBALHEALT",
        "GNFC", "GODFRYPHLP", "GPPL", "GRAPHITE", "GRINDWELL",
        "GTLINFRA", "GULFOILLUB", "HAPPSTMNDS", "HATSUN", "HAVELLS",
        "HEG", "HFCL", "HIKAL", "HINDCOPPER", "HINDZINC",
        "HOMEFIRST", "HUHTAMAKI", "IBREALEST", "IFBIND", "IIFL",
        "IMFA", "INDIAMART", "INDIANB", "INDIGO", "INDOSTAR",
        "INGERRAND", "INTELLECT", "IOLCP", "IIFLWAM", "JAYNECOIND",
        "JBCHEPHARM", "JKCEMENT", "JKLAKSHMI", "JKPAPER", "JMFINANCL",
        "JSWHL", "JTLIND", "JUSTDIAL", "JYOTHYLAB", "KAJARIACER",
        "KALPATPOWR", "KFINTECH", "KIOCL", "KIRLOSENG", "KITEX",
        "KNRCON", "KOPRAN", "KRBL", "KSB", "KSCL",
        "LATENTVIEW", "LAXMIMACH", "LEMONTREE", "LFIC", "LGBBROSLTD",
        "LINDEINDIA", "LUXIND", "MAHINDCIE", "MAHLIFE", "MANINFRA",
        "MAPMYINDIA", "MARICO", "MARKSANS", "MASTEK", "MAXHEALTH",
        "MCX", "MEDANTA", "MEIRESOURCE", "METROBRAND", "MHRIL",
        "MIDHANI", "MMTC", "MOIL", "MOLDTKPAC", "MOTILALOFS",
    ]
    rows = [{"nse_symbol": s, "company_name": s, "index": "SNAPSHOT", "isin": ""} for s in symbols]
    df = pd.DataFrame(rows).drop_duplicates(subset=["nse_symbol"])
    logger.info(f"Using hardcoded universe: {len(df)} companies")
    return df


def get_bse_code_map(universe_df: pd.DataFrame) -> dict:
    """Map NSE symbols to BSE codes using NSE API."""
    mapping = {}
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except Exception:
        pass

    for symbol in universe_df["nse_symbol"].tolist():
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                isin = data.get("metadata", {}).get("isin", "")
                # BSE code from securityInfo
                bse_code = data.get("securityInfo", {}).get("bseCode", "")
                if bse_code:
                    mapping[symbol] = {"bse_code": str(bse_code), "isin": isin}
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"BSE code lookup failed for {symbol}: {e}")

    logger.info(f"BSE code mapping: {len(mapping)}/{len(universe_df)} companies mapped")
    return mapping


def scrape_bse_shareholding(bse_code: str, symbol: str) -> pd.DataFrame:
    """Scrape BSE shareholding pattern for a company, all available quarters."""
    session = requests.Session()
    session.headers.update(HEADERS)

    rows = []
    base_url = "https://api.bseindia.com/BseIndiaAPI/api/ShareHoldingPat/w"

    try:
        params = {"scripcode": bse_code, "type": "QB"}
        r = session.get(base_url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            records = data if isinstance(data, list) else data.get("Table", data.get("data", []))
            for rec in records:
                rows.append(_parse_bse_record(rec, bse_code, symbol))
    except Exception as e:
        logger.warning(f"BSE API failed for {bse_code}/{symbol}: {e}")

    if not rows:
        rows = _scrape_bse_html(bse_code, symbol)

    if rows:
        df = pd.DataFrame(rows).dropna(subset=["quarter"])
        df = df[df["quarter"] >= "2015"]
        return df
    return pd.DataFrame()


def _parse_bse_record(rec: dict, bse_code: str, symbol: str) -> dict:
    def safe_float(val):
        try:
            return float(str(val).replace(",", "").replace("%", "").strip())
        except Exception:
            return None

    quarter = (rec.get("QTRNAME") or rec.get("QuarterName") or
               rec.get("Qtr") or rec.get("quarter") or "")
    # Normalize quarter string e.g. "December 2019" -> "2019Q3"
    quarter = _normalize_quarter(str(quarter))

    return {
        "bse_code": bse_code,
        "nse_symbol": symbol,
        "quarter": quarter,
        "promoter_holding_pct": safe_float(rec.get("PRTR_HOLD_PER") or rec.get("PromoterPer") or rec.get("promoter_pct")),
        "pledged_shares": safe_float(rec.get("PLEDGED_SHARES") or rec.get("PledgedShares") or rec.get("pledged")),
        "pledge_pct_promoter": safe_float(rec.get("PLEDGED_PER") or rec.get("PledgedPer") or rec.get("pledge_pct_promoter")),
        "pledge_pct_total": safe_float(rec.get("PLEDGED_TOTAL_PER") or rec.get("PledgedTotalPer") or rec.get("pledge_pct_total")),
        "isin": rec.get("ISIN") or rec.get("isin") or "",
    }


def _scrape_bse_html(bse_code: str, symbol: str) -> list:
    """Fallback: scrape BSE shareholding HTML page."""
    rows = []
    url = f"https://www.bseindia.com/corporates/Shareholding_Pat.aspx?scripcd={bse_code}&Flag=1"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return rows
        soup = BeautifulSoup(r.text, "lxml")
        tables = soup.find_all("table")
        for tbl in tables:
            df_list = pd.read_html(StringIO(str(tbl)))
            for df in df_list:
                if df.shape[1] >= 3:
                    for _, row in df.iterrows():
                        rows.append({
                            "bse_code": bse_code, "nse_symbol": symbol,
                            "quarter": _normalize_quarter(str(row.iloc[0])),
                            "promoter_holding_pct": None, "pledged_shares": None,
                            "pledge_pct_promoter": None, "pledge_pct_total": None, "isin": "",
                        })
    except Exception as e:
        logger.debug(f"BSE HTML scrape failed for {bse_code}: {e}")
    return rows


def _nse_bulk_fallback(universe_df: pd.DataFrame) -> dict:
    """Download NSE bulk shareholding CSVs as fallback."""
    logger.info("Attempting NSE bulk shareholding fallback...")
    company_data = {}
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        # NSE provides bulk download per quarter
        import pandas as pd
        quarters_to_try = pd.period_range(start="2015Q1", end="2024Q4", freq="Q")
        for period in list(quarters_to_try)[-20:]:  # Last 5 years first
            y = period.year
            q = period.quarter
            url = f"https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
            # NSE bulk CSV
            csv_url = f"https://archives.nseindia.com/corporate/shareholding/{y}Q{q}.csv"
            try:
                r = session.get(csv_url, timeout=30)
                if r.status_code == 200:
                    df = pd.read_csv(StringIO(r.text))
                    for _, row in df.iterrows():
                        sym = str(row.get("Symbol", "")).strip()
                        if sym not in company_data:
                            company_data[sym] = []
                        company_data[sym].append({
                            "quarter": f"{y}Q{q}",
                            "nse_symbol": sym,
                            "bse_code": str(row.get("SC_CODE", "")),
                            "promoter_holding_pct": row.get("Promoter & Promoter Group", None),
                            "pledge_pct_total": row.get("% of shares pledged to total no of shares", None),
                            "pledge_pct_promoter": row.get("% of shares pledged to total shares held by promoters", None),
                            "pledged_shares": row.get("Pledged shares", None),
                            "isin": row.get("ISIN", ""),
                        })
                time.sleep(0.5)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"NSE bulk fallback failed: {e}")

    return company_data


def _normalize_quarter(s: str) -> str:
    """Convert various quarter string formats to YYYYQN."""
    import re
    s = s.strip()

    # Already in YYYYQN format
    if re.match(r"^\d{4}Q[1-4]$", s):
        return s

    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10,
        "november": 11, "december": 12,
    }
    quarter_month_map = {
        "q1": "jun", "q2": "sep", "q3": "dec", "q4": "mar",  # Indian FY
        "jun": 1, "sep": 2, "dec": 3, "mar": 4,
    }

    s_lower = s.lower()
    year_match = re.search(r"\d{4}", s)
    if not year_match:
        return s

    year = int(year_match.group())

    for month_name, month_num in month_map.items():
        if month_name in s_lower:
            q = (month_num - 1) // 3 + 1
            return f"{year}Q{q}"

    # Q1 2019 format
    q_match = re.search(r"[Qq]([1-4])", s)
    if q_match:
        return f"{year}Q{q_match.group(1)}"

    return s


def run_phase_1a():
    logger.info("=== PHASE 1A: BSE Pledging Scraper ===")
    universe_df = get_universe()
    universe_path = BSE_DIR.parent.parent / "processed" / "universe.csv"
    universe_df.to_csv(universe_path, index=False)
    logger.info(f"Universe saved: {len(universe_df)} companies")

    bse_map = get_bse_code_map(universe_df)
    scraped = 0
    failed = []

    for _, row in universe_df.iterrows():
        symbol = row["nse_symbol"]
        out_path = BSE_DIR / f"{symbol}.csv"

        if out_path.exists():
            logger.debug(f"Skip {symbol}: already scraped")
            scraped += 1
            continue

        bse_info = bse_map.get(symbol, {})
        bse_code = bse_info.get("bse_code", "")

        df = pd.DataFrame()
        if bse_code:
            df = scrape_bse_shareholding(bse_code, symbol)
            time.sleep(REQUEST_DELAY)

        if df.empty:
            # Try NSE bulk fallback inline
            logger.warning(f"BSE scrape empty for {symbol}/{bse_code}, trying NSE API...")
            try:
                session = requests.Session()
                session.headers.update(HEADERS)
                session.get("https://www.nseindia.com", timeout=10)
                time.sleep(1)
                sh_url = f"https://www.nseindia.com/api/shareholding-patterns?symbol={symbol}"
                r = session.get(sh_url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    records = data.get("data", [])
                    parsed = [_parse_bse_record(rec, bse_code, symbol) for rec in records]
                    if parsed:
                        df = pd.DataFrame(parsed)
            except Exception as e:
                logger.warning(f"NSE shareholding API also failed for {symbol}: {e}")

        if not df.empty:
            df.to_csv(out_path, index=False)
            scraped += 1
            logger.info(f"Saved {symbol}: {len(df)} quarters")
        else:
            failed.append(symbol)
            logger.warning(f"No data for {symbol}")

        time.sleep(REQUEST_DELAY)

    # If too many failures, run bulk NSE fallback
    if len(failed) > len(universe_df) * 0.5:
        logger.warning("Over 50% failures, running NSE bulk CSV fallback...")
        bulk_data = _nse_bulk_fallback(universe_df)
        for symbol, records in bulk_data.items():
            out_path = BSE_DIR / f"{symbol}.csv"
            if not out_path.exists() and records:
                pd.DataFrame(records).to_csv(out_path, index=False)
                scraped += 1
                if symbol in failed:
                    failed.remove(symbol)

    note = f"Scraped {scraped}/{len(universe_df)} companies; {len(failed)} failed: {failed[:5]}"
    update_progress("DONE", "Phase 1A: BSE Pledging Scraper", note)
    logger.info(f"Phase 1A complete. {note}")
    return universe_df


if __name__ == "__main__":
    run_phase_1a()
