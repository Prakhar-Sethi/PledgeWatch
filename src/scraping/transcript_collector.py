"""
Phase 1C: Earnings Call Transcript Collector
Sources: Screener.in concalls -> BSE announcements API fallback.
Splits transcript into prepared_statement + qa_ceo_text.
"""

import sys
import re
import json
import time
import logging
import requests
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import TRANSCRIPTS_DIR, PROCESSED_DIR
from src.utils.helpers import logger, update_progress, log_assumption

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

QA_KEYWORDS = re.compile(
    r"(?:question[s]?[\s\&]*answer[s]?|q\s*&\s*a|q&a|open\s+the\s+floor|queries|"
    r"analyst[\s]+questions?|investor[\s]+questions?|we\s+will\s+now\s+begin|"
    r"operator:|we\s+now\s+open|floor\s+open|questions\s+from)",
    re.IGNORECASE
)

CEO_SPEAKER = re.compile(
    r"^(MD|CEO|Chairman|Managing\s+Director|Chief\s+Executive|CFO|"
    r"Executive\s+Director|Promoter|Management|[A-Z][a-z]+\s+[A-Z][a-z]+)\s*:",
    re.MULTILINE
)


def split_transcript(text: str) -> tuple[str, str, str]:
    """Return (prepared_text, qa_section, qa_ceo_text)."""
    lines = text.split("\n")
    qa_start_idx = None

    for i, line in enumerate(lines):
        if QA_KEYWORDS.search(line):
            # Verify it looks like a section header (short line or all-caps)
            stripped = line.strip()
            if len(stripped) < 120 or stripped.isupper():
                qa_start_idx = i
                break

    if qa_start_idx is None:
        return text, "", ""

    prepared = "\n".join(lines[:qa_start_idx])
    qa_section = "\n".join(lines[qa_start_idx:])

    # Extract CEO/MD turns from Q&A
    qa_ceo_parts = []
    current_speaker = None
    current_lines = []

    for line in lines[qa_start_idx:]:
        speaker_match = CEO_SPEAKER.match(line.strip())
        if speaker_match:
            if current_speaker and current_lines:
                qa_ceo_parts.append("\n".join(current_lines))
            current_speaker = speaker_match.group(0)
            current_lines = [line]
        elif current_speaker:
            # Check if this looks like a new non-CEO speaker (analyst question)
            analyst_match = re.match(r"^(Analyst|Investor|Moderator|Operator)\s*:", line.strip(), re.IGNORECASE)
            if analyst_match:
                if current_lines:
                    qa_ceo_parts.append("\n".join(current_lines))
                current_speaker = None
                current_lines = []
            else:
                current_lines.append(line)

    if current_lines and current_speaker:
        qa_ceo_parts.append("\n".join(current_lines))

    qa_ceo_text = "\n\n".join(qa_ceo_parts) if qa_ceo_parts else qa_section

    return prepared, qa_section, qa_ceo_text


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""


def download_pdf(url: str, session: requests.Session) -> bytes:
    """Try AttachHis first, fallback to AttachLive."""
    urls_to_try = [url]
    if "AttachHis" in url:
        urls_to_try.append(url.replace("AttachHis", "AttachLive"))
    elif "AttachLive" in url:
        urls_to_try.append(url.replace("AttachLive", "AttachHis"))

    for u in urls_to_try:
        try:
            r = session.get(u, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                return r.content
        except Exception as e:
            logger.debug(f"PDF download failed {u}: {e}")
    return b""


def scrape_bse_transcripts(symbol: str, bse_code: str, session: requests.Session) -> list[dict]:
    """Fetch transcript PDFs from BSE announcements API using scrip code.
    Paginates year-by-year (API returns max 50 items). Tries AttachHis then AttachLive for PDFs.
    Coverage: 2022+ (SEBI mandate for transcript filing started ~2022).
    """
    results = []
    if not bse_code:
        return results

    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    seen_pdfs: set = set()

    for year in range(2015, 2025):
        params = {
            "strCat": "-1",
            "strPrevDate": f"{year}0101",
            "strScrip": bse_code,
            "strSearch": "P",
            "strToDate": f"{year}1231",
            "strType": "C",
            "subcategory": "-1",
        }
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code != 200:
                time.sleep(0.3)
                continue
            items = r.json().get("Table", [])
            for ann in items:
                sub = str(ann.get("NEWSSUB", "")).lower()
                if "transcript" not in sub and "earnings call" not in sub and "concall" not in sub:
                    continue
                pdf_name = ann.get("ATTACHMENTNAME", "")
                if not pdf_name or pdf_name in seen_pdfs:
                    continue
                seen_pdfs.add(pdf_name)
                # AttachHis for older; AttachLive for recent — try both in download
                pdf_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{pdf_name}"
                quarter = _extract_quarter_from_text(
                    str(ann.get("DT_TM", "")) + " " + str(ann.get("NEWSSUB", ""))
                )
                results.append({"url": pdf_url, "quarter": quarter, "source": "bse", "pdf_name": pdf_name})
            time.sleep(0.3)
        except Exception as e:
            logger.debug(f"BSE transcript API {symbol} {year}: {e}")

    return results


def _extract_quarter_from_text(text: str) -> str:
    """Extract YYYYQN from text containing dates or quarter mentions."""
    text = str(text)

    # Q1 FY2020 or Q1FY20 format
    m = re.search(r"[Qq]([1-4])\s*[Ff][Yy][\s]?(\d{2,4})", text)
    if m:
        q = m.group(1)
        yr_raw = m.group(2)
        yr = int(yr_raw) if len(yr_raw) == 4 else 2000 + int(yr_raw)
        # Indian FY: Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar
        # FY2020 Q1 = April 2019 to June 2019 -> calendar Q2 2019
        cal_yr = yr - 1 if int(q) == 4 else yr - 1
        cal_q_map = {"1": 2, "2": 3, "3": 4, "4": 1}
        cal_yr_map = {"1": yr - 1, "2": yr - 1, "3": yr - 1, "4": yr}
        return f"{cal_yr_map[q]}Q{cal_q_map[q]}"

    # Month Year pattern
    month_map = {
        "jan": (1, 1), "feb": (1, 1), "mar": (1, 1),
        "apr": (2, 0), "may": (2, 0), "jun": (2, 0),
        "jul": (3, 0), "aug": (3, 0), "sep": (3, 0),
        "oct": (4, 0), "nov": (4, 0), "dec": (4, 0),
    }
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+(\d{4})", text, re.IGNORECASE)
    if m:
        mon = m.group(1).lower()[:3]
        yr = int(m.group(2))
        q, offset = month_map.get(mon, (1, 0))
        return f"{yr + offset}Q{q}"

    # YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-\d{2}", text)
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        q = (mo - 1) // 3 + 1
        return f"{yr}Q{q}"

    return "UNKNOWN"


def run_phase_1c(universe_df: pd.DataFrame = None):
    logger.info("=== PHASE 1C: Transcript Collector ===")

    if universe_df is None:
        universe_path = PROCESSED_DIR / "universe.csv"
        if universe_path.exists():
            universe_df = pd.read_csv(universe_path)
        else:
            logger.error("No universe.csv. Run Phase 1A first.")
            return

    # BSE session (needs bseindia.com cookies for API to return JSON)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
    })
    session.get("https://www.bseindia.com", timeout=10)
    time.sleep(1.5)

    # Build BSE scrip code map via NSE quote API
    bse_code_map: dict = {}
    bse_code_cache = PROCESSED_DIR / "bse_code_map.json"
    if bse_code_cache.exists():
        import json as _json
        bse_code_map = _json.loads(bse_code_cache.read_text())
        logger.info(f"Loaded BSE code map: {len(bse_code_map)} entries")
    else:
        # Build BSE scrip code map in ONE call via BSE's full equity list
        bse_session = requests.Session()
        bse_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.bseindia.com/",
        })
        bse_session.get("https://www.bseindia.com", timeout=10)
        time.sleep(1.5)
        try:
            r = bse_session.get(
                "https://api.bseindia.com/BseIndiaAPI/api/listofscripdata/w"
                "?Group=&Scripcode=&industry=&segment=Equity&status=Active",
                timeout=20
            )
            if r.status_code == 200:
                for item in r.json():
                    nse_sym = str(item.get("scrip_id", "")).strip()
                    bse_code_val = str(item.get("SCRIP_CD", "")).strip()
                    if nse_sym and bse_code_val:
                        bse_code_map[nse_sym] = bse_code_val
        except Exception as e:
            logger.warning(f"BSE list API failed: {e}")
        import json as _json
        bse_code_cache.write_text(_json.dumps(bse_code_map))
        logger.info(f"Built BSE code map: {len(bse_code_map)} entries")

    collected = 0
    total_links = 0

    for _, row in universe_df.iterrows():
        symbol = row["nse_symbol"]
        bse_code = bse_code_map.get(symbol, "")

        # Check existing transcripts
        existing = list(TRANSCRIPTS_DIR.glob(f"{symbol}_*.json"))

        # Collect PDF links via BSE announcements API
        links = scrape_bse_transcripts(symbol, bse_code, session)
        time.sleep(0.5)

        total_links += len(links)

        for link_info in links:
            pdf_url = link_info["url"]
            quarter = link_info["quarter"]
            out_path = TRANSCRIPTS_DIR / f"{symbol}_{quarter}.json"

            if out_path.exists():
                collected += 1
                continue

            pdf_bytes = download_pdf(pdf_url, session)
            if not pdf_bytes:
                continue

            text = extract_text_from_pdf(pdf_bytes)
            if len(text) < 500:
                continue

            prepared, qa_section, qa_ceo_text = split_transcript(text)

            record = {
                "company": str(row.get("company_name", symbol)),
                "symbol": symbol,
                "quarter": quarter,
                "prepared_text": prepared,
                "qa_section": qa_section,
                "qa_ceo_text": qa_ceo_text,
                "full_text": text,
                "pdf_url": pdf_url,
                "source": link_info["source"],
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            collected += 1
            logger.info(f"Saved transcript: {symbol} {quarter}")
            time.sleep(0.5)

    coverage = collected / max(len(universe_df), 1) * 100
    note = f"Collected {collected} transcripts for {len(universe_df)} companies ({coverage:.1f}% coverage); {total_links} links found"

    if coverage < 40:
        log_assumption(f"Transcript coverage only {coverage:.1f}%; continuing with available data per project rules.")

    update_progress("DONE", "Phase 1C: Transcript Collector", note)
    logger.info(f"Phase 1C complete. {note}")


if __name__ == "__main__":
    run_phase_1c()
