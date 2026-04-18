import logging
import time
import json
import functools
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
PIPELINE_LOG = ROOT / "pipeline.log"
PROGRESS_MD = ROOT / "PROGRESS.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(PIPELINE_LOG),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("pledge_warning")


def log_assumption(text: str):
    assumptions_path = ROOT / "assumptions.log"
    ts = datetime.now().strftime("%Y-%m-%d")
    with open(assumptions_path, "a") as f:
        f.write(f"[{ts}] {text}\n")


def update_progress(status: str, phase: str, note: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"[{status}] {ts} | {phase} | {note}\n"

    lines = []
    if PROGRESS_MD.exists():
        lines = PROGRESS_MD.read_text().splitlines(keepends=True)

    # Count done steps
    done_count = sum(1 for l in lines if l.startswith("[DONE]"))
    if status == "DONE":
        done_count += 1

    # Rebuild status block
    new_status = (
        "=== CURRENT STATUS ===\n"
        f"Last completed step: {phase}\n"
        f"Next step to run: (see log)\n"
        f"Overall progress: {done_count}/18 steps done\n"
        f"Blocking issues: None\n"
        "=====================\n\n"
    )

    # Strip old status block
    content = "".join(lines)
    import re
    content = re.sub(r"=== CURRENT STATUS ===.*?=====================\n\n", "", content, flags=re.DOTALL)

    PROGRESS_MD.write_text(new_status + content + entry)
    logger.info(f"PROGRESS {status}: {phase} | {note}")


def retry_request(func, retries=3, backoff=1.5):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = backoff ** (attempt + 1)
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
    return wrapper


def quarter_from_date(date_str: str) -> str:
    from pandas import Timestamp
    ts = Timestamp(date_str)
    q = (ts.month - 1) // 3 + 1
    return f"{ts.year}Q{q}"


def date_range_quarters(start: str, end: str):
    import pandas as pd
    periods = pd.period_range(start=start, end=end, freq="Q")
    return [str(p).replace("-", "Q").replace("Q0", "Q") for p in periods]
