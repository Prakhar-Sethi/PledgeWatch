import yaml
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

cfg = load_config()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
BSE_DIR = RAW_DIR / "bse_pledging"
NSE_PRICES_DIR = RAW_DIR / "nse_prices"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
TONE_CACHE_DIR = TRANSCRIPTS_DIR / "tone_cache"
MODELS_DIR = PROCESSED_DIR / "models"
PAPER_DIR = ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"

for d in [BSE_DIR, NSE_PRICES_DIR, TRANSCRIPTS_DIR, TONE_CACHE_DIR,
          PROCESSED_DIR, LABELS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_DATE = cfg["data"]["start_date"]
END_DATE = cfg["data"]["end_date"]
TRAIN_END = cfg["data"]["train_end"]
VAL_END = cfg["data"]["val_end"]

CRASH_THRESHOLD = cfg["model"]["crash_drawdown_threshold"]
MIN_POSITIVE = cfg["model"]["min_positive_class_size"]
RANDOM_SEED = cfg["model"]["random_seed"]

CLAUDE_MODEL = cfg["claude"]["model"]
MAX_TRANSCRIPT_WORDS = cfg["claude"]["max_transcript_words"]
