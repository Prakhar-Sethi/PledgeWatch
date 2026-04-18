# PledgeWatch

**Early warning system for Indian stock crashes driven by promoter share pledging.**

Predicts equity distress 3-6 months before it materialises, using only publicly available data. Combines BSE shareholding disclosures, NSE price history, and LLM-extracted tone from earnings call transcripts in a multi-modal XGBoost classifier with full SHAP explainability.

---

## Why This Exists

Promoter share pledging is a structural feature of Indian capital markets. When pledged collateral falls below loan-to-value thresholds, lenders trigger forced sales, accelerating the very price decline that caused the margin call. This reflexive mechanism drove some of the worst crashes in recent Indian market history: DHFL, Zee Entertainment, Reliance Capital, IL&FS Engineering, Yes Bank.

SEBI mandates quarterly disclosure of pledged shares, yet no accessible, predictive tool existed to act on this data systematically. PledgeWatch fills that gap.

---

## Architecture

```
pledge_warning/
├── data/
│   ├── raw/
│   │   ├── bse_pledging/          # Per-company quarterly shareholding CSVs
│   │   ├── nse_prices/            # Daily OHLCV + return features
│   │   └── transcripts/
│   │       └── tone_cache/        # Cached LLM responses (no re-calls)
│   ├── processed/                 # Feature parquets, model artifacts
│   └── labels/
│       └── crash_events.csv       # Labeled company-quarters
├── src/
│   ├── scraping/
│   │   ├── bse_scraper.py         # BSE shareholding scraper + NSE fallback
│   │   ├── nse_price_fetcher.py   # yfinance + nsepython fallback
│   │   └── transcript_collector.py# Screener.in + BSE API, Q&A/CEO split
│   ├── features/
│   │   ├── pledge_features.py     # Delta, acceleration, momentum features
│   │   ├── financial_features.py  # Price + Screener.in ratio features
│   │   ├── tone_extractor.py      # LLM tone extraction with disk cache
│   │   └── assemble_dataset.py    # Master dataset assembly + temporal split
│   ├── modeling/
│   │   ├── train.py               # M1-M5 XGBoost ablation training
│   │   ├── evaluate.py            # AUC-ROC/PR + bootstrap CIs
│   │   └── shap_analysis.py       # SHAP global, beeswarm, waterfall plots
│   └── utils/
│       ├── config.py              # Centralised paths and config loading
│       └── helpers.py             # Logging, progress tracking, retries
├── paper/
│   ├── main.tex                   # Full IEEE conference paper (IEEEtran)
│   └── figures/                   # Auto-generated SHAP and timeseries plots
├── config.yaml
├── requirements.txt
├── run_all.py                     # Resumable end-to-end pipeline
└── PROGRESS.md                    # Live step-by-step execution log
```

---

## Data Sources

| Source | What | How |
|---|---|---|
| BSE Corporate Portal | Quarterly promoter shareholding + pledge counts | `requests` + `BeautifulSoup`, NSE bulk CSV fallback |
| NSE / yfinance | Daily OHLCV price history 2014-2024 | `yfinance (.NS)`, `nsepython` fallback |
| Screener.in | Earnings call transcript PDFs + financial ratios | HTTP scraper, `PyMuPDF` for PDF text extraction |
| BSE Announcements API | Transcript PDF fallback | JSON API filtered by announcement type |
| Claude API | Tone feature extraction from CEO speech | `anthropic` SDK, 9 structured scores per transcript |

---

## Feature Groups

### Pledge Features (9)
| Feature | Description |
|---|---|
| `pledge_pct_promoter` | % of promoter shares pledged |
| `pledge_pct_total` | % of total shares pledged |
| `pledge_change_1q` | Quarter-on-quarter delta |
| `pledge_change_2q` | Two-quarter delta |
| `pledge_acceleration` | Change-in-change (second derivative) |
| `pledge_high_flag` | Binary: pledge > 50% |
| `pledge_rising_3q` | Binary: rising 3 consecutive quarters |
| `pledge_pct_4q_max` | Rolling 4-quarter maximum |
| `pledge_pct_vs_4q_max` | Distance from rolling max |

### Price + Financial Features (11)
`price_return_3m`, `price_return_6m`, `price_volatility_60d`, `price_vs_52w_high`, `volume_spike_flag`, `market_cap_log`, `debt_to_equity`, `interest_coverage`, `current_ratio`, `roe_ttm`, `revenue_growth_yoy`

### LLM Tone Features (9)
Extracted from CEO/MD Q&A speech turns via Claude API:

| Score | What it measures |
|---|---|
| `evasiveness_score` | Deflection and non-answers under analyst questioning |
| `confidence_score` | Forward-looking positive language |
| `debt_mention_count` | Frequency of debt/pledging/borrowing references |
| `reassurance_score` | Over-reassurance language ("everything is fine") |
| `analyst_tension_score` | Hostility or defensiveness in Q&A responses |
| `guidance_vagueness_score` | Vagueness about future plans or guidance |
| `liquidity_stress_mentions` | Cash flow / liquidity / working capital references |
| `tone_shift_flag` | Q&A tone noticeably different from prepared statement |
| `overall_distress_score` | Holistic financial distress signal (0-10) |

---

## Model Variants

| Model | Features | Dataset |
|---|---|---|
| M1 | Pledge only | Full universe |
| M2 | Pledge + Price + Financial | Full universe |
| M3 | Tone only | Transcript-available subset |
| M4 | All features (full model) | Transcript-available subset |
| M5 | M2 features, same subset as M4 | Transcript-available subset |

M4 vs M5 is the critical ablation: identical data, identical XGBoost config, only difference is presence of tone features. This isolates the incremental predictive value of LLM tone extraction.

**XGBoost config:** `n_estimators=500`, `max_depth=4`, `lr=0.05`, `subsample=0.8`, `colsample=0.8`, early stopping on AUC-PR (patience=50), `scale_pos_weight` auto-set from class ratio.

---

## Crash Labels

A company-quarter is labeled as a crash event if:

```
max_drawdown(t, t+180d) >= 30%
```

Ten seed cases are always labeled positive regardless of drawdown magnitude (DHFL 2019Q3, ZEEL 2019Q3, RELCAPITAL 2019Q3, YESBANK 2020Q1, FRETAIL 2020Q3, ZEEMEDIA 2019Q3, ILFSENGG 2018Q4, ADANIENT 2023Q1, RCOM 2019Q2, JETAIRWAYS 2019Q1). If the positive class remains below 20 events after full collection, threshold auto-lowers to 25%.

**Temporal split:** Train 2015Q1-2020Q4 | Val 2021Q1-2022Q4 | Test 2023Q1-2024Q4. No look-ahead leakage.

---

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Set Claude API key (required for tone extraction only)
export ANTHROPIC_API_KEY=sk-ant-...

# Run full pipeline (resumable at any point)
cd pledge_warning
python run_all.py
```

The pipeline is fully resumable. `PROGRESS.md` tracks every completed step. Re-running skips completed steps automatically. If a phase fails, the pipeline continues with available data and logs the failure.

Individual phases can be run standalone:

```bash
python -m src.scraping.bse_scraper
python -m src.scraping.nse_price_fetcher
python -m src.scraping.transcript_collector
python -m src.scraping.build_crash_labels
python -m src.features.pledge_features
python -m src.features.financial_features
python -m src.features.tone_extractor
python -m src.features.assemble_dataset
python -m src.modeling.train
python -m src.modeling.evaluate
python -m src.modeling.shap_analysis
```

---

## Outputs

| Output | Path |
|---|---|
| Master dataset | `data/processed/master_dataset.parquet` |
| Model artifacts | `data/processed/models/*.pkl` |
| Evaluation results | `data/processed/results_table.csv` |
| SHAP importance | `data/processed/shap_importance.csv` |
| Figures | `paper/figures/*.png` |
| Research paper | `paper/main.tex` |
| Execution log | `pipeline.log` |
| Assumptions | `assumptions.log` |

---

## Research Paper

`paper/main.tex` is a complete IEEE two-column conference paper targeting DSAA 2025 Application Track. All result placeholders (`\INPUT{...}`) are filled automatically by `run_all.py` from actual evaluation outputs. Figures are generated by `shap_analysis.py` and referenced directly from `paper/figures/`.

---

## Requirements

Python 3.10+. Key dependencies:

```
xgboost>=2.0.0        scikit-learn>=1.4.0    shap>=0.44.0
pandas>=2.0.0         numpy>=1.26.0          pyarrow>=14.0.0
anthropic>=0.34.0     yfinance>=0.2.36       PyMuPDF>=1.23.0
requests>=2.31.0      beautifulsoup4>=4.12.0 matplotlib>=3.8.0
nsepython>=2.0        seaborn>=0.13.0        plotly>=5.18.0
```

---

## Limitations

- Positive class is small in absolute terms (historically significant but rare events)
- Transcript coverage is incomplete for smaller-cap companies
- Threshold-based labeling may mislabel distress events that were subsequently reversed
- LLM tone extraction quality degrades for Hindi-English code-switched earnings calls

---

## License

MIT
