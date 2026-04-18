# Agent Handoff Summary

## Agent 1 (Complete)
Built full ML pipeline codebase from scratch. 24 files, 5 phases scaffolded.
Pushed to GitHub: https://github.com/Prakhar-Sethi/PledgeWatch (branch: master)

## Agent 2 (Complete — This Session)
Fixed broken scrapers, ran full pipeline, collected data. Ready for tone extraction.

---

## Current State (as of 2026-04-18 ~18:00)

### PROGRESS.md status: 13/18 steps done

| Phase | Status | Notes |
|-------|--------|-------|
| 1A BSE Pledging | ✓ DONE | 247/262 companies, NSE XBRL approach |
| 1B Price Fetcher | ✓ DONE | 243/262 companies via yfinance |
| 1C Transcripts | ✓ DONE | 1,135 transcripts, 262 companies |
| 1D Crash Labels | ✓ DONE | 9,885 rows, 1,648 positives |
| 2A Pledge Features | ✓ DONE | 9,885 rows |
| 2B Financial Features | ✓ DONE | 9,885 rows |
| 2C Tone Features | ✗ FAILED | Needs OPENROUTER_API_KEY |
| 2D Assemble Dataset | ✓ DONE | 0 tone features (2C skipped) |
| 3A Train Models | ✓ DONE | M1+M2 only (no tone data yet) |
| 3B Evaluate | ✓ DONE | M2 best AUC-ROC=0.6664 |
| 3C SHAP | ✓ DONE | 6 figures in paper/figures/ |
| 4A Paper | ✓ DONE | Placeholders filled |

### What needs to happen (IN ORDER):
1. Set `OPENROUTER_API_KEY` env var
2. Remove stale 2D/3A/3B/3C/4A/5 entries from PROGRESS.md so they re-run
3. Run `python3 run_all.py` → 2C runs (tone on 1135 transcripts), then 2D-3C auto-follow
4. Compile paper: `cd paper && pdflatex main.tex && pdflatex main.tex`
5. Commit + push to GitHub (branch: master)

---

## Key Fixes Made This Session

### Fix 1: BSE Scraper (bse_scraper.py) — COMPLETE REWRITE
- **Problem**: `api.bseindia.com/BseIndiaAPI/api/ShareHoldingPat/w` returns HTML not JSON. NSE `shareholding-patterns` API returns 404.
- **Fix**: NSE `corporate-share-holdings-master` API (1 call/company) + parallel XBRL download from `nsearchives.nseindia.com` for pledge %.
- **XBRL field**: `PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares` with `ShareholdingOfPromoterAndPromoterGroupI` context.

### Fix 2: Transcript Collector (transcript_collector.py) — COMPLETE REWRITE
- **Problem**: Screener.in `/concalls/` URL returns 404. BSE announcements API used wrong params + ISIN instead of scrip code.
- **Fix**: BSE `listofscripdata` API (1 call → 4866 NSE→BSE mappings). Then `AnnSubCategoryGetData` per company per year, filtering for "Earnings Call Transcript" entries. AttachHis→AttachLive fallback for PDF download.
- **Coverage**: 1,135 transcripts (2022+ only — SEBI mandate for transcript filing started 2022).

### Fix 3: XGBoost + SHAP compat (shap_analysis.py)
- **Problem**: XGBoost 3.x stores `base_score` as `[5E-1]` string → SHAP 0.49 parse fails.
- **Fix**: Downgraded xgboost to 2.1.4.

### Fix 4: pandas fillna(ndarray) (shap_analysis.py line 173)
- **Problem**: `results_df[col].fillna(vals)` where vals=ndarray → TypeError in newer pandas.
- **Fix**: Wrap ndarray as `pd.Series(vals, index=results_df.index)`.

### Fix 5: Tone extractor (tone_extractor.py) — OpenRouter support
- **Problem**: Uses `anthropic` SDK, needs `ANTHROPIC_API_KEY`.
- **Fix**: Swapped to `openai` SDK pointed at OpenRouter (`https://openrouter.ai/api/v1`). Reads `OPENROUTER_API_KEY` env var (falls back to `ANTHROPIC_API_KEY`).
- **Model in config.yaml**: `anthropic/claude-sonnet-4-5` (OpenRouter format).

---

## Key Data Facts
- Universe: 262 companies (hardcoded Midcap/Smallcap snapshot — NSE index API returns 0)
- Pledging data: 247 companies, 2015-2025, quarterly
- Price data: 243 companies, daily OHLCV
- Crash labels: 1,648 positives out of 9,885 rows (~16.7%)
- Transcripts: 1,135 files in `data/raw/transcripts/`, 2022-2024 coverage
- Tone cache: `data/raw/transcripts/tone_cache/` — already-processed transcripts cached here

## Current Model Results (M1+M2 only, no tone yet)
| Model | AUC-ROC | AUC-PR |
|-------|---------|--------|
| M1 (pledge only) | 0.518 | 0.086 |
| M2 (pledge+price+fin) | 0.666 | 0.084 |

## After 2C runs, expect:
- M3 (tone only), M4 (all features), M5 (no tone, same subset as M4) trained
- M4 vs M5 ablation = paper's core contribution
- SHAP re-runs on M4

---

## Repository
- Remote: https://github.com/Prakhar-Sethi/PledgeWatch
- Branch: **master** (NOT main)
- Git user: Anish (anishgupta198@gmail.com)

## File Structure
```
pledge_warning/         ← run from here
  run_all.py            ← orchestrator, reads PROGRESS.md, resumable
  config.yaml           ← model config, claude model ID
  PROGRESS.md           ← tracks done steps
  assumptions.log       ← assumption log
  pipeline.log          ← full run log
  src/
    scraping/           ← 1A-1D
    features/           ← 2A-2D
    modeling/           ← 3A-3C
  data/
    raw/bse_pledging/   ← 247 CSVs
    raw/prices/         ← 243 CSVs
    raw/transcripts/    ← 1135 JSONs + tone_cache/
    processed/          ← parquets, models/, results_table.csv
  paper/
    main.tex            ← IEEE paper
    figures/            ← 6 SHAP figures
```
