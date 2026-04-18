=== CURRENT STATUS ===
Last completed step: Phase 5: Full Pipeline Orchestration
Next step to run: (see log)
Overall progress: 17/18 steps done
Blocking issues: None
=====================

[DONE] 2026-04-18 00:00 | Phase 0: Project Scaffold | All source files, configs, LaTeX paper, run_all.py written
[DONE] 2026-04-18 19:37 | Phase 1A: BSE Pledging Scraper | Scraped 247/262 companies via NSE XBRL; 15 failed
[DONE] 2026-04-18 19:43 | Phase 1B: NSE Price Fetcher | Downloaded prices for 243/262 companies; 19 failed
[DONE] 2026-04-18 17:46 | Phase 1C: Transcript Collector | Collected 973 transcripts for 262 companies
[DONE] 2026-04-18 16:41 | Phase 1D: Crash Event Labels | Labeled 10480 company-quarters; crash=1: 1707, crash=0: 8773; threshold=30%
[DONE] 2026-04-18 16:41 | Phase 2A: Pledge Features | Computed pledge features for 247 companies, 9885 rows
[DONE] 2026-04-18 16:45 | Phase 2B: Financial Features | Financial features for 247 companies, 9885 rows
[FAILED] 2026-04-18 21:03 | Phase 2C: Claude Tone Features | API key missing
[DONE] 2026-04-18 22:27 | Phase 2C: Claude Tone Features | Tone features for 961 transcripts; 750 API calls, 223 cache hits
[DONE] 2026-04-18 22:27 | Phase 2D: Assemble Dataset | Master dataset: 9885 rows, 1648 positives, 948 with tone features
[DONE] 2026-04-18 22:27 | Phase 2C: Claude Tone Features | Tone features for 955 transcripts; 607 API calls, 366 cache hits
[DONE] 2026-04-18 22:27 | Phase 2D: Assemble Dataset | Master dataset: 9885 rows, 1648 positives, 942 with tone features
[DONE] 2026-04-18 22:27 | Phase 3A: Train Models | Trained 5/5 models: ['M1_pledge_only', 'M2_pledge_price_fin', 'M3_tone_only', 'M4_full', 'M5_baseline_no_tone']
[DONE] 2026-04-18 22:27 | Phase 3B: Evaluate Models | Evaluated 5 models; best AUC-ROC=0.6550
[DONE] 2026-04-18 22:27 | Phase 3B: Evaluate Models | Evaluated 5 models; best AUC-ROC=0.6550
[DONE] 2026-04-18 22:27 | Phase 3C: SHAP Analysis | SHAP analysis complete; 6 figures saved to paper/figures/
[DONE] 2026-04-18 22:27 | Phase 4A: LaTeX Paper | Placeholders filled from actual results
[DONE] 2026-04-18 22:27 | Phase 5: Full Pipeline Orchestration | 12/12 phases done
