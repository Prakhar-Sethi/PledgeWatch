=== CURRENT STATUS ===
Last completed step: Phase 1C: Transcript Collector
Next step to run: (see log)
Overall progress: 13/18 steps done
Blocking issues: None
=====================

[DONE] 2026-04-18 00:00 | Phase 0: Project Scaffold | All source files, configs, LaTeX paper, run_all.py written
[DONE] 2026-04-18 16:24 | Phase 1A: BSE Pledging Scraper | Scraped 247/262 companies via NSE XBRL; 15 failed; first failures: ['ILFSENGG', 'ADANITRANS', 'IBULHSGFIN', 'JUBILANT', 'MOTHERSUMI', 'TATAMOTORS', 'CENTURYTEX', 'GARWARE', 'GLOBALHEALT', 'IBREALEST']
[DONE] 2026-04-18 16:29 | Phase 1B: NSE Price Fetcher | Downloaded prices for 243/262 companies; 19 failed
[DONE] 2026-04-18 16:41 | Phase 1D: Crash Event Labels | Labeled 10480 company-quarters; crash=1: 1707, crash=0: 8773; threshold=30%; 0 news-verified
[DONE] 2026-04-18 16:41 | Phase 2A: Pledge Features | Computed pledge features for 247 companies, 9885 rows
[DONE] 2026-04-18 16:45 | Phase 2B: Financial Features | Financial features for 247 companies, 9885 rows
[FAILED] 2026-04-18 16:45 | Phase 2C: Claude Tone Features | ANTHROPIC_API_KEY missing
[FAILED] 2026-04-18 17:01 | Phase 2C: Claude Tone Features | ANTHROPIC_API_KEY missing
[DONE] 2026-04-18 17:01 | Phase 2D: Assemble Dataset | Master dataset: 9885 rows, 1648 positives, 0 with tone features
[DONE] 2026-04-18 17:01 | Phase 3A: Train Models | Trained 2/5 models: ['M1_pledge_only', 'M2_pledge_price_fin']
[DONE] 2026-04-18 17:01 | Phase 3B: Evaluate Models | Evaluated 2 models; best AUC-ROC=0.6664
[DONE] 2026-04-18 17:01 | Phase 3C: SHAP Analysis | SHAP analysis complete; 6 figures saved to paper/figures/
[DONE] 2026-04-18 17:01 | Phase 4A: LaTeX Paper | Placeholders filled from actual results
[DONE] 2026-04-18 17:01 | Phase 5: Full Pipeline Orchestration | 12/12 phases done
[DONE] 2026-04-18 17:46 | Phase 1C: Transcript Collector | Collected 1135 transcripts for 262 companies (433.2% coverage); 1330 links found
