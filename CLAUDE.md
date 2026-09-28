# CLAUDE.md — project conventions

Predicts calibrated P(≥1 first-half goal) for football matches. Data comes from football-data.co.uk.

## Layout
- `src/fh/data/`: downloader (`download.py`), cleaner (`clean.py`, tidy schema `TIDY_COLUMNS`), fixture providers (`providers.py`), and a synthetic generator used only for tests and smoke runs.
- `src/fh/features/`: target, team form, league base rates, odds. `build.py` assembles everything.
- `src/fh/models/`: splits, baselines, LR/LightGBM training, calibration, registry, and `pipeline.py` for the CV and bundle fitting.
- `src/fh/evaluation/`: metrics, plots, betting sim, and report tables.
- `scripts/`: entry points. `config.yaml` holds all leagues, seasons, splits and params.

## Hard rules
- **No leakage.** Every history lookup uses post-match state joined with `merge_asof(..., allow_exact_matches=False)` on **date**. Same-day matches never see each other. Any new feature must go through this pattern and must keep `tests/test_leakage.py` passing. Add a perturbation case for the new feature.
- **Closing odds (`close_*`) are never features.** They are for CLV analysis only. Features use pre-closing `odds_*` only.
- **Splits are by season, never random.** CV uses `splits.cv_validation`, calibration uses `splits.calibration`, and the test season is `splits.test`.
- **The test season is evaluated once**, by `scripts/evaluate_test.py`, and only after the user has reviewed the backtest. `scripts/backtest.py` drops it before modelling. Never tune anything on test results.
- **No real FH O/U 0.5 odds exist in football-data.** The market benchmark is *derived* from O/U 2.5. Betting results on synthetic odds are labelled HYPOTHETICAL. Never pool REAL and HYPOTHETICAL bets in one ROI.
- **Report honestly.** Verdicts in the reports are computed from bootstrap CIs. Don't rephrase them more favourably.
- **Synthetic data** (`fh.data.synthetic`) is for tests and `--synthetic` smoke runs only. Its output goes to the gitignored `reports/_scratch/`.

## Conventions
- Seasons are football-data codes (`"2425"`). League codes are football-data `Div` codes. Toggle leagues with `enabled` in config.yaml.
- Kick-off timezone is set in config (`data.kickoff_timezone`). It must be verified with `scripts/verify_timezone.py`. Until `kickoff_timezone_verified: true`, treat it as an assumption.
- **Retraining cadence:** retrain and recalibrate once per season, after it ends. To do that, shift `splits` / `seasons` forward by one. Rolling features and base rates refresh on every predict run.
- API keys only come from `.env` (gitignored). See `.env.example`.
- Dependencies are pinned in `pyproject.toml`. Use Python 3.11+ and a venv at `.venv`.
- Run tests with `pytest -q`. Commit in small logical steps.
