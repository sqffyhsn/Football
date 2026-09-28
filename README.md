# Football — first-half over 0.5 goals

This project gives calibrated probabilities that a match has **at least one first-half goal** (`HTHG + HTAG ≥ 1`). It has three parts:

- a walk-forward backtest with no data leakage
- honest comparison against baselines and a bookmaker-derived benchmark
- forward testing ("paper trading"), which is still in progress

## Setup
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # only needed for future API providers
pytest -q
```

## Pipeline
```bash
python scripts/download_data.py      # football-data.co.uk CSVs -> data/raw/ (cached)
python scripts/build_dataset.py      # -> data/processed/matches.parquet + reports/cleaning_report.json
python scripts/verify_timezone.py    # checks the kick-off timezone of the data -> reports/timezone_check.md
python scripts/backtest.py           # walk-forward CV + calibration -> reports/backtest/
```
`python scripts/backtest.py --synthetic` runs the whole pipeline on synthetic data. It writes to `reports/_scratch/`, and those numbers are meaningless.

```bash
python scripts/evaluate_test.py      # ONE-TIME held-out season evaluation -> reports/test/ (already run; refuses to re-run)
```

## Results so far
- **Backtest** (walk-forward CV, 2020/21–2023/24, 20,887 matches), in `reports/backtest/`:
  - every model beats the baselines
  - the primary model, LogReg (form+odds), beats the market-derived benchmark by Δ log loss −0.00063, with 95% CI [−0.00116, −0.00012]. That figure is slightly optimistic because of model selection.
- **Test season 2025/26** (5,101 matches, evaluated once), in `reports/test/`: Δ = +0.00145, with 95% CI [−0.00059, +0.00396]. The CI spans 0. The pre-written interpretation reads: *"Cannot confirm backtest edge. Forward testing is the tiebreaker. Do not adjust the model."*

## Forward testing (paper trading)
```bash
python scripts/train_live_model.py   # once per season: refit the same specs, calibrate on the last full season
python scripts/predict_upcoming.py   # before each round: log predictions for fixtures that haven't kicked off
python scripts/settle_results.py     # after matches: fill results + closing odds, refresh reports/live/
```
- **Model version:** `predictions/models_live_manifest.json` records the live model version and the backtest and test reference deltas.
- **`predictions/log.csv` is committed.**
  - The first prediction for a fixture is final and is never overwritten.
  - Fixtures that have already kicked off are never added.
  - Kick-off times in football-data.co.uk are UK time. This was verified with `scripts/verify_timezone.py` and converted to UTC.
- **Real first-half odds (optional):**
  - Set `ODDS_PROVIDER=manual_csv` in `.env`.
  - Add rows to `predictions/manual_fh_odds.csv`: `div,date,home,away,fh_o05_odds,fh_u05_odds,source,ts_utc`, plus `fh_o05_close_odds,fh_u05_close_odds` for REAL CLV. `date` is YYYY-MM-DD and team names use football-data spelling.
  - Rows are validated when loaded. Odds must be > 1, 1/over + 1/under must fall in [1.00, 1.20], and `source` and `ts_utc` are required. Any rejected row is printed with its reason.
  - Pre-match odds stamped at or after kick-off are rejected.
- **Live report** (`reports/live/live_report.md`):
  - sample size and per-league tables
  - metrics with bootstrap CIs
  - CLV, in two separate sections: DERIVED/INDIRECT, and REAL when real FH odds exist
  - betting, with REAL and HYPOTHETICAL results reported separately and never pooled
  - a power calculation
- **Convergence note** (top of the live report, plus `convergence.png` and `convergence.csv`): tracks the running log-loss difference of the primary model against the market benchmark.
  - Below 200 settled matches it shows the running Δ only.
  - From 200 matches it adds a 95% CI band and a status line: **HOLDING** (CI below 0), **INCONCLUSIVE** (CI covers both 0 and the backtest Δ), or **INCONSISTENT WITH BACKTEST** (CI lies entirely above the backtest Δ).
  - It also shows how many matches are still needed. Resolving an edge the size of the backtest's at 80% power takes about 29,500 settled matches, roughly six seasons across all 14 leagues. Expect "INCONCLUSIVE" for a long time.
  - `convergence_history.csv` keeps one row per settle run.

## Configuration
Everything is in `config.yaml`:

- **Leagues:** 14 leagues, each toggled with `enabled`.
- **Seasons:** 2016/17–2025/26.
- **Splits:** CV validation seasons 2020/21–2023/24, calibration season 2024/25, test season 2025/26.
- **Other settings:** feature windows (5, 10), model grids, betting margin and thresholds.

## Method
- **Features, all strictly pre-match:**
  - rolling team stats over the last 5/10 matches, both overall and venue-specific: FH/FT goals for/against and FH-goal rate
  - days since the last match (league matches only, so approximate)
  - expanding league and season base rates
  - de-vigged 1X2 and O/U 2.5 probabilities, overround, and the implied goal expectancy λ

  Each match only sees data from **strictly earlier dates**. `tests/test_leakage.py` enforces this: it perturbs a match's result and all later results and odds, and checks that the match's features are unchanged.
- **Baselines:**
  - A: the league's historical rate
  - B: an FH Poisson model built from team attack/defence rates
  - a **market-derived** benchmark: p = 1 − exp(−s·λ), with λ from O/U 2.5 odds and s fitted on training data
- **Models:** logistic regression and LightGBM, each with and without odds features. They are selected by mean walk-forward CV log loss. Calibration (none, Platt or isotonic) is chosen on a time-ordered split of the calibration season.
- **Evaluation:**
  - log loss, Brier, ROC AUC, ECE and reliability curves
  - date-block bootstrap CIs on the log-loss differences, with verdicts computed from those CIs
  - per-league tables

## Interpreting results
- **Lower log loss and Brier are better.** For a base rate near 72%, always predicting the base rate gives a log loss of about 0.59. Real improvements are small, typically in the third decimal place.
- **Read the confidence interval.** A Δ whose 95% CI spans 0 is *not* evidence of improvement.
- **The market-derived benchmark is strong.** Failing to beat it is the expected outcome, and the report says so plainly when it happens.
- **The betting simulation is HYPOTHETICAL.** football-data.co.uk has no first-half O/U 0.5 odds, so bets are priced at synthetic odds derived from the market benchmark plus an assumed margin. Treat it as informational only.

## Retraining cadence
- **Once per season, after it ends:** retrain and recalibrate by shifting the season/split config forward by one season.
- **Every prediction run:** rolling team features and base rates refresh from the newly downloaded results.
