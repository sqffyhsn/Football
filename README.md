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

Coming after the backtest review: `evaluate_test.py`, which evaluates the held-out season once, followed by `train_live_model.py`, `predict_upcoming.py` and `settle_results.py`.

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
