"""Season-based, time-ordered splits. Never random."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    fit_seasons: tuple[str, ...]   # model is fitted here
    stop_season: str | None        # early stopping (LightGBM) — the last season before `eval`
    eval_season: str


def season_order(cfg: dict) -> list[str]:
    return list(cfg["seasons"]["history"])


def dev_seasons(cfg: dict) -> list[str]:
    """Seasons usable for model development: strictly before the calibration season."""
    order = season_order(cfg)
    return order[: order.index(cfg["splits"]["calibration"])]


def _fold(order: list[str], eval_season: str, name: str) -> Fold:
    prior = order[: order.index(eval_season)]
    return Fold(name=name, fit_seasons=tuple(prior[:-1]), stop_season=prior[-1], eval_season=eval_season)


def cv_folds(cfg: dict) -> list[Fold]:
    order = season_order(cfg)
    folds = [_fold(order, v, f"cv_{v}") for v in cfg["splits"]["cv_validation"]]
    forbidden = {cfg["splits"]["calibration"], cfg["splits"]["test"]}
    for f in folds:
        assert forbidden.isdisjoint({*f.fit_seasons, f.stop_season, f.eval_season}), f
    return folds


def calibration_fold(cfg: dict) -> Fold:
    """Model fitted before the calibration season, calibrator fitted on it."""
    f = _fold(season_order(cfg), cfg["splits"]["calibration"], "calibration")
    assert cfg["splits"]["test"] not in {*f.fit_seasons, f.stop_season}
    return f


def test_fold(cfg: dict) -> Fold:
    """Same model as calibration_fold; the test season is only ever predicted, never fitted."""
    f = calibration_fold(cfg)
    return Fold("test", f.fit_seasons, f.stop_season, cfg["splits"]["test"])


def live_fold(cfg: dict) -> Fold:
    """The recipe shifted forward one season: calibrate on the test season, fit on everything before."""
    return _fold(season_order(cfg), cfg["splits"]["test"], "live")


def rows(df: pd.DataFrame, seasons) -> pd.DataFrame:
    seasons = [seasons] if isinstance(seasons, str) else list(seasons)
    return df[df["season"].isin(seasons) & df["target"].notna()]
