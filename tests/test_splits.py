import pandas as pd

from fh.config import load_config
from fh.models.splits import calibration_fold, cv_folds, live_fold
from fh.models.splits import test_fold as final_test_fold


def test_folds_are_time_ordered_and_never_touch_test():
    cfg = load_config()
    order = cfg["seasons"]["history"]
    test, cal = cfg["splits"]["test"], cfg["splits"]["calibration"]
    for f in cv_folds(cfg):
        used = [*f.fit_seasons, f.stop_season]
        assert max(order.index(s) for s in used) < order.index(f.eval_season)
        assert test not in used + [f.eval_season] and cal not in used + [f.eval_season]
    c = calibration_fold(cfg)
    assert c.eval_season == cal and test not in c.fit_seasons and c.stop_season != test
    t = final_test_fold(cfg)
    assert t.eval_season == test and test not in t.fit_seasons and t.stop_season != test
    live = live_fold(cfg)
    assert live.eval_season == test and order.index(live.stop_season) == order.index(test) - 1


def test_backtest_guard_drops_test_season():
    cfg = load_config()
    order = cfg["seasons"]["history"]
    allowed = order[: order.index(cfg["splits"]["test"])]
    assert cfg["splits"]["test"] not in allowed and cfg["seasons"]["current"] not in allowed
