"""Post-hoc probability calibration on a held-out, time-ordered slice."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fh.evaluation.metrics import log_loss

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str):
        assert method in ("none", "platt", "isotonic")
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            self.m_ = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.m_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.m_.predict(p)
        return np.asarray(p, dtype=float)


def choose_calibrator(p: np.ndarray, y: np.ndarray, order: np.ndarray) -> tuple[str, dict]:
    """Pick none/platt/isotonic by fitting on the first half of the (time-ordered)
    calibration slice and scoring on the second half. Test data is never involved."""
    idx = np.argsort(order, kind="mergesort")
    half = len(idx) // 2
    a, b = idx[:half], idx[half:]
    scores = {m: log_loss(y[b], Calibrator(m).fit(p[a], y[a]).transform(p[b]))
              for m in ("none", "platt", "isotonic")}
    return min(scores, key=scores.get), scores
