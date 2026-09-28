"""Logistic regression and LightGBM classifiers behind one small interface."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Spec:
    family: str                 # "logreg" | "lightgbm"
    windows: tuple[int, ...]
    with_odds: bool
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        odds = "odds" if self.with_odds else "form"
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}[{odds};N={'+'.join(map(str, self.windows))};{p}]"

    @property
    def label(self) -> str:
        return f"{'LogReg' if self.family == 'logreg' else 'LightGBM'} ({'form+odds' if self.with_odds else 'form only'})"


def fit(spec: Spec, features: list[str], train: pd.DataFrame, stop: pd.DataFrame | None,
        random_state: int, lgb_cfg: dict | None = None):
    X, y = train[features], train["target"].to_numpy(int)
    if spec.family == "logreg":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=spec.params["C"], max_iter=2000),
        )
        # LR has no early stopping; fold the stop season in so it sees the same data span.
        if stop is not None and len(stop):
            X = pd.concat([X, stop[features]])
            y = np.concatenate([y, stop["target"].to_numpy(int)])
        return model.fit(X, y)
    if spec.family == "lightgbm":
        lgb_cfg = lgb_cfg or {}
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=lgb_cfg.get("n_estimators", 2000),
            subsample=lgb_cfg.get("subsample", 0.8), subsample_freq=1,
            colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
            random_state=random_state, verbose=-1, **spec.params,
        )
        callbacks = [lgb.early_stopping(lgb_cfg.get("early_stopping_rounds", 100), verbose=False)]
        model.fit(X, y, eval_X=(stop[features],), eval_y=(stop["target"].to_numpy(int),),
                  eval_metric="binary_logloss", callbacks=callbacks)
        return model
    raise ValueError(spec.family)


def predict(model, features: list[str], df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


def candidate_specs(cfg: dict) -> list[Spec]:
    windows = cfg["features"]["windows"]
    window_sets = [(w,) for w in windows] + ([tuple(windows)] if len(windows) > 1 else [])
    specs = []
    for with_odds in (False, True):
        for ws in window_sets:
            for C in cfg["models"]["logreg"]["C_grid"]:
                specs.append(Spec("logreg", ws, with_odds, {"C": C}))
            for params in cfg["models"]["lightgbm"]["grid"]:
                specs.append(Spec("lightgbm", ws, with_odds, dict(params)))
    return specs
