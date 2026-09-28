"""Persist a fitted model bundle (model + calibrator + metadata)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from fh.models.train import Spec


@dataclass
class Bundle:
    spec: Spec
    features: list[str]
    model: object
    calibrator: object
    market: object                       # fitted MarketDerived benchmark
    meta: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.meta["version"]


def new_version(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def save(bundle: Bundle, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{bundle.version}.joblib"
    joblib.dump(bundle, p)
    meta = {**bundle.meta, "spec": asdict(bundle.spec), "features": bundle.features}
    (directory / f"{bundle.version}.json").write_text(json.dumps(meta, indent=2, default=str))
    (directory / "LATEST").write_text(bundle.version)
    return p


def load(directory: Path, version: str | None = None) -> Bundle:
    version = version or (directory / "LATEST").read_text().strip()
    return joblib.load(directory / f"{version}.joblib")
