"""Load config.yaml and .env. All paths in the config are relative to the repo root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def enabled_leagues(cfg: dict[str, Any]) -> list[str]:
    return [code for code, spec in cfg["leagues"].items() if spec.get("enabled", True)]


def path(rel: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def env(name: str, default: str | None = None) -> str | None:
    load_dotenv(ROOT / ".env")
    return os.environ.get(name, default)
