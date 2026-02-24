"""Configuration loading with YAML and .env support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _project_root() -> Path:
    """Resolve project root (parent of src)."""
    return Path(__file__).resolve().parents[3]


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load YAML config with path resolution from project root."""
    p = Path(path) if not Path(path).is_absolute() else Path(path)
    if not p.is_absolute():
        p = _project_root() / "configs" / p.name
    with open(p) as f:
        return yaml.safe_load(f) or {}


def env(key: str, default: str | None = None) -> str | None:
    """Get env var with optional default."""
    return os.environ.get(key, default)


def get_paths() -> PathsConfig:
    """Return resolved paths config."""
    root = _project_root()
    raw = Path(env("DATA_RAW_DIR") or str(root / "data" / "raw"))
    processed = Path(env("DATA_PROCESSED_DIR") or str(root / "data" / "processed"))
    return PathsConfig(raw=raw, processed=processed)


class PathsConfig(BaseModel):
    """Resolved data paths."""

    raw: Path = Field(default_factory=lambda: Path("data/raw"))
    processed: Path = Field(default_factory=lambda: Path("data/processed"))
