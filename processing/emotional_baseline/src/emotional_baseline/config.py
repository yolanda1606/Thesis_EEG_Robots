from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE.parents[1]
CONFIG_DIR = WORKSPACE / "configs"
OUTPUT_DIR = WORKSPACE / "outputs"


def load_config(name: str) -> dict[str, Any]:
    """Load a version-controlled JSON configuration."""
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
