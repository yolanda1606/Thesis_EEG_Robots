"""Small YAML-driven Matplotlib style loader for thesis figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


def load_thesis_style(path: Path) -> dict[str, Any]:
    """Load palette YAML and resolve semantic role names to hex colors."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("palette"), dict):
        raise ValueError(f"{path}: expected a palette mapping")
    palette = config["palette"]

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, str) and value in palette:
            return palette[value]
        return value

    roles = resolve(config.get("semantic_roles", {}))
    return {"palette": palette, "roles": roles, "source": str(path.resolve())}


def apply_thesis_style(style: dict[str, Any]) -> None:
    """Apply restrained academic defaults without changing fonts or dependencies."""
    neutral = style["roles"]["neutral"]
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": neutral["text"], "axes.labelcolor": neutral["text"],
        "axes.edgecolor": neutral["edge"], "xtick.color": neutral["text"], "ytick.color": neutral["text"],
        "axes.linewidth": .8, "grid.color": neutral["grid"], "grid.linewidth": .6,
        "grid.alpha": .65, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.frameon": False, "legend.fontsize": 9,
    })
