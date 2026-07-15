from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import load_ratings


def attach_ratings(features: pd.DataFrame, inventory_path: Path) -> pd.DataFrame:
    """Attach observed ratings to feature rows by participant and trigger code."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rating_rows = []
    for record in inventory["records"]:
        selected = record.get("authoritative_rating")
        if not selected:
            continue
        ratings = load_ratings(Path(selected)).copy()
        ratings = ratings.rename(columns={
            "trigger_sent": "trigger",
            "valence_rating": "valence",
            "arousal_rating": "arousal",
        })
        ratings["participant"] = record["participant"]
        rating_rows.append(ratings[["participant", "trigger", "stim_id", "category", "valence", "arousal"]])
    if not rating_rows:
        raise ValueError("Inventory contains no authoritative rating exports")
    targets = pd.concat(rating_rows, ignore_index=True)
    if targets.duplicated(["participant", "trigger"]).any():
        raise ValueError("Behavioral targets are not unique by participant and trigger")
    labeled = features.merge(
        targets,
        on=["participant", "trigger"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    labeled["rating_match"] = labeled.pop("_merge").astype(str)
    return labeled
