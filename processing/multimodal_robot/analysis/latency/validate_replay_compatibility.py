"""Compatibility comparisons for latency replay outputs.

This is intentionally independent of benchmark timing: validation must not
pollute the primary timing measurements.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


IDENTIFIERS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]


def recorded_model_input_features(replay_input: pd.DataFrame) -> list[str]:
    """Return the exact probability-input schema recorded for one model call.

    ``DataFrame`` construction unions columns across different frozen models.
    The explicit schema prevents unrelated columns (and their structural NaNs)
    from being mistaken for inputs to this model.
    """
    if "probability_input_features" not in replay_input:
        raise ValueError("Replay model-input rows lack their recorded probability-input schema")
    schemas = replay_input["probability_input_features"].dropna().unique().tolist()
    if len(schemas) != 1:
        raise ValueError("A target/rank/modality group must have exactly one probability-input schema")
    features = json.loads(schemas[0])
    if not isinstance(features, list) or not features or not all(isinstance(name, str) for name in features):
        raise ValueError("Recorded probability-input schema must be a non-empty JSON string list")
    missing = sorted(set(features) - set(replay_input.columns))
    if missing:
        raise ValueError(f"Replay model-input rows lack recorded features: {missing}")
    if replay_input.loc[:, features].isna().any().any():
        raise ValueError("Replay model-input rows contain NaN in a feature actually supplied to the frozen probability model")
    return features


def compare_frames(replay: pd.DataFrame, canonical: pd.DataFrame, feature_columns: list[str], *, tolerance: float = 1e-9, comparison: str) -> pd.DataFrame:
    """Compare selected values by canonical window identifier and report all differences."""
    keys = [name for name in IDENTIFIERS if name in replay and name in canonical]
    if not keys:
        raise ValueError("Replay and canonical data have no shared window identifiers")
    missing = sorted(set(feature_columns) - set(replay.columns) | (set(feature_columns) - set(canonical.columns)))
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")
    left = replay.loc[:, [*keys, *feature_columns]].copy()
    right = canonical.loc[:, [*keys, *feature_columns]].copy()
    merged = left.merge(right, on=keys, suffixes=("_replay", "_canonical"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        replay_values = pd.to_numeric(merged[f"{feature}_replay"], errors="coerce")
        canonical_values = pd.to_numeric(merged[f"{feature}_canonical"], errors="coerce")
        difference = (replay_values - canonical_values).abs()
        for index, value in difference.items():
            rows.append({**{key: merged.at[index, key] for key in keys}, "comparison": comparison,
                         "feature": feature, "replay_value": replay_values.at[index],
                         "canonical_value": canonical_values.at[index], "absolute_difference": value,
                         "exact_equal": bool(replay_values.at[index] == canonical_values.at[index]),
                         "tolerance": tolerance, "pass": bool(np.isfinite(value) and value <= tolerance)})
    return pd.DataFrame(rows)


def validation_passes(results: pd.DataFrame) -> bool:
    """Require at least one comparison and a pass for every compared value."""
    return bool(len(results) and results["pass"].all())
