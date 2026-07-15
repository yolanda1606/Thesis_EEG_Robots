from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METADATA = {
    "participant", "segment", "epoch_index", "trigger", "stim_id",
    "valence", "arousal", "category", "rating_match"
}


def train(features: pd.DataFrame, target: str, model_path) -> dict:
    """Fit a baseline model; final performance must be obtained with grouped evaluation."""
    clean = features.dropna(subset=[target]).copy()
    columns = [column for column in clean.columns if column not in METADATA]
    if not columns:
        raise ValueError("No feature columns found")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(clean[columns], clean[target])
    artifact = {"model": model, "features": columns, "target": target}
    joblib.dump(artifact, model_path)
    return {"rows": len(clean), "participants": clean["participant"].nunique(), "features": len(columns)}
