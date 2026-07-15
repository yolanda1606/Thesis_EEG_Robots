from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .trainer import METADATA


def evaluate_loso(features: pd.DataFrame, target: str) -> tuple[dict, pd.DataFrame]:
    clean = features.dropna(subset=[target]).copy()
    columns = [column for column in clean.columns if column not in METADATA]
    base = Pipeline([
        ("scale", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    predictions = []
    splitter = LeaveOneGroupOut()
    groups = clean["participant"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(clean[columns], clean[target], groups)):
        model = clone(base)
        model.fit(clean.iloc[train_idx][columns], clean.iloc[train_idx][target])
        predicted = model.predict(clean.iloc[test_idx][columns])
        for index, value in zip(test_idx, predicted):
            predictions.append({"fold": fold, "participant": clean.iloc[index]["participant"],
                                "observed": float(clean.iloc[index][target]), "predicted": float(value)})
    result = pd.DataFrame(predictions)
    metrics = {
        "target": target,
        "participants": int(clean["participant"].nunique()),
        "rows": int(len(result)),
        "mae": float(mean_absolute_error(result["observed"], result["predicted"])),
        "rmse": float(np.sqrt(mean_squared_error(result["observed"], result["predicted"]))),
        "correlation": float(result[["observed", "predicted"]].corr().iloc[0, 1]),
    }
    return metrics, result
