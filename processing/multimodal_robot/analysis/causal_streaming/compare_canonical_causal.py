"""Read-only comparison helpers for canonical versus causal Robot outputs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


KEYS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]


def canonical_eeg_wide(path: str | Any) -> pd.DataFrame:
    """Pivot the canonical long export into transfer-compatible EEG columns."""
    long = pd.read_csv(path)
    features = [name for name in long if name.startswith("eeg_") and name != "eeg_coverage"]
    if set(KEYS + ["channel"]) - set(long) or not features:
        raise ValueError("Canonical EEG input is not the expected long-format Robot feature table")
    wide = long.pivot(index=KEYS, columns="channel", values=features)
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    return wide.reset_index()


def feature_family(name: str) -> str:
    """Return the feature family before its channel separator."""
    return name.split("__", 1)[0]


def feature_channel(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else ""


def _correlations(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3 or pair.left.nunique() < 2 or pair.right.nunique() < 2:
        return np.nan, np.nan
    return float(pair.left.corr(pair.right, method="pearson")), float(spearmanr(pair.left, pair.right).statistic)


def compare_features(canonical: pd.DataFrame, causal: pd.DataFrame, *, epsilon: float = 1e-12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare matching warm-up-eligible windows without requiring equality."""
    features = sorted(set(canonical.columns).intersection(causal.columns) - set(KEYS))
    features = [name for name in features if name.startswith("eeg_")]
    causal_keys = causal.loc[~causal["causal_warmup_excluded"], KEYS + features]
    merged = canonical.loc[:, KEYS + features].merge(causal_keys, on=KEYS, suffixes=("_canonical", "_causal"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in features:
        left, right = merged[f"{name}_canonical"], merged[f"{name}_causal"]
        signed = right - left
        absolute = signed.abs()
        symmetric_relative = 2 * absolute / (left.abs() + right.abs() + epsilon)
        pearson, spearman = _correlations(left, right)
        summaries.append({"feature": name, "feature_family": feature_family(name), "channel": feature_channel(name),
                          "n_windows": int(left.notna().sum()), "pearson_r": pearson, "spearman_rho": spearman,
                          "mean_absolute_difference": float(absolute.mean()), "median_absolute_difference": float(absolute.median()),
                          "median_symmetric_relative_difference": float(symmetric_relative.median())})
        rows.extend({**{key: merged.at[index, key] for key in KEYS}, "feature": name,
                     "feature_family": feature_family(name), "channel": feature_channel(name),
                     "canonical_value": left.at[index], "causal_value": right.at[index],
                     "signed_difference": signed.at[index], "absolute_difference": absolute.at[index],
                     "symmetric_relative_difference": symmetric_relative.at[index]}
                    for index in merged.index)
    detail = pd.DataFrame(rows)
    by_feature = pd.DataFrame(summaries)
    family = by_feature.groupby("feature_family", as_index=False).agg(
        feature_count=("feature", "size"), median_pearson_r=("pearson_r", "median"),
        median_spearman_rho=("spearman_rho", "median"),
        median_absolute_difference=("median_absolute_difference", "median"),
        median_symmetric_relative_difference=("median_symmetric_relative_difference", "median"),
    )
    return detail, family


def compare_predictions(canonical_path: str | Any, causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare canonical and causal frozen outputs by window, target, and rank."""
    canonical = pd.read_csv(canonical_path)
    fields = KEYS + ["target", "model_rank", "classifier", "high_probability", "original_hard_prediction"]
    required = set(fields)
    if required - set(canonical) or required - set(causal):
        raise ValueError("Prediction inputs lack canonical window/model fields")
    merged = canonical.loc[:, fields].merge(causal.loc[:, fields], on=KEYS + ["target", "model_rank"], suffixes=("_canonical", "_causal"), validate="one_to_one")
    merged["signed_probability_difference"] = merged.high_probability_causal - merged.high_probability_canonical
    merged["absolute_probability_difference"] = merged.signed_probability_difference.abs()
    merged["hard_agree"] = merged.original_hard_prediction_canonical.eq(merged.original_hard_prediction_causal)
    rows = []
    for keys, frame in merged.groupby(["target", "model_rank", "classifier_canonical"], sort=False):
        pearson, spearman = _correlations(frame.high_probability_canonical, frame.high_probability_causal)
        rows.append({"target": keys[0], "model_rank": keys[1], "classifier": keys[2], "n_windows": len(frame),
                     "pearson_r": pearson, "spearman_rho": spearman,
                     "mae": float(frame.absolute_probability_difference.mean()),
                     "median_absolute_difference": float(frame.absolute_probability_difference.median()),
                     "hard_agreement_percent": float(frame.hard_agree.mean() * 100)})
    disagreements = merged.loc[~merged.hard_agree].copy()
    return merged, pd.DataFrame(rows), disagreements
