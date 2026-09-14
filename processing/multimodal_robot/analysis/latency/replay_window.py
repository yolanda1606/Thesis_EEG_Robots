"""Small, benchmark-local replay primitives.

Nothing in this module writes to raw data or alters the canonical pipeline.
The functions deliberately reuse the canonical preprocessing and feature
definitions, while exposing the per-window boundaries that are embedded in
the production extractor.
"""
from __future__ import annotations

from time import perf_counter_ns
from typing import Any, Iterator

import numpy as np
import pandas as pd

from processing.multimodal_robot.prep.continuous import _extract_eeg_window_features, prepare_continuous_robot_eeg


WINDOW_SECONDS = 2.0
WINDOW_STEP_SECONDS = 1.0
CANONICAL_WINDOW_SAMPLES = 500


def milliseconds(start_ns: int, end_ns: int | None = None) -> float:
    """Return an elapsed duration in milliseconds from ``perf_counter_ns``."""
    return ((perf_counter_ns() if end_ns is None else end_ns) - start_ns) / 1_000_000.0


def preprocess_continuous(raw: Any) -> tuple[Any, float]:
    """Copy and canonically CAR/filter a continuous recording once.

    This preserves the production order and its zero-phase filtering behavior.
    The returned duration is a recording-level prerequisite, never an online
    per-window latency claim.
    """
    prepared = raw.copy()
    started = perf_counter_ns()
    prepare_continuous_robot_eeg(prepared)
    return prepared, milliseconds(started)


def eeg_window_starts(raw: Any) -> Iterator[tuple[float, int]]:
    """Yield exactly the canonical 2-s windows at 1-s increments."""
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, 250.0):
        raise ValueError(f"Canonical Robot replay requires 250 Hz EEG; found {sfreq:g} Hz")
    duration = raw.n_times / sfreq
    for start in np.arange(0.0, duration - WINDOW_SECONDS + 1e-9, WINDOW_STEP_SECONDS):
        yield float(start), int(round(start * sfreq))


def extract_eeg_window(raw: Any, eeg_channels: list[str], start_s: float) -> tuple[np.ndarray, float, float]:
    """Slice one canonical window and calculate its channel-wise features."""
    first = int(round(start_s * float(raw.info["sfreq"])))
    started = perf_counter_ns()
    data = raw.copy().pick(eeg_channels).get_data(start=first, stop=first + CANONICAL_WINDOW_SAMPLES)
    extract_ms = milliseconds(started)
    if data.shape[1] != CANONICAL_WINDOW_SAMPLES:
        raise ValueError(f"Expected {CANONICAL_WINDOW_SAMPLES} EEG samples, found {data.shape[1]}")
    started = perf_counter_ns()
    rows = _extract_eeg_window_features(data, float(raw.info["sfreq"]))
    feature_ms = milliseconds(started)
    return rows, extract_ms, feature_ms


def eeg_wide_row(metadata: dict[str, Any], channels: list[str], channel_features: list[dict[str, float]]) -> dict[str, Any]:
    """Create the exact feature-name convention used by Robot transfer."""
    row = dict(metadata)
    for channel, features in zip(channels, channel_features, strict=True):
        row.update({f"{feature}__{channel}": value for feature, value in features.items()})
    return row


def assemble_and_predict(record: dict[str, Any], feature_row: dict[str, Any]) -> tuple[float, int, float, float, float, float]:
    """Apply the canonical frozen-record inference path to one feature row.

    The original fitted pipeline always supplies the canonical hard prediction
    from candidate features.  Non-SVM probability models are fitted pipelines
    and can be split at their final classifier for timing.  Calibrated SVM
    probability objects are ``CalibratedClassifierCV`` instances, rather than
    Pipelines, so their fitted internals remain opaque and their complete
    ``predict_proba`` call is timed as one combined operation.
    """
    started = perf_counter_ns()
    candidate_features = record["candidate_features"]
    values = pd.DataFrame([{name: feature_row[name] for name in candidate_features}], columns=candidate_features)
    selected_values = values.loc[:, record["selected_features"]]
    assembly_ms = milliseconds(started)
    started = perf_counter_ns()
    hard_prediction = int(record["pipeline"].predict(values)[0])
    hard_predict_ms = milliseconds(started)
    model = record["probability_model"]
    column = record.get("calibrated_high_probability_column", record["high_probability_column"])
    if str(record["classifier"]).lower() == "svm":
        # This is exactly the canonical transfer path. Do not unwrap or
        # reconstruct the calibrated frozen object merely to expose scaling.
        transform_ms = float("nan")
        started = perf_counter_ns()
        probabilities = model.predict_proba(selected_values)
        predict_ms = milliseconds(started)
    else:
        if not hasattr(model, "steps") or len(model.steps) < 1:
            raise TypeError("Canonical non-SVM probability model is not a fitted sklearn Pipeline")
        started = perf_counter_ns()
        transformed = model[:-1].transform(values) if len(model.steps) > 1 else values
        transform_ms = milliseconds(started)
        started = perf_counter_ns()
        probabilities = model.steps[-1][1].predict_proba(transformed)
        predict_ms = milliseconds(started)
    probability = float(probabilities[0, column])
    return probability, hard_prediction, assembly_ms, transform_ms, hard_predict_ms, predict_ms


def summarize_timings(frame: pd.DataFrame, fields: list[str], groups: list[str]) -> pd.DataFrame:
    """Produce required descriptive timing statistics without timing output I/O."""
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for field in fields:
            values = pd.to_numeric(part[field], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(dict(zip(groups, keys)) | {
                "metric": field, "count": int(values.size), "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "median": float(values.median()), "p95": float(values.quantile(0.95)),
                "min": float(values.min()), "max": float(values.max()),
            })
    return pd.DataFrame(rows)
