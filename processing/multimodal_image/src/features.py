from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _median_frequency(frequencies: np.ndarray, psd: np.ndarray) -> float:
    cumulative = np.cumsum((psd[1:] + psd[:-1]) * np.diff(frequencies) / 2.0)
    if not len(cumulative) or cumulative[-1] <= 0:
        return float("nan")
    return float(frequencies[np.searchsorted(cumulative, cumulative[-1] / 2.0) + 1])


def _feature_window_epochs(epochs, config: dict[str, Any]):
    """Return a safely cropped copy of cleaned epochs for feature calculation."""
    window = config["features"].get("eeg_feature_window_s")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError("features.eeg_feature_window_s must contain [tmin_s, tmax_s]")
    tmin_s, tmax_s = (float(value) for value in window)
    if not np.isfinite([tmin_s, tmax_s]).all() or tmin_s >= tmax_s:
        raise ValueError("features.eeg_feature_window_s must be finite with tmin_s < tmax_s")
    if tmin_s < epochs.tmin or tmax_s > epochs.tmax:
        raise ValueError(
            "features.eeg_feature_window_s is outside the available cleaned epoch "
            f"[{epochs.tmin:.6f}, {epochs.tmax:.6f}] s"
        )
    # MNE performs time-aware selection. include_tmax=True explicitly retains
    # the +2.0 s sample when it lies on the sampling grid.
    feature_epochs = epochs.copy().crop(tmin=tmin_s, tmax=tmax_s, include_tmax=True)
    expected_samples = int(round((tmax_s - tmin_s) * float(epochs.info["sfreq"]))) + 1
    if abs(len(feature_epochs.times) - expected_samples) > 1:
        raise ValueError(
            "Cropped EEG feature window has an unexpected sample count: "
            f"expected approximately {expected_samples}, got {len(feature_epochs.times)}"
        )
    return feature_epochs


def extract_eeg_features(epochs, config: dict[str, Any]) -> pd.DataFrame:
    """One transparent feature row per retained epoch and EEG channel."""
    bands = config["features"]["bands_hz"]
    feature_epochs = _feature_window_epochs(epochs, config)
    rows: list[dict[str, Any]] = []
    for epoch_index, (epoch, event) in enumerate(zip(feature_epochs.get_data(copy=True), feature_epochs.events)):
        for channel, signal in zip(feature_epochs.ch_names, epoch):
            frequencies, psd = welch(signal, fs=float(feature_epochs.info["sfreq"]), nperseg=min(len(signal), int(feature_epochs.info["sfreq"])))
            row: dict[str, Any] = {"epoch_index": epoch_index, "trigger": int(event[2]), "channel": channel,
                                   "eeg_sd": float(np.std(signal)), "eeg_se": _entropy(psd), "eeg_hm": _hjorth_mobility(signal),
                                   "eeg_hc": _hjorth_complexity(signal), "eeg_mf_hz": _median_frequency(frequencies, psd)}
            for band, (low, high) in bands.items():
                mask = (frequencies >= float(low)) & (frequencies <= float(high))
                row[f"eeg_bp_{band}"] = float(np.trapezoid(psd[mask], frequencies[mask])) if mask.any() else float("nan")
                row[f"eeg_se_{band}"] = _entropy(psd[mask]) if mask.any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def extract_eeg_window_features(data: np.ndarray, sampling_hz: float) -> list[dict[str, float]]:
    """Frozen feature formulas for one continuous window, one row per channel.

    The robot pipeline calls this with exactly 500 samples (2 s at 250 Hz).
    Keeping the numerical implementation here prevents a feature-definition
    fork between the image and robot experiments.
    """
    bands = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0), "gamma": (30.0, 40.0)}
    rows = []
    for signal in data:
        frequencies, psd = welch(signal, fs=float(sampling_hz), nperseg=min(len(signal), int(sampling_hz)))
        row = {"eeg_sd": float(np.std(signal)), "eeg_se": _entropy(psd), "eeg_hm": _hjorth_mobility(signal),
               "eeg_hc": _hjorth_complexity(signal), "eeg_mf_hz": _median_frequency(frequencies, psd)}
        for band, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies <= high)
            row[f"eeg_bp_{band}"] = float(np.trapezoid(psd[mask], frequencies[mask])) if mask.any() else float("nan")
            row[f"eeg_se_{band}"] = _entropy(psd[mask]) if mask.any() else float("nan")
        rows.append(row)
    return rows


def _hjorth_mobility(signal: np.ndarray) -> float:
    signal_variance = np.var(signal)
    return float(np.sqrt(np.var(np.diff(signal)) / signal_variance)) if signal_variance else float("nan")


def _hjorth_complexity(signal: np.ndarray) -> float:
    first = np.diff(signal)
    first_variance = np.var(first)
    mobility = _hjorth_mobility(signal)
    if not first_variance or not np.isfinite(mobility) or mobility == 0:
        return float("nan")
    return float(np.sqrt(np.var(np.diff(first)) / first_variance) / mobility)
