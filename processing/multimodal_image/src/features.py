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


def extract_eeg_features(epochs, config: dict[str, Any]) -> pd.DataFrame:
    """One transparent feature row per retained epoch and EEG channel."""
    bands = config["features"]["bands_hz"]
    rows: list[dict[str, Any]] = []
    for epoch_index, (epoch, event) in enumerate(zip(epochs.get_data(copy=True), epochs.events)):
        for channel, signal in zip(epochs.ch_names, epoch):
            frequencies, psd = welch(signal, fs=float(epochs.info["sfreq"]), nperseg=min(len(signal), int(epochs.info["sfreq"])))
            row: dict[str, Any] = {"epoch_index": epoch_index, "trigger": int(event[2]), "channel": channel,
                                   "eeg_sd": float(np.std(signal)), "eeg_se": _entropy(psd), "eeg_hm": _hjorth_mobility(signal),
                                   "eeg_hc": _hjorth_complexity(signal), "eeg_mf_hz": _median_frequency(frequencies, psd)}
            for band, (low, high) in bands.items():
                mask = (frequencies >= float(low)) & (frequencies <= float(high))
                row[f"eeg_bp_{band}"] = float(np.trapezoid(psd[mask], frequencies[mask])) if mask.any() else float("nan")
                row[f"eeg_se_{band}"] = _entropy(psd[mask]) if mask.any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


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
