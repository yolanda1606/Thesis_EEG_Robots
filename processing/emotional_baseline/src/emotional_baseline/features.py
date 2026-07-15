from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def extract_epoch_features(epochs, participant: str, segment: str) -> pd.DataFrame:
    """Extract transparent baseline PSD and Hjorth features per epoch/channel."""
    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])
    rows = []
    for epoch_index, (epoch, event) in enumerate(zip(data, epochs.events)):
        row = {"participant": participant, "segment": segment,
               "epoch_index": epoch_index, "trigger": int(event[2])}
        for channel, signal in zip(epochs.ch_names, epoch):
            first = np.diff(signal)
            second = np.diff(first)
            activity = float(np.var(signal))
            mobility = float(np.sqrt(np.var(first) / activity)) if activity else 0.0
            first_var = float(np.var(first))
            complexity = float(np.sqrt(np.var(second) / first_var) / mobility) if first_var and mobility else 0.0
            row[f"{channel}__hjorth_activity"] = activity
            row[f"{channel}__hjorth_mobility"] = mobility
            row[f"{channel}__hjorth_complexity"] = complexity
            frequencies, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq)))
            total = np.trapezoid(psd[(frequencies >= 1) & (frequencies <= 40)],
                                 frequencies[(frequencies >= 1) & (frequencies <= 40)])
            for band, (low, high) in BANDS.items():
                mask = (frequencies >= low) & (frequencies < high)
                absolute = float(np.trapezoid(psd[mask], frequencies[mask]))
                row[f"{channel}__{band}_absolute"] = absolute
                row[f"{channel}__{band}_relative"] = absolute / total if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
