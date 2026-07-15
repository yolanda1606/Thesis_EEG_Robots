from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from .config import load_config


def load_ratings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"stim_id", "category", "trigger_sent", "valence_rating", "arousal_rating"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Rating file is missing columns: {sorted(missing)}")
    if frame["trigger_sent"].dropna().duplicated().any():
        raise ValueError("Rating file contains duplicated image triggers")
    return frame


def load_unicorn_bdf(path: Path, preload: bool = True) -> mne.io.BaseRaw:
    config = load_config("channels.json")
    raw = mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    expected = set(config["channel_types"])
    missing = expected.difference(raw.ch_names)
    if missing:
        raise ValueError(f"BDF is missing expected channels: {sorted(missing)}")
    raw.set_channel_types(config["channel_types"], verbose=False)
    raw.rename_channels(config["eeg_mapping"])
    montage = mne.channels.make_standard_montage(config["montage"])
    raw.set_montage(montage, match_case=False, on_missing="raise")
    return raw
