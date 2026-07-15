from __future__ import annotations

from pathlib import Path

import mne

from .config import load_config
from .io import load_unicorn_bdf


def preprocess_segment(path: Path) -> mne.Epochs:
    """Create conservatively preprocessed image-onset epochs from one BDF segment."""
    channels = load_config("channels.json")
    events_config = load_config("events.json")
    config = load_config("preprocessing.json")
    raw = load_unicorn_bdf(path, preload=True)
    events = mne.find_events(raw, stim_channel=channels["stim_channel"], verbose=False)
    image_codes = {
        code
        for first, last in events_config["image_ranges"].values()
        for code in range(int(first), int(last) + 1)
    }
    selected = events[[int(code) in image_codes for code in events[:, 2]]]
    if not len(selected):
        raise ValueError(f"No image-onset events found in {path}")
    raw.pick("eeg")
    raw.filter(
        l_freq=config["l_freq_hz"],
        h_freq=config["h_freq_hz"],
        method=config["filter_method"],
        iir_params={"order": config["iir_order"], "ftype": "butter", "output": "sos"},
        verbose=False,
    )
    if config["notch_hz"] is not None:
        raw.notch_filter(config["notch_hz"], verbose=False)
    if config["reference"] == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    event_id = {str(code): code for code in sorted(set(selected[:, 2]))}
    return mne.Epochs(
        raw,
        selected,
        event_id=event_id,
        tmin=config["epoch_tmin_s"],
        tmax=config["epoch_tmax_s"],
        baseline=tuple(config["baseline_s"]),
        preload=True,
        on_missing="raise",
        verbose=False,
    )
