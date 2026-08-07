from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import mne
import numpy as np

from .eeg_sources import load_eeg_session
from .validation import available_image_codes


def build_alignment(paths: dict[str, Path], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    with paths["vision_log"].open(encoding="utf-8-sig", newline="") as handle:
        log_rows = list(csv.DictReader(handle))
    video_onsets = {
        int(float(row["Trigger"])): float(row["Experiment_Time"])
        for row in log_rows
        if row.get("Trigger", "").strip() not in ("", "0", "0.0")
    }
    raw, events, _ = load_eeg_session(paths, config, preload=False)
    expected = available_image_codes(config)
    eeg_onsets = {int(event[2]): float(event[0]) / raw.info["sfreq"] for event in events if int(event[2]) in expected}
    codes = sorted(expected.intersection(video_onsets, eeg_onsets))
    if len(codes) != len(expected):
        raise ValueError("Cannot build alignment: not all image triggers are shared")
    x = np.asarray([eeg_onsets[code] for code in codes])
    offsets = np.asarray([video_onsets[code] - eeg_onsets[code] for code in codes])
    slope, intercept = np.polyfit(x, offsets, 1)
    rows = [{"trigger": code, "eeg_onset_s": eeg_onsets[code], "video_onset_s": video_onsets[code],
             "video_minus_eeg_s": video_onsets[code] - eeg_onsets[code]} for code in codes]
    model = {"offset_intercept_s": float(intercept), "offset_drift_s_per_s": float(slope),
             "offset_min_s": float(offsets.min()), "offset_max_s": float(offsets.max()), "matched_trials": len(codes)}
    return rows, model
