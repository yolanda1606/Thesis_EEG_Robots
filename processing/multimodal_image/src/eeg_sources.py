"""Read-only loading and validation of ordered EEG source recordings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np


def eeg_source_paths(paths: dict[str, Any]) -> list[Path]:
    """Return one or more configured EEG files without changing their order."""
    if "eeg_files" in paths:
        return list(paths["eeg_files"])
    return [paths["eeg"]]


def load_eeg_session(paths: dict[str, Any], config: dict[str, Any], *, preload: bool):
    """Load validated fragments and return one MNE Raw plus offset events.

    MNE's supported concatenation adds boundary annotations and offsets events.
    No samples are inserted and no combined raw recording is written.
    """
    source_paths = eeg_source_paths(paths)
    raws = [mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR") for path in source_paths]
    first = raws[0]
    required = (set(config["channels"]["eeg_mapping"])
                | set(config["channels"]["motion_channels"])
                | {config["channels"]["stim_channel"]})
    missing = sorted(required.difference(first.ch_names))
    if missing:
        raise ValueError(f"EEG source is missing required channels: {missing}")
    events_list = []
    metadata = []
    seen: set[int] = set()
    expected = {
        code for low, high in config["events"]["image_ranges"].values()
        for code in range(int(low), int(high) + 1)
    }
    cumulative_offset = 0
    previous_end = None
    for index, (path, raw) in enumerate(zip(source_paths, raws)):
        if raw.ch_names != first.ch_names:
            raise ValueError(f"EEG fragment channel names/order differ: {path}")
        if float(raw.info["sfreq"]) != float(first.info["sfreq"]):
            raise ValueError(f"EEG fragment sampling frequency differs: {path}")
        fragment_missing = sorted(required.difference(raw.ch_names))
        if fragment_missing:
            raise ValueError(f"EEG fragment is missing required channels: {path}: {fragment_missing}")
        events = mne.find_events(raw, stim_channel=config["channels"]["stim_channel"], verbose=False)
        image_ids = [int(event[2]) for event in events if int(event[2]) in expected]
        overlap = sorted(seen.intersection(image_ids))
        if overlap:
            raise ValueError(f"Duplicate image trigger IDs across EEG fragments: {overlap}")
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Duplicate image trigger IDs within EEG fragment: {path}")
        seen.update(image_ids)
        events_list.append(events)
        start = raw.info.get("meas_date")
        gap = None
        if previous_end is not None and start is not None:
            gap = float((start - previous_end).total_seconds())
        if start is not None:
            previous_end = start + __import__("datetime").timedelta(seconds=raw.n_times / raw.info["sfreq"])
        metadata.append({
            "source_index": index,
            "source_filename": path.name,
            "source_sample_count": int(raw.n_times),
            "source_duration_s": float(raw.n_times / raw.info["sfreq"]),
            "source_trigger_ids": image_ids,
            "cumulative_sample_offset": cumulative_offset,
            "recording_gap_s": gap,
        })
        cumulative_offset += int(raw.n_times)
    if len(raws) == 1:
        return first, events_list[0], metadata
    combined, events = mne.concatenate_raws(raws, events_list=events_list, preload=preload, verbose="ERROR")
    if not np.all(np.diff(events[:, 0]) > 0):
        raise ValueError("Concatenated EEG events are not strictly time ordered")
    return combined, events, metadata
