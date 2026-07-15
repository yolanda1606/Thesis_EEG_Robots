from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import pandas as pd

from .config import PROJECT_ROOT, load_config


IMAGE_FOLDER_NAMES = {
    "Image Experiment",
    "Image_Experiment",
    "Image Emotion Experiment",
}
REQUIRED_RATING_COLUMNS = {
    "stim_id", "category", "trigger_sent", "valence_rating", "arousal_rating"
}


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _rating_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_RATING_COLUMNS.difference(frame.columns))
    return {
        "path": str(path.resolve()),
        "sha256": _digest(path),
        "rows": int(len(frame)),
        "missing_columns": missing_columns,
        "missing_valence": int(frame["valence_rating"].isna().sum()) if "valence_rating" in frame else None,
        "missing_arousal": int(frame["arousal_rating"].isna().sum()) if "arousal_rating" in frame else None,
        "unique_triggers": int(frame["trigger_sent"].nunique()) if "trigger_sent" in frame else None,
        "unique_stimuli": int(frame["stim_id"].nunique()) if "stim_id" in frame else None,
    }


def choose_rating_export(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the structurally complete export with the fewest missing targets."""
    usable = [item for item in summaries if not item["missing_columns"]]
    if not usable:
        return None
    return min(
        usable,
        key=lambda item: (
            item["missing_valence"] + item["missing_arousal"],
            -item["rows"],
            item["path"],
        ),
    )


def _image_codes() -> set[int]:
    ranges = load_config("events.json")["image_ranges"]
    return {
        code
        for first, last in ranges.values()
        for code in range(int(first), int(last) + 1)
    }


def _bdf_summary(path: Path) -> dict[str, Any]:
    channel_config = load_config("channels.json")
    raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    events = mne.find_events(
        raw, stim_channel=channel_config["stim_channel"], verbose=False
    )
    counts = Counter(int(code) for code in events[:, 2])
    expected = _image_codes()
    present = expected.intersection(counts)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "channels": list(raw.ch_names),
        "duration_s": float(raw.times[-1]),
        "events_total": int(len(events)),
        "image_events": int(sum(counts[code] for code in expected)),
        "unique_image_events": int(len(present)),
        "missing_image_codes": sorted(expected.difference(present)),
        "repeated_image_events": int(sum(max(0, counts[code] - 1) for code in expected)),
    }


def build_inventory(data_root: Path | None = None) -> dict[str, Any]:
    """Inspect Image Experiment metadata without modifying source data."""
    root = (data_root or PROJECT_ROOT / "data").resolve()
    records: list[dict[str, Any]] = []
    for participant_dir in sorted(root.glob("P[0-9]*_*")):
        experiment_dirs = [
            path for path in participant_dir.iterdir()
            if path.is_dir() and path.name in IMAGE_FOLDER_NAMES
        ]
        for experiment_dir in experiment_dirs:
            ratings = [_rating_summary(path) for path in sorted(experiment_dir.glob("emotion_ratings_*.csv"))]
            eeg_dir = experiment_dir / "EEG"
            bdfs = []
            if eeg_dir.exists():
                for path in sorted(eeg_dir.glob("UnicornRecorder_*.bdf")):
                    try:
                        bdfs.append(_bdf_summary(path))
                    except Exception as exc:  # preserve failure in the QC manifest
                        bdfs.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
            chosen = choose_rating_export(ratings)
            records.append({
                "participant": participant_dir.name,
                "experiment_directory": str(experiment_dir.resolve()),
                "rating_exports": ratings,
                "authoritative_rating": chosen["path"] if chosen else None,
                "eeg_segments": bdfs,
                "video_files": [str(path.resolve()) for path in sorted(experiment_dir.glob("vision_video_*.avi"))],
            })
    return {"schema_version": 1, "data_root": str(root), "records": records}


def write_inventory(inventory: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)
        handle.write("\n")
