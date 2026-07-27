from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mne
import pandas as pd


def resolve_inputs(config: dict[str, Any], root: Path, resource_root: Path | None = None) -> dict[str, Path]:
    inputs = config["inputs"]
    experiment = root / inputs["participant_dir"] / inputs["experiment_dir"]
    if resource_root is None:
        legacy_model = inputs.get("face_landmarker_model")
        if not legacy_model:
            raise ValueError("No resource root or legacy face-landmarker model path was supplied")
        model = root / legacy_model
    else:
        model = resource_root / config["resources"]["face_landmarker_filename"]
    paths = {
        "experiment_dir": experiment,
        "eeg": experiment / inputs["eeg_file"],
        "ratings": experiment / inputs["ratings_file"],
        "video": experiment / inputs["video_file"],
        "vision_log": experiment / inputs["vision_log_file"],
        "model": model,
    }
    for label, path in paths.items():
        if label != "experiment_dir" and not path.is_file():
            raise FileNotFoundError(f"Required {label} file does not exist: {path}")
    return paths


def image_codes(config: dict[str, Any]) -> set[int]:
    return {
        code
        for low, high in config["events"]["image_ranges"].values()
        for code in range(int(low), int(high) + 1)
    }


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(), "sha256": digest.hexdigest()}


def inspect_inputs(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    ratings = pd.read_csv(paths["ratings"], encoding="utf-8-sig")
    required = {"stim_id", "category", "trigger_sent", "valence_rating", "arousal_rating"}
    missing = sorted(required.difference(ratings.columns))
    if missing:
        raise ValueError(f"Ratings file is missing columns: {missing}")
    rating_codes = ratings["trigger_sent"].dropna().astype(int).tolist()
    if len(rating_codes) != len(set(rating_codes)):
        raise ValueError("Ratings contain duplicate trigger_sent values")

    raw = mne.io.read_raw_bdf(paths["eeg"], preload=False, verbose="ERROR")
    events = mne.find_events(raw, stim_channel=config["channels"]["stim_channel"], verbose=False)
    codes = image_codes(config)
    eeg_codes = [int(event[2]) for event in events if int(event[2]) in codes]
    if len(eeg_codes) != len(set(eeg_codes)):
        raise ValueError("EEG contains duplicate image trigger values")

    capture = cv2.VideoCapture(str(paths["video"]))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {paths['video']}")
    video = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if video["fps"] <= 0 or video["frame_count"] <= 0:
        raise ValueError("Video header has no usable frame rate or frame count")

    with paths["vision_log"].open(encoding="utf-8-sig", newline="") as handle:
        log_rows = list(csv.DictReader(handle))
    log_codes = [int(float(row["Trigger"])) for row in log_rows if row.get("Trigger", "").strip() not in ("", "0", "0.0")]
    log_image_codes = [code for code in log_codes if code in codes]
    if len(log_image_codes) != len(set(log_image_codes)):
        raise ValueError("Vision log contains duplicate image trigger values")

    expected = image_codes(config)
    summary = {
        "ratings_rows": int(len(ratings)),
        "ratings_missing_valence": int(ratings["valence_rating"].isna().sum()),
        "ratings_missing_arousal": int(ratings["arousal_rating"].isna().sum()),
        "rating_image_triggers": len(set(rating_codes).intersection(expected)),
        "eeg_total_events": int(len(events)),
        "eeg_image_events": len(eeg_codes),
        "vision_log_rows": len(log_rows),
        "vision_log_image_events": len(log_image_codes),
        "video": video,
        "eeg_sampling_hz": float(raw.info["sfreq"]),
        "eeg_duration_s": float(raw.times[-1]),
        "eeg_channels": raw.ch_names,
        "matchable_image_triggers": len(set(rating_codes).intersection(eeg_codes, log_image_codes)),
    }
    if summary["matchable_image_triggers"] != len(expected):
        raise ValueError(f"Only {summary['matchable_image_triggers']} of {len(expected)} planned image triggers match")
    return summary
