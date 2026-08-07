"""Read-only P01 EEG, ratings, video, and synchronization health checks."""
from __future__ import annotations

import csv
from typing import Any

import cv2
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from .eeg_sources import load_eeg_session
from .validation import available_image_codes, image_codes


def _status(reasons: list[str]) -> str:
    return "PASS" if not reasons else "PASS_WITH_WARNINGS"


def eeg_health(paths, config: dict[str, Any], logger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw, events, source_metadata = load_eeg_session(paths, config, preload=True)
    mapping = config["channels"]["eeg_mapping"]
    names = list(mapping)
    missing = sorted(set(names).difference(raw.ch_names))
    duplicates = sorted({name for name in raw.ch_names if raw.ch_names.count(name) > 1})
    if missing or duplicates:
        raise ValueError(f"EEG channel structure invalid; missing={missing}, duplicates={duplicates}")
    data = raw.get_data(picks=names)
    sfreq = float(raw.info["sfreq"])
    cfg = config["health"]
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    for name, signal in zip(names, data):
        variance = float(np.var(signal))
        maximum = float(np.max(np.abs(signal)))
        finite = bool(np.isfinite(signal).all())
        flat = variance == 0.0
        near_flat = variance < float(cfg["near_flat_variance_v2"])
        clipped_fraction = float(max(np.mean(signal == signal.max()), np.mean(signal == signal.min())))
        frequencies, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq) * 4))
        band = lambda low, high: float(np.mean(psd[(frequencies >= low) & (frequencies <= high)]))
        line_ratio = band(49, 51) / max((band(47, 49) + band(51, 53)) / 2.0, np.finfo(float).tiny)
        warning = not finite or flat or near_flat or maximum > float(cfg["large_amplitude_v"]) or clipped_fraction > float(cfg["clipping_fraction_warning"]) or line_ratio > float(cfg["line_noise_ratio_warning"])
        if warning:
            reasons.append(f"EEG {name}: finite={finite}, flat={flat}, near_flat={near_flat}, max_abs_v={maximum:.3g}, clipping_fraction={clipped_fraction:.3g}, line_noise_ratio={line_ratio:.3g}")
        rows.append({"domain": "eeg_channel", "name": name, "finite": finite, "variance_v2": variance, "max_abs_v": maximum,
                     "flat": flat, "near_flat": near_flat, "clipping_fraction": clipped_fraction, "line_noise_ratio_50hz": line_ratio,
                     "status": "PASS_WITH_WARNINGS" if warning else "PASS"})
    expected = available_image_codes(config)
    image = events[np.isin(events[:, 2], list(expected))]
    observed = [int(event[2]) for event in image]
    duplicate_triggers = sorted({code for code in observed if observed.count(code) > 1})
    missing_triggers = sorted(expected.difference(observed))
    event_ordered = bool(np.all(np.diff(events[:, 0]) > 0))
    if duplicate_triggers or missing_triggers or not event_ordered:
        reasons.append(f"Trigger integrity: duplicates={duplicate_triggers}, missing={missing_triggers}, time_ordered={event_ordered}")
    if config.get("trials", {}).get("allow_incomplete_image_trials", False):
        reasons.append(f"Approved incomplete session: known_missing_triggers={config['trials']['known_missing_triggers']}")
    summary = {"status": _status(reasons), "reasons": reasons, "format": "BDF", "sampling_hz": sfreq,
               "sample_count": int(raw.n_times), "duration_s": float(raw.times[-1]), "sample_interval_s": 1.0 / sfreq,
               "sample_count_duration_consistent": bool(abs(raw.n_times / sfreq - (raw.times[-1] + 1.0 / sfreq)) < 1e-9),
               "expected_eeg_channels": names, "motion_channels_present": [name for name in config["channels"]["motion_channels"] if name in raw.ch_names],
               "stim_channel_present": config["channels"]["stim_channel"] in raw.ch_names, "missing_channels": missing,
               "duplicate_channel_names": duplicates, "total_events": int(len(events)), "image_trigger_count": int(len(image)), "eeg_sources": source_metadata,
               "duplicate_image_triggers": duplicate_triggers, "missing_image_triggers": missing_triggers, "events_time_ordered": event_ordered}
    logger.info("EEG health: %s; %d image triggers", summary["status"], len(image))
    return summary, rows


def ratings_health(path, config: dict[str, Any], logger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ratings_cfg = config["ratings"]
    frame = pd.read_csv(path, encoding="utf-8-sig", na_values=ratings_cfg["missing_tokens"], keep_default_na=True)
    required = ratings_cfg["required_columns"]
    missing_columns = sorted(set(required).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Ratings missing required columns: {missing_columns}")
    trigger = frame["trigger_sent"]
    duplicate_triggers = sorted(trigger[trigger.duplicated(keep=False)].dropna().astype(int).unique().tolist())
    duplicate_rows = int(frame.duplicated().sum())
    invalid: dict[str, list[int]] = {}
    for column in (ratings_cfg["rating_columns"]["valence"], ratings_cfg["rating_columns"]["arousal"]):
        values = pd.to_numeric(frame[column], errors="coerce")
        # DataFrame index 0 is the first data record; CSV row 1 is the header.
        invalid_indices = frame.index[values.notna() & ~values.between(1, 7)].to_numpy(dtype=int)
        invalid[column] = (invalid_indices + 2).tolist()
    reasons = []
    missing_arousal = frame[ratings_cfg["rating_columns"]["arousal"]].isna()
    missing_valence = frame[ratings_cfg["rating_columns"]["valence"]].isna()
    if missing_arousal.any() or missing_valence.any():
        reasons.append(f"Missing ratings: valence triggers={trigger[missing_valence].dropna().astype(int).tolist()}, arousal triggers={trigger[missing_arousal].dropna().astype(int).tolist()}")
    if duplicate_triggers or duplicate_rows or any(invalid.values()):
        reasons.append(f"Ratings integrity: duplicate_triggers={duplicate_triggers}, duplicate_rows={duplicate_rows}, invalid_rows={invalid}")
    rows = [{"domain": "ratings_category", "name": str(category), "count": int(count), "status": "PASS"} for category, count in frame["category"].value_counts().sort_index().items()]
    summary = {"status": _status(reasons), "reasons": reasons, "rows": int(len(frame)), "required_columns": required,
               "duplicate_rows": duplicate_rows, "duplicate_triggers": duplicate_triggers, "missing_valence_triggers": trigger[missing_valence].dropna().astype(int).tolist(),
               "missing_arousal_triggers": trigger[missing_arousal].dropna().astype(int).tolist(), "invalid_rating_rows": invalid,
               "category_counts": {str(key): int(value) for key, value in frame["category"].value_counts().sort_index().items()},
               "trigger_order_unique": bool(trigger.dropna().is_unique)}
    logger.info("Ratings health: %s; %d rows", summary["status"], len(frame))
    return summary, rows


def video_health(video_path, log_path, config: dict[str, Any], logger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps, frames = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    positions = np.unique(np.linspace(0, max(frames - 1, 0), min(frames, int(config["health"]["video_frame_samples"])), dtype=int))
    unreadable = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, _ = capture.read()
        if not ok:
            unreadable.append(int(position + 1))
    capture.release()
    with log_path.open(encoding="utf-8-sig", newline="") as handle:
        log = pd.DataFrame(csv.DictReader(handle))
    experiment_time = pd.to_numeric(log["Experiment_Time"], errors="coerce")
    timestamps = pd.to_datetime(log["Timestamp"], errors="coerce")
    numeric_frames = pd.to_numeric(log["Frame_Count"], errors="coerce")
    gaps = experiment_time.diff()
    median_gap = float(gaps[gaps > 0].median()) if (gaps > 0).any() else float("nan")
    large_gaps = gaps[gaps > median_gap * float(config["health"]["video_timestamp_gap_factor"])] if np.isfinite(median_gap) else pd.Series(dtype=float)
    expected = image_codes(config)
    trigger_values = pd.to_numeric(log["Trigger"], errors="coerce").dropna().astype(int)
    image_values = trigger_values[trigger_values.isin(expected)]
    duplicate_triggers = sorted(image_values[image_values.duplicated(keep=False)].unique().tolist())
    missing_triggers = sorted(expected.difference(image_values.tolist()))
    reasons = []
    if unreadable or len(log) != frames or not experiment_time.is_monotonic_increasing or timestamps.duplicated().any() or len(large_gaps) or duplicate_triggers or missing_triggers:
        reasons.append(f"Video/log integrity: unreadable_sampled_frames={unreadable}, log_rows={len(log)}, video_frames={frames}, monotonic={experiment_time.is_monotonic_increasing}, duplicate_timestamps={int(timestamps.duplicated().sum())}, large_gaps={len(large_gaps)}, duplicate_image_triggers={duplicate_triggers}, missing_image_triggers={missing_triggers}")
    summary = {"status": _status(reasons), "reasons": reasons, "resolution": [width, height], "fps": fps, "frame_count": frames,
               "duration_s": frames / fps if fps else None, "sampled_readability_frames": int(len(positions)), "unreadable_sampled_frames": unreadable,
               "frame_log_rows": int(len(log)), "frame_log_matches_video": len(log) == frames, "experiment_time_monotonic": bool(experiment_time.is_monotonic_increasing),
               "duplicate_timestamps": int(timestamps.duplicated().sum()), "median_frame_log_gap_s": median_gap,
               "large_timestamp_gaps": int(len(large_gaps)), "frame_count_contiguous": bool(numeric_frames.dropna().astype(int).tolist() == list(range(1, len(log) + 1))),
               "image_trigger_count": int(len(image_values)), "duplicate_image_triggers": duplicate_triggers, "missing_image_triggers": missing_triggers}
    rows = [{"domain": "video", "name": "container", "count": frames, "status": summary["status"]}]
    logger.info("Video health: %s; %d logged frames", summary["status"], len(log))
    return summary, rows


def synchronization_health(paths, config: dict[str, Any], logger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = available_image_codes(config)
    ratings = pd.read_csv(paths["ratings"], encoding="utf-8-sig", na_values=config["ratings"]["missing_tokens"])
    rating_codes = set(ratings["trigger_sent"].dropna().astype(int))
    raw, events, _ = load_eeg_session(paths, config, preload=False)
    eeg = {int(event[2]): float(event[0]) / raw.info["sfreq"] for event in events if int(event[2]) in expected}
    with paths["vision_log"].open(encoding="utf-8-sig", newline="") as handle:
        log = list(csv.DictReader(handle))
    video = {int(float(row["Trigger"])): float(row["Experiment_Time"]) for row in log if row.get("Trigger", "").strip() not in ("", "0", "0.0") and int(float(row["Trigger"])) in expected}
    shared = sorted(expected.intersection(rating_codes, eeg, video))
    x = np.asarray([eeg[code] for code in shared]); offset = np.asarray([video[code] - eeg[code] for code in shared])
    constant = float(offset.mean()); constant_residual = offset - constant
    slope, intercept = np.polyfit(x, offset, 1); linear_residual = offset - (intercept + slope * x)
    cfg = config["health"]["sync"]
    pairs = []
    for code, eeg_time, video_time, off, c_res, l_res in zip(shared, x, [video[c] for c in shared], offset, constant_residual, linear_residual):
        pairs.append({"trigger": int(code), "eeg_onset_s": float(eeg_time), "video_onset_s": float(video_time), "video_minus_eeg_s": float(off),
                      "constant_residual_s": float(c_res), "linear_predicted_offset_s": float(intercept + slope * eeg_time), "linear_residual_s": float(l_res),
                      "linear_outlier": bool(abs(l_res) > float(cfg["outlier_residual_s"]))})
    linear_rmse = float(np.sqrt(np.mean(linear_residual ** 2))); linear_max = float(np.max(np.abs(linear_residual)))
    reasons = []
    if config.get("trials", {}).get("allow_incomplete_image_trials", False):
        reasons.append(f"Approved incomplete session: known_missing_triggers={config['trials']['known_missing_triggers']}")
    if len(shared) != len(expected): reasons.append(f"Only {len(shared)} of {len(expected)} expected triggers are shared")
    if linear_rmse > float(cfg["linear_rmse_warning_s"]) or linear_max > float(cfg["linear_max_residual_warning_s"]): reasons.append(f"Linear residuals exceed threshold: rmse={linear_rmse:.4f}s, max={linear_max:.4f}s")
    summary = {"status": _status(reasons), "reasons": reasons, "expected_trials": len(expected), "ratings_trigger_count": len(rating_codes.intersection(expected)),
               "eeg_trigger_count": len(set(eeg)), "video_trigger_count": len(set(video)), "shared_trigger_count": len(shared),
               "missing_from_ratings": sorted(expected.difference(rating_codes)), "missing_from_eeg": sorted(expected.difference(eeg)), "missing_from_video": sorted(expected.difference(video)),
               "constant_offset": {"offset_s": constant, "rmse_s": float(np.sqrt(np.mean(constant_residual ** 2))), "max_abs_residual_s": float(np.max(np.abs(constant_residual)))},
               "linear_drift": {"intercept_s": float(intercept), "drift_s_per_s": float(slope), "rmse_s": linear_rmse, "max_abs_residual_s": linear_max,
                                "outlier_triggers": [row["trigger"] for row in pairs if row["linear_outlier"]]},
               "recommended_model": "linear_drift" if not reasons else "manual_review"}
    logger.info("Synchronization health: %s; linear RMSE %.4f s", summary["status"], linear_rmse)
    return summary, pairs


def markdown_report(participant: str, summary: dict[str, Any]) -> str:
    lines = [f"# {participant} Image Experiment health report", "", "## Overall status", "", f"`{summary['status']}`", ""]
    for name in ("eeg", "ratings", "video", "synchronization"):
        item = summary[name]
        lines.extend([f"## {name.title()}", "", f"Status: `{item['status']}`", ""])
        for reason in item.get("reasons", []): lines.append(f"- {reason}")
        if not item.get("reasons"): lines.append("- No warnings produced by the configured checks.")
        lines.append("")
    lines.extend(["## Synchronization model", "", f"Recommended model: `{summary['synchronization']['recommended_model']}`", "",
                  f"Linear residual RMSE: {summary['synchronization']['linear_drift']['rmse_s']:.4f} s", ""])
    return "\n".join(lines)
