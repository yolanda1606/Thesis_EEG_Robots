"""Continuous robot EEG/video feature extraction.

This module deliberately uses Status events only as clock landmarks.  It never
creates event-locked EEG epochs.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning

IMAGE_ROOT = Path(__file__).resolve().parents[2] / "multimodal_image"
if str(IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGE_ROOT))
from src.eeg import _prepare_raw
from src.video import _facial_measures, _landmarker
import mediapipe as mp
from mediapipe.tasks.python import vision


FIELDS = ("participant", "task", "category", "actual_condition", "actual_speed", "segment_id", "window_id", "window_start_s", "window_end_s", "alignment_method", "alignment_qc", "eeg_coverage", "video_coverage")

PRIMARY_MAX_RESIDUAL_S = 0.125
ISOLATED_RESIDUAL_EXCEPTION = {
    "minimum_retained_anchors": 20,
    "maximum_rmse_s": 0.070,
    "maximum_median_absolute_residual_s": 0.050,
    "maximum_p95_absolute_residual_s": 0.100,
    "maximum_anchors_above_primary": 1,
    "maximum_residual_s": 0.200,
}


@dataclass
class ProgressReporter:
    """Opt-in concise terminal reporting for long Robot processing runs."""

    show_progress: bool = False
    video_progress: bool = False
    eeg_progress: bool = False
    timing: bool = False
    writer: Callable[[str], None] = print
    started_at: float = field(default_factory=perf_counter)
    task_started_at: float | None = None
    completed_tasks: list[str] = field(default_factory=list)
    skipped_tasks: list[tuple[str, str]] = field(default_factory=list)

    def participant_start(self, participant: str, task_count: int) -> None:
        if self.show_progress:
            self.writer(f"[{participant}] Starting Robot processing: {task_count} tasks")

    def task_start(self, index: int, total: int, task_id: str) -> None:
        self.task_started_at = perf_counter()
        if self.show_progress or self.timing:
            self.writer(f"[{index}/{total}] {task_id}")

    def eeg(self, message: str) -> None:
        if self.eeg_progress:
            self.writer(f"[EEG] {message}")

    def alignment(self, qc: dict[str, Any]) -> None:
        if not self.show_progress:
            return
        labels = {
            "status_constant_offset": "constant offset",
            "status_linear": "linear offset",
            "filename_fallback": "filename fallback",
            "non_trigger_absolute_start": "absolute-start fallback",
        }
        rmse = qc.get("rmse_s")
        rmse_text = f", RMSE={float(rmse):.3f} s" if rmse is not None else ""
        self.writer(f"[ALIGN] {qc['matched_anchor_count']} anchors, {labels.get(qc['method'], qc['method'])}{rmse_text}")

    def video_start(self, total_frames: int) -> None:
        if self.video_progress:
            suffix = f" ({total_frames} frames)" if total_frames > 0 else ""
            self.writer(f"[VIDEO] MediaPipe face processing{suffix}")

    def video_frame(self, processed: int, total_frames: int, interval: int) -> None:
        if not self.video_progress or processed % interval != 0 and processed != total_frames:
            return
        if total_frames > 0:
            self.writer(f"[VIDEO] {processed}/{total_frames} frames ({100.0 * processed / total_frames:.1f}%)")
        else:
            self.writer(f"[VIDEO] {processed} frames")

    def features(self, windows: int) -> None:
        if self.show_progress:
            self.writer(f"[FEATURES] {windows} windows")

    def elapsed(self, label: str, started_at: float) -> None:
        if self.timing:
            self.writer(f"[TIMING] {label}: {perf_counter() - started_at:.1f} s")

    def task_done(self, task_id: str) -> None:
        self.completed_tasks.append(task_id)
        if (self.show_progress or self.timing) and self.task_started_at is not None:
            self.writer(f"[DONE] {task_id} in {perf_counter() - self.task_started_at:.1f} s")

    def task_skipped(self, task_id: str, reason: str) -> None:
        self.skipped_tasks.append((task_id, reason))
        if self.show_progress or self.timing:
            self.writer(f"[SKIPPED] {task_id}: {reason}")

    def participant_done(self, participant: str, destination: Path) -> None:
        if not (self.show_progress or self.timing):
            return
        self.writer(f"[{participant}] OK tasks: {', '.join(self.completed_tasks) or 'none'}")
        self.writer(f"[{participant}] Skipped tasks: {len(self.skipped_tasks)}")
        for task_id, reason in self.skipped_tasks:
            self.writer(f"  - {task_id}: {reason}")
        self.writer(f"[{participant}] Failed tasks: none")
        self.writer(f"[{participant}] Total runtime: {perf_counter() - self.started_at:.1f} s")
        self.writer(f"[{participant}] Output: {destination}")


def _continuous_ica_motion_edit(raw: mne.io.BaseRaw, eeg_names: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Apply the original continuous ICA/ACC source edit before windowing.

    This compatibility helper is retained here because the Image EEG module no
    longer exports it. Its implementation is unchanged from the robot
    pipeline's imported version, preserving the established ICA behavior.
    """
    if not config["eeg"]["ica"]["enabled"]:
        return {"enabled": False, "corrected_source_count": 0}
    eeg = raw.copy().pick(eeg_names)
    motion_names = [name for name in ("ACC X", "ACC Y", "ACC Z") if name in raw.ch_names]
    if len(motion_names) != 3:
        raise ValueError(f"ICA requires ACC X/Y/Z motion references; found {motion_names}")
    data = eeg.get_data().copy()
    rank = int(np.linalg.matrix_rank(data))
    if rank < 1:
        raise RuntimeError("ICA cannot be fitted: post-CAR EEG rank is zero")
    ica_config = config["eeg"]["ica"]
    ica = mne.preprocessing.ICA(n_components=rank, method=ica_config["method"], random_state=ica_config["random_seed"], verbose=False)
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        ica.fit(eeg, verbose=False)
    if any(issubclass(item.category, ConvergenceWarning) for item in fit_warnings):
        raise RuntimeError("Rank-aware continuous ICA did not converge; preprocessing stopped")
    if any("unstable mixing matrix" in str(item.message) for item in fit_warnings):
        raise RuntimeError("Rank-aware continuous ICA reported an unstable mixing matrix; preprocessing stopped")
    sources = ica.get_sources(eeg).get_data()
    motion = raw.copy().pick(motion_names).get_data()
    correlations = np.full((ica.n_components_, 3), np.nan)
    for axis in range(3):
        for component in range(ica.n_components_):
            correlation, _ = pearsonr(sources[component], motion[axis])
            correlations[component, axis] = correlation if np.isfinite(correlation) else np.nan
    means = np.nanmean(correlations, axis=0)
    standard_deviations = np.nanstd(correlations, axis=0)
    highpass_sos = butter(4, 3.0, btype="highpass", fs=float(raw.info["sfreq"]), output="sos")
    selected = []
    for component in range(ica.n_components_):
        axes = [motion_names[index] for index in range(3) if np.isfinite(correlations[component, index])
                and correlations[component, index] > means[index] + 2.0 * standard_deviations[index]]
        if axes:
            sources[component] = sosfiltfilt(highpass_sos, sources[component])
            selected.append({"component": component, "axes": axes})
    prewhitened = ica.pca_components_[:ica.n_components_].T @ (ica.mixing_matrix_ @ sources)
    if ica.pca_mean_ is not None:
        prewhitened += ica.pca_mean_[:, None]
    reconstructed = prewhitened * ica.pre_whitener_ if ica.noise_cov is None else np.linalg.pinv(ica.pre_whitener_, rcond=1e-14) @ prewhitened
    raw._data[[raw.ch_names.index(name) for name in eeg_names]] = reconstructed
    return {"enabled": True, "detected_eeg_rank": rank, "fitted_ica_components": int(ica.n_components_),
            "corrected_source_count": len(selected), "motion_related_components": selected}


def _extract_eeg_window_features(data: np.ndarray, sampling_hz: float) -> list[dict[str, float]]:
    """Calculate the original frozen continuous-window feature definitions."""
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


def image_defaults() -> dict[str, Any]:
    with (IMAGE_ROOT / "configs" / "pipeline" / "eeg_video_defaults.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def robot_eeg_config() -> dict[str, Any]:
    """Robot configuration for continuous no-ICA preparation.

    The shared Image defaults provide the channel mapping and filter settings,
    but Robot continuous processing intentionally never uses their ICA stage.
    """
    cfg = image_defaults()
    cfg["allow_single_sample_status"] = True
    cfg["eeg"]["ica"]["enabled"] = False
    # _prepare_raw/load_eeg_session requires an image-event range even though
    # robot Status values are neither image events nor unique IDs.  Keep the
    # sentinel outside the unsigned 16-bit BDF Status range so its image-only
    # duplicate guard does not reject repeated robot landmarks.
    cfg["events"] = {"image_ranges": {"robot_sentinel": [65536, 65536]}}
    return cfg


def prepare_continuous_robot_eeg(raw: mne.io.BaseRaw, progress: ProgressReporter | None = None) -> mne.io.BaseRaw:
    """Apply the validated Robot continuous no-ICA signal preparation.

    ``raw`` has already been loaded, mapped, and assigned its montage by
    ``_prepare_raw``.  Keep the recording continuous: no epoching, baseline
    correction, resampling, AutoReject, or ICA is applied here.
    """
    started_at = perf_counter()
    if progress:
        progress.eeg("Applying CAR + 1–40 Hz filter...")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    raw.filter(
        1.0,
        40.0,
        method="iir",
        iir_params={"order": 4, "ftype": "butter", "output": "sos"},
        verbose=False,
    )
    if progress:
        progress.eeg("CAR + 1–40 Hz filter complete")
        progress.elapsed("EEG CAR + filtering", started_at)
    return raw


def _files(item: Any) -> list[Path]:
    if isinstance(item, str): return [Path(item)]
    if isinstance(item, dict) and isinstance(item.get("segments"), list):
        return [Path(x["file"] if isinstance(x, dict) else x) for x in item["segments"]]
    return [Path(item["file"])] if isinstance(item, dict) and isinstance(item.get("file"), str) else []


def _number(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    lookup = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value not in (None, ""):
            try: return float(value)
            except ValueError: pass
    return None


def _parse_absolute(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Vision logs are recorded as local laboratory wall-clock time, while
        # BDF measurement dates are timezone-aware UTC values.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("Europe/Berlin"))
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def read_vision_log(path: Path) -> tuple[list[dict[str, str]], np.ndarray, list[tuple[str, float]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    times, anchors = [], []
    for row in rows:
        time = _number(row, ("Experiment_Time", "experiment_time", "elapsed_time_s", "time_s", "timestamp_s"))
        if time is not None: times.append(time)
        code = row.get("Trigger") or row.get("trigger") or row.get("Event") or row.get("event")
        if code and str(code).strip() not in ("", "0", "0.0") and time is not None:
            anchors.append((str(code).strip(), time))
    return rows, np.asarray(times, dtype=float), anchors


def _status_events(raw: mne.io.BaseRaw, cfg: dict[str, Any]) -> list[tuple[str, float]]:
    events = mne.find_events(raw, stim_channel=cfg["channels"]["stim_channel"], shortest_event=1, verbose="ERROR")
    return [(str(int(code)), float(sample) / raw.info["sfreq"]) for sample, _, code in events]


def _linear_model(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    slope, intercept = np.polyfit(x, y - x, 1)
    predicted = x + intercept + slope * x
    residuals = y - predicted
    return {"offset_s": float(intercept), "drift_s_per_s": float(slope), "rmse_s": float(np.sqrt(np.mean(residuals ** 2))), "max_residual_s": float(np.max(np.abs(residuals))), "residuals_s": residuals.tolist()}


def _filename_start(path: Path, session_date: str) -> datetime | None:
    """Parse recorder/video start stamps without participant-specific rules."""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", session_date)
    if not date_match: return None
    date = date_match.group(1)
    match = re.search(r"(?:_|-)(\d{2})[_-](\d{2})[_-](\d{2})(?:\.|_)", path.name)
    if not match: return None
    try: return datetime.fromisoformat(f"{date}T{':'.join(match.groups())}")
    except ValueError: return None


def _apply_status_acceptance(selected: dict[str, Any], retained_anchor_count: int, *, allow_isolated_residual_exception: bool) -> dict[str, Any]:
    """Apply the primary residual rule and the documented isolated exception.

    The exception is intentionally unavailable to segmented recordings, where
    each independent recorder segment retains the original strict criterion.
    ``residuals_s`` is internal fitting evidence and is not exported directly.
    """
    residuals = np.abs(np.asarray(selected.pop("residuals_s"), dtype=float))
    p95 = float(np.quantile(residuals, 0.95))
    above_primary = int(np.sum(residuals > PRIMARY_MAX_RESIDUAL_S))
    selected["p95_absolute_residual_s"] = p95
    selected["retained_anchors_above_primary_count"] = above_primary
    selected["fitted_method"] = selected["method"]
    if selected["max_residual_s"] <= PRIMARY_MAX_RESIDUAL_S:
        selected["acceptance_mode"] = "primary"
        selected["acceptance_reason"] = "maximum residual is within the primary 0.125 s criterion"
        return selected
    criteria = {
        "retained_anchors": retained_anchor_count >= ISOLATED_RESIDUAL_EXCEPTION["minimum_retained_anchors"],
        "rmse": selected["rmse_s"] <= ISOLATED_RESIDUAL_EXCEPTION["maximum_rmse_s"],
        "median_absolute_residual": selected["median_absolute_residual_s"] <= ISOLATED_RESIDUAL_EXCEPTION["maximum_median_absolute_residual_s"],
        "p95_absolute_residual": p95 <= ISOLATED_RESIDUAL_EXCEPTION["maximum_p95_absolute_residual_s"],
        "anchors_above_primary": above_primary <= ISOLATED_RESIDUAL_EXCEPTION["maximum_anchors_above_primary"],
        "maximum_residual": selected["max_residual_s"] <= ISOLATED_RESIDUAL_EXCEPTION["maximum_residual_s"],
    }
    selected["isolated_residual_exception_metrics"] = {
        "criteria": ISOLATED_RESIDUAL_EXCEPTION,
        "criteria_satisfied": criteria,
    }
    if allow_isolated_residual_exception and all(criteria.values()):
        selected["acceptance_mode"] = "isolated_residual_exception"
        selected["acceptance_reason"] = "primary maximum residual failed; all isolated-residual exception criteria satisfied"
        return selected
    selected["method"] = "invalid_residual"
    selected["acceptance_mode"] = "rejected"
    selected["acceptance_reason"] = "primary maximum residual failed; isolated-residual exception was unavailable or criteria were not all satisfied"
    return selected


def alignment_for_task(task: dict[str, Any], task_id: str, session_date: str, progress: ProgressReporter | None = None, *, allow_isolated_residual_exception: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[Path, mne.io.BaseRaw]]]:
    """Validate task timing and choose an offset/linear mapping conservatively."""
    cfg = robot_eeg_config(); eeg_files = _files(task["eeg"]); video = _files(task["video"])[0]; log = _files(task["vision_log"])[0]
    raws: list[tuple[Path, mne.io.BaseRaw]] = []
    anchors: list[tuple[str, float]] = []
    offset = 0.0
    eeg_started_at = perf_counter()
    if progress:
        progress.eeg("Loading...")
    for file in eeg_files:
        # Calling the frozen setup for each source avoids concatenating actual
        # recorder-off gaps.
        raw, _, _ = _prepare_raw({"eeg": file}, cfg)
        raws.append((file, raw)); anchors.extend(_status_events(raw, cfg))
    if progress:
        progress.eeg("Channel/montage preparation complete")
        progress.elapsed("EEG loading and channel/montage preparation", eeg_started_at)
    _, log_times, log_anchors = read_vision_log(log)
    cap = cv2.VideoCapture(str(video)); fps = float(cap.get(cv2.CAP_PROP_FPS)); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if fps <= 0: raise ValueError(f"{task_id}: video has no usable frame rate")
    # Some AVI codecs report a zero frame count although sequential decoding
    # works.  The vision log is the preferred task clock in that case.
    if frames > 0:
        video_duration = frames / fps
    elif len(log_times):
        video_duration = float(np.max(log_times))
    else:
        raise ValueError(f"{task_id}: video has no usable duration or vision-log clock")
    eeg_duration = sum(raw.n_times / raw.info["sfreq"] for _, raw in raws)
    # Match within each identity by order. Repeated labels are deliberately not
    # collapsed into a dictionary.
    log_by_code: dict[str, list[float]] = {}
    for code, time in log_anchors: log_by_code.setdefault(code, []).append(time)
    eeg_by_code: dict[str, list[float]] = {}
    for code, time in anchors: eeg_by_code.setdefault(code, []).append(time)
    pairs = [(code, occurrence, a, b) for code in sorted(set(log_by_code) & set(eeg_by_code))
             for occurrence, (a, b) in enumerate(zip(eeg_by_code[code], log_by_code[code]), start=1)]
    if pairs:
        x_all = np.asarray([p[2] for p in pairs]); y_all = np.asarray([p[3] for p in pairs]); offsets = y_all - x_all
        initial_offset = float(np.median(offsets)); initial_residuals = offsets - initial_offset
        mad = float(np.median(np.abs(initial_residuals - np.median(initial_residuals))))
        robust_threshold = max(0.050, 3 * 1.4826 * mad)
        retained = np.abs(initial_residuals - np.median(initial_residuals)) <= robust_threshold
        x, y = x_all[retained], y_all[retained]
        constant = y - x; fitted_offset = float(np.median(constant)); final_residuals = constant - fitted_offset
        constant_model = {"offset_s": fitted_offset, "drift_s_per_s": 0.0,
                          "rmse_s": float(np.sqrt(np.mean(final_residuals ** 2))), "max_residual_s": float(np.max(np.abs(final_residuals))),
                          "median_absolute_residual_s": float(np.median(np.abs(final_residuals))), "residuals_s": final_residuals.tolist()}
        selected = {**constant_model, "method": "status_constant_offset"}
        if len(x) >= 2 and np.ptp(x) > 0:
            linear = _linear_model(x, y)
            # Require a meaningful, not merely numerical, improvement.
            if abs(linear["drift_s_per_s"]) >= 1e-5 and linear["rmse_s"] + 0.005 < constant_model["rmse_s"]:
                selected = linear | {"median_absolute_residual_s": float(np.median(np.abs(linear["residuals_s"]))), "method": "status_linear"}
        offset = selected["offset_s"]
    elif len(log_times):
        # This is only valid when both recordings have explicit absolute starts.
        eeg_start = raws[0][1].info.get("meas_date")
        absolute_values = [_parse_absolute(row.get("Timestamp", "")) for row in read_vision_log(log)[0]]
        absolute_values = [x for x in absolute_values if x is not None]
        header_offset = float(eeg_start.timestamp() - absolute_values[0]) if eeg_start is not None and absolute_values else None
        eeg_file_start = _filename_start(eeg_files[0], session_date); video_file_start = _filename_start(video, session_date)
        if eeg_file_start and video_file_start and video_file_start.date() == eeg_file_start.date():
            filename_offset = float((eeg_file_start - video_file_start).total_seconds())
            video_eeg_start = -filename_offset
            plausible = video_eeg_start >= 0 and video_eeg_start < eeg_duration and video_eeg_start + video_duration > 0
            header_conflict = header_offset is None or abs(header_offset - filename_offset) > 5.0
            if plausible:
                selected = {"method": "filename_fallback" if header_conflict else "non_trigger_absolute_start", "offset_s": filename_offset,
                            # No anchors exist, so residual statistics are not
                            # measurable and must not imply perfect sync.
                            "drift_s_per_s": 0.0, "rmse_s": None, "max_residual_s": None, "median_absolute_residual_s": None,
                            "eeg_filename_start": eeg_file_start.isoformat(), "video_filename_start": video_file_start.isoformat(),
                            "header_offset_s": header_offset, "header_rejected_reason": "BDF/header and filename clocks conflict" if header_conflict else None}
                offset = filename_offset
            else:
                selected = {"method": "unsupported_non_trigger", "offset_s": None, "drift_s_per_s": None, "rmse_s": None, "max_residual_s": None, "median_absolute_residual_s": None}
        else:
            selected = {"method": "unsupported_non_trigger", "offset_s": None, "drift_s_per_s": None, "rmse_s": None, "max_residual_s": None, "median_absolute_residual_s": None}
    else:
        selected = {"method": "unsupported", "offset_s": None, "drift_s_per_s": None, "rmse_s": None, "max_residual_s": None, "median_absolute_residual_s": None}
    if selected["method"].startswith("status"):
        selected = _apply_status_acceptance(selected, int(np.sum(retained)), allow_isolated_residual_exception=allow_isolated_residual_exception)
    else:
        selected["acceptance_mode"] = "not_status_based"
        selected["acceptance_reason"] = "isolated-residual exception applies only to Status-based alignment"
    video_eeg_start = -offset if offset is not None else 0.0
    overlap = max(0.0, min(eeg_duration, video_eeg_start + video_duration) - max(0.0, video_eeg_start)) if offset is not None else 0.0
    qc = {"task": task_id, "eeg_duration_s": eeg_duration, "video_duration_s": video_duration, "status_event_count": len(anchors), "original_anchor_count": len(pairs), "retained_anchor_count": int(np.sum(retained)) if pairs else 0, "rejected_anchor_count": int(len(pairs)-np.sum(retained)) if pairs else 0, "matched_anchor_count": int(np.sum(retained)) if pairs else 0, "aligned_overlap_s": overlap, **selected}
    pair_rows = []
    for index, (c, occurrence, a, b) in enumerate(pairs):
        rejected = not bool(retained[index])
        pair_rows.append({"task": task_id, "event_identity": c, "occurrence": occurrence, "eeg_time_s": a, "vision_time_s": b, "vision_minus_eeg_s": b-a, "initial_residual_s": float(initial_residuals[index]), "retained": not rejected, "rejection_reason": "MAD residual exceeds robust threshold" if rejected else ""})
    return qc, pair_rows, raws


def has_eeg_segments(task: dict[str, Any]) -> bool:
    """Return whether a task explicitly declares independent EEG recordings."""
    eeg = task.get("eeg")
    return isinstance(eeg, dict) and isinstance(eeg.get("segments"), list)


def segment_accepted(qc: dict[str, Any]) -> bool:
    """Apply the existing task acceptance rule to one independent EEG segment."""
    return not str(qc["method"]).startswith(("unsupported", "invalid")) and float(qc["aligned_overlap_s"]) >= 2.0


def _segment_rejection_reason(qc: dict[str, Any]) -> str | None:
    if str(qc["method"]).startswith("invalid"):
        return "synchronization_qc_failed"
    if str(qc["method"]).startswith("unsupported"):
        return "synchronization_model_unsupported"
    if float(qc["aligned_overlap_s"]) < 2.0:
        return "aligned_overlap_below_2s"
    return None


def alignment_for_segmented_task(task: dict[str, Any], task_id: str, session_date: str, progress: ProgressReporter | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Align each explicitly configured EEG BDF independently to one vision log.

    This intentionally never concatenates recordings or assigns time to a
    recorder-off gap.  Each BDF retains its local sample clock and receives its
    own alignment model against the shared vision-log clock.
    """
    if not has_eeg_segments(task):
        raise ValueError("Segmented alignment requires eeg.segments")
    segment_results: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(task["eeg"]["segments"], start=1):
        if not isinstance(segment, dict) or not isinstance(segment.get("file"), str):
            raise ValueError(f"{task_id}: eeg.segments[{segment_id}] requires a file")
        segment_task = dict(task)
        segment_task["eeg"] = segment["file"]
        qc, pairs, raws = alignment_for_task(segment_task, task_id, session_date, progress, allow_isolated_residual_exception=False)
        accepted = segment_accepted(qc)
        segment_qc = {
            **qc,
            "segment_id": segment_id,
            "eeg_file": segment["file"],
            "accepted": accepted,
            "rejection_reason": None if accepted else _segment_rejection_reason(qc),
        }
        segment_results.append({"qc": segment_qc, "raws": raws})
        all_pairs.extend({**pair, "segment_id": segment_id} for pair in pairs)

    qcs = [result["qc"] for result in segment_results]
    accepted_qcs = [qc for qc in qcs if qc["accepted"]]
    rejected_qcs = [qc for qc in qcs if not qc["accepted"]]
    task_qc = {
        "task": task_id,
        "method": "segmented",
        "segment_count": len(qcs),
        "accepted_segment_count": len(accepted_qcs),
        "rejected_segment_count": len(rejected_qcs),
        "accepted": bool(accepted_qcs),
        # Aggregate only segments that passed QC; rejected recordings are
        # preserved below but never represented as valid task coverage.
        "eeg_duration_s": float(sum(qc["eeg_duration_s"] for qc in accepted_qcs)),
        "rejected_eeg_duration_s": float(sum(qc["eeg_duration_s"] for qc in rejected_qcs)),
        "video_duration_s": qcs[0]["video_duration_s"] if qcs else 0.0,
        "status_event_count": int(sum(qc["status_event_count"] for qc in qcs)),
        "original_anchor_count": int(sum(qc["original_anchor_count"] for qc in qcs)),
        "retained_anchor_count": int(sum(qc["retained_anchor_count"] for qc in accepted_qcs)),
        "rejected_anchor_count": int(sum(qc["rejected_anchor_count"] for qc in qcs)),
        "matched_anchor_count": int(sum(qc["matched_anchor_count"] for qc in accepted_qcs)),
        "aligned_overlap_s": float(sum(qc["aligned_overlap_s"] for qc in accepted_qcs)),
        "segments": qcs,
        "rejected_segments": [
            {"segment_id": qc["segment_id"], "reason": qc["rejection_reason"]}
            for qc in rejected_qcs
        ],
    }
    return task_qc, all_pairs, segment_results


def _metadata(participant: str, task_id: str, task: dict[str, Any], segment_id: int, start: float, end: float, qc: dict[str, Any], eeg_coverage: float, video_coverage: float) -> dict[str, Any]:
    condition = task.get("condition", {}) if isinstance(task.get("condition"), dict) else {}
    return {"participant": participant, "task": task_id, "category": task["category"], "actual_condition": condition.get("actual"), "actual_speed": condition.get("speed_actual"), "segment_id": segment_id, "window_id": f"{task_id}_s{segment_id}_{start:.3f}", "window_start_s": start, "window_end_s": end, "alignment_method": qc["method"], "alignment_qc": "PASS", "eeg_coverage": eeg_coverage, "video_coverage": video_coverage}


def extract_task(participant: str, task_id: str, task: dict[str, Any], qc: dict[str, Any], raws: list[tuple[Path, mne.io.BaseRaw]], resource_root: Path, progress: ProgressReporter | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if qc["method"].startswith(("unsupported", "invalid")) or qc["aligned_overlap_s"] < 2.0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cfg = robot_eeg_config(); defaults = image_defaults(); crop = task.get("_crop")
    if not crop: raise ValueError("Robot task requires a participant approved crop")
    video_file = _files(task["video"])[0]; cap = cv2.VideoCapture(str(video_file)); fps = float(cap.get(cv2.CAP_PROP_FPS)); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_rows, _, _ = read_vision_log(_files(task["vision_log"])[0])
    log_time_by_frame = {}
    for row in log_rows:
        frame = _number(row, ("Frame_Count", "frame_count", "frame_number"))
        moment = _number(row, ("Experiment_Time", "experiment_time", "elapsed_time_s", "time_s", "timestamp_s"))
        if frame is not None and moment is not None:
            log_time_by_frame[int(frame)] = moment
    frame_times: list[tuple[float, np.ndarray, bool, dict[str, float]]] = []
    model = resource_root / defaults["resources"]["face_landmarker_filename"]
    video_cfg = {**defaults, "video": {**defaults["video"], "crop": {"left": crop["x"], "top": crop["y"], "right": crop["x"]+crop["width"], "bottom": crop["y"]+crop["height"], "scale": 1.0, "approved": True},}}
    video_started_at = perf_counter()
    video_interval = max(1, total_frames // 20) if total_frames > 0 else 250
    if progress:
        progress.video_start(total_frames)
    with _landmarker(video_cfg, model, vision.RunningMode.VIDEO) as detector:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            decode_time = index / fps
            video_time = log_time_by_frame.get(index + 1, decode_time); index += 1
            eeg_time = (video_time - qc["offset_s"]) / (1 + qc["drift_s_per_s"])
            region = frame[crop["y"]:crop["y"]+crop["height"], crop["x"]:crop["x"]+crop["width"]]
            # MediaPipe VIDEO mode requires strictly monotonic timestamps.
            # Vision-log time remains the synchronization/window clock.
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(region, cv2.COLOR_BGR2RGB)), int(round(decode_time*1000)))
            found = bool(result.face_landmarks); values: dict[str, float] = {}
            if found:
                points = np.asarray([[p.x,p.y,p.z] for p in result.face_landmarks[0]], dtype=np.float32)
                if points.shape == (478,3): values = _facial_measures(points, defaults["video"]["landmark_indices"], region.shape[1], region.shape[0])
                else: found = False
            frame_times.append((eeg_time, np.empty(0), found, values))
            if progress and progress.video_progress:
                progress.video_frame(index, total_frames, video_interval)
    cap.release()
    if progress:
        progress.elapsed("Video/MediaPipe processing", video_started_at)
    eeg_rows: list[dict[str, Any]]=[]; video_rows: list[dict[str, Any]]=[]
    window_count = 0
    feature_started_at = perf_counter()
    if progress:
        progress.eeg("Window feature extraction...")
    for segment_id, (_, raw) in enumerate(raws, start=1):
        prepare_continuous_robot_eeg(raw, progress)
        if not task.get("bad_channels", []): pass
        duration = raw.n_times/raw.info["sfreq"]
        for start in np.arange(0, duration-2.0+1e-9, 1.0):
            first=int(round(start*raw.info["sfreq"])); data=raw.copy().pick(list(cfg["channels"]["eeg_mapping"].values())).get_data(start=first, stop=first+500)
            if data.shape[1] != 500: continue
            window_count += 1
            selected_frames=[item for item in frame_times if start <= item[0] < start+2.0]
            meta=_metadata(participant,task_id,task,segment_id,float(start),float(start+2),qc,1.0,1.0 if selected_frames else 0.0)
            for channel, row in zip(raw.copy().pick(list(cfg["channels"]["eeg_mapping"].values())).ch_names, _extract_eeg_window_features(data, raw.info["sfreq"])): eeg_rows.append({**meta,"channel":channel,"sample_count":500,**row})
            # Filename-fallback video has an explicit task-relative origin.
            # Emit only complete video [k, k+2) intervals at their mapped EEG
            # starts, rather than partial windows on either edge.
            if qc["method"] == "filename_fallback":
                video_start_eeg = -float(qc["offset_s"])
                if start < video_start_eeg or start + 2.0 > video_start_eeg + qc["video_duration_s"]:
                    continue
            detected=[x[3] for x in selected_frames if x[2]]; vr={**meta,"video_frame_count":len(selected_frames),"face_detected_frames":len(detected),"face_detection_rate":len(detected)/len(selected_frames) if selected_frames else np.nan}
            for name in ("irisdo_norm","eso_norm","enso_norm","mnso_norm","mwo_norm"):
                values=[d[name] for d in detected if name in d]; vr[f"video_{name}_mean"]=float(np.mean(values)) if values else np.nan; vr[f"video_{name}_std"]=float(np.std(values,ddof=1)) if len(values)>1 else np.nan
            video_rows.append(vr)
    eeg_df=pd.DataFrame(eeg_rows); video_df=pd.DataFrame(video_rows)
    if len(eeg_df):
        eeg_feature_names = [c for c in eeg_df if c.startswith("eeg_") and c != "eeg_coverage"]
        eeg_wide=eeg_df.pivot(index="window_id",columns="channel",values=eeg_feature_names)
        eeg_wide.columns=[f"{a}__{b}" for a,b in eeg_wide.columns]
        eeg_wide=eeg_wide.reset_index()
    else: eeg_wide=pd.DataFrame(columns=["window_id"])
    # A multimodal row is valid only when both streams cover the same interval.
    merged=video_df[video_df["video_coverage"] == 1.0].merge(eeg_wide,on="window_id",how="inner") if len(video_df) else pd.DataFrame()
    if progress:
        progress.features(window_count)
        progress.elapsed("EEG window feature extraction", feature_started_at)
    return eeg_df,video_df,merged


def _replace_segment_id(frame: pd.DataFrame, task_id: str, segment_id: int) -> pd.DataFrame:
    """Relabel an independently extracted single-BDF result with its real ID."""
    if segment_id == 1 or frame.empty:
        return frame
    result = frame.copy()
    if "segment_id" in result:
        result["segment_id"] = segment_id
    if "window_id" in result:
        result["window_id"] = result["window_id"].str.replace(f"{task_id}_s1_", f"{task_id}_s{segment_id}_", regex=False)
    return result


def _complete_segment_video_windows(video: pd.DataFrame, qc: dict[str, Any]) -> pd.DataFrame:
    """Keep only complete two-second video intervals within one segment model."""
    if video.empty:
        return video
    video_eeg_start = -float(qc["offset_s"])
    video_eeg_end = video_eeg_start + float(qc["video_duration_s"])
    mask = (video["window_start_s"] >= video_eeg_start - 1e-9) & (video["window_end_s"] <= video_eeg_end + 1e-9)
    return video.loc[mask].copy()


def extract_segmented_task(participant: str, task_id: str, task: dict[str, Any], segment_results: list[dict[str, Any]], resource_root: Path, progress: ProgressReporter | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract accepted BDF segments with their own synchronization models.

    The frozen single-segment extractor is called independently for each
    accepted recording.  This preserves its EEG/video/MediaPipe behavior while
    preventing a restarted recorder clock from being applied to another BDF.
    """
    eeg_tables: list[pd.DataFrame] = []
    video_tables: list[pd.DataFrame] = []
    merged_tables: list[pd.DataFrame] = []
    for result in segment_results:
        qc = result["qc"]
        if not qc["accepted"]:
            continue
        eeg, video, merged = extract_task(participant, task_id, task, qc, result["raws"], resource_root, progress)
        complete_video = _complete_segment_video_windows(video, qc)
        complete_ids = set(complete_video.get("window_id", pd.Series(dtype=str)))
        complete_merged = merged.loc[merged["window_id"].isin(complete_ids)].copy() if not merged.empty else merged
        segment_id = int(qc["segment_id"])
        eeg_tables.append(_replace_segment_id(eeg, task_id, segment_id))
        video_tables.append(_replace_segment_id(complete_video, task_id, segment_id))
        merged_tables.append(_replace_segment_id(complete_merged, task_id, segment_id))
    return (
        pd.concat(eeg_tables, ignore_index=True) if eeg_tables else pd.DataFrame(),
        pd.concat(video_tables, ignore_index=True) if video_tables else pd.DataFrame(),
        pd.concat(merged_tables, ignore_index=True) if merged_tables else pd.DataFrame(),
    )
