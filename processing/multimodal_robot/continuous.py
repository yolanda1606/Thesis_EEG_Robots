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
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import cv2
import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning

IMAGE_ROOT = Path(__file__).resolve().parents[1] / "multimodal_image"
if str(IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGE_ROOT))
from src.eeg import _prepare_raw
from src.video import _facial_measures, _landmarker
import mediapipe as mp
from mediapipe.tasks.python import vision


FIELDS = ("participant", "task", "category", "actual_condition", "actual_speed", "segment_id", "window_id", "window_start_s", "window_end_s", "alignment_method", "alignment_qc", "eeg_coverage", "video_coverage")


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
    """Minimal frozen-image configuration required by ``_prepare_raw``."""
    cfg = image_defaults()
    cfg["allow_single_sample_status"] = True
    # _prepare_raw/load_eeg_session requires an image-event range even though
    # robot Status values are neither image events nor unique IDs.  Keep the
    # sentinel outside the unsigned 16-bit BDF Status range so its image-only
    # duplicate guard does not reject repeated robot landmarks.
    cfg["events"] = {"image_ranges": {"robot_sentinel": [65536, 65536]}}
    return cfg


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


def alignment_for_task(task: dict[str, Any], task_id: str, session_date: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[Path, mne.io.BaseRaw]]]:
    """Validate task timing and choose an offset/linear mapping conservatively."""
    cfg = robot_eeg_config(); eeg_files = _files(task["eeg"]); video = _files(task["video"])[0]; log = _files(task["vision_log"])[0]
    raws: list[tuple[Path, mne.io.BaseRaw]] = []
    anchors: list[tuple[str, float]] = []
    offset = 0.0
    for file in eeg_files:
        # Calling the frozen setup for each source avoids concatenating actual
        # recorder-off gaps.
        raw, _, _ = _prepare_raw({"eeg": file}, cfg)
        raws.append((file, raw)); anchors.extend(_status_events(raw, cfg))
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
                          "median_absolute_residual_s": float(np.median(np.abs(final_residuals)))}
        selected = {**constant_model, "method": "status_constant_offset"}
        if len(x) >= 2 and np.ptp(x) > 0:
            linear = _linear_model(x, y)
            # Require a meaningful, not merely numerical, improvement.
            if abs(linear["drift_s_per_s"]) >= 1e-5 and linear["rmse_s"] + 0.005 < constant_model["rmse_s"]:
                selected = {k: v for k, v in linear.items() if k != "residuals_s"} | {"median_absolute_residual_s": float(np.median(np.abs(linear["residuals_s"]))), "method": "status_linear"}
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
    if selected["method"].startswith("status") and selected["max_residual_s"] > cfg["health"]["sync"]["linear_max_residual_warning_s"]:
        selected["method"] = "invalid_residual"
    video_eeg_start = -offset if offset is not None else 0.0
    overlap = max(0.0, min(eeg_duration, video_eeg_start + video_duration) - max(0.0, video_eeg_start)) if offset is not None else 0.0
    qc = {"task": task_id, "eeg_duration_s": eeg_duration, "video_duration_s": video_duration, "status_event_count": len(anchors), "original_anchor_count": len(pairs), "retained_anchor_count": int(np.sum(retained)) if pairs else 0, "rejected_anchor_count": int(len(pairs)-np.sum(retained)) if pairs else 0, "matched_anchor_count": int(np.sum(retained)) if pairs else 0, "aligned_overlap_s": overlap, **selected}
    pair_rows = []
    for index, (c, occurrence, a, b) in enumerate(pairs):
        rejected = not bool(retained[index])
        pair_rows.append({"task": task_id, "event_identity": c, "occurrence": occurrence, "eeg_time_s": a, "vision_time_s": b, "vision_minus_eeg_s": b-a, "initial_residual_s": float(initial_residuals[index]), "retained": not rejected, "rejection_reason": "MAD residual exceeds robust threshold" if rejected else ""})
    return qc, pair_rows, raws


def _metadata(participant: str, task_id: str, task: dict[str, Any], segment_id: int, start: float, end: float, qc: dict[str, Any], eeg_coverage: float, video_coverage: float) -> dict[str, Any]:
    condition = task.get("condition", {}) if isinstance(task.get("condition"), dict) else {}
    return {"participant": participant, "task": task_id, "category": task["category"], "actual_condition": condition.get("actual"), "actual_speed": condition.get("speed_actual"), "segment_id": segment_id, "window_id": f"{task_id}_s{segment_id}_{start:.3f}", "window_start_s": start, "window_end_s": end, "alignment_method": qc["method"], "alignment_qc": "PASS", "eeg_coverage": eeg_coverage, "video_coverage": video_coverage}


def extract_task(participant: str, task_id: str, task: dict[str, Any], qc: dict[str, Any], raws: list[tuple[Path, mne.io.BaseRaw]], resource_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if qc["method"].startswith(("unsupported", "invalid")) or qc["aligned_overlap_s"] < 2.0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cfg = robot_eeg_config(); defaults = image_defaults(); crop = task.get("_crop")
    if not crop: raise ValueError("Robot task requires a participant approved crop")
    video_file = _files(task["video"])[0]; cap = cv2.VideoCapture(str(video_file)); fps = float(cap.get(cv2.CAP_PROP_FPS))
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
    cap.release()
    eeg_rows: list[dict[str, Any]]=[]; video_rows: list[dict[str, Any]]=[]
    for segment_id, (_, raw) in enumerate(raws, start=1):
        eeg = raw.copy().pick(list(cfg["channels"]["eeg_mapping"].values()))
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
        raw.filter(1.0, 40.0, method="iir", iir_params={"order":4,"ftype":"butter","output":"sos"}, verbose=False)
        _continuous_ica_motion_edit(raw, list(cfg["channels"]["eeg_mapping"].values()), cfg)
        if not task.get("bad_channels", []): pass
        duration = raw.n_times/raw.info["sfreq"]
        for start in np.arange(0, duration-2.0+1e-9, 1.0):
            first=int(round(start*raw.info["sfreq"])); data=raw.copy().pick(list(cfg["channels"]["eeg_mapping"].values())).get_data(start=first, stop=first+500)
            if data.shape[1] != 500: continue
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
    return eeg_df,video_df,merged
