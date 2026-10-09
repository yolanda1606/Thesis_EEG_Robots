#!/usr/bin/env python3
"""Causal modality-agnostic replay with a frozen early Status offset.

This isolated experiment uses raw EEG/video but never fits a model.  Its
synchronization policy is pre-specified: match anchors available during the
first 15 seconds of the vision task clock, freeze the median vision-minus-EEG
offset, and never estimate or update drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
ROBOT_ROOT = ROOT / "processing/multimodal_robot"
for value in (ROOT, ROBOT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision

from processing.multimodal_image.src.eeg import _prepare_raw
from processing.multimodal_image.src.video import _facial_measures, _landmarker
from processing.multimodal_robot.analysis.causal_streaming.causal_filter import (
    SAMPLE_RATE_HZ, car_filter_stream, causal_sos, filter_design_metadata,
)
from processing.multimodal_robot.analysis.latency.replay_window import assemble_and_predict, summarize_timings
from processing.multimodal_robot.prep.continuous import (
    _extract_eeg_window_features, _files, _number, _status_events, image_defaults,
    read_vision_log, robot_eeg_config,
)
from processing.multimodal_robot.run_robot_pipeline import load, localize_task_paths
from processing.multimodal_robot.transfer import run_final_modality_agnostic_transfer as transfer


TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")
CALIBRATION_SECONDS = 15.0
WINDOW_SECONDS = 2.0
WINDOW_STEP_SECONDS = 1.0
CHUNK_MS = 100.0
KEYS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]
REFERENCE_DIAGNOSTIC_EXAMPLE_LIMIT = 5
VALIDATION_MODES = ("strict_historical", "corrected_timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default="P27")
    parser.add_argument("--participant-config", type=Path)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--image-reference-csv", type=Path)
    parser.add_argument("--alignment-metadata", type=Path)
    parser.add_argument("--reference-causal-output", type=Path,
                        help="Existing causal replay directory used only to validate frozen P(HIGH) and consensus equivalence.")
    parser.add_argument("--validation-mode", choices=VALIDATION_MODES, default="strict_historical",
                        help="strict_historical rejects every v1 difference; corrected_timestamp permits only face/multimodal differences in windows whose membership changed from the legacy trailing-frame fallback.")
    return parser.parse_args()


def participant_paths(participant: str) -> dict[str, Path]:
    """Resolve established participant-specific inputs without P27 assumptions."""
    lower = participant.lower()
    robot_run = ROOT / "derived" / participant / "Robot_Experiment/runs" / f"{lower}_robot_final_features"
    return {
        "participant_config": ROOT / "processing/multimodal_robot/configs/participants" / f"{participant}.yaml",
        "canonical_root": ROOT / "outputs/robot_transfer/modality_agnostic" / participant,
        "image_reference_csv": ROOT / "derived" / participant / "Image_Experiment/runs" / f"{lower}_no_ica/merged/{lower}_image_trial_dataset.csv",
        "alignment_metadata": robot_run / "alignment/alignment_metadata.json",
        "synchronization_anchors": robot_run / "alignment/synchronization_anchors.csv",
    }


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill participant paths while preserving the validated P27 pilot default."""
    paths = participant_paths(args.participant)
    for name in ("participant_config", "canonical_root", "image_reference_csv", "alignment_metadata"):
        if getattr(args, name) is None:
            setattr(args, name, paths[name])
    if args.output_dir is None:
        if args.participant != "P27":
            raise ValueError("--output-dir is required outside the preserved P27 pilot invocation")
        args.output_dir = ROOT / "outputs/robot_transfer/causal_modality_agnostic/v1/p27_status_calibrated"
    return args


def model_args(participant: str) -> SimpleNamespace:
    return SimpleNamespace(
        participant=participant,
        top3_csv=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target_modality.csv",
        image_model_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1",
        calibration_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1/modality_agnostic_probability_calibration_v1",
    )


def matched_anchor_pairs(eeg_anchors: list[tuple[str, float]], vision_anchors: list[tuple[str, float]]) -> pd.DataFrame:
    """Match identities and occurrences exactly as the canonical aligner does."""
    eeg_by_code: dict[str, list[float]] = {}
    vision_by_code: dict[str, list[float]] = {}
    for code, moment in eeg_anchors:
        eeg_by_code.setdefault(str(code), []).append(float(moment))
    for code, moment in vision_anchors:
        vision_by_code.setdefault(str(code), []).append(float(moment))
    rows = [
        {"event_identity": code, "occurrence": occurrence, "eeg_time_s": eeg_time,
         "vision_time_s": vision_time, "vision_minus_eeg_s": vision_time - eeg_time}
        for code in sorted(set(eeg_by_code).intersection(vision_by_code))
        for occurrence, (eeg_time, vision_time) in enumerate(zip(eeg_by_code[code], vision_by_code[code]), start=1)
    ]
    return pd.DataFrame(rows).sort_values(["vision_time_s", "eeg_time_s"], kind="stable").reset_index(drop=True) if rows else pd.DataFrame(columns=["event_identity", "occurrence", "eeg_time_s", "vision_time_s", "vision_minus_eeg_s"])


def fixed_offset_calibration(eeg_anchors: list[tuple[str, float]], vision_anchors: list[tuple[str, float]], calibration_seconds: float = CALIBRATION_SECONDS) -> tuple[float, pd.DataFrame]:
    """Freeze the median offset from anchors available by the calibration end."""
    pairs = matched_anchor_pairs(eeg_anchors, vision_anchors)
    early = pairs.loc[pairs.vision_time_s <= calibration_seconds + 1e-12].copy()
    if early.empty:
        raise ValueError(f"No matched Status/vision anchor is available in the first {calibration_seconds:g} s")
    offset = float(early.vision_minus_eeg_s.median())
    early["used_for_frozen_offset"] = True
    return offset, early


def prediction_window_starts(duration_s: float, calibration_eeg_end_s: float) -> list[float]:
    """Return canonical-grid windows wholly after calibration has finished."""
    return [float(start) for start in np.arange(0.0, duration_s - WINDOW_SECONDS + 1e-9, WINDOW_STEP_SECONDS)
            if start >= calibration_eeg_end_s - 1e-12]


def metadata(participant: str, task: str, segment_id: int, start: float) -> dict[str, Any]:
    return {"participant": participant, "task": task, "segment_id": segment_id,
            "window_id": f"{task}_s{segment_id}_{start:.3f}", "window_start_s": start, "window_end_s": start + WINDOW_SECONDS}


def face_window_features(frames: pd.DataFrame, start_s: float, end_s: float) -> dict[str, float]:
    """Aggregate only already-observed frames in the half-open current window."""
    selected = frames.loc[(frames.eeg_time_s >= start_s) & (frames.eeg_time_s < end_s)]
    if len(selected) and not bool((selected.eeg_time_s < end_s).all()):
        raise AssertionError("Face window accessed a future frame")
    detected = selected.loc[selected.face_detected]
    result: dict[str, float] = {"video_frame_count": float(len(selected)), "face_detected_frames": float(len(detected)),
                                "face_detection_rate": float(len(detected) / len(selected)) if len(selected) else np.nan}
    for name in ("irisdo_norm", "eso_norm", "enso_norm", "mnso_norm", "mwo_norm"):
        values = pd.to_numeric(detected[name], errors="coerce").dropna() if name in detected else pd.Series(dtype=float)
        result[f"video_{name}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"video_{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return result


def source_video_timestamp(
    frame_index: int,
    source_fps: float,
    log_times: dict[int, float],
    previous_video_time_s: float | None,
) -> tuple[float, str]:
    """Return the causal source-video timestamp and its provenance.

    Vision-log timestamps remain authoritative whenever available.  If a video
    has trailing frames without a corresponding log row, continue from the
    last observed log-clock timestamp by one source-frame interval.  This does
    not use a later log entry or fit a new synchronization model.
    """
    mapped_time = log_times.get(frame_index + 1)
    if mapped_time is not None:
        return float(mapped_time), "vision_log"
    if previous_video_time_s is not None:
        return float(previous_video_time_s + 1.0 / source_fps), "source_fps_continuation"
    return float(frame_index / source_fps), "source_fps_no_vision_log"


def historical_source_video_timestamp(frame_index: int, source_fps: float, log_times: dict[int, float]) -> float:
    """Return the v1 source-video timestamp for assignment-audit comparison only."""
    return float(log_times.get(frame_index + 1, frame_index / source_fps))


class SequentialFaceReplay:
    """Decode and landmark frames only when their mapped EEG time is due."""
    def __init__(self, task: dict[str, Any], frozen_offset_s: float, crop: dict[str, int]):
        self.offset = frozen_offset_s; self.crop = crop; self.defaults = image_defaults()
        self.video_path = _files(task["video"])[0]; self.capture = cv2.VideoCapture(str(self.video_path))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0: raise ValueError(f"Video has no usable frame rate: {self.video_path}")
        rows, _, _ = read_vision_log(_files(task["vision_log"])[0])
        self.log_times = {int(frame): float(moment) for row in rows
                          for frame, moment in [(_number(row, ("Frame_Count", "frame_count", "frame_number")), _number(row, ("Experiment_Time", "experiment_time", "elapsed_time_s", "time_s", "timestamp_s")))]
                          if frame is not None and moment is not None}
        cfg = {**self.defaults, "video": {**self.defaults["video"], "crop": {
            "left": crop["x"], "top": crop["y"], "right": crop["x"] + crop["width"],
            "bottom": crop["y"] + crop["height"], "scale": 1.0, "approved": True}}}
        model = ROOT / "processing/multimodal_image/models" / self.defaults["resources"]["face_landmarker_filename"]
        self.detector = _landmarker(cfg, model, vision.RunningMode.VIDEO)
        self.frame_index = 0
        self.last_video_time_s: float | None = None
        # The bounded buffer is the only structure used for prediction
        # features.  Telemetry is retained separately for the requested audit.
        self.buffer: deque[dict[str, Any]] = deque()
        self.telemetry: list[dict[str, Any]] = []

    def _next_times(self) -> tuple[float, float, float, str]:
        decode_time = self.frame_index / self.fps
        video_time, provenance = source_video_timestamp(
            frame_index=self.frame_index,
            source_fps=self.fps,
            log_times=self.log_times,
            previous_video_time_s=self.last_video_time_s,
        )
        return decode_time, video_time - self.offset, video_time, provenance

    def advance_to(self, eeg_end_s: float) -> dict[str, float]:
        """Process frames strictly before ``eeg_end_s`` and return new work."""
        timing = {name: 0.0 for name in (
            "video_frame_access_decode_ms", "video_crop_ms", "face_image_prepare_ms",
            "face_landmark_ms", "face_geometric_measure_ms",
        )}
        timing["new_video_frame_count"] = 0.0
        while True:
            decode_time, eeg_time, video_time, timestamp_provenance = self._next_times()
            legacy_video_time = historical_source_video_timestamp(self.frame_index, self.fps, self.log_times)
            legacy_eeg_time = legacy_video_time - self.offset
            if eeg_time >= eeg_end_s - 1e-12:
                break
            started = perf_counter_ns(); ok, frame = self.capture.read(); decode_ms = (perf_counter_ns() - started) / 1e6; timing["video_frame_access_decode_ms"] += decode_ms
            if not ok: break
            started = perf_counter_ns(); region = frame[self.crop["y"]:self.crop["y"] + self.crop["height"], self.crop["x"]:self.crop["x"] + self.crop["width"]]; crop_ms = (perf_counter_ns() - started) / 1e6; timing["video_crop_ms"] += crop_ms
            started = perf_counter_ns(); rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB); image_prepare_ms = (perf_counter_ns() - started) / 1e6; timing["face_image_prepare_ms"] += image_prepare_ms
            started = perf_counter_ns()
            detection = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_region), int(round(decode_time * 1000)))
            landmark_ms = (perf_counter_ns() - started) / 1e6; timing["face_landmark_ms"] += landmark_ms
            values: dict[str, float] = {}
            geometry_ms = 0.0
            if detection.face_landmarks:
                points = np.asarray([[point.x, point.y, point.z] for point in detection.face_landmarks[0]], dtype=np.float32)
                if points.shape == (478, 3):
                    started = perf_counter_ns(); values = _facial_measures(points, self.defaults["video"]["landmark_indices"], region.shape[1], region.shape[0]); geometry_ms = (perf_counter_ns() - started) / 1e6; timing["face_geometric_measure_ms"] += geometry_ms
            row = {"frame_index": self.frame_index, "source_fps": self.fps, "decode_time_s": decode_time,
                   "eeg_time_s": eeg_time, "video_time_s": video_time,
                   "legacy_eeg_time_s": legacy_eeg_time, "legacy_video_time_s": legacy_video_time,
                   "timestamp_provenance": timestamp_provenance, "face_detected": bool(values),
                   "video_frame_access_decode_ms": decode_ms, "video_crop_ms": crop_ms,
                   "face_image_prepare_ms": image_prepare_ms, "face_landmark_ms": landmark_ms,
                   "face_geometric_measure_ms": geometry_ms,
                   "frame_total_compute_ms": decode_ms + crop_ms + image_prepare_ms + landmark_ms + geometry_ms, **values}
            self.buffer.append(row); self.telemetry.append(row)
            self.last_video_time_s = video_time
            self.frame_index += 1
            timing["new_video_frame_count"] += 1.0
        return timing

    def window_features(self, start_s: float, end_s: float) -> tuple[dict[str, float], dict[str, float]]:
        """Return exact half-open window features from a bounded causal buffer."""
        started = perf_counter_ns()
        while self.buffer and self.buffer[0]["eeg_time_s"] < start_s:
            self.buffer.popleft()
        # ``advance_to`` has already stopped before ``end_s``.  This assertion
        # is a guard against accidental future-frame access during refactors.
        if self.buffer and self.buffer[-1]["eeg_time_s"] >= end_s - 1e-12:
            raise AssertionError("Face buffer contains a future frame")
        frame_table = pd.DataFrame(list(self.buffer))
        if frame_table.empty:
            # Preserve the established missing-face representation rather than
            # filling values when a valid EEG window has no mapped frames.
            frame_table = pd.DataFrame(columns=["eeg_time_s", "face_detected"])
        buffer_ms = (perf_counter_ns() - started) / 1e6
        started = perf_counter_ns()
        values = face_window_features(frame_table, start_s, end_s)
        aggregate_ms = (perf_counter_ns() - started) / 1e6
        return values, {"face_buffer_prepare_ms": buffer_ms, "face_window_aggregation_ms": aggregate_ms,
                        "face_buffer_frame_count": float(len(self.buffer))}

    def frame_telemetry(self) -> pd.DataFrame:
        return pd.DataFrame(self.telemetry)

    def close(self) -> None:
        self.capture.release(); self.detector.close()


def route_source(modality: str, eeg: dict[str, Any], face: dict[str, Any], multimodal: dict[str, Any] | None = None) -> dict[str, Any]:
    if modality == "eeg": return eeg
    if modality == "face": return face
    if modality == "multimodal": return multimodal if multimodal is not None else eeg | face
    raise ValueError(f"Unsupported frozen modality: {modality}")


def validate_route_schema(record: dict[str, Any], source: dict[str, Any]) -> None:
    missing = [name for name in record["candidate_features"] if name not in source]
    if missing: raise ValueError(f"{record['target']} rank {record['model_rank']} {record['modality']} schema missing: {missing}")
    ordered = [name for name in record["candidate_features"] if name in source]
    if ordered != record["candidate_features"]: raise ValueError("Frozen candidate-feature order changed")


def causal_eeg(raw: Any, task: str) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, float]]:
    """Chronologically CAR/filter raw EEG and retain per-chunk compute timing."""
    eeg_names = list(robot_eeg_config()["channels"]["eeg_mapping"].values())
    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE_HZ): raise ValueError(f"{task}: expected 250 Hz EEG")
    started = perf_counter_ns()
    data = raw.copy().pick(eeg_names).get_data()
    raw_extract_ms = (perf_counter_ns() - started) / 1e6
    filtered, chunks = car_filter_stream(data, causal_sos(), int(round(CHUNK_MS * SAMPLE_RATE_HZ / 1000)))
    return filtered, eeg_names, pd.DataFrame(chunks), {"raw_eeg_extract_ms": raw_extract_ms}


def eeg_features(filtered: np.ndarray, channels: list[str], start_s: float) -> tuple[dict[str, float], float]:
    first = int(round(start_s * SAMPLE_RATE_HZ)); data = filtered[:, first:first + int(WINDOW_SECONDS * SAMPLE_RATE_HZ)]
    if data.shape[1] != int(WINDOW_SECONDS * SAMPLE_RATE_HZ): raise ValueError("Incomplete EEG window")
    started = perf_counter_ns(); values = _extract_eeg_window_features(data, SAMPLE_RATE_HZ); elapsed = (perf_counter_ns() - started) / 1e6
    row = {f"{name}__{channel}": value for channel, features in zip(channels, values, strict=True) for name, value in features.items()}
    return row, elapsed


def canonical_alignments(task: str, path: Path) -> list[dict[str, Any]]:
    """Return canonical recording-level alignment decisions for one task."""
    items = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    item = next(row for row in items if row["task"] == task)
    if item.get("method") == "segmented":
        return [dict(segment) for segment in item.get("segments", [])]
    return [{**item, "segment_id": 1, "accepted": not str(item.get("method", "")).startswith(("invalid", "unsupported"))}]


def configured_segments(task: dict[str, Any]) -> list[tuple[int, Path]]:
    """Preserve the canonical order of independent configured EEG recordings."""
    return [(index, Path(path)) for index, path in enumerate(_files(task["eeg"]), start=1)]


def ambiguous_segment_mapping_reason(task: dict[str, Any]) -> str | None:
    """Reject multi-BDF tasks without an explicit segment-to-video mapping.

    The current participant schema supplies task-level video/log paths, not a
    verified mapping for each independently restarted EEG recording.  Reusing
    the first video/log would be scientifically unsafe.
    """
    if len(configured_segments(task)) > 1:
        return "ambiguous_multi_segment_video_log_mapping"
    return None


def exclusion_reason(canonical: dict[str, Any] | None, has_canonical_windows: bool,
                     eeg_path: Path | None, video_path: Path | None, log_path: Path | None) -> str | None:
    """Apply the pre-specified cohort eligibility rules before loading raw data."""
    if canonical is None or not bool(canonical.get("accepted", True)) or not has_canonical_windows:
        return "canonical_unavailable"
    if not str(canonical.get("method", "")).startswith("status_"):
        return "no_status_alignment"
    missing = [label for label, path in (("EEG", eeg_path), ("video", video_path), ("vision log", log_path))
               if path is None or not path.is_file()]
    if missing:
        return "missing_source:" + ",".join(missing)
    return None


def sync_diagnostics(task: str, segment_id: int, duration_s: float, frozen_offset: float, canonical: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    times = np.arange(0.0, duration_s + 1e-9, 1.0)
    canonical_offset = float(canonical["offset_s"]); drift = float(canonical.get("drift_s_per_s") or 0.0)
    signed = (times + frozen_offset) - (times + canonical_offset + drift * times); absolute = np.abs(signed)
    frame = pd.DataFrame({"task": task, "segment_id": segment_id, "eeg_time_s": times, "frozen_mapping_vision_time_s": times + frozen_offset,
                          "canonical_mapping_vision_time_s": times + canonical_offset + drift * times,
                          "signed_synchronization_error_s": signed, "absolute_synchronization_error_s": absolute})
    summary = {"task": task, "segment_id": segment_id, "canonical_alignment_type": canonical["method"], "canonical_offset_s": canonical_offset,
               "canonical_drift_s_per_s": drift, "synchronization_error_median_s": float(np.median(absolute)),
               "synchronization_error_p95_s": float(np.quantile(absolute, .95)), "synchronization_error_max_s": float(np.max(absolute))}
    return frame, summary


def timed_window_consensus(metadata_row: dict[str, Any], predictions: list[dict[str, Any]], target: str) -> dict[str, Any]:
    """Time the existing Definition-B per-window median without changing it."""
    rows = [row for row in predictions if row["target"] == target]
    if {row["model_rank"] for row in rows} != {1, 2, 3}:
        return metadata_row | {"target": target, "complete_top3": False, "consensus_compute_ms": np.nan,
                               "top3_consensus_probability": np.nan, "top3_consensus_class": pd.NA}
    started = perf_counter_ns()
    ordered = sorted(rows, key=lambda row: row["model_rank"])
    probability = float(np.median([row["high_probability"] for row in ordered]))
    hard = [int(row["original_hard_prediction"]) for row in ordered]
    elapsed = (perf_counter_ns() - started) / 1e6
    return metadata_row | {"target": target, "complete_top3": True, "consensus_compute_ms": elapsed,
                           "top3_consensus_probability": probability,
                           "top3_consensus_class": "HIGH" if probability >= 0.5 else "LOW",
                           "all_three_hard_agree": len(set(hard)) == 1}


def single_worker_schedule(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    """Deterministically schedule chronological frame/chunk/prediction jobs.

    Jobs use the EEG clock.  At the same timestamp, arriving data work is
    available before a released prediction; all jobs then share one
    non-preemptive worker and therefore propagate backlog exactly.
    """
    priority = {"eeg_chunk": 0, "video_frame": 0, "prediction_update": 1}
    worker_free_s = 0.0; rows = []
    for order, job in enumerate(sorted(jobs, key=lambda item: (float(item["arrival_s"]), priority[item["job_type"]], int(item["sequence"])))):
        arrival = max(0.0, float(job["arrival_s"]))
        start = max(worker_free_s, arrival)
        service_s = float(job["service_ms"]) / 1000.0
        completion = start + service_s
        deadline = float(job["deadline_s"]) if pd.notna(job.get("deadline_s", np.nan)) else np.nan
        lateness_ms = max(0.0, completion - deadline) * 1000.0 if np.isfinite(deadline) else np.nan
        rows.append(job | {"schedule_order": order, "arrival_s": arrival, "start_s": start,
                           "completion_s": completion, "waiting_ms": (start - arrival) * 1000.0,
                           "deadline_s": deadline, "lateness_ms": lateness_ms,
                           "deadline_met": bool(completion <= deadline + 1e-12) if np.isfinite(deadline) else pd.NA,
                           "response_ms": (completion - arrival) * 1000.0,
                           "backlog_after_ms": max(0.0, completion - arrival) * 1000.0})
        worker_free_s = completion
    return pd.DataFrame(rows)


def build_realtime_schedule(frame_timing: pd.DataFrame, chunk_timing: pd.DataFrame,
                            window_timing: pd.DataFrame) -> pd.DataFrame:
    """Build one shared-worker schedule from recorded compute durations."""
    jobs: list[dict[str, Any]] = []
    for row in frame_timing.itertuples(index=False):
        jobs.append({"job_type": "video_frame", "sequence": int(row.frame_index), "participant": row.participant,
                     "task": row.task, "segment_id": row.segment_id, "window_id": pd.NA,
                     "arrival_s": float(row.eeg_time_s), "service_ms": float(row.frame_total_compute_ms), "deadline_s": np.nan})
    for row in chunk_timing.itertuples(index=False):
        jobs.append({"job_type": "eeg_chunk", "sequence": int(row.chunk_start_sample), "participant": row.participant,
                     "task": row.task, "segment_id": row.segment_id, "window_id": pd.NA,
                     "arrival_s": float(row.chunk_end_sample) / SAMPLE_RATE_HZ,
                     "service_ms": float(row.causal_car_ms + row.causal_filter_ms), "deadline_s": np.nan})
    for row in window_timing.itertuples(index=False):
        jobs.append({"job_type": "prediction_update", "sequence": int(round(float(row.window_start_s) * 1000)),
                     "participant": row.participant, "task": row.task, "segment_id": row.segment_id,
                     "window_id": row.window_id, "arrival_s": float(row.window_end_s),
                     "service_ms": float(row.prediction_compute_ms), "deadline_s": float(row.window_end_s) + 1.0})
    return single_worker_schedule(jobs)


def delivery_accounting(window_timing: pd.DataFrame, consensus_predictions: pd.DataFrame,
                        realtime_schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report validity and timing jointly without changing scheduler semantics."""
    scheduled = realtime_schedule.loc[realtime_schedule.job_type.eq("prediction_update")].copy()
    scheduler_keys = ["participant", "task", "segment_id", "window_id"]
    schedule_columns = scheduler_keys + ["deadline_s", "completion_s", "waiting_ms", "response_ms", "lateness_ms", "deadline_met"]
    scheduled = scheduled.loc[:, schedule_columns]
    base = window_timing.loc[:, KEYS + ["valid_model_count"]].merge(scheduled, on=scheduler_keys, how="left", validate="one_to_one")
    present = consensus_predictions.loc[:, KEYS + ["target"]].drop_duplicates()
    flags = present.assign(valid_consensus=True).pivot(index=KEYS, columns="target", values="valid_consensus").reset_index()
    flags = flags.rename(columns={"valence": "valid_valence_consensus", "arousal": "valid_arousal_consensus"})
    delivery = base.merge(flags, on=KEYS, how="left", validate="one_to_one")
    for column in ("valid_valence_consensus", "valid_arousal_consensus"):
        if column not in delivery:
            delivery[column] = False
        else:
            delivery[column] = delivery[column].eq(True)
    delivery["scheduled_update"] = True
    delivery["computational_deadline_met"] = delivery["deadline_met"].fillna(False).astype(bool)
    delivery["complete_paired_consensus"] = delivery.valid_valence_consensus & delivery.valid_arousal_consensus
    delivery["complete_paired_delivered_on_time"] = delivery.complete_paired_consensus & delivery.computational_deadline_met
    summary = delivery.groupby(["participant", "task"], as_index=False).agg(
        scheduled_updates=("scheduled_update", "sum"),
        computationally_on_time_updates=("computational_deadline_met", "sum"),
        valid_valence_consensuses=("valid_valence_consensus", "sum"),
        valid_arousal_consensuses=("valid_arousal_consensus", "sum"),
        complete_paired_consensuses=("complete_paired_consensus", "sum"),
        complete_paired_delivered_on_time=("complete_paired_delivered_on_time", "sum"),
    )
    for column in ("computationally_on_time_updates", "valid_valence_consensuses", "valid_arousal_consensuses",
                   "complete_paired_consensuses", "complete_paired_delivered_on_time"):
        summary[f"{column}_rate"] = summary[column] / summary.scheduled_updates
    return delivery, summary


def validate_frame_telemetry(frame_timing: pd.DataFrame) -> None:
    """Ensure each chronological source frame is represented exactly once."""
    if frame_timing.empty:
        return
    keys = ["participant", "task", "segment_id", "frame_index"]
    if frame_timing.duplicated(keys).any():
        raise RuntimeError("Frame telemetry contains a decoded frame more than once")
    for _, part in frame_timing.groupby(["participant", "task", "segment_id"], sort=False):
        if not part.frame_index.is_monotonic_increasing or not part.eeg_time_s.is_monotonic_increasing:
            raise RuntimeError("Frame telemetry is not in chronological source order")


def summarize_realtime_timings(frame: pd.DataFrame, fields: list[str], groups: list[str]) -> pd.DataFrame:
    """Use the established summary schema and add the requested p99 column."""
    summary = summarize_timings(frame, fields, groups)
    p99: list[float] = []
    for row in summary.itertuples(index=False):
        mask = pd.Series(True, index=frame.index)
        for group in groups:
            mask &= frame[group].eq(getattr(row, group))
        values = pd.to_numeric(frame.loc[mask, row.metric], errors="coerce").dropna()
        p99.append(float(values.quantile(.99)) if len(values) else np.nan)
    summary["p99"] = p99
    return summary


def timestamp_assignment_audit(frame_timing: pd.DataFrame, window_timing: pd.DataFrame) -> pd.DataFrame:
    """Identify window memberships changed only by the corrected source clock."""
    columns = ["participant", "task", "segment_id", "frame_index", "timestamp_provenance",
               "legacy_video_time_s", "video_time_s", "legacy_eeg_time_s", "eeg_time_s",
               "window_id", "window_start_s", "window_end_s", "legacy_member", "corrected_member"]
    if frame_timing.empty or window_timing.empty or "legacy_eeg_time_s" not in frame_timing:
        return pd.DataFrame(columns=columns)
    changed = frame_timing.loc[~np.isclose(frame_timing.legacy_eeg_time_s, frame_timing.eeg_time_s,
                                             rtol=0.0, atol=1e-12)].copy()
    rows: list[dict[str, Any]] = []
    window_keys = window_timing.loc[:, KEYS].drop_duplicates()
    for frame in changed.itertuples(index=False):
        windows = window_keys.loc[(window_keys.participant.eq(frame.participant)) &
                                  (window_keys.task.eq(frame.task)) &
                                  (window_keys.segment_id.eq(frame.segment_id))]
        for window in windows.itertuples(index=False):
            legacy_member = bool(window.window_start_s <= frame.legacy_eeg_time_s < window.window_end_s)
            corrected_member = bool(window.window_start_s <= frame.eeg_time_s < window.window_end_s)
            if legacy_member != corrected_member:
                rows.append({"participant": frame.participant, "task": frame.task,
                             "segment_id": frame.segment_id, "frame_index": int(frame.frame_index),
                             "timestamp_provenance": frame.timestamp_provenance,
                             "legacy_video_time_s": float(frame.legacy_video_time_s),
                             "video_time_s": float(frame.video_time_s),
                             "legacy_eeg_time_s": float(frame.legacy_eeg_time_s),
                             "eeg_time_s": float(frame.eeg_time_s),
                             "window_id": window.window_id, "window_start_s": float(window.window_start_s),
                             "window_end_s": float(window.window_end_s), "legacy_member": legacy_member,
                             "corrected_member": corrected_member})
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["participant", "task", "segment_id", "frame_index", "window_start_s"], kind="stable"
    ).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)


def validate_reference_invariants(models: list[dict[str, Any]], calibration: pd.DataFrame,
                                  reference_dir: Path | None) -> dict[str, Any]:
    """Check frozen routes/features and fixed early-offset calibration against v1."""
    if reference_dir is None:
        return {"performed": False, "reason": "no --reference-causal-output supplied", "pass": True}
    reference_dir = reference_dir.resolve()
    manifest = json.loads((reference_dir / "run_manifest.json").read_text(encoding="utf-8"))
    fields = ("target", "rank", "modality", "classifier", "candidate_features", "selected_features",
              "model_path", "probability_source")
    current_routes = [{"target": item["target"], "rank": item["model_rank"], "modality": item["modality"],
                       "classifier": item["classifier"], "candidate_features": item["candidate_features"],
                       "selected_features": item["selected_features"], "model_path": str(item["model_path"]),
                       "probability_source": item["probability_source"]} for item in models]
    reference_routes = [{field: item.get(field) for field in fields} for item in manifest.get("routes", [])]
    routes_match = sorted(current_routes, key=lambda item: (item["target"], item["rank"])) == sorted(reference_routes, key=lambda item: (item["target"], item["rank"]))
    reference_offsets = pd.read_csv(reference_dir / "calibration_offsets.csv")
    calibration_keys = ["participant", "task", "segment_id"]
    columns = calibration_keys + ["frozen_offset_s", "early_calibration_anchor_count", "calibration_vision_end_s", "calibration_eeg_end_s"]
    observed = calibration.loc[:, columns].merge(reference_offsets.loc[:, columns], on=calibration_keys,
                                                   suffixes=("_replay", "_reference"), validate="one_to_one")
    coverage_match = len(observed) == len(calibration) == len(reference_offsets)
    numeric_fields = ("frozen_offset_s", "calibration_vision_end_s", "calibration_eeg_end_s")
    numeric_match = all(np.allclose(observed[f"{field}_replay"], observed[f"{field}_reference"], rtol=1e-12, atol=1e-12)
                        for field in numeric_fields)
    anchor_match = observed.early_calibration_anchor_count_replay.eq(observed.early_calibration_anchor_count_reference).all()
    return {"performed": True, "pass": bool(routes_match and coverage_match and numeric_match and anchor_match),
            "routes_match": bool(routes_match), "calibration_coverage_match": bool(coverage_match),
            "calibration_numeric_match": bool(numeric_match), "calibration_anchor_count_match": bool(anchor_match),
            "calibration_row_count": int(len(observed))}


def compare_reference_predictions(predictions: pd.DataFrame, consensus: pd.DataFrame,
                                  reference_dir: Path | None, validation_mode: str = "strict_historical",
                                  affected_windows: pd.DataFrame | None = None) -> dict[str, Any]:
    """Reject a pilot if it changes saved frozen causal P(HIGH) outputs."""
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(f"Unknown validation mode: {validation_mode}")
    if reference_dir is None:
        return {"performed": False, "reason": "no --reference-causal-output supplied"}

    def coverage_diagnostics(replay: pd.DataFrame, reference: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
        coverage = replay.loc[:, keys].merge(reference.loc[:, keys], on=keys, how="outer", indicator=True)
        replay_only = coverage.loc[coverage._merge.eq("left_only"), keys]
        reference_only = coverage.loc[coverage._merge.eq("right_only"), keys]
        ordered = lambda frame: frame.sort_values(keys, kind="stable").head(REFERENCE_DIAGNOSTIC_EXAMPLE_LIMIT).to_dict("records")
        return {"replay_only_count": int(len(replay_only)), "reference_only_count": int(len(reference_only)),
                "replay_only_examples": ordered(replay_only), "reference_only_examples": ordered(reference_only),
                # Coverage changes are small but scientifically important.  Keep
                # every key in the manifest rather than only bounded examples.
                "replay_only_keys": replay_only.sort_values(keys, kind="stable").to_dict("records"),
                "reference_only_keys": reference_only.sort_values(keys, kind="stable").to_dict("records")}

    def mismatch_examples(frame: pd.DataFrame, mask: np.ndarray, sort_keys: list[str], columns: list[str]) -> list[dict[str, Any]]:
        examples = frame.loc[mask, columns].sort_values(sort_keys, kind="stable").head(REFERENCE_DIAGNOSTIC_EXAMPLE_LIMIT)
        return examples.to_dict("records")

    def mismatch_records(frame: pd.DataFrame, mask: np.ndarray, sort_keys: list[str], columns: list[str]) -> list[dict[str, Any]]:
        return frame.loc[mask, columns].sort_values(sort_keys, kind="stable").to_dict("records")

    def maximum_absolute_difference(values: pd.Series | np.ndarray) -> float:
        values = pd.Series(values)
        finite = values.loc[np.isfinite(values)]
        return float(finite.max()) if len(finite) else (float("nan") if len(values) else 0.0)

    reference_dir = reference_dir.resolve()
    reference_predictions = pd.read_csv(reference_dir / "causal_window_predictions.csv")
    probability_keys = KEYS + ["target", "model_rank", "modality", "classifier"]
    replay_probabilities = predictions.loc[:, probability_keys + ["high_probability"]]
    saved_probabilities = reference_predictions.loc[:, probability_keys + ["high_probability"]]
    probability_coverage = coverage_diagnostics(replay_probabilities, saved_probabilities, probability_keys)
    left = replay_probabilities.merge(
        saved_probabilities, on=probability_keys,
        suffixes=("_replay", "_reference"), validate="one_to_one")
    replay_probability = pd.to_numeric(left.high_probability_replay, errors="coerce").to_numpy(dtype=float)
    reference_probability = pd.to_numeric(left.high_probability_reference, errors="coerce").to_numpy(dtype=float)
    probability_difference = np.abs(replay_probability - reference_probability)
    probability_close = np.isclose(replay_probability, reference_probability, rtol=1e-12, atol=1e-12)
    probability_mismatch_columns = probability_keys + ["high_probability_reference", "high_probability_replay"]
    left["absolute_probability_difference"] = probability_difference
    probability_mismatch_columns.append("absolute_probability_difference")

    reference_consensus = pd.read_csv(reference_dir / "causal_window_consensus.csv")
    consensus_keys = KEYS + ["target"]
    replay_consensus = consensus.loc[:, consensus_keys + ["top3_consensus_probability", "top3_consensus_class"]]
    saved_consensus = reference_consensus.loc[:, consensus_keys + ["top3_consensus_probability", "top3_consensus_class"]]
    consensus_coverage = coverage_diagnostics(replay_consensus, saved_consensus, consensus_keys)
    right = replay_consensus.merge(
        saved_consensus, on=consensus_keys,
        suffixes=("_replay", "_reference"), validate="one_to_one")
    replay_consensus_probability = pd.to_numeric(right.top3_consensus_probability_replay, errors="coerce").to_numpy(dtype=float)
    reference_consensus_probability = pd.to_numeric(right.top3_consensus_probability_reference, errors="coerce").to_numpy(dtype=float)
    consensus_difference = np.abs(replay_consensus_probability - reference_consensus_probability)
    consensus_close = np.isclose(replay_consensus_probability, reference_consensus_probability, rtol=1e-12, atol=1e-12)
    class_equal = right.top3_consensus_class_replay.eq(right.top3_consensus_class_reference)
    right["absolute_consensus_difference"] = consensus_difference
    consensus_mismatch_columns = consensus_keys + ["top3_consensus_probability_reference", "top3_consensus_probability_replay", "absolute_consensus_difference"]
    class_mismatch_columns = consensus_keys + ["top3_consensus_class_reference", "top3_consensus_class_replay"]
    coverage_complete = all(value == 0 for value in (
        probability_coverage["replay_only_count"], probability_coverage["reference_only_count"],
        consensus_coverage["replay_only_count"], consensus_coverage["reference_only_count"],
    ))
    affected_keys = (affected_windows.loc[:, KEYS].drop_duplicates() if affected_windows is not None and not affected_windows.empty
                     else pd.DataFrame(columns=KEYS))
    affected_key_set = set(map(tuple, affected_keys.itertuples(index=False, name=None)))

    def affected_window_mask(frame: pd.DataFrame) -> np.ndarray:
        return np.array([tuple(row) in affected_key_set for row in frame.loc[:, KEYS].itertuples(index=False, name=None)], dtype=bool)

    replay_only_model = replay_probabilities.merge(saved_probabilities.loc[:, probability_keys], on=probability_keys,
                                                    how="left", indicator=True).loc[lambda frame: frame._merge.eq("left_only")].drop(columns="_merge")
    replay_only_consensus = replay_consensus.merge(saved_consensus.loc[:, consensus_keys], on=consensus_keys,
                                                    how="left", indicator=True).loc[lambda frame: frame._merge.eq("left_only")].drop(columns="_merge")
    model_coverage_allowed = (affected_window_mask(replay_only_model)
                              & replay_only_model.modality.isin(("face", "multimodal")).to_numpy(dtype=bool)
                              & np.isfinite(pd.to_numeric(replay_only_model.high_probability, errors="coerce").to_numpy(dtype=float)))

    def complete_definition_b_consensus(row: pd.Series) -> bool:
        """Verify a newly complete consensus from its frozen prediction rows."""
        matching = predictions.loc[(predictions[KEYS] == row[KEYS]).all(axis=1) & predictions.target.eq(row.target)]
        if matching.model_rank.duplicated().any() or set(matching.model_rank) != {1, 2, 3}:
            return False
        probabilities = pd.to_numeric(matching.high_probability, errors="coerce")
        hard = pd.to_numeric(matching.original_hard_prediction, errors="coerce")
        if not np.isfinite(probabilities).all() or not np.isfinite(hard).all() or not hard.isin((0, 1)).all():
            return False
        expected_probability = float(np.median(probabilities.to_numpy()))
        expected_class = "HIGH" if expected_probability >= 0.5 else "LOW"
        return bool(np.isclose(float(row.top3_consensus_probability), expected_probability, rtol=1e-12, atol=1e-12)
                    and row.top3_consensus_class == expected_class)

    consensus_coverage_allowed = (affected_window_mask(replay_only_consensus)
                                  & np.array([complete_definition_b_consensus(row)
                                              for _, row in replay_only_consensus.iterrows()], dtype=bool))
    unexpected_model_coverage = ~model_coverage_allowed
    unexpected_consensus_coverage = ~consensus_coverage_allowed
    model_timestamp_affected = pd.Series(
        [tuple(row) in affected_key_set for row in left.loc[:, KEYS].itertuples(index=False, name=None)], index=left.index
    ) & left.modality.isin(("face", "multimodal"))
    consensus_timestamp_affected = pd.Series(
        [tuple(row) in affected_key_set for row in right.loc[:, KEYS].itertuples(index=False, name=None)], index=right.index
    )
    unexpected_model_mismatch = ~probability_close & ~model_timestamp_affected.to_numpy()
    unexpected_consensus_mismatch = ~consensus_close & ~consensus_timestamp_affected.to_numpy()
    unexpected_class_mismatch = ~class_equal.to_numpy() & ~consensus_timestamp_affected.to_numpy()
    strict_pass = bool(coverage_complete
                       and np.allclose(replay_probability, reference_probability, rtol=1e-12, atol=1e-12)
                       and np.allclose(replay_consensus_probability, reference_consensus_probability, rtol=1e-12, atol=1e-12)
                       and class_equal.all())
    corrected_timestamp_pass = bool(probability_coverage["reference_only_count"] == 0
                                    and consensus_coverage["reference_only_count"] == 0
                                    and not unexpected_model_coverage.any()
                                    and not unexpected_consensus_coverage.any()
                                    and not unexpected_model_mismatch.any()
                                    and not unexpected_consensus_mismatch.any() and not unexpected_class_mismatch.any())
    passed = strict_pass if validation_mode == "strict_historical" else corrected_timestamp_pass
    return {"performed": True, "probability_row_count": int(len(left)), "consensus_row_count": int(len(right)),
            "probability_maximum_absolute_difference": maximum_absolute_difference(probability_difference),
            "consensus_maximum_absolute_difference": maximum_absolute_difference(consensus_difference),
            "rtol": 1e-12, "atol": 1e-12, "pass": passed,
            "coverage": {"model": probability_coverage, "consensus": consensus_coverage},
            "validation_mode": validation_mode, "timestamp_affected_window_count": int(len(affected_keys)),
            "allowed_replay_only_model_row_count": int(model_coverage_allowed.sum()),
            "unexpected_replay_only_model_row_count": int(unexpected_model_coverage.sum()),
            "allowed_replay_only_consensus_row_count": int(consensus_coverage_allowed.sum()),
            "unexpected_replay_only_consensus_row_count": int(unexpected_consensus_coverage.sum()),
            "timestamp_affected_model_difference_count": int((~probability_close & model_timestamp_affected.to_numpy()).sum()),
            "timestamp_affected_consensus_difference_count": int((~consensus_close & consensus_timestamp_affected.to_numpy()).sum()),
            "unexpected_model_difference_count": int(unexpected_model_mismatch.sum()),
            "unexpected_consensus_difference_count": int(unexpected_consensus_mismatch.sum()),
            "unexpected_consensus_class_difference_count": int(unexpected_class_mismatch.sum()),
            "model_differing_probability_count": int((~probability_close).sum()),
            "model_probability_mismatch_examples": mismatch_examples(left, ~probability_close, probability_keys, probability_mismatch_columns),
            "model_probability_mismatches": mismatch_records(left, ~probability_close, probability_keys, probability_mismatch_columns),
            "consensus_differing_probability_count": int((~consensus_close).sum()),
            "consensus_probability_mismatch_examples": mismatch_examples(right, ~consensus_close, consensus_keys, consensus_mismatch_columns),
            "consensus_probability_mismatches": mismatch_records(right, ~consensus_close, consensus_keys, consensus_mismatch_columns),
            "changed_consensus_class_count": int((~class_equal).sum()),
            "consensus_class_mismatch_examples": mismatch_examples(right, ~class_equal, consensus_keys, class_mismatch_columns),
            "consensus_class_mismatches": mismatch_records(right, ~class_equal, consensus_keys, class_mismatch_columns)}


def reference_equivalence_failure_summary(result: dict[str, Any]) -> str:
    """Render a bounded deterministic diagnostic for the failure log."""
    details = {key: result[key] for key in (
        "probability_row_count", "consensus_row_count", "model_differing_probability_count",
        "probability_maximum_absolute_difference", "consensus_differing_probability_count",
        "consensus_maximum_absolute_difference", "changed_consensus_class_count", "coverage", "validation_mode",
        "timestamp_affected_window_count", "timestamp_affected_model_difference_count",
        "timestamp_affected_consensus_difference_count", "unexpected_model_difference_count",
        "unexpected_consensus_difference_count", "unexpected_consensus_class_difference_count",
        "allowed_replay_only_model_row_count", "unexpected_replay_only_model_row_count",
        "allowed_replay_only_consensus_row_count", "unexpected_replay_only_consensus_row_count",
        "model_probability_mismatch_examples", "consensus_probability_mismatch_examples",
        "consensus_class_mismatch_examples", "model_probability_mismatches",
        "consensus_probability_mismatches", "consensus_class_mismatches",
    )}
    return json.dumps(details, sort_keys=True, default=str)


def make_figures(output: Path, comparison: pd.DataFrame, synchronization: pd.DataFrame, latency: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for modality, part in comparison.groupby("modality"): ax.scatter(part.canonical_p_high, part.causal_p_high, s=13, alpha=.55, label=modality.title())
    ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="Canonical P(HIGH)", ylabel="Fixed-offset causal P(HIGH)", xlim=(0, 1), ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(output / "canonical_vs_causal_probability.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for task, part in synchronization.groupby("task"): ax.plot(part.eeg_time_s, 1000 * part.signed_synchronization_error_s, label=task)
    ax.axhline(0, color="black", ls=":"); ax.set(xlabel="EEG task time (s)", ylabel="Fixed − canonical mapping error (ms)"); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output / "synchronization_error_over_time.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(latency.total_prediction_path_compute_ms, bins=40, color="#4c78a8"); ax.axvline(1000, color="black", ls=":", label="1 s update budget"); ax.set(xlabel="Offline replay compute per prediction update (ms)", ylabel="Windows"); ax.legend(); fig.tight_layout(); fig.savefig(output / "latency_distribution.png", dpi=180); plt.close(fig)


def main() -> int:
    args = resolve_args(parse_args()); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    if len(set(args.tasks)) != len(args.tasks): raise ValueError("Tasks must not be repeated")
    config = localize_task_paths(load(args.participant_config.resolve()), args.raw_root.resolve())
    models = transfer.load_models(model_args(args.participant))
    routes = {(item["target"], item["model_rank"]): item["modality"] for item in models}
    expected_keys = {(target, rank) for target in ("arousal", "valence") for rank in (1, 2, 3)}
    if set(routes) != expected_keys: raise ValueError(f"Frozen modality routes are incomplete for {args.participant}: {routes}")
    prefix = args.participant
    canonical_predictions = pd.read_csv(args.canonical_root / f"{prefix}_window_predictions.csv")
    canonical_consensus = pd.read_csv(args.canonical_root / f"{prefix}_window_consensus.csv")
    canonical_tasks = pd.read_csv(args.canonical_root / f"{prefix}_task_consensus_analysis.csv")
    prediction_rows=[]; latency_rows=[]; calibration_rows=[]; anchor_rows=[]; sync_rows=[]; filter_rows=[]; excluded_rows=[]
    frame_tables=[]; timed_consensus_rows=[]; eeg_setup_rows=[]
    for task_name in args.tasks:
        task = config["tasks"][task_name]
        alignments = {int(item.get("segment_id", 1)): item for item in canonical_alignments(task_name, args.alignment_metadata.resolve())}
        segments = configured_segments(task)
        video_files = [Path(path) for path in _files(task.get("video"))]
        log_files = [Path(path) for path in _files(task.get("vision_log"))]
        video_path = video_files[0] if video_files else None; log_path = log_files[0] if log_files else None
        canonical_ids = set(pd.to_numeric(canonical_predictions.loc[canonical_predictions.task.eq(task_name), "segment_id"], errors="coerce").dropna().astype(int))
        mapping_reason = ambiguous_segment_mapping_reason(task)
        for segment_id, eeg_path in segments:
            canonical = alignments.get(segment_id)
            reason = mapping_reason or exclusion_reason(canonical, segment_id in canonical_ids, eeg_path, video_path, log_path)
            if reason:
                excluded_rows.append({"participant": args.participant, "task": task_name, "segment_id": segment_id,
                                      "reason": reason, "canonical_alignment_type": None if canonical is None else canonical.get("method"),
                                      "eeg_path": str(eeg_path), "video_path": str(video_path or ""), "vision_log_path": str(log_path or "")})
                continue
            raw, _, _ = _prepare_raw({"eeg": eeg_path}, robot_eeg_config())
            _, _, vision_anchors = read_vision_log(log_path)
            try:
                frozen_offset, early = fixed_offset_calibration(_status_events(raw, robot_eeg_config()), vision_anchors)
            except ValueError as error:
                excluded_rows.append({"participant": args.participant, "task": task_name, "segment_id": segment_id,
                                      "reason": "no_early_anchor", "detail": str(error), "canonical_alignment_type": canonical["method"],
                                      "eeg_path": str(eeg_path), "video_path": str(video_path), "vision_log_path": str(log_path)})
                continue
            calibration_eeg_end = CALIBRATION_SECONDS - frozen_offset
            early.insert(0, "segment_id", segment_id); early.insert(0, "task", task_name); early.insert(0, "participant", args.participant); anchor_rows.append(early)
            sync, summary = sync_diagnostics(task_name, segment_id, raw.n_times / SAMPLE_RATE_HZ, frozen_offset, canonical); sync_rows.append(sync)
            calibration_rows.append(summary | {"participant": args.participant, "early_calibration_anchor_count": len(early), "frozen_offset_s": frozen_offset,
                                               "calibration_vision_end_s": CALIBRATION_SECONDS, "calibration_eeg_end_s": calibration_eeg_end,
                                               "task_duration_s": raw.n_times / SAMPLE_RATE_HZ, "eeg_path": str(eeg_path),
                                               "video_path": str(video_path), "vision_log_path": str(log_path)})
            filtered, channels, chunk_timing, eeg_setup = causal_eeg(raw, task_name)
            eeg_setup_rows.append({"participant": args.participant, "task": task_name, "segment_id": segment_id, **eeg_setup})
            chunk_timing.insert(0, "segment_id", segment_id); chunk_timing.insert(0, "task", task_name); chunk_timing.insert(0, "participant", args.participant); filter_rows.append(chunk_timing)
            starts = prediction_window_starts(filtered.shape[1] / SAMPLE_RATE_HZ, calibration_eeg_end)
            face = SequentialFaceReplay(task, frozen_offset, config["video_settings"]["crop"])
            try:
                calibration_face_timing = face.advance_to(calibration_eeg_end)
                previous_end = calibration_eeg_end
                for start in starts:
                    end = start + WINDOW_SECONDS; meta = metadata(args.participant, task_name, segment_id, start)
                    new_face_timing = face.advance_to(end)
                    face_values, face_window_timing = face.window_features(start, end)
                    eeg_values, eeg_feature_ms = eeg_features(filtered, channels, start); eeg_row = meta | eeg_values; face_row = meta | face_values
                    combine_started = perf_counter_ns(); multimodal_row = eeg_row | face_values; multimodal_combine_ms = (perf_counter_ns() - combine_started) / 1e6
                    model_assembly_ms = multimodal_assembly_ms = inference_ms = routing_ms = schema_validation_ms = input_validation_ms = 0.0; valid_model_count = 0
                    window_prediction_rows: list[dict[str, Any]] = []
                    for record in models:
                        started = perf_counter_ns(); source = route_source(record["modality"], eeg_row, face_row, multimodal_row); routing_ms += (perf_counter_ns() - started) / 1e6
                        started = perf_counter_ns()
                        validate_route_schema(record, source)
                        schema_validation_ms += (perf_counter_ns() - started) / 1e6
                        started = perf_counter_ns()
                        candidates = pd.Series([source[name] for name in record["candidate_features"]], dtype=float)
                        finite = bool(np.isfinite(candidates).all())
                        input_validation_ms += (perf_counter_ns() - started) / 1e6
                        if not finite: continue
                        probability, hard, assembly_ms, transform_ms, hard_ms, probability_ms = assemble_and_predict(record, source)
                        transform_value = 0.0 if not np.isfinite(transform_ms) else transform_ms
                        model_inference_ms = transform_value + hard_ms + probability_ms
                        prediction = meta | {"target": record["target"], "model_rank": record["model_rank"], "modality": record["modality"],
                            "classifier": record["classifier"], "feature_family": record["feature_family"], "selected_feature_count": len(record["selected_features"]),
                            "frozen_image_balanced_accuracy": record["mean_outer_cv_balanced_accuracy"], "original_hard_prediction": hard, "high_probability": probability,
                            "feature_assembly_ms": assembly_ms, "fitted_transform_and_inference_ms": model_inference_ms}
                        prediction_rows.append(prediction); window_prediction_rows.append(prediction)
                        model_assembly_ms += assembly_ms; inference_ms += model_inference_ms; valid_model_count += 1
                        if record["modality"] == "multimodal": multimodal_assembly_ms += assembly_ms
                    chunk_start = max(previous_end, calibration_eeg_end) * SAMPLE_RATE_HZ; chunk_stop = end * SAMPLE_RATE_HZ
                    car_ms = float(chunk_timing.loc[(chunk_timing.chunk_start_sample >= chunk_start - 1e-9) & (chunk_timing.chunk_start_sample < chunk_stop - 1e-9), "causal_car_ms"].sum())
                    filter_ms = float(chunk_timing.loc[(chunk_timing.chunk_start_sample >= chunk_start - 1e-9) & (chunk_timing.chunk_start_sample < chunk_stop - 1e-9), "causal_filter_ms"].sum())
                    consensus_rows = [timed_window_consensus(meta, window_prediction_rows, target) for target in ("valence", "arousal")]
                    timed_consensus_rows.extend(consensus_rows)
                    consensus_ms = float(np.nansum([row["consensus_compute_ms"] for row in consensus_rows]))
                    face_new_frame_ms = sum(new_face_timing[name] for name in (
                        "video_frame_access_decode_ms", "video_crop_ms", "face_image_prepare_ms",
                        "face_landmark_ms", "face_geometric_measure_ms",
                    ))
                    face_total = face_new_frame_ms + face_window_timing["face_buffer_prepare_ms"] + face_window_timing["face_window_aggregation_ms"]
                    prediction_compute = eeg_feature_ms + face_window_timing["face_buffer_prepare_ms"] + face_window_timing["face_window_aggregation_ms"] + multimodal_combine_ms + routing_ms + schema_validation_ms + input_validation_ms + model_assembly_ms + inference_ms + consensus_ms
                    total = car_ms + filter_ms + prediction_compute + face_new_frame_ms
                    latency_rows.append(meta | {"valid_model_count": valid_model_count, "causal_car_increment_ms": car_ms, "causal_eeg_filter_increment_ms": filter_ms, "eeg_feature_ms": eeg_feature_ms,
                        **new_face_timing, **face_window_timing, "face_new_frame_processing_ms": face_new_frame_ms, "face_processing_ms": face_total,
                        "multimodal_feature_combine_ms": multimodal_combine_ms, "all_model_feature_assembly_ms": model_assembly_ms,
                        "multimodal_model_assembly_ms": multimodal_assembly_ms, "all_model_routing_ms": routing_ms,
                        "all_model_schema_validation_ms": schema_validation_ms, "all_model_input_validation_ms": input_validation_ms,
                        "valence_consensus_ms": consensus_rows[0]["consensus_compute_ms"], "arousal_consensus_ms": consensus_rows[1]["consensus_compute_ms"],
                        "all_consensus_ms": consensus_ms, "frozen_inference_ms": inference_ms, "prediction_compute_ms": prediction_compute,
                        "total_prediction_path_compute_ms": total})
                    previous_end = end
                telemetry = face.frame_telemetry()
                if not telemetry.empty:
                    telemetry.insert(0, "segment_id", segment_id); telemetry.insert(0, "task", task_name); telemetry.insert(0, "participant", args.participant); frame_tables.append(telemetry)
                calibration_rows[-1].update({f"calibration_{name}": value for name, value in calibration_face_timing.items() if name.endswith("_ms")})
            finally: face.close()
    predictions = pd.DataFrame(prediction_rows); latency = pd.DataFrame(latency_rows)
    if predictions.empty:
        raise ValueError(f"{args.participant}: no eligible recording produced a frozen-model prediction")
    complete, completeness = transfer.consensus(predictions, models)
    timed_consensus = pd.DataFrame(timed_consensus_rows)
    timed_complete = timed_consensus.loc[timed_consensus.complete_top3].copy()
    consensus_keys = KEYS + ["target"]
    timed_check = complete.loc[:, consensus_keys + ["top3_consensus_probability", "top3_consensus_class"]].merge(
        timed_complete.loc[:, consensus_keys + ["top3_consensus_probability", "top3_consensus_class"]], on=consensus_keys,
        suffixes=("_transfer", "_timed"), validate="one_to_one")
    if len(timed_check) != len(complete) or not np.allclose(timed_check.top3_consensus_probability_transfer, timed_check.top3_consensus_probability_timed, rtol=1e-15, atol=1e-15) or not timed_check.top3_consensus_class_transfer.eq(timed_check.top3_consensus_class_timed).all():
        raise RuntimeError("Timed consensus diverged from the existing frozen Definition-B implementation")
    consensus_predictions = complete.merge(timed_complete.loc[:, consensus_keys + ["consensus_compute_ms"]], on=consensus_keys, validate="one_to_one")
    frame_timing = pd.concat(frame_tables, ignore_index=True) if frame_tables else pd.DataFrame()
    validate_frame_telemetry(frame_timing)
    calibration = pd.DataFrame(calibration_rows)
    timestamp_audit = timestamp_assignment_audit(frame_timing, latency)
    reference_invariants = validate_reference_invariants(models, calibration, args.reference_causal_output)
    if reference_invariants.get("performed") and not reference_invariants.get("pass"):
        raise RuntimeError("Frozen route/feature or fixed-calibration validation failed against the historical reference. "
                           + json.dumps(reference_invariants, sort_keys=True, default=str))
    chunk_timing_all = pd.concat(filter_rows, ignore_index=True) if filter_rows else pd.DataFrame()
    realtime_schedule = build_realtime_schedule(frame_timing, chunk_timing_all, latency)
    scheduled_predictions = realtime_schedule.loc[realtime_schedule.job_type.eq("prediction_update")].copy()
    deadline_flags = scheduled_predictions.deadline_met.astype("boolean").fillna(False)
    deadline_summary = pd.DataFrame([{
        "participant": args.participant, "prediction_updates": int(len(scheduled_predictions)),
        "deadline_s_after_release": 1.0,
        "deadline_met_count": int(deadline_flags.sum()),
        "deadline_missed_count": int((~deadline_flags).sum()),
        "deadline_success_rate": float(deadline_flags.mean()) if len(scheduled_predictions) else np.nan,
        "maximum_prediction_waiting_ms": float(scheduled_predictions.waiting_ms.max()) if len(scheduled_predictions) else np.nan,
        "maximum_prediction_lateness_ms": float(scheduled_predictions.lateness_ms.max()) if len(scheduled_predictions) else np.nan,
        "maximum_shared_worker_backlog_ms": float(realtime_schedule.waiting_ms.max()) if len(realtime_schedule) else np.nan,
    }])
    delivery_by_window, delivery_summary = delivery_accounting(latency, consensus_predictions, realtime_schedule)
    reference_equivalence = compare_reference_predictions(
        predictions, complete, args.reference_causal_output, args.validation_mode, timestamp_audit
    )
    if reference_equivalence.get("performed") and not reference_equivalence.get("pass"):
        raise RuntimeError("Frozen prediction equivalence failed; refusing to write real-time feasibility output. "
                           + reference_equivalence_failure_summary(reference_equivalence))
    comparison_fields = KEYS + ["target", "model_rank", "modality", "classifier", "high_probability", "original_hard_prediction"]
    comparison = canonical_predictions.loc[canonical_predictions.task.isin(args.tasks), comparison_fields].merge(predictions[comparison_fields], on=KEYS + ["target", "model_rank", "modality", "classifier"], suffixes=("_canonical", "_causal"), validate="one_to_one")
    comparison = comparison.rename(columns={"high_probability_canonical":"canonical_p_high", "high_probability_causal":"causal_p_high", "original_hard_prediction_canonical":"canonical_hard_prediction", "original_hard_prediction_causal":"causal_hard_prediction"})
    comparison["probability_difference"] = comparison.causal_p_high - comparison.canonical_p_high; comparison["absolute_probability_difference"] = comparison.probability_difference.abs(); comparison["hard_agree"] = comparison.canonical_hard_prediction.eq(comparison.causal_hard_prediction)
    consensus_compare = canonical_consensus.loc[canonical_consensus.task.isin(args.tasks)].merge(complete, on=KEYS + ["target"], suffixes=("_canonical", "_causal"), validate="one_to_one")
    consensus_compare["consensus_probability_difference"] = consensus_compare.top3_consensus_probability_causal - consensus_compare.top3_consensus_probability_canonical; consensus_compare["consensus_class_agree"] = consensus_compare.top3_consensus_class_canonical.eq(consensus_compare.top3_consensus_class_causal)
    causal_task = complete.groupby(["participant", "task", "target"], as_index=False).agg(causal_P_task=("top3_consensus_probability", "median"), n_complete_causal_windows=("window_id", "size")); causal_task["causal_verdict"] = np.where(causal_task.causal_P_task >= .5, "HIGH", "LOW")
    paired_canonical_task = consensus_compare.groupby(["participant", "task", "target"], as_index=False).agg(canonical_paired_P_task=("top3_consensus_probability_canonical", "median")); paired_canonical_task["canonical_paired_verdict"] = np.where(paired_canonical_task.canonical_paired_P_task >= .5, "HIGH", "LOW")
    final_canonical = canonical_tasks.loc[canonical_tasks.task.isin(args.tasks), ["participant","task","target","median_top3_consensus_probability","descriptive_task_verdict"]].rename(columns={"median_top3_consensus_probability":"canonical_final_P_task","descriptive_task_verdict":"canonical_final_verdict"})
    task_compare = final_canonical.merge(paired_canonical_task, on=["participant","task","target"], validate="one_to_one").merge(causal_task, on=["participant","task","target"], validate="one_to_one"); task_compare["final_verdict_agree"] = task_compare.canonical_final_verdict.eq(task_compare.causal_verdict); task_compare["paired_verdict_agree"] = task_compare.canonical_paired_verdict.eq(task_compare.causal_verdict)
    synchronization = pd.concat(sync_rows, ignore_index=True) if sync_rows else pd.DataFrame()
    anchors = pd.concat(anchor_rows, ignore_index=True) if anchor_rows else pd.DataFrame()
    exclusions = pd.DataFrame(excluded_rows)
    timing_fields = [column for column in latency if column.endswith("_ms")]
    window_summary = summarize_realtime_timings(latency.assign(scope="window"), timing_fields, ["scope", "task"])
    frame_fields = [column for column in frame_timing if column.endswith("_ms")]
    frame_summary = summarize_realtime_timings(frame_timing.assign(scope="frame"), frame_fields, ["scope", "task"]) if len(frame_timing) else pd.DataFrame()
    chunk_fields = [column for column in chunk_timing_all if column.endswith("_ms")]
    chunk_summary = summarize_realtime_timings(chunk_timing_all.assign(scope="eeg_chunk"), chunk_fields, ["scope", "task"]) if len(chunk_timing_all) else pd.DataFrame()
    latency_summary = pd.concat([window_summary, frame_summary, chunk_summary], ignore_index=True, sort=False)
    budget_rows = []
    for scope, part in [("overall", latency), *[(str(task), group) for task, group in latency.groupby("task", sort=False)]]:
        values = pd.to_numeric(part["total_prediction_path_compute_ms"], errors="coerce").dropna()
        within = int(values.le(1000.0).sum()); over = int(values.gt(1000.0).sum())
        budget_rows.append({"scope": scope, "count": int(len(values)), "deadline_ms": 1000.0,
                            "n_le_deadline": within, "proportion_le_deadline": within / len(values) if len(values) else np.nan,
                            "n_gt_deadline": over, "proportion_gt_deadline": over / len(values) if len(values) else np.nan})
    latency_budget = pd.DataFrame(budget_rows)
    output.mkdir(parents=True)
    calibration.to_csv(output / "calibration_offsets.csv", index=False); anchors.to_csv(output / "calibration_anchors.csv", index=False); synchronization.to_csv(output / "synchronization_error.csv", index=False)
    comparison.to_csv(output / "model_window_comparison.csv", index=False); predictions.to_csv(output / "causal_window_predictions.csv", index=False); complete.to_csv(output / "causal_window_consensus.csv", index=False); completeness.to_csv(output / "top3_window_completeness.csv", index=False); consensus_compare.to_csv(output / "window_consensus_comparison.csv", index=False); task_compare.to_csv(output / "task_consensus_comparison.csv", index=False)
    # Stable aliases make the real-time feasibility artifacts self-contained.
    frame_timing.to_csv(output / "frame_timing.csv", index=False)
    timestamp_audit.to_csv(output / "timestamp_assignment_audit.csv", index=False)
    latency.to_csv(output / "window_timing.csv", index=False)
    predictions.to_csv(output / "model_probabilities.csv", index=False)
    consensus_predictions.to_csv(output / "consensus_predictions.csv", index=False)
    realtime_schedule.to_csv(output / "realtime_schedule.csv", index=False)
    deadline_summary.to_csv(output / "deadline_summary.csv", index=False)
    delivery_by_window.to_csv(output / "prediction_delivery_by_window.csv", index=False)
    delivery_summary.to_csv(output / "prediction_delivery_summary.csv", index=False)
    latency.to_csv(output / "latency_by_window.csv", index=False); latency_summary.to_csv(output / "latency_summary.csv", index=False); latency_budget.to_csv(output / "latency_budget_summary.csv", index=False); chunk_timing_all.to_csv(output / "causal_filter_chunk_timing.csv", index=False)
    pd.DataFrame(eeg_setup_rows).to_csv(output / "eeg_setup_timing.csv", index=False)
    exclusions.to_csv(output / "excluded_cases.csv", index=False)
    make_figures(output, comparison, synchronization, latency)
    manifest = {"purpose":"Causal modality-agnostic offline replay under a fixed pre-deployment Status calibration offset", "participant":args.participant, "tasks_requested":args.tasks,
        "eligible_recording_segments":int(len(calibration)), "eligible_participant_task_cases":int(calibration[["participant","task"]].drop_duplicates().shape[0]),
        "excluded_cases":exclusions.to_dict("records"), "calibration_policy":{"vision_task_clock_seconds":CALIBRATION_SECONDS,"matching":"event identity and occurrence","offset":"median(vision_time - eeg_time)","drift_model":None,"later_updates":False,"predictions":"window start must be at or after calibration end mapped to EEG time"},
        "routes":[{"target":m["target"],"rank":m["model_rank"],"modality":m["modality"],"classifier":m["classifier"],"candidate_features":m["candidate_features"],"selected_features":m["selected_features"],"model_path":str(m["model_path"]),"probability_source":m["probability_source"]} for m in models],
        "filter":filter_design_metadata(causal_sos()), "chunk_ms":CHUNK_MS, "window_seconds":WINDOW_SECONDS,"window_step_seconds":WINDOW_STEP_SECONDS,"threshold":0.5,"definition_b":"median across complete rank-1/2/3 window probabilities, then median across complete windows; HIGH if >=0.5",
        "robot_ratings_used":False,"training_refitting_or_selection":False,"latency_scope":"OFFLINE REPLAY COMPUTATION; excludes camera/EEG hardware, Bluetooth, driver/OS transport, and physical synchronization hardware",
        "realtime_scheduler":{"worker":"single non-preemptive shared compute worker","data_arrival":"recorded frame/EEG chunk timestamps on the EEG clock","prediction_release":"window end","prediction_deadline":"release plus 1.0 s","backlog":"carried across all frame, EEG chunk, and prediction jobs","sensor_or_driver_latency_simulated":False},
        "validation_mode":args.validation_mode, "reference_invariants":reference_invariants,
        "timestamp_assignment_audit":{"changed_membership_rows":int(len(timestamp_audit)), "affected_windows":int(len(timestamp_audit.loc[:, KEYS].drop_duplicates()))},
        "frozen_prediction_equivalence":reference_equivalence,
        "delivery_reporting":"scheduled, computational deadline, target validity, paired validity, and paired-on-time fields are reported per window and per task",
        "inputs":{"participant_config":str(args.participant_config.resolve()),"canonical_root":str(args.canonical_root.resolve()),"image_reference_csv":str(args.image_reference_csv.resolve()),"alignment_metadata":str(args.alignment_metadata.resolve()),"raw_root":str(args.raw_root.resolve())},
        "python_executable":sys.executable,"command":[sys.executable,*sys.argv],"calibration_records":calibration.to_dict("records")}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{args.participant} fixed-offset causal modality-agnostic replay complete: {output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
