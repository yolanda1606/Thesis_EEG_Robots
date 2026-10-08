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
    SAMPLE_RATE_HZ, causal_sos, filter_design_metadata, filter_stream,
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
        self.frame_index = 0; self.records: list[dict[str, Any]] = []

    def _next_times(self) -> tuple[float, float]:
        decode_time = self.frame_index / self.fps
        video_time = self.log_times.get(self.frame_index + 1, decode_time)
        return decode_time, video_time - self.offset

    def advance_to(self, eeg_end_s: float) -> dict[str, float]:
        """Process frames strictly before ``eeg_end_s`` and return new work."""
        timing = {name: 0.0 for name in ("video_frame_access_decode_ms", "face_landmark_ms", "face_geometric_measure_ms")}
        while True:
            decode_time, eeg_time = self._next_times()
            if eeg_time >= eeg_end_s - 1e-12:
                break
            started = perf_counter_ns(); ok, frame = self.capture.read(); timing["video_frame_access_decode_ms"] += (perf_counter_ns() - started) / 1e6
            if not ok: break
            region = frame[self.crop["y"]:self.crop["y"] + self.crop["height"], self.crop["x"]:self.crop["x"] + self.crop["width"]]
            started = perf_counter_ns()
            detection = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(region, cv2.COLOR_BGR2RGB)), int(round(decode_time * 1000)))
            timing["face_landmark_ms"] += (perf_counter_ns() - started) / 1e6
            values: dict[str, float] = {}
            if detection.face_landmarks:
                points = np.asarray([[point.x, point.y, point.z] for point in detection.face_landmarks[0]], dtype=np.float32)
                if points.shape == (478, 3):
                    started = perf_counter_ns(); values = _facial_measures(points, self.defaults["video"]["landmark_indices"], region.shape[1], region.shape[0]); timing["face_geometric_measure_ms"] += (perf_counter_ns() - started) / 1e6
            self.records.append({"eeg_time_s": eeg_time, "video_time_s": eeg_time + self.offset, "face_detected": bool(values), **values})
            self.frame_index += 1
        return timing

    def frame_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def close(self) -> None:
        self.capture.release(); self.detector.close()


def route_source(modality: str, eeg: dict[str, Any], face: dict[str, Any]) -> dict[str, Any]:
    if modality == "eeg": return eeg
    if modality == "face": return face
    if modality == "multimodal": return eeg | face
    raise ValueError(f"Unsupported frozen modality: {modality}")


def validate_route_schema(record: dict[str, Any], source: dict[str, Any]) -> None:
    missing = [name for name in record["candidate_features"] if name not in source]
    if missing: raise ValueError(f"{record['target']} rank {record['model_rank']} {record['modality']} schema missing: {missing}")
    ordered = [name for name in record["candidate_features"] if name in source]
    if ordered != record["candidate_features"]: raise ValueError("Frozen candidate-feature order changed")


def causal_eeg(raw: Any, task: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    eeg_names = list(robot_eeg_config()["channels"]["eeg_mapping"].values())
    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE_HZ): raise ValueError(f"{task}: expected 250 Hz EEG")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    data = raw.copy().pick(eeg_names).get_data()
    filtered, chunks = filter_stream(data, causal_sos(), int(round(CHUNK_MS * SAMPLE_RATE_HZ / 1000)))
    return filtered, eeg_names, pd.DataFrame(chunks)


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
    for task_name in args.tasks:
        task = config["tasks"][task_name]
        alignments = {int(item.get("segment_id", 1)): item for item in canonical_alignments(task_name, args.alignment_metadata.resolve())}
        segments = configured_segments(task)
        video_files = [Path(path) for path in _files(task.get("video"))]
        log_files = [Path(path) for path in _files(task.get("vision_log"))]
        video_path = video_files[0] if video_files else None; log_path = log_files[0] if log_files else None
        canonical_ids = set(pd.to_numeric(canonical_predictions.loc[canonical_predictions.task.eq(task_name), "segment_id"], errors="coerce").dropna().astype(int))
        for segment_id, eeg_path in segments:
            canonical = alignments.get(segment_id)
            reason = exclusion_reason(canonical, segment_id in canonical_ids, eeg_path, video_path, log_path)
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
            filtered, channels, chunk_timing = causal_eeg(raw, task_name)
            chunk_timing.insert(0, "segment_id", segment_id); chunk_timing.insert(0, "task", task_name); chunk_timing.insert(0, "participant", args.participant); filter_rows.append(chunk_timing)
            starts = prediction_window_starts(filtered.shape[1] / SAMPLE_RATE_HZ, calibration_eeg_end)
            face = SequentialFaceReplay(task, frozen_offset, config["video_settings"]["crop"])
            try:
                calibration_face_timing = face.advance_to(calibration_eeg_end)
                previous_end = calibration_eeg_end
                for start in starts:
                    end = start + WINDOW_SECONDS; meta = metadata(args.participant, task_name, segment_id, start)
                    new_face_timing = face.advance_to(end)
                    frame_table = face.frame_table(); aggregate_started = perf_counter_ns(); face_values = face_window_features(frame_table, start, end); face_aggregation_ms = (perf_counter_ns() - aggregate_started) / 1e6
                    eeg_values, eeg_feature_ms = eeg_features(filtered, channels, start); eeg_row = meta | eeg_values; face_row = meta | face_values
                    combine_started = perf_counter_ns(); _ = eeg_row | face_values; multimodal_combine_ms = (perf_counter_ns() - combine_started) / 1e6
                    model_assembly_ms = multimodal_assembly_ms = inference_ms = 0.0; valid_model_count = 0
                    for record in models:
                        source = route_source(record["modality"], eeg_row, face_row)
                        validate_route_schema(record, source)
                        candidates = pd.Series([source[name] for name in record["candidate_features"]], dtype=float)
                        if not np.isfinite(candidates).all(): continue
                        probability, hard, assembly_ms, transform_ms, hard_ms, probability_ms = assemble_and_predict(record, source)
                        transform_value = 0.0 if not np.isfinite(transform_ms) else transform_ms
                        model_inference_ms = transform_value + hard_ms + probability_ms
                        prediction_rows.append(meta | {"target": record["target"], "model_rank": record["model_rank"], "modality": record["modality"],
                            "classifier": record["classifier"], "feature_family": record["feature_family"], "selected_feature_count": len(record["selected_features"]),
                            "frozen_image_balanced_accuracy": record["mean_outer_cv_balanced_accuracy"], "original_hard_prediction": hard, "high_probability": probability,
                            "feature_assembly_ms": assembly_ms, "fitted_transform_and_inference_ms": model_inference_ms})
                        model_assembly_ms += assembly_ms; inference_ms += model_inference_ms; valid_model_count += 1
                        if record["modality"] == "multimodal": multimodal_assembly_ms += assembly_ms
                    chunk_start = max(previous_end, calibration_eeg_end) * SAMPLE_RATE_HZ; chunk_stop = end * SAMPLE_RATE_HZ
                    filter_ms = float(chunk_timing.loc[(chunk_timing.chunk_start_sample >= chunk_start - 1e-9) & (chunk_timing.chunk_start_sample < chunk_stop - 1e-9), "causal_filter_ms"].sum())
                    face_total = sum(new_face_timing.values()) + face_aggregation_ms
                    total = filter_ms + eeg_feature_ms + face_total + multimodal_combine_ms + model_assembly_ms + inference_ms
                    latency_rows.append(meta | {"valid_model_count": valid_model_count, "causal_eeg_filter_increment_ms": filter_ms, "eeg_feature_ms": eeg_feature_ms,
                        **new_face_timing, "face_window_aggregation_ms": face_aggregation_ms, "face_processing_ms": face_total,
                        "multimodal_feature_combine_ms": multimodal_combine_ms, "all_model_feature_assembly_ms": model_assembly_ms,
                        "multimodal_model_assembly_ms": multimodal_assembly_ms, "frozen_inference_ms": inference_ms, "total_prediction_path_compute_ms": total})
                    previous_end = end
                calibration_rows[-1].update({f"calibration_{name}": value for name, value in calibration_face_timing.items()})
            finally: face.close()
    predictions = pd.DataFrame(prediction_rows); latency = pd.DataFrame(latency_rows)
    if predictions.empty:
        raise ValueError(f"{args.participant}: no eligible recording produced a frozen-model prediction")
    complete, completeness = transfer.consensus(predictions, models)
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
    calibration = pd.DataFrame(calibration_rows)
    synchronization = pd.concat(sync_rows, ignore_index=True) if sync_rows else pd.DataFrame()
    anchors = pd.concat(anchor_rows, ignore_index=True) if anchor_rows else pd.DataFrame()
    exclusions = pd.DataFrame(excluded_rows)
    timing_fields = [column for column in latency if column.endswith("_ms")]
    latency_summary = summarize_timings(latency, timing_fields, ["task"])
    overall_timing = summarize_timings(latency.assign(scope="overall"), timing_fields, ["scope"]); latency_summary = pd.concat([overall_timing, latency_summary], ignore_index=True)
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
    latency.to_csv(output / "latency_by_window.csv", index=False); latency_summary.to_csv(output / "latency_summary.csv", index=False); latency_budget.to_csv(output / "latency_budget_summary.csv", index=False); pd.concat(filter_rows, ignore_index=True).to_csv(output / "causal_filter_chunk_timing.csv", index=False)
    exclusions.to_csv(output / "excluded_cases.csv", index=False)
    make_figures(output, comparison, synchronization, latency)
    manifest = {"purpose":"Causal modality-agnostic offline replay under a fixed pre-deployment Status calibration offset", "participant":args.participant, "tasks_requested":args.tasks,
        "eligible_recording_segments":int(len(calibration)), "eligible_participant_task_cases":int(calibration[["participant","task"]].drop_duplicates().shape[0]),
        "excluded_cases":exclusions.to_dict("records"), "calibration_policy":{"vision_task_clock_seconds":CALIBRATION_SECONDS,"matching":"event identity and occurrence","offset":"median(vision_time - eeg_time)","drift_model":None,"later_updates":False,"predictions":"window start must be at or after calibration end mapped to EEG time"},
        "routes":[{"target":m["target"],"rank":m["model_rank"],"modality":m["modality"],"classifier":m["classifier"],"candidate_features":m["candidate_features"],"selected_features":m["selected_features"],"model_path":str(m["model_path"]),"probability_source":m["probability_source"]} for m in models],
        "filter":filter_design_metadata(causal_sos()), "chunk_ms":CHUNK_MS, "window_seconds":WINDOW_SECONDS,"window_step_seconds":WINDOW_STEP_SECONDS,"threshold":0.5,"definition_b":"median across complete rank-1/2/3 window probabilities, then median across complete windows; HIGH if >=0.5",
        "robot_ratings_used":False,"training_refitting_or_selection":False,"latency_scope":"OFFLINE REPLAY COMPUTATION; excludes camera/EEG hardware, Bluetooth, driver/OS transport, and physical synchronization hardware",
        "inputs":{"participant_config":str(args.participant_config.resolve()),"canonical_root":str(args.canonical_root.resolve()),"image_reference_csv":str(args.image_reference_csv.resolve()),"alignment_metadata":str(args.alignment_metadata.resolve()),"raw_root":str(args.raw_root.resolve())},
        "python_executable":sys.executable,"command":[sys.executable,*sys.argv],"calibration_records":calibration.to_dict("records")}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{args.participant} fixed-offset causal modality-agnostic replay complete: {output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
