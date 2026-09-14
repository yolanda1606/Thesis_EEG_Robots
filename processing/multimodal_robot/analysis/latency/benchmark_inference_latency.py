#!/usr/bin/env python3
"""Offline replay benchmark from a canonically preprocessed Robot recording to P(HIGH).

This analysis-only entry point never trains, selects, calibrates, or writes to
the raw-data tree.  It deliberately excludes acquisition, driver, Bluetooth,
disk/model loading, alignment, and one-time MediaPipe construction from its
primary per-window timings.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ``run_robot_pipeline.py`` retains its established direct-execution imports.
# Make that package root visible before importing its read-only path helpers.
ROBOT_ROOT = ROOT / "processing" / "multimodal_robot"
if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision

from processing.multimodal_image.src.video import _facial_measures, _landmarker
from processing.multimodal_robot.analysis.latency.replay_window import (
    assemble_and_predict, eeg_window_starts, eeg_wide_row, extract_eeg_window,
    milliseconds, preprocess_continuous, summarize_timings,
)
from processing.multimodal_robot.analysis.latency.validate_replay_compatibility import (
    compare_frames, recorded_model_input_features, validation_passes,
)
from processing.multimodal_robot.prep.continuous import (
    _files, _number, alignment_for_task, image_defaults, read_vision_log,
)
from processing.multimodal_robot.run_robot_pipeline import load, localize_task_paths
from processing.multimodal_robot.transfer import run_final_frozen_eeg_transfer as eeg_transfer
from processing.multimodal_robot.transfer import run_final_modality_agnostic_transfer as modality_transfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--task", required=True, help="One configured Robot task; this first implementation is deliberately single-task.")
    parser.add_argument("--participant-config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("eeg_only", "modality_agnostic"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_transfer/latency_benchmark/v1")
    parser.add_argument("--max-windows", type=int, default=10, help="Small replay sample; use a positive number. No cohort default exists.")
    parser.add_argument("--canonical-eeg-csv", type=Path, help="Optional established EEG feature CSV for compatibility validation.")
    parser.add_argument("--canonical-video-csv", type=Path, help="Optional established face feature CSV for compatibility validation.")
    parser.add_argument("--canonical-multimodal-csv", type=Path, help="Optional established multimodal feature CSV for compatibility validation.")
    parser.add_argument("--canonical-predictions-csv", type=Path, help="Optional established transfer predictions CSV for P(HIGH) validation.")
    parser.add_argument("--skip-compatibility", action="store_true", help="Do not write compatibility results; not recommended for latency conclusions.")
    return parser.parse_args()


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    """Supply exactly the loader attributes without invoking either transfer CLI."""
    if args.mode == "eeg_only":
        return SimpleNamespace(
            participant=args.participant,
            top3_csv=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv",
            model_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_models",
            calibration_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_probability_calibration_v1",
        )
    return SimpleNamespace(
        participant=args.participant,
        top3_csv=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target_modality.csv",
        image_model_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1",
        calibration_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1/modality_agnostic_probability_calibration_v1",
    )


class FaceReplay:
    """Process a task's video once in canonical sequential VIDEO mode.

    Frame compute is measured at the frame level and attributed to each 2-s
    window that contains the frame.  Since consecutive windows overlap, this
    attribution is descriptive; it is not an assertion of independent camera
    work for every window.
    """
    def __init__(self, task: dict[str, Any], qc: dict[str, Any], crop: dict[str, int], resource_root: Path):
        self.task, self.qc, self.crop = task, qc, crop
        defaults = image_defaults()
        self.defaults = defaults
        self.video_file = _files(task["video"])[0]
        self.cap = cv2.VideoCapture(str(self.video_file))  # opening excluded from primary timings
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            raise ValueError(f"Video has no usable frame rate: {self.video_file}")
        log_rows, _, _ = read_vision_log(_files(task["vision_log"])[0])
        self.log_times = {
            int(frame): moment for row in log_rows
            for frame, moment in [(_number(row, ("Frame_Count", "frame_count", "frame_number")), _number(row, ("Experiment_Time", "experiment_time", "elapsed_time_s", "time_s", "timestamp_s")))]
            if frame is not None and moment is not None
        }
        video_config = {**defaults, "video": {**defaults["video"], "crop": {
            "left": crop["x"], "top": crop["y"], "right": crop["x"] + crop["width"],
            "bottom": crop["y"] + crop["height"], "scale": 1.0, "approved": True,
        }}}
        model = resource_root / defaults["resources"]["face_landmarker_filename"]
        self.detector = _landmarker(video_config, model, vision.RunningMode.VIDEO)  # excluded one-time initialization
        self.frames: pd.DataFrame | None = None

    def close(self) -> None:
        self.cap.release()
        self.detector.close()

    def process_stream(self, until_eeg_s: float | None = None) -> pd.DataFrame:
        """Sequentially replay decode/landmark/measure operations once."""
        records: list[dict[str, Any]] = []
        index = 0
        while True:
            started = perf_counter_ns(); ok, frame = self.cap.read(); decode_ms = milliseconds(started)
            if not ok:
                break
            decode_time = index / self.fps
            video_time = self.log_times.get(index + 1, decode_time)
            eeg_time = (video_time - self.qc["offset_s"]) / (1 + self.qc["drift_s_per_s"])
            region = frame[self.crop["y"]:self.crop["y"] + self.crop["height"], self.crop["x"]:self.crop["x"] + self.crop["width"]]
            started = perf_counter_ns()
            result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(region, cv2.COLOR_BGR2RGB)), int(round(decode_time * 1000)))
            landmark_ms = milliseconds(started)
            values: dict[str, float] = {}; measure_ms = 0.0
            if result.face_landmarks:
                points = np.asarray([[point.x, point.y, point.z] for point in result.face_landmarks[0]], dtype=np.float32)
                if points.shape == (478, 3):
                    started = perf_counter_ns()
                    values = _facial_measures(points, self.defaults["video"]["landmark_indices"], region.shape[1], region.shape[0])
                    measure_ms = milliseconds(started)
            records.append({"eeg_time": eeg_time, "face_detected": bool(values), "face_decode_crop_ms": decode_ms,
                            "face_landmarker_ms": landmark_ms, "facial_measure_ms": measure_ms, **values})
            index += 1
            if until_eeg_s is not None and eeg_time >= until_eeg_s:
                break
        self.frames = pd.DataFrame(records)
        return self.frames

    def window_features(self, start_s: float) -> tuple[dict[str, float], dict[str, float]]:
        if self.frames is None:
            raise RuntimeError("Face stream has not been replayed")
        selected = self.frames[(self.frames.eeg_time >= start_s) & (self.frames.eeg_time < start_s + 2.0)]
        started = perf_counter_ns()
        detected = selected[selected.face_detected]
        values: dict[str, float] = {"video_frame_count": len(selected), "face_detected_frames": len(detected),
            "face_detection_rate": len(detected) / len(selected) if len(selected) else np.nan}
        for name in ("irisdo_norm", "eso_norm", "enso_norm", "mnso_norm", "mwo_norm"):
            series = detected[name].dropna() if name in detected else pd.Series(dtype=float)
            values[f"video_{name}_mean"] = float(series.mean()) if len(series) else np.nan
            values[f"video_{name}_std"] = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        aggregate_ms = milliseconds(started)
        return values, {"face_decode_crop_ms": float(selected.face_decode_crop_ms.sum()),
                         "face_landmarker_ms": float(selected.face_landmarker_ms.sum()),
                         "facial_measure_ms": float(selected.facial_measure_ms.sum()),
                         "face_aggregation_ms": aggregate_ms}


def _canonical_eeg_wide(path: Path) -> pd.DataFrame:
    """Pivot the canonical long EEG export using the transfer feature naming."""
    long = pd.read_csv(path)
    keys = eeg_transfer.KEYS
    features = [name for name in long if name.startswith("eeg_") and name != "eeg_coverage"]
    required = set(keys + ["channel"])
    if not features or required - set(long):
        raise ValueError("Canonical EEG compatibility CSV is not the expected long-format Robot feature table")
    wide = long.pivot(index=keys, columns="channel", values=features)
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    return wide.reset_index()


def canonical_compatibility(base_rows: pd.DataFrame, model_inputs: pd.DataFrame, predictions: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if args.skip_compatibility or args.canonical_eeg_csv is None:
        return {"performed": False, "reason": "skipped or no --canonical-eeg-csv supplied"}
    canonical = _canonical_eeg_wide(args.canonical_eeg_csv)
    features = [column for column in base_rows if column.startswith("eeg_")]
    results = compare_frames(base_rows, canonical, features, comparison="eeg_features")
    source_tables: dict[str, pd.DataFrame] = {"eeg": canonical}
    if args.canonical_video_csv is not None:
        video = pd.read_csv(args.canonical_video_csv)
        video_features = [column for column in base_rows if column.startswith("video_")]
        if video_features:
            results = pd.concat([results, compare_frames(base_rows, video, video_features, comparison="face_features")], ignore_index=True)
        source_tables["face"] = video
    if args.canonical_multimodal_csv is not None:
        source_tables["multimodal"] = pd.read_csv(args.canonical_multimodal_csv)
    for (target, rank, modality), replay_input in model_inputs.groupby(["target", "model_rank", "model_modality"], sort=False):
        canonical_input = source_tables.get(modality)
        if canonical_input is None:
            continue
        input_features = recorded_model_input_features(replay_input)
        if input_features:
            comparison = compare_frames(replay_input, canonical_input, input_features, comparison="final_model_input")
            comparison.insert(0, "model_modality", modality)
            comparison.insert(0, "model_rank", rank)
            comparison.insert(0, "target", target)
            results = pd.concat([results, comparison], ignore_index=True)
    if args.canonical_predictions_csv is not None:
        predicted = pd.read_csv(args.canonical_predictions_csv)
        replay_probability = predictions.loc[:, ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s", "target", "model_rank", "p_high"]]
        canonical_probability = predicted.rename(columns={"high_probability": "p_high"})
        for target, rank in replay_probability[["target", "model_rank"]].drop_duplicates().itertuples(index=False):
            left = replay_probability[(replay_probability.target == target) & (replay_probability.model_rank == rank)]
            right = canonical_probability[(canonical_probability.target == target) & (canonical_probability.model_rank == rank)]
            if len(left) and len(right):
                results = pd.concat([results, compare_frames(left, right, ["p_high"], comparison="p_high")], ignore_index=True)
    results.to_csv(output_dir / "compatibility_validation.csv", index=False)
    return {"performed": True, "tolerance": 1e-9, "pass": validation_passes(results), "row_count": int(len(results)),
            "maximum_absolute_difference": float(results.absolute_difference.max()) if len(results) else None}


def main() -> int:
    args = parse_args()
    if args.max_windows < 1:
        raise ValueError("--max-windows must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output directory: {output_dir}")
    config = localize_task_paths(load(args.participant_config.resolve()), args.raw_root.resolve())
    if config.get("participant") != args.participant or args.task not in config.get("tasks", {}):
        raise ValueError("Participant or task does not match the resolved participant configuration")
    task = config["tasks"][args.task]
    if isinstance(task.get("eeg"), dict) and "segments" in task["eeg"]:
        raise NotImplementedError("This initial isolated replay supports one BDF per task; segmented replay requires explicit per-segment output selection")
    qc, _, raws = alignment_for_task(task, args.task, str(config.get("raw_participant_dir", "")))
    if str(qc["method"]).startswith(("unsupported", "invalid")) or qc["aligned_overlap_s"] < 2.0:
        raise ValueError(f"Task is not eligible for canonical replay: {qc['method']}")
    raw_path, raw = raws[0]
    prepared, preprocessing_ms = preprocess_continuous(raw)
    eeg_channels = list(image_defaults()["channels"]["eeg_mapping"].values())
    selected_starts = list(eeg_window_starts(prepared))[:args.max_windows]
    if not selected_starts:
        raise ValueError("Recording has no complete canonical 2-s windows")
    records = eeg_transfer.load_frozen_models(model_args(args)) if args.mode == "eeg_only" else modality_transfer.load_models(model_args(args))
    # Face stream is initialized and replayed outside primary per-window timing.
    face = None
    if args.mode == "modality_agnostic" and any(record["modality"] in {"face", "multimodal"} for record in records):
        face = FaceReplay(task, qc, config["video_settings"]["crop"], ROOT / "processing/multimodal_image/models")
        try:
            face.process_stream(until_eeg_s=selected_starts[-1][0] + 2.0)
        finally:
            face.close()
    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for start_s, _ in selected_starts:
        features, extract_ms, feature_ms = extract_eeg_window(prepared, eeg_channels, start_s)
        metadata = {"participant": args.participant, "task": args.task, "segment_id": 1,
                    "window_id": f"{args.task}_s1_{start_s:.3f}", "window_start_s": start_s, "window_end_s": start_s + 2.0}
        eeg_row = eeg_wide_row(metadata, eeg_channels, features)
        face_values, face_times = ({}, {name: np.nan for name in ("face_decode_crop_ms", "face_landmarker_ms", "facial_measure_ms", "face_aggregation_ms")}) if face is None else face.window_features(start_s)
        multimodal_row = eeg_row | face_values
        base_rows.append(multimodal_row)
        for record in records:
            source = eeg_row if record.get("modality", "eeg") == "eeg" else (face_values | metadata if record["modality"] == "face" else multimodal_row)
            # Canonical transfer calls the original pipeline's predict() on
            # all candidate features even for calibrated SVM probability rows.
            missing = sorted(set(record["candidate_features"]) - set(source))
            if missing:
                raise ValueError(f"{record['model_path'].name}: replay feature schema missing {missing}")
            input_features = record["selected_features"] if str(record["classifier"]).lower() == "svm" else record["candidate_features"]
            input_rows.append(metadata | {"target": record["target"], "model_rank": record["model_rank"],
                         "model_modality": record.get("modality", "eeg"),
                         "probability_input_features": json.dumps(input_features)} |
                        {name: source[name] for name in input_features})
            if not np.isfinite(pd.Series([source[name] for name in record["candidate_features"]], dtype=float)).all():
                exclusion = "non_finite_feature"
                probability = hard_prediction = assembly_ms = transform_ms = hard_predict_ms = predict_ms = np.nan
            else:
                exclusion = ""
                probability, hard_prediction, assembly_ms, transform_ms, hard_predict_ms, predict_ms = assemble_and_predict(record, source)
                face_total = 0.0 if face is None else sum(face_times.values())
                transform_total = 0.0 if not np.isfinite(transform_ms) else transform_ms
                total_ms = extract_ms + feature_ms + face_total + assembly_ms + hard_predict_ms + transform_total + predict_ms
            rows.append(metadata | {"mode": args.mode, "target": record["target"], "model_rank": record["model_rank"],
                         "model_modality": record.get("modality", "eeg"), "classifier": record["classifier"],
                         "selected_feature_count": len(record["selected_features"]), "eeg_window_extract_ms": extract_ms,
                         "eeg_feature_ms": feature_ms, **face_times, "face_feature_ms": (face_times["face_decode_crop_ms"] + face_times["face_landmarker_ms"] + face_times["facial_measure_ms"] + face_times["face_aggregation_ms"]) if face is not None else np.nan,
                         "feature_assembly_ms": assembly_ms, "feature_selection_scaling_ms": transform_ms,
                         "model_predict_ms": hard_predict_ms, "model_predict_proba_ms": predict_ms,
                         "total_window_compute_ms": total_ms if exclusion == "" else np.nan,
                         "p_high": probability, "original_hard_prediction": hard_prediction,
                         "window_valid": exclusion == "", "exclusion_reason": exclusion})
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True)
    frame.to_csv(output_dir / "latency_windows.csv", index=False)
    metrics = [name for name in frame if name.endswith("_ms") and name != "amortized_preprocessing_ms_per_window"]
    summarize_timings(frame[frame.window_valid], metrics, ["mode", "model_modality", "target", "model_rank"]).to_csv(output_dir / "latency_summary.csv", index=False)
    valid_windows = int(frame[["window_id", "window_valid"]].drop_duplicates().window_valid.sum())
    pd.DataFrame([{"participant": args.participant, "task": args.task, "raw_duration_s": raw.n_times / raw.info["sfreq"],
                   "valid_window_count": valid_windows, "car_filter_total_ms": preprocessing_ms,
                   "amortized_preprocessing_ms_per_window": preprocessing_ms / valid_windows if valid_windows else np.nan}]).to_csv(output_dir / "continuous_preprocessing_summary.csv", index=False)
    compatibility = canonical_compatibility(pd.DataFrame(base_rows), pd.DataFrame(input_rows), frame, args, output_dir)
    manifest = {"purpose": "Offline replay computational latency only; excludes acquisition, Bluetooth, RealSense driver and live synchronization.",
                "participant": args.participant, "task": args.task, "mode": args.mode, "raw_eeg": str(raw_path),
                "window_policy": "2 s windows with 1 s step from canonically continuous CAR + 1-40 Hz fourth-order Butterworth IIR preprocessing; ICA disabled",
                "preprocessing_timing_policy": "continuous recording measured once; amortized value is not online per-window latency",
                "per_window_exclusions": ["BDF loading", "video opening", "alignment fitting", "configuration loading", "model loading/hashing", "MediaPipe construction", "CSV/manifest writing"],
                "face_attribution": "sequential canonical VIDEO-mode frame times are summed for every overlapping 2-s window; this is descriptive replay attribution", "compatibility": compatibility}
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if compatibility.get("performed") and not compatibility.get("pass"):
        raise RuntimeError("Replay compatibility validation failed; do not use this benchmark for latency conclusions")
    print(f"Latency replay complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
