#!/usr/bin/env python3
"""Apply frozen EEG-only Image models to canonical no-ICA Robot EEG windows.

This entry point deliberately has no model selection, model fitting, ICA,
Face, or Multimodal branches.  Robot ratings are loaded only after frozen
inference and are reported as descriptive comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
KEYS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]
DISPLAY_TASKS = {
    "pick_place": "PnP",
    "stack": "St",
    "shape_sorter_observation": "SSObs",
    "sisyphus": "Sisyphus",
    "shape_sorter_interaction": "SSInt",
    "shape_sorter_alone": "SSAlone",
}
TASK_ORDER = list(DISPLAY_TASKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument(
        "--top3-csv",
        type=Path,
        default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_models",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_probability_calibration_v1",
        help="Image-only calibrated SVM deployment wrappers; used only for frozen SVM rows.",
    )
    parser.add_argument("--robot-eeg-csv", type=Path, required=True)
    parser.add_argument("--image-reference-csv", type=Path, required=True)
    parser.add_argument("--participant-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate all frozen inputs without writing outputs.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = pd.read_csv(args.top3_csv)
    rows = rows[(rows.participant == args.participant) & rows.target.isin(["valence", "arousal"])].copy()
    if len(rows) != 6 or rows.groupby("target").size().to_dict() != {"arousal": 3, "valence": 3}:
        raise ValueError(f"Expected exactly three frozen rows per target for {args.participant}; found {len(rows)}")
    records: list[dict[str, Any]] = []
    for row in rows.sort_values(["target", "transfer_rank"]).itertuples(index=False):
        selected_csv = json.loads(row.actual_selected_feature_names)
        if not selected_csv or any(not name.startswith("eeg_") for name in selected_csv):
            raise ValueError(f"{row.target} rank {row.transfer_rank} is not an EEG-only frozen model")
        path = Path(row.final_model_path).resolve()
        if path.parent != args.model_root.resolve() or not path.is_file():
            raise ValueError(f"Frozen model path is absent or outside --model-root: {path}")
        pipeline = joblib.load(path)
        if not hasattr(pipeline, "feature_names_in_"):
            raise ValueError(f"{path.name} is not a fitted frozen pipeline")
        selector = pipeline.named_steps.get("selector")
        candidates = list(pipeline.feature_names_in_)
        selected = list(selector.get_feature_names_out(pipeline.feature_names_in_)) if selector is not None else candidates.copy()
        if selected != selected_csv:
            raise ValueError(f"{path.name}: serialized selected features do not match authoritative top-3 CSV")
        classes = [int(value) for value in pipeline.named_steps["classifier"].classes_]
        if 1 not in classes:
            raise ValueError(f"{path.name}: HIGH class is absent")
        record = {
            "target": row.target,
            "model_rank": int(row.transfer_rank),
            "classifier": row.classifier,
            "feature_family": row.feature_family,
            "channel_subset": row.channel_subset,
            "mean_outer_cv_balanced_accuracy": float(row.mean_outer_cv_balanced_accuracy),
            "model_path": path,
            "model_sha256": sha256(path),
            "pipeline": pipeline,
            "candidate_features": candidates,
            "selected_features": selected,
            "high_probability_column": classes.index(1),
            "probability_source": "original_frozen_pipeline",
        }
        if row.classifier == "svm":
            calibrated_path = args.calibration_root / "artifacts" / f"{row.participant}_{row.target}_rank{int(row.transfer_rank)}_calibrated.joblib"
            if not calibrated_path.is_file():
                raise FileNotFoundError(f"Missing Image-only calibrated SVM wrapper: {calibrated_path}")
            payload = joblib.load(calibrated_path)
            metadata = payload.get("deployment_metadata", {})
            probability_model = payload.get("calibrated_probability_model")
            if not hasattr(probability_model, "predict_proba"):
                raise ValueError(f"{calibrated_path.name}: calibrated wrapper lacks predict_proba()")
            if metadata.get("original_frozen_model_sha256") != record["model_sha256"]:
                raise ValueError(f"{calibrated_path.name}: original frozen model hash does not match")
            if metadata.get("selected_features") != selected:
                raise ValueError(f"{calibrated_path.name}: selected features do not match frozen SVM")
            if list(probability_model.feature_names_in_) != selected:
                raise ValueError(f"{calibrated_path.name}: calibrated wrapper input schema does not match selected features")
            calibrated_classes = [int(value) for value in probability_model.classes_]
            if 1 not in calibrated_classes:
                raise ValueError(f"{calibrated_path.name}: calibrated HIGH class is absent")
            record.update({
                "probability_model": probability_model,
                "probability_source": "image_only_calibrated_svm_wrapper",
                "calibrated_artifact_path": calibrated_path.resolve(),
                "calibrated_artifact_sha256": sha256(calibrated_path),
                "calibrated_high_probability_column": calibrated_classes.index(1),
            })
        else:
            if not hasattr(pipeline, "predict_proba"):
                raise ValueError(f"{path.name}: frozen non-SVM pipeline lacks predict_proba()")
            record["probability_model"] = pipeline
        records.append(record)
    return records


def robot_eeg_wide(path: Path) -> pd.DataFrame:
    long = pd.read_csv(path)
    required = set(KEYS + ["channel"])
    if missing := sorted(required.difference(long.columns)):
        raise ValueError(f"Robot EEG table missing required columns: {missing}")
    feature_columns = [name for name in long.columns if name.startswith("eeg_")]
    if not feature_columns:
        raise ValueError("Robot EEG table contains no long-format EEG feature columns")
    duplicates = long.duplicated(KEYS + ["channel"])
    if duplicates.any():
        raise ValueError(f"Robot EEG table has {int(duplicates.sum())} duplicate window/channel rows")
    wide = long.pivot(index=KEYS, columns="channel", values=feature_columns)
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    return wide.reset_index().sort_values(["task", "window_start_s", "segment_id", "window_id"]).reset_index(drop=True)


def image_reference(path: Path, selected: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise ValueError(f"Image reference table missing selected features: {missing}")
    values = frame.loc[:, selected].apply(pd.to_numeric, errors="raise")
    values = values.loc[np.isfinite(values.to_numpy()).all(axis=1)].reset_index(drop=True)
    if values.empty:
        raise ValueError("No finite Image reference rows for selected features")
    return values


def shift_diagnostics(model: Any, candidates: list[str], selected: list[str], robot: pd.DataFrame, image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_indices = [candidates.index(name) for name in selected]
    scaler = model.named_steps["scale"]
    robot_values = robot.loc[:, selected].to_numpy(dtype=float)
    image_values = image.loc[:, selected].to_numpy(dtype=float)
    mean = scaler.mean_[selected_indices]
    scale = scaler.scale_[selected_indices]
    robot_z = (robot_values - mean) / scale
    image_z = (image_values - mean) / scale
    distances = np.sqrt(((robot_z[:, None, :] - image_z[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    outside = (robot_values < image_values.min(axis=0)) | (robot_values > image_values.max(axis=0))
    windows = robot.loc[:, KEYS].copy()
    windows["nearest_image_distance"] = distances
    windows["outside_selected_count"] = outside.sum(axis=1)
    windows["outside_selected_fraction"] = outside.mean(axis=1)
    windows["median_abs_image_z"] = np.median(np.abs(robot_z), axis=1)
    windows["max_abs_image_z"] = np.abs(robot_z).max(axis=1)
    summary = windows.groupby("task", dropna=False).agg(
        window_count=("task", "size"),
        median_nearest_image_distance=("nearest_image_distance", "median"),
        fraction_windows_with_outside=("outside_selected_count", lambda value: float((value > 0).mean())),
        fraction_feature_values_outside=("outside_selected_fraction", "mean"),
        median_abs_image_z=("median_abs_image_z", "median"),
        maximum_abs_image_z=("max_abs_image_z", "max"),
    ).reset_index()
    return windows, summary


def run_inference(models: list[dict[str, Any]], robot: pd.DataFrame, image_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    window_shift_rows: list[pd.DataFrame] = []
    task_shift_rows: list[pd.DataFrame] = []
    for record in models:
        missing = sorted(set(record["candidate_features"]).difference(robot.columns))
        if missing:
            raise ValueError(f"{record['model_path'].name}: Robot feature columns missing: {missing}")
        values = robot.loc[:, record["candidate_features"]].apply(pd.to_numeric, errors="raise")
        valid = np.isfinite(values.to_numpy()).all(axis=1)
        if not valid.any():
            raise ValueError(f"{record['model_path'].name}: no finite Robot windows")
        frame = robot.loc[valid].reset_index(drop=True)
        values = values.loc[valid].reset_index(drop=True)
        pipeline = record["pipeline"]
        original_hard_prediction = pipeline.predict(values)
        prediction = frame.loc[:, KEYS].copy()
        prediction["target"] = record["target"]
        prediction["model_rank"] = record["model_rank"]
        prediction["classifier"] = record["classifier"]
        prediction["feature_family"] = record["feature_family"]
        prediction["selected_feature_count"] = len(record["selected_features"])
        prediction["frozen_image_balanced_accuracy"] = record["mean_outer_cv_balanced_accuracy"]
        prediction["predicted_class"] = original_hard_prediction
        prediction["original_hard_prediction"] = original_hard_prediction
        if record["classifier"] == "svm":
            prediction["original_svm_hard_prediction"] = original_hard_prediction
            calibrated_input = frame.loc[:, record["selected_features"]].apply(pd.to_numeric, errors="raise")
            prediction["high_probability"] = record["probability_model"].predict_proba(calibrated_input)[:, record["calibrated_high_probability_column"]]
        else:
            prediction["original_svm_hard_prediction"] = pd.NA
            prediction["high_probability"] = record["probability_model"].predict_proba(values)[:, record["high_probability_column"]]
        prediction_rows.append(prediction)
        reference = image_reference(image_path, record["selected_features"])
        window_shift, task_shift = shift_diagnostics(pipeline, record["candidate_features"], record["selected_features"], frame, reference)
        for output in (window_shift, task_shift):
            output.insert(0, "model_rank", record["model_rank"])
            output.insert(0, "target", record["target"])
        window_shift_rows.append(window_shift)
        task_shift_rows.append(task_shift)
    return pd.concat(prediction_rows, ignore_index=True), pd.concat(window_shift_rows, ignore_index=True), pd.concat(task_shift_rows, ignore_index=True)


def probability_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(["target", "model_rank", "classifier", "feature_family", "task"], dropna=False).agg(
        n_windows=("high_probability", "size"),
        mean_high_probability=("high_probability", "mean"),
        median_high_probability=("high_probability", "median"),
        sd_high_probability=("high_probability", "std"),
        fraction_predicted_high=("predicted_class", "mean"),
    ).reset_index()


def complete_top3_windows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align ranks and retain only windows with all three valid frozen outputs."""
    complete_rows: list[pd.DataFrame] = []
    completeness_rows: list[dict[str, Any]] = []
    for (participant, target, task), part in predictions.groupby(["participant", "target", "task"], sort=False):
        probabilities = part.pivot(index=KEYS, columns="model_rank", values="high_probability").reindex(columns=[1, 2, 3])
        hard_predictions = part.pivot(index=KEYS, columns="model_rank", values="original_hard_prediction").reindex(columns=[1, 2, 3])
        complete_mask = probabilities.notna().all(axis=1) & hard_predictions.notna().all(axis=1)
        n_candidate = int(len(probabilities))
        n_complete = int(complete_mask.sum())
        completeness_rows.append({
            "participant": participant, "target": target, "task": task,
            "n_candidate_windows": n_candidate,
            "n_complete_top3_windows": n_complete,
            "n_incomplete_top3_windows": n_candidate - n_complete,
            "fraction_complete_top3_windows": n_complete / n_candidate if n_candidate else np.nan,
        })
        if not n_complete:
            continue
        probability_complete = probabilities.loc[complete_mask]
        hard_complete = hard_predictions.loc[complete_mask]
        frame = probability_complete.index.to_frame(index=False)
        frame["target"] = target
        frame["window_center_s"] = (frame["window_start_s"] + frame["window_end_s"]) / 2
        for rank in [1, 2, 3]:
            frame[f"rank{rank}_high_probability"] = probability_complete[rank].to_numpy()
            frame[f"rank{rank}_hard_prediction"] = hard_complete[rank].astype(int).to_numpy()
        frame["top3_consensus_probability"] = probability_complete.median(axis=1).to_numpy()
        frame["top3_probability_min"] = probability_complete.min(axis=1).to_numpy()
        frame["top3_probability_max"] = probability_complete.max(axis=1).to_numpy()
        frame["top3_probability_range"] = frame["top3_probability_max"] - frame["top3_probability_min"]
        frame["top3_consensus_class"] = np.where(frame["top3_consensus_probability"] >= 0.5, "HIGH", "LOW")
        frame["all_three_hard_agree"] = hard_complete.nunique(axis=1).eq(1).to_numpy()
        complete_rows.append(frame)
    complete = pd.concat(complete_rows, ignore_index=True) if complete_rows else pd.DataFrame(
        columns=[*KEYS, "target", "window_center_s", "rank1_high_probability", "rank2_high_probability", "rank3_high_probability",
                 "rank1_hard_prediction", "rank2_hard_prediction", "rank3_hard_prediction", "top3_consensus_probability",
                 "top3_probability_min", "top3_probability_max", "top3_probability_range", "top3_consensus_class",
                 "all_three_hard_agree"]
    )
    if not complete.empty:
        complete = complete.sort_values(
            ["participant", "target", "task", "segment_id", "window_center_s"], kind="stable"
        ).reset_index(drop=True)
    return complete, pd.DataFrame(completeness_rows)


def agreement(complete: pd.DataFrame, completeness: pd.DataFrame) -> pd.DataFrame:
    """Summarize only the explicit complete-top-three window set."""
    rows: list[dict[str, Any]] = []
    for item in completeness.itertuples(index=False):
        part = complete[(complete.target.eq(item.target)) & (complete.task.eq(item.task))]
        row: dict[str, Any] = {
            "participant": item.participant, "target": item.target, "task": item.task,
            "n_candidate_windows": item.n_candidate_windows,
            "n_complete_top3_windows": item.n_complete_top3_windows,
            "n_incomplete_top3_windows": item.n_incomplete_top3_windows,
            "fraction_complete_top3_windows": item.fraction_complete_top3_windows,
            "n_all_three_windows": item.n_complete_top3_windows,
        }
        if part.empty:
            for name in ["unanimous_agreement_rate", "unanimous_high_rate", "unanimous_low_rate", "disagreement_rate",
                         "majority_vote_high_fraction", "class_agreement_1_2", "class_agreement_1_3", "class_agreement_2_3",
                         "probability_correlation_1_2", "probability_correlation_1_3", "probability_correlation_2_3"]:
                row[name] = np.nan
        else:
            hard = part[["rank1_hard_prediction", "rank2_hard_prediction", "rank3_hard_prediction"]]
            row.update({
                "unanimous_agreement_rate": float(part["all_three_hard_agree"].mean()),
                "unanimous_high_rate": float((hard == 1).all(axis=1).mean()),
                "unanimous_low_rate": float((hard == 0).all(axis=1).mean()),
                "disagreement_rate": float((~part["all_three_hard_agree"]).mean()),
                "majority_vote_high_fraction": float((hard.sum(axis=1) >= 2).mean()),
            })
            for left, right in [(1, 2), (1, 3), (2, 3)]:
                row[f"class_agreement_{left}_{right}"] = float((hard[f"rank{left}_hard_prediction"] == hard[f"rank{right}_hard_prediction"]).mean())
                row[f"probability_correlation_{left}_{right}"] = float(part[f"rank{left}_high_probability"].corr(part[f"rank{right}_high_probability"]))
        rows.append(row)
    return pd.DataFrame(rows)


def robot_ratings(participant: str) -> pd.DataFrame:
    ratings = pd.read_csv(ROOT / "data/Robot Ratings - Robot Ratings.csv")
    labels = {
        "pick_place": "Pick and Place", "shape_sorter_observation": "Shape Sorter Observation", "stack": "Stack",
        "sisyphus": "Sisyphus", "shape_sorter_interaction": "Shape Sorter Interaction", "shape_sorter_alone": "Shape Sorter Alone",
    }
    rows = []
    participant_rows = ratings[ratings["Participant ID"].astype(str).eq(participant)]
    for task, label in labels.items():
        row = participant_rows[participant_rows["Robot Experiment"].astype(str).str.casefold().eq(label.casefold())]
        if len(row) != 1:
            raise ValueError(f"Expected one canonical Robot rating for {participant} {label}; found {len(row)}")
        for target, column in [("valence", "Valence"), ("arousal", "Arousal")]:
            value = float(row.iloc[0][column])
            rows.append({"task": task, "target": target, "robot_rating": value, "robot_rating_class": "HIGH" if value >= 4 else "LOW"})
    return pd.DataFrame(rows)


def selected_configured_paths(item: Any) -> list[Path]:
    """Match the canonical Robot pipeline's selected_files() path convention."""
    if isinstance(item, str):
        return [Path(item)]
    if not isinstance(item, dict):
        return []
    if "segments" in item:
        segments = item["segments"]
        if not isinstance(segments, list):
            return []
        return [
            Path(value)
            for segment in segments
            for value in [segment.get("file") if isinstance(segment, dict) else segment]
            if isinstance(value, str)
        ]
    return [Path(item["file"])] if isinstance(item.get("file"), str) else []


def event_table(config_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Resolve the exact semantic events later supplied to trigger plotting."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    events = []
    sources = []
    meaningful = ("START", "END", "GRASP", "RELEASE", "COLLISION", "WRONG LOCATION", "MID-AIR DROP")
    for task, specification in config["tasks"].items():
        metrics_item = specification.get("robot_metrics")
        vision_item = specification.get("vision_log")
        metric_paths = selected_configured_paths(metrics_item)
        vision_paths = selected_configured_paths(vision_item)
        if not metric_paths:
            reason = metrics_item.get("reason", "no selected robot-metrics path") if isinstance(metrics_item, dict) else "no selected robot-metrics path"
            sources.append({"task": task, "vision_log": str((ROOT / "data" / vision_paths[0]).resolve()) if len(vision_paths) == 1 else None,
                            "robot_metrics": None, "status": "not_available", "reason": str(reason)})
            continue
        if len(metric_paths) != 1 or len(vision_paths) != 1:
            sources.append({"task": task, "vision_log": str(vision_paths[0].resolve()) if len(vision_paths) == 1 else None,
                            "robot_metrics": str(metric_paths[0].resolve()) if len(metric_paths) == 1 else None,
                            "status": "not_resolved", "reason": "semantic event resolution requires exactly one selected vision log and robot-metrics file"})
            continue
        metrics_path = ROOT / "data" / metric_paths[0]
        vision_path = ROOT / "data" / vision_paths[0]
        metrics = pd.read_csv(metrics_path)
        definitions = metrics.loc[metrics.Event_Trigger.ne(0), ["Event_Trigger", "Controller_Type", "Waypoint_Stage"]].drop_duplicates("Event_Trigger")
        definitions["Event_Trigger"] = definitions.Event_Trigger.astype(str)
        definition_map = definitions.set_index("Event_Trigger").to_dict("index")
        vision = pd.read_csv(vision_path)
        vision["Trigger"] = vision.Trigger.astype(str)
        changes = vision[(vision.Trigger.ne("0")) & vision.Trigger.ne(vision.Trigger.shift())].copy()
        for row in changes.itertuples(index=False):
            definition = definition_map.get(str(row.Trigger))
            if not definition:
                continue
            stage = str(definition["Waypoint_Stage"])
            if any(term in stage.upper() for term in meaningful):
                upper = stage.upper()
                category = "start" if "START" in upper else "end" if "END" in upper else (
                    "fault" if any(word in upper for word in ["COLLISION", "WRONG", "MID-AIR"]) else "normal"
                )
                event_time = float(row.Experiment_Time)
                events.append({
                    "participant": str(config["participant"]), "task": task, "segment_id": pd.NA,
                    "event_time_s": event_time, "time_coordinate": "task_local_time_s",
                    "trigger_code": str(row.Trigger), "semantic_label": stage,
                    "semantic_category": category,
                    # Plotting aliases retain the established visual semantics unchanged.
                    "time_s": event_time, "event_label": stage,
                })
        sources.append({"task": task, "vision_log": str(vision_path.resolve()), "robot_metrics": str(metrics_path.resolve()), "status": "resolved", "reason": ""})
    columns = ["participant", "task", "segment_id", "event_time_s", "time_coordinate", "trigger_code", "semantic_label", "semantic_category", "time_s", "event_label"]
    return pd.DataFrame(events, columns=columns), sources


def robot_conditions(config_path: Path) -> dict[str, str]:
    """Read observation-task condition labels from the authoritative Robot config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    conditions: dict[str, str] = {}
    for task, specification in config["tasks"].items():
        if specification.get("category") != "observation":
            conditions[task] = "N/A"
            continue
        condition = specification.get("condition", {})
        actual = str(condition.get("actual", "")).upper()
        speed = str(condition.get("speed_actual", "")).upper()
        conditions[task] = "FAULTY" if actual == "FAULTY" else speed if speed in {"FAST", "SLOW"} else "N/A"
    return conditions


def task_statistics(complete: pd.DataFrame, agreement_data: pd.DataFrame, ratings: pd.DataFrame,
                    conditions: dict[str, str], shift: pd.DataFrame) -> pd.DataFrame:
    """Compute final consensus-first task statistics from complete top-three windows only."""
    rows: list[dict[str, Any]] = []
    for agreement_row in agreement_data.itertuples(index=False):
        target, task = agreement_row.target, agreement_row.task
        part = complete[(complete.target.eq(target)) & (complete.task.eq(task))]
        rating = ratings[(ratings.target.eq(target)) & (ratings.task.eq(task))]
        if len(rating) != 1:
            raise ValueError(f"{agreement_row.participant} {target} {task}: expected one Robot rating")
        rating_row = rating.iloc[0]
        shift_part = shift[(shift.target.eq(target)) & (shift.task.eq(task))]
        shift_warning = shift_part["median_nearest_image_distance"] > 10.0
        row: dict[str, Any] = {
            "participant": agreement_row.participant, "target": target, "task": task, "task_display": DISPLAY_TASKS[task],
            "robot_condition": conditions.get(task, "N/A"), "robot_rating": float(rating_row.robot_rating),
            "robot_rating_1_to_7": float(rating_row.robot_rating), "robot_rating_class": rating_row.robot_rating_class,
            "n_candidate_windows": agreement_row.n_candidate_windows,
            "n_complete_top3_windows": agreement_row.n_complete_top3_windows,
            "n_incomplete_top3_windows": agreement_row.n_incomplete_top3_windows,
            "fraction_complete_top3_windows": agreement_row.fraction_complete_top3_windows,
            "n_consensus_windows": agreement_row.n_complete_top3_windows,
            "unanimous_agreement_rate": agreement_row.unanimous_agreement_rate,
            "unanimous_high_rate": agreement_row.unanimous_high_rate,
            "unanimous_low_rate": agreement_row.unanimous_low_rate,
            "fraction_unanimous_top3_windows": agreement_row.unanimous_agreement_rate,
            "fraction_unanimous_high_windows": agreement_row.unanimous_high_rate,
            "fraction_unanimous_low_windows": agreement_row.unanimous_low_rate,
            "majority_vote_high_fraction": agreement_row.majority_vote_high_fraction,
            "models_with_shift_warning": ";".join("R" + str(value) for value in shift_part.loc[shift_warning, "model_rank"]),
        }
        if part.empty:
            row.update({
                "median_top3_consensus_probability": np.nan, "mean_top3_consensus_probability": np.nan,
                "fraction_consensus_windows_ge_0_5": np.nan, "descriptive_task_verdict": pd.NA,
                "consensus_verdict": pd.NA, "verdict_matches_robot_rating": pd.NA,
                "consensus_matches_robot_rating": pd.NA, "mean_top3_probability_range": np.nan,
                "median_top3_probability_range": np.nan, "mean_pairwise_absolute_probability_difference": np.nan,
            })
        else:
            consensus = part["top3_consensus_probability"]
            median_probability = float(consensus.median())
            verdict = "HIGH" if median_probability >= 0.5 else "LOW"
            pair_differences = pd.DataFrame({
                "r1_r2": (part["rank1_high_probability"] - part["rank2_high_probability"]).abs(),
                "r1_r3": (part["rank1_high_probability"] - part["rank3_high_probability"]).abs(),
                "r2_r3": (part["rank2_high_probability"] - part["rank3_high_probability"]).abs(),
            })
            row.update({
                "median_top3_consensus_probability": median_probability,
                "mean_top3_consensus_probability": float(consensus.mean()),
                "fraction_consensus_windows_ge_0_5": float((consensus >= 0.5).mean()),
                "descriptive_task_verdict": verdict, "consensus_verdict": verdict,
                "verdict_matches_robot_rating": bool(verdict == rating_row.robot_rating_class),
                "consensus_matches_robot_rating": bool(verdict == rating_row.robot_rating_class),
                "mean_top3_probability_range": float(part["top3_probability_range"].mean()),
                "median_top3_probability_range": float(part["top3_probability_range"].median()),
                "mean_pairwise_absolute_probability_difference": float(pair_differences.mean(axis=1).mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "task"]).reset_index(drop=True)


def event_style(label: str) -> tuple[str, str]:
    """Use distinct visual treatments for boundaries, normal actions, and faults."""
    upper = label.upper()
    if "START" in upper or "END" in upper:
        return "#4d4d4d", "--"
    if any(word in upper for word in ["COLLISION", "WRONG", "MID-AIR"]):
        return "#b2182b", "-"
    return "#2166ac", "-"


def plot_task_trigger_consensus(consensus_windows: pd.DataFrame, statistics: pd.DataFrame, events: pd.DataFrame,
                                participant: str, figures: Path) -> None:
    """Save one wide, annotated semantic-trigger plot per target and task."""
    for target, data in consensus_windows.groupby("target", sort=False):
        for task in [item for item in TASK_ORDER if item in set(data.task)]:
            part = data[data.task.eq(task)].copy()
            segment_parts = consensus_segments(part)
            can_overlay_triggers = len(segment_parts) == 1
            task_events = events[events.task.eq(task)]
            event_types = list(task_events.event_label.drop_duplicates()) if can_overlay_triggers else []
            figure_width = 16 if len(event_types) >= 5 or task in {"stack", "sisyphus"} else 13
            fig, axis = plt.subplots(figsize=(figure_width, 5.2))
            for index, segment in enumerate(segment_parts):
                axis.fill_between(segment.window_center_s, segment.top3_probability_min, segment.top3_probability_max,
                                  color="0.35", alpha=0.18, label="Top-3 range" if index == 0 else None)
                axis.plot(segment.window_center_s, segment.top3_consensus_probability, color="0.15", lw=1.8,
                          label="Top-3 median" if index == 0 else None)
            labels_seen: set[str] = set()
            if can_overlay_triggers:
                for event in task_events.itertuples(index=False):
                    label = event.event_label if event.event_label not in labels_seen else None
                    color, linestyle = event_style(event.event_label)
                    axis.axvline(event.time_s, color=color, ls=linestyle, lw=0.9, alpha=0.8, label=label)
                    labels_seen.add(event.event_label)
            else:
                axis.text(0.01, 0.03, "Trigger overlay omitted: segment-local times lack a verified task-global mapping.",
                          transform=axis.transAxes, ha="left", va="bottom", fontsize=6.5,
                          bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9})
            stat = statistics[(statistics.target.eq(target)) & (statistics.task.eq(task))].iloc[0]
            annotation = (
                f"Self-report: {stat.robot_rating_1_to_7:.0f}/7 ({stat.robot_rating_class})\n"
                f"Task median P(HIGH): {stat.median_top3_consensus_probability:.2f} → {stat.descriptive_task_verdict}\n"
                f"Unanimous top-3 windows: {stat.fraction_unanimous_top3_windows:.0%}"
            )
            axis.text(0.995, 0.04, annotation, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9})
            axis.axhline(0.5, color="black", ls=":", lw=0.8)
            axis.set_ylim(0, 1); axis.set_ylabel("P(HIGH)"); axis.set_xlabel("Time within task (s)")
            condition = stat.robot_condition
            task_title = DISPLAY_TASKS[task] if condition == "N/A" else f"{DISPLAY_TASKS[task]} ({condition})"
            fig.suptitle(f"{participant} {target} {task_title}: descriptive top-3 consensus with semantic Robot events", y=0.97, fontsize=13)
            axis.legend(fontsize=7, ncol=min(4, max(1, len(labels_seen) + 2)), loc="lower center",
                        bbox_to_anchor=(0.5, 1.03), frameon=True)
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.70)
            fig.savefig(figures / f"{participant}_{target}_{DISPLAY_TASKS[task]}_consensus_triggers.png", dpi=170)
            plt.close(fig)


def segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    frame = frame.sort_values("window_center_s")
    breaks = np.r_[True, np.diff(frame.window_start_s.to_numpy(dtype=float)) > 1.000001]
    return [part for _, part in frame.assign(_segment=breaks.cumsum()).groupby("_segment")]


def consensus_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Return independently plotted consensus segments in numeric time order."""
    return [
        segment.sort_values("window_center_s", kind="stable")
        for _, segment in frame.groupby("segment_id", sort=True)
    ]


def model_legend_label(rank_data: pd.DataFrame, rank: int) -> str:
    """Render concise, frozen Image-model metadata for timeline legends."""
    classifier = {"svm": "SVM", "logreg": "LogReg"}.get(str(rank_data.classifier.iloc[0]).lower(),
                                                           str(rank_data.classifier.iloc[0]).upper())
    family = {
        "all_eeg": "All EEG",
        "paper_compact": "Paper compact",
        "hjorth": "Hjorth",
    }.get(str(rank_data.feature_family.iloc[0]), str(rank_data.feature_family.iloc[0]).replace("_", " ").title())
    parts = [f"R{rank}", "EEG", classifier]
    if "selected_feature_count" in rank_data and pd.notna(rank_data.selected_feature_count.iloc[0]):
        parts.append(f"{family} (k={int(rank_data.selected_feature_count.iloc[0])})")
    if "frozen_image_balanced_accuracy" in rank_data and pd.notna(rank_data.frozen_image_balanced_accuracy.iloc[0]):
        parts.append(f"BA={float(rank_data.frozen_image_balanced_accuracy.iloc[0]):.3f}")
    return " · ".join(parts)


def plot_target(predictions: pd.DataFrame, consensus_windows: pd.DataFrame, target: str, participant: str, figures: Path) -> None:
    data = predictions[predictions.target.eq(target)].copy()
    tasks = [task for task in TASK_ORDER if task in set(data.task)]
    colors = {1: "#1b9e77", 2: "#377eb8", 3: "#984ea3"}

    fig, axes = plt.subplots(len(tasks), 1, figsize=(14.5, 2.15 * len(tasks)))
    for axis, task in zip(np.atleast_1d(axes), tasks):
        part = data[data.task.eq(task)].copy()
        part["window_center_s"] = (part.window_start_s + part.window_end_s) / 2
        for rank, rank_data in part.groupby("model_rank"):
            first = True
            for piece in segments(rank_data):
                axis.plot(piece.window_center_s, piece.high_probability, color=colors[rank], lw=0.9, marker="o", ms=2.1,
                          alpha=0.75, label=model_legend_label(rank_data, rank) if first else None)
                first = False
        axis.axhline(0.5, color="black", ls=":", lw=0.8)
        axis.set_ylim(0, 1); axis.set_ylabel(DISPLAY_TASKS[task]); axis.set_title(DISPLAY_TASKS[task], loc="left", fontsize=9)
    axes[0].legend(ncol=1, fontsize=6.7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True)
    axes[-1].set_xlabel("Time within task (s)")
    fig.suptitle(f"{participant} {target}: frozen-model P(HIGH)")
    fig.tight_layout(rect=(0, 0, 0.68, 0.98))
    fig.savefig(figures / f"{participant}_{target}_probability_timeline.png", dpi=170)
    plt.close(fig)

    def consensus_plot() -> None:
        consensus_data = consensus_windows[consensus_windows.target.eq(target)].copy()
        fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 1.75 * len(tasks)))
        for axis, task in zip(np.atleast_1d(axes), tasks):
            part = consensus_data[consensus_data.task.eq(task)]
            for index, segment in enumerate(consensus_segments(part)):
                axis.fill_between(segment.window_center_s, segment.top3_probability_min, segment.top3_probability_max,
                                  color="0.35", alpha=0.18, label="Top-3 range" if index == 0 else None)
                axis.plot(segment.window_center_s, segment.top3_consensus_probability, color="0.15", lw=1.8,
                          label="Top-3 median" if index == 0 else None)
            axis.axhline(0.5, color="black", ls=":", lw=0.8)
            axis.set_ylim(0, 1); axis.set_ylabel(DISPLAY_TASKS[task]); axis.set_title(DISPLAY_TASKS[task], loc="left", fontsize=9)
        axes[-1].set_xlabel("Time within task (s)")
        axes[0].legend(fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.suptitle(f"{participant} {target}: descriptive top-3 median P(HIGH)")
        fig.savefig(figures / f"{participant}_{target}_probability_consensus.png", dpi=170)
        plt.close(fig)

    consensus_plot()


def main() -> int:
    args = parse_args()
    args.top3_csv, args.model_root = args.top3_csv.resolve(), args.model_root.resolve()
    args.calibration_root = args.calibration_root.resolve()
    args.robot_eeg_csv, args.image_reference_csv = args.robot_eeg_csv.resolve(), args.image_reference_csv.resolve()
    args.participant_config, args.output_dir = args.participant_config.resolve(), args.output_dir.resolve()
    models = load_frozen_models(args)
    robot = robot_eeg_wide(args.robot_eeg_csv)
    all_candidates = sorted(set().union(*(set(record["candidate_features"]) for record in models)))
    compatibility = {
        "robot_window_count": int(len(robot)),
        "robot_pivoted_eeg_column_count": int(len([name for name in robot if name.startswith("eeg_")])),
        "robot_model_feature_count": int(len(set(all_candidates).intersection(robot.columns))),
        "missing_any_pipeline_feature": sorted(set(all_candidates).difference(robot.columns)),
        "models": [{"target": record["target"], "model_rank": record["model_rank"], "candidate_feature_count": len(record["candidate_features"]),
                    "selected_feature_count": len(record["selected_features"]), "probability_source": record["probability_source"],
                    "missing_pipeline_features": sorted(set(record["candidate_features"]).difference(robot.columns))} for record in models],
    }
    if compatibility["missing_any_pipeline_feature"]:
        raise ValueError(f"Robot schema incompatible: {compatibility['missing_any_pipeline_feature']}")
    events, event_sources = event_table(args.participant_config)
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", "participant": args.participant, "compatibility": compatibility,
                          "event_count": int(len(events)), "event_sources": event_sources}, indent=2))
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    predictions, shift_windows, shift_tasks = run_inference(models, robot, args.image_reference_csv)
    summary = probability_summary(predictions)
    consensus_windows, completeness = complete_top3_windows(predictions)
    agreement_data = agreement(consensus_windows, completeness)
    ratings = robot_ratings(args.participant)
    conditions = robot_conditions(args.participant_config)
    ratings = ratings.assign(robot_condition=ratings.task.map(conditions).fillna("N/A"))
    task_analysis = task_statistics(consensus_windows, agreement_data, ratings, conditions, shift_tasks)
    verdict_data = task_analysis[[
        "task", "task_display", "target", "n_candidate_windows", "n_complete_top3_windows", "n_incomplete_top3_windows",
        "fraction_complete_top3_windows", "median_top3_consensus_probability", "consensus_verdict",
        "unanimous_agreement_rate", "robot_rating", "robot_rating_class", "consensus_matches_robot_rating",
        "models_with_shift_warning",
    ]].copy()
    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    figures.mkdir()
    predictions.to_csv(args.output_dir / f"{args.participant}_window_predictions.csv", index=False)
    consensus_windows.to_csv(args.output_dir / f"{args.participant}_window_consensus.csv", index=False)
    summary.to_csv(args.output_dir / f"{args.participant}_task_probability_summary.csv", index=False)
    agreement_data.to_csv(args.output_dir / f"{args.participant}_model_agreement.csv", index=False)
    shift_windows.to_csv(args.output_dir / f"{args.participant}_selected_feature_shift_window_diagnostics.csv", index=False)
    shift_tasks.to_csv(args.output_dir / f"{args.participant}_selected_feature_shift_task_summary.csv", index=False)
    verdict_data.to_csv(args.output_dir / f"{args.participant}_task_verdict_summary.csv", index=False)
    ratings.to_csv(args.output_dir / f"{args.participant}_robot_rating_comparison.csv", index=False)
    task_analysis.to_csv(args.output_dir / f"{args.participant}_task_consensus_analysis.csv", index=False)
    semantic_events_path = args.output_dir / f"{args.participant}_semantic_events.csv"
    events.to_csv(semantic_events_path, index=False)
    for target in ["valence", "arousal"]:
        plot_target(predictions, consensus_windows, target, args.participant, figures)
    plot_task_trigger_consensus(consensus_windows, task_analysis, events, args.participant, figures)
    manifest = {
        "participant": args.participant,
        "purpose": "Final frozen EEG-only Image-to-Robot transfer",
        "command": " ".join([sys.executable, *sys.argv]),
        "top3_csv": str(args.top3_csv), "top3_csv_sha256": sha256(args.top3_csv),
        "robot_eeg_input": str(args.robot_eeg_csv), "robot_eeg_sha256": sha256(args.robot_eeg_csv),
        "image_reference_input": str(args.image_reference_csv), "image_reference_sha256": sha256(args.image_reference_csv),
        "participant_config": str(args.participant_config), "participant_config_sha256": sha256(args.participant_config),
        "calibration_root": str(args.calibration_root),
        "models": [{key: value for key, value in record.items() if key not in {"pipeline", "probability_model", "model_path"}} | {"model_path": str(record["model_path"])} for record in models],
        "feature_compatibility": compatibility,
        "event_sources": event_sources,
        "semantic_event_count": int(len(events)),
        "semantic_events_csv": str(semantic_events_path.resolve()),
        "semantic_events_csv_sha256": sha256(semantic_events_path),
        "semantic_events_csv_row_count": int(len(events)),
        "task_probability_definition": "median_windows(median_models(P(HIGH))) over complete ranks 1-3 windows only",
        "top3_window_requirement": "All three frozen ranks must have finite P(HIGH) and a frozen hard prediction.",
        "historical_logic_excluded": ["Stage-B selection", "model preparation/refitting", "GridSearchCV", "ICA/no-ICA comparison", "Face inference", "Multimodal inference"],
        "ratings_use": "Descriptive post-inference comparison only; never used for model selection.",
    }
    (args.output_dir / f"{args.participant}_transfer_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    compact = task_analysis[["task_display", "target", "median_top3_consensus_probability", "descriptive_task_verdict", "unanimous_agreement_rate", "robot_rating", "robot_rating_class", "verdict_matches_robot_rating", "models_with_shift_warning"]].copy()
    compact.columns = ["Task", "Target", "Median P(HIGH)", "Verdict", "Agreement", "Robot rating", "Rating class", "Match", "Shift warnings"]
    (args.output_dir / f"{args.participant}_transfer_summary.md").write_text(
        f"# {args.participant} final frozen EEG-only Image-to-Robot transfer\n\n"
        "Predictions use frozen Image pipelines. Robot ratings are descriptive post-inference comparisons only.\n\n"
        + compact.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"Final frozen transfer complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
