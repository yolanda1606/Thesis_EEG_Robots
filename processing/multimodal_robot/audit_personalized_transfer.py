#!/usr/bin/env python3
"""Read-only domain-shift audit for frozen P19 Image-to-Robot EEG models.

The audit never fits or changes a classifier, scaler, selector, robot feature
table, Image table, or primary inference output. It only loads frozen artifacts
and writes separate diagnostic outputs when explicitly invoked.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import pairwise_distances


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.multimodal_image.modeling.train_classification import (  # noqa: E402
    EEG_COLUMNS,
    load_participant_table,
    prepare_modality_data,
)
from processing.multimodal_robot.run_personalized_inference import (  # noqa: E402
    EXCLUDED_TASKS,
    PARTICIPANT,
    TASK_LABELS,
    TASK_ORDER,
    prepare_robot_windows,
)


INFERENCE_DIR = Path("outputs/robot_personalized_inference/P19")
AUDIT_DIR = INFERENCE_DIR / "domain_shift_audit"
ROBOT_FEATURE_PATH = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/features/eeg_window_features.csv")
DERIVED_ROOT = Path("derived")
TARGETS = ("valence", "arousal")
EPSILON = np.finfo(float).eps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default=PARTICIPANT, choices=(PARTICIPANT,))
    parser.add_argument("--derived-root", type=Path, default=DERIVED_ROOT)
    parser.add_argument("--robot-features", type=Path, default=ROBOT_FEATURE_PATH)
    parser.add_argument("--inference-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate frozen inputs without calculating or writing diagnostics.")
    return parser.parse_args(argv)


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value))


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_sd": float(np.std(values, ddof=0)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q1": percentile(values, 25),
        f"{prefix}_q3": percentile(values, 75),
        f"{prefix}_iqr": percentile(values, 75) - percentile(values, 25),
    }


def transform_selected(pipeline, values: pd.DataFrame) -> np.ndarray:
    """Apply only the already-fitted Image scaler and selector."""
    scaled = pipeline.named_steps["scale"].transform(values)
    return pipeline.named_steps["selector"].transform(scaled)


def selected_features(pipeline) -> list[str]:
    support = pipeline.named_steps["selector"].get_support(indices=True)
    return [EEG_COLUMNS[index] for index in support]


def load_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.inference_dir.is_dir():
        raise FileNotFoundError(f"Frozen P19 inference directory does not exist: {args.inference_dir}")
    prediction_path = args.inference_dir / "P19_robot_window_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Frozen robot prediction file does not exist: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    all_robot, included_robot = prepare_robot_windows(args.robot_features)
    required_prediction = {"window_id", "task", "valence_high_score", "arousal_high_score", "post_task_valence_rating", "post_task_arousal_rating"}
    if missing := sorted(required_prediction.difference(predictions.columns)):
        raise ValueError(f"Frozen prediction file missing columns: {missing}")
    if predictions.duplicated("window_id").any():
        raise ValueError("Frozen prediction file contains duplicate window IDs")
    expected_ids = set(included_robot["window_id"])
    if set(predictions["window_id"]) != expected_ids:
        raise ValueError("Frozen prediction windows do not exactly match the approved robot windows")
    robot = predictions.merge(included_robot[["window_id", *EEG_COLUMNS]], on="window_id", how="inner", validate="one_to_one")
    robot["task"] = pd.Categorical(robot["task"], categories=TASK_ORDER, ordered=True)
    robot = robot.sort_values(["task", "window_index", "window_id"]).reset_index(drop=True)
    models: dict[str, Any] = {}
    image: dict[str, Any] = {}
    image_table, image_path = load_participant_table(args.derived_root, PARTICIPANT)
    for target in TARGETS:
        model_path = args.inference_dir / f"P19_{target}_image_calibration_model.joblib"
        manifest_path = args.inference_dir / f"P19_{target}_image_calibration_manifest.json"
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Frozen {target} model or manifest is missing")
        pipeline = joblib.load(model_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features, labels, columns = prepare_modality_data(image_table, PARTICIPANT, target, "eeg")
        if columns != EEG_COLUMNS:
            raise RuntimeError(f"Unexpected Image EEG input schema for {target}")
        manifest_features = manifest.get("selected_top_features")
        model_features = selected_features(pipeline)
        if set(manifest_features) != set(model_features) or len(manifest_features) != len(model_features):
            raise ValueError(f"Frozen {target} manifest selected feature set does not match its serialized model")
        if pipeline.named_steps["scale"].n_features_in_ != len(EEG_COLUMNS):
            raise ValueError(f"Frozen {target} scaler has an unexpected feature count")
        models[target] = pipeline
        image[target] = {"features": features, "labels": labels.to_numpy(dtype=int), "path": image_path,
                         "manifest": manifest, "selected_features": model_features}
    return {"all_robot": all_robot, "robot": robot, "models": models, "image": image, "predictions": predictions}


def feature_domain_rows(state: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    task_z_rows: list[dict[str, Any]] = []
    robot: pd.DataFrame = state["robot"]
    for target in TARGETS:
        image_values: pd.DataFrame = state["image"][target]["features"]
        labels: np.ndarray = state["image"][target]["labels"]
        pipeline = state["models"][target]
        scaler = pipeline.named_steps["scale"]
        selected = state["image"][target]["selected_features"]
        image_scaled = pd.DataFrame(scaler.transform(image_values), columns=EEG_COLUMNS)
        robot_scaled = pd.DataFrame(scaler.transform(robot[EEG_COLUMNS]), columns=EEG_COLUMNS, index=robot.index)
        for feature in selected:
            image_raw = image_values[feature].to_numpy(dtype=float)
            image_low = image_values.loc[labels == 0, feature].to_numpy(dtype=float)
            image_high = image_values.loc[labels == 1, feature].to_numpy(dtype=float)
            for scope, subset in [("ALL_ANALYZED_TASKS", robot), *[(task, robot[robot["task"].astype(str) == task]) for task in TASK_ORDER if task not in EXCLUDED_TASKS]]:
                raw = subset[feature].to_numpy(dtype=float)
                z = robot_scaled.loc[subset.index, feature].to_numpy(dtype=float)
                image_sd = float(np.std(image_raw, ddof=0))
                row = {"target": target, "feature": feature, "robot_scope": scope, **describe(image_raw, "image"),
                       **describe(image_low, "image_low"), **describe(image_high, "image_high"), **describe(raw, "robot")}
                row.update({
                    "robot_mean_minus_image_mean_in_image_sd": float((np.mean(raw) - np.mean(image_raw)) / image_sd) if image_sd > 0 else np.nan,
                    "robot_pct_outside_image_min_max": float(np.mean((raw < np.min(image_raw)) | (raw > np.max(image_raw))) * 100),
                    "robot_pct_outside_image_q1_q3": float(np.mean((raw < percentile(image_raw, 25)) | (raw > percentile(image_raw, 75))) * 100),
                    "robot_pct_outside_image_mean_pm_2sd": float(np.mean(np.abs(raw - np.mean(image_raw)) > 2 * image_sd) * 100) if image_sd > 0 else np.nan,
                    "robot_pct_outside_image_mean_pm_3sd": float(np.mean(np.abs(raw - np.mean(image_raw)) > 3 * image_sd) * 100) if image_sd > 0 else np.nan,
                    "robot_z_median_abs": float(np.median(np.abs(z))),
                    "robot_z_p95_abs": percentile(np.abs(z), 95),
                    "robot_z_max_abs": float(np.max(np.abs(z))),
                    "robot_z_pct_abs_gt_2": float(np.mean(np.abs(z) > 2) * 100),
                    "robot_z_pct_abs_gt_3": float(np.mean(np.abs(z) > 3) * 100),
                    "robot_z_pct_abs_gt_5": float(np.mean(np.abs(z) > 5) * 100),
                })
                rows.append(row)
            for task in TASK_ORDER:
                if task in EXCLUDED_TASKS:
                    continue
                subset = robot[robot["task"].astype(str) == task]
                z = robot_scaled.loc[subset.index, feature].to_numpy(dtype=float)
                task_z_rows.append({"target": target, "task": task, "feature": feature,
                                    "median_abs_z": float(np.median(np.abs(z))), "p95_abs_z": percentile(np.abs(z), 95),
                                    "max_abs_z": float(np.max(np.abs(z))), "pct_abs_z_gt_2": float(np.mean(np.abs(z) > 2) * 100),
                                    "pct_abs_z_gt_3": float(np.mean(np.abs(z) > 3) * 100), "pct_abs_z_gt_5": float(np.mean(np.abs(z) > 5) * 100)})
    return pd.DataFrame(rows), pd.DataFrame(task_z_rows)


def distance_audit(selected_image: np.ndarray, selected_robot: np.ndarray, k: int = 11) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compare robot-to-Image distances with leave-self-out Image reference distances."""
    k = min(k, len(selected_image) - 1)
    image_distances = pairwise_distances(selected_image, selected_image, metric="euclidean")
    np.fill_diagonal(image_distances, np.inf)
    image_sorted = np.sort(image_distances, axis=1)[:, :k]
    robot_sorted = np.sort(pairwise_distances(selected_robot, selected_image, metric="euclidean"), axis=1)[:, :k]
    return ({"nearest": image_sorted[:, 0], "mean_k": image_sorted.mean(axis=1), "max_k": image_sorted[:, -1]},
            {"nearest": robot_sorted[:, 0], "mean_k": robot_sorted.mean(axis=1), "max_k": robot_sorted[:, -1]})


def arousal_neighbor_audit(state: dict[str, Any], scaled_image: np.ndarray, scaled_robot: np.ndarray, reference: dict[str, np.ndarray]) -> pd.DataFrame:
    """Expose exact frozen-kNN neighbour geometry for every robot window."""
    robot: pd.DataFrame = state["robot"]
    labels: np.ndarray = state["image"]["arousal"]["labels"]
    pipeline = state["models"]["arousal"]
    classifier = pipeline.named_steps["classifier"]
    expected_training = classifier._fit_X
    if expected_training.shape != scaled_image.shape or not np.allclose(expected_training, scaled_image):
        raise RuntimeError("Frozen arousal kNN training geometry does not match the reconstructed Image feature space")
    distances, indices = classifier.kneighbors(scaled_robot, n_neighbors=classifier.n_neighbors, return_distance=True)
    scores = pipeline.predict_proba(state["robot"][EEG_COLUMNS])[:, list(classifier.classes_).index(1)]
    rows: list[dict[str, Any]] = []
    for index, (window, distance_row, index_row) in enumerate(zip(robot.itertuples(index=False), distances, indices)):
        neighbor_labels = labels[index_row]
        weights = 1.0 / np.maximum(distance_row, EPSILON)
        weighted_high = float(weights[neighbor_labels == 1].sum() / weights.sum())
        rows.append({
            "participant": PARTICIPANT, "task": str(window.task), "window_id": window.window_id, "window_index": int(window.window_index),
            "window_start_s": float(window.window_start_s), "arousal_high_score": float(scores[index]),
            "closest_neighbor_distance": float(distance_row[0]), "mean_11_neighbor_distance": float(distance_row.mean()),
            "max_11_neighbor_distance": float(distance_row.max()), "high_neighbor_count": int((neighbor_labels == 1).sum()),
            "low_neighbor_count": int((neighbor_labels == 0).sum()), "distance_weighted_high_fraction": weighted_high,
            "nearest_distance_above_image_reference_p95": bool(distance_row[0] > percentile(reference["nearest"], 95)),
            "nearest_distance_above_image_reference_max": bool(distance_row[0] > np.max(reference["nearest"])),
            "image_neighbor_indices": json.dumps([int(value) for value in index_row]),
            "image_neighbor_labels": json.dumps(["HIGH" if value == 1 else "LOW" for value in neighbor_labels]),
            "image_neighbor_distances": json.dumps([float(value) for value in distance_row]),
        })
    frame = pd.DataFrame(rows)
    if not np.allclose(frame["distance_weighted_high_fraction"], frame["arousal_high_score"], atol=1e-12):
        raise RuntimeError("Reconstructed frozen-kNN weighted votes do not match arousal probabilities")
    return frame


def valence_gnb_audit(state: dict[str, Any], selected_robot: np.ndarray) -> pd.DataFrame:
    """Use frozen GaussianNB parameters and stable log likelihoods, without refitting."""
    robot: pd.DataFrame = state["robot"]
    pipeline = state["models"]["valence"]
    classifier = pipeline.named_steps["classifier"]
    features = state["image"]["valence"]["selected_features"]
    jll = classifier._joint_log_likelihood(selected_robot)
    feature_logpdf = -0.5 * (np.log(2.0 * np.pi * classifier.var_[None, :, :])
                             + ((selected_robot[:, None, :] - classifier.theta_[None, :, :]) ** 2) / classifier.var_[None, :, :])
    differences = feature_logpdf[:, 1, :] - feature_logpdf[:, 0, :]
    high_scores = pipeline.predict_proba(robot[EEG_COLUMNS])[:, list(classifier.classes_).index(1)]
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(robot.itertuples(index=False)):
        ranked = np.argsort(np.abs(differences[index]))[::-1]
        rows.append({
            "participant": PARTICIPANT, "task": str(window.task), "window_id": window.window_id, "window_index": int(window.window_index),
            "window_start_s": float(window.window_start_s), "valence_high_score": float(high_scores[index]),
            "joint_log_likelihood_low": float(jll[index, 0]), "joint_log_likelihood_high": float(jll[index, 1]),
            "high_minus_low_joint_log_likelihood": float(jll[index, 1] - jll[index, 0]),
            "dominant_feature": features[int(ranked[0])],
            "dominant_feature_high_minus_low_logpdf": float(differences[index, ranked[0]]),
            "feature_high_minus_low_logpdf": json.dumps({feature: float(value) for feature, value in zip(features, differences[index])}),
        })
    return pd.DataFrame(rows)


def task_summary(state: dict[str, Any], z_by_task: pd.DataFrame, distance_results: dict[str, dict[str, np.ndarray]], references: dict[str, dict[str, np.ndarray]], neighbor: pd.DataFrame, gnb: pd.DataFrame) -> pd.DataFrame:
    robot: pd.DataFrame = state["robot"]
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        probability_column = f"{target}_high_score"
        for task in TASK_ORDER:
            if task in EXCLUDED_TASKS:
                continue
            subset = robot[robot["task"].astype(str) == task].copy()
            scores = subset[probability_column].to_numpy(dtype=float)
            z = z_by_task[(z_by_task["target"] == target) & (z_by_task["task"] == task)]
            positions = subset.index.to_numpy(dtype=int)
            distances = distance_results[target]
            reference = references[target]
            row = {
                "participant": PARTICIPANT, "target": target, "task": task, "task_label": TASK_LABELS[task], "window_count": len(subset),
                "post_task_valence_rating": float(subset["post_task_valence_rating"].iloc[0]),
                "post_task_arousal_rating": float(subset["post_task_arousal_rating"].iloc[0]),
                "score_first": float(scores[0]), "score_last": float(scores[-1]), "score_median": float(np.median(scores)),
                "score_iqr": percentile(scores, 75) - percentile(scores, 25), "score_mean": float(np.mean(scores)),
                "pct_score_lt_0_05": float(np.mean(scores < 0.05) * 100), "pct_score_gt_0_95": float(np.mean(scores > 0.95) * 100),
                "selected_feature_median_abs_z_median": float(z["median_abs_z"].median()),
                "selected_feature_p95_abs_z_max": float(z["p95_abs_z"].max()),
                "selected_feature_pct_abs_z_gt_3_mean": float(z["pct_abs_z_gt_3"].mean()),
                "nearest_image_distance_median": float(np.median(distances["nearest"][positions])),
                "mean_11_image_distance_median": float(np.median(distances["mean_k"][positions])),
                "pct_nearest_distance_above_image_reference_p95": float(np.mean(distances["nearest"][positions] > percentile(reference["nearest"], 95)) * 100),
                "pct_nearest_distance_above_image_reference_max": float(np.mean(distances["nearest"][positions] > np.max(reference["nearest"])) * 100),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def plot_feature_distributions(state: dict[str, Any], output_dir: Path) -> Path:
    robot: pd.DataFrame = state["robot"]
    features = [(target, feature) for target in TARGETS for feature in state["image"][target]["selected_features"]]
    columns = 3
    figure, axes = plt.subplots(int(np.ceil(len(features) / columns)), columns, figsize=(15, 3.2 * int(np.ceil(len(features) / columns))))
    for axis, (target, feature) in zip(np.ravel(axes), features):
        image_values = state["image"][target]["features"]
        labels_for_target = state["image"][target]["labels"]
        series = [image_values.loc[labels_for_target == 0, feature].to_numpy(), image_values.loc[labels_for_target == 1, feature].to_numpy()]
        labels = ["Image LOW", "Image HIGH"]
        for task in TASK_ORDER:
            if task not in EXCLUDED_TASKS:
                series.append(robot.loc[robot["task"].astype(str) == task, feature].to_numpy())
                labels.append(TASK_LABELS[task])
        axis.boxplot(series, labels=labels, showfliers=False)
        axis.set_title(f"{target}: {feature}")
        axis.tick_params(axis="x", rotation=50, labelsize=7)
    for axis in np.ravel(axes)[len(features):]:
        axis.set_visible(False)
    figure.suptitle("P19 raw selected-feature distributions: Image calibration vs Robot windows")
    path = output_dir / "P19_feature_distribution_comparison.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_z_scores(feature_rows: pd.DataFrame, output_dir: Path) -> Path:
    task_rows = feature_rows[feature_rows["robot_scope"] != "ALL_ANALYZED_TASKS"].copy()
    feature_labels = list(dict.fromkeys(task_rows.apply(lambda row: f"{row.target}:{row.feature}", axis=1)))
    tasks = [task for task in TASK_ORDER if task not in EXCLUDED_TASKS]
    values = np.full((len(feature_labels), len(tasks)), np.nan)
    for row in task_rows.itertuples(index=False):
        values[feature_labels.index(f"{row.target}:{row.feature}"), tasks.index(row.robot_scope)] = row.robot_z_median_abs
    figure, axis = plt.subplots(figsize=(9, max(5, 0.38 * len(feature_labels))))
    image = axis.imshow(values, aspect="auto", cmap="magma")
    axis.set(xticks=range(len(tasks)), xticklabels=[TASK_LABELS[task] for task in tasks], yticks=range(len(feature_labels)), yticklabels=feature_labels,
             title="Median absolute z-score under frozen Image scaler")
    axis.tick_params(axis="x", rotation=35)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Median |z|")
    figure.tight_layout()
    path = output_dir / "P19_image_scaled_z_score_diagnostic.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_distances(state: dict[str, Any], distances: dict[str, dict[str, np.ndarray]], references: dict[str, dict[str, np.ndarray]], output_dir: Path) -> Path:
    robot: pd.DataFrame = state["robot"]
    tasks = [task for task in TASK_ORDER if task not in EXCLUDED_TASKS]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, target in zip(axes, TARGETS):
        series = [references[target]["nearest"]] + [distances[target]["nearest"][robot.index[robot["task"].astype(str) == task]] for task in tasks]
        labels = ["Image leave-self-out"] + [TASK_LABELS[task] for task in tasks]
        axis.boxplot(series, labels=labels, showfliers=False)
        axis.axhline(percentile(references[target]["nearest"], 95), color="black", linestyle="--", linewidth=1, label="Image reference 95th percentile")
        axis.set(title=f"{target.capitalize()} selected space", ylabel="Nearest Image distance")
        axis.tick_params(axis="x", rotation=45, labelsize=7)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Frozen Image-scaled selected-feature distance geometry")
    figure.tight_layout()
    path = output_dir / "P19_distance_ood_diagnostic.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_interaction_shift(state: dict[str, Any], output_dir: Path) -> Path:
    robot: pd.DataFrame = state["robot"]
    interaction = "shape_sorter_interaction"
    other_tasks = [task for task in TASK_ORDER if task not in EXCLUDED_TASKS and task != interaction]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        pipeline = state["models"][target]
        selected = state["image"][target]["selected_features"]
        image_features = state["image"][target]["features"]
        labels = state["image"][target]["labels"]
        scale = pipeline.named_steps["scale"]
        image_z = pd.DataFrame(scale.transform(image_features), columns=EEG_COLUMNS)
        robot_z = pd.DataFrame(scale.transform(robot[EEG_COLUMNS]), columns=EEG_COLUMNS, index=robot.index)
        x = np.arange(len(selected))
        width = 0.18
        groups = [("Image LOW", image_z.loc[labels == 0, selected].mean()), ("Image HIGH", image_z.loc[labels == 1, selected].mean()),
                  ("Other robot tasks", robot_z.loc[robot["task"].astype(str).isin(other_tasks), selected].mean()),
                  ("Interaction", robot_z.loc[robot["task"].astype(str) == interaction, selected].mean())]
        for offset, (name, means) in enumerate(groups):
            axis.bar(x + (offset - 1.5) * width, means.to_numpy(), width, label=name)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xticks=x, xticklabels=selected, ylabel="Mean Image-scaled z-score", title=f"{target.capitalize()} selected features")
        axis.tick_params(axis="x", rotation=40, labelsize=8)
        axis.legend(frameon=False, ncol=2)
    path = output_dir / "P19_shape_sorter_interaction_comparison.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def verdict(feature_rows: pd.DataFrame, task_rows: pd.DataFrame) -> tuple[str, list[str]]:
    overall = feature_rows[feature_rows["robot_scope"] == "ALL_ANALYZED_TASKS"]
    high_z_fraction = float(np.mean(overall["robot_z_p95_abs"] > 3) * 100)
    high_range_fraction = float(np.mean(overall["robot_pct_outside_image_min_max"] > 25) * 100)
    distance_fraction = float(np.mean(task_rows["pct_nearest_distance_above_image_reference_p95"] > 50) * 100)
    evidence = [
        f"{high_z_fraction:.1f}% of selected feature/model rows have robot 95th-percentile |z| above 3.",
        f"{high_range_fraction:.1f}% have more than 25% of robot windows outside the Image observed min/max range.",
        f"{distance_fraction:.1f}% of task/model summaries have more than half of windows beyond the Image leave-self-out nearest-distance 95th percentile.",
    ]
    if high_z_fraction >= 50 or high_range_fraction >= 50 or distance_fraction >= 50:
        return "TRANSFER_SHOWS_SUBSTANTIAL_DOMAIN_SHIFT", evidence
    return "TRANSFER_LOOKS_REASONABLE_FOR_EXPLORATORY_USE", evidence


def write_report(state: dict[str, Any], feature_rows: pd.DataFrame, task_rows: pd.DataFrame, neighbor: pd.DataFrame, gnb: pd.DataFrame, references: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    status, evidence = verdict(feature_rows, task_rows)
    overall = feature_rows[feature_rows["robot_scope"] == "ALL_ANALYZED_TASKS"].copy()
    shifted = overall.sort_values(["robot_z_p95_abs", "robot_pct_outside_image_min_max"], ascending=False).head(8)
    interaction = feature_rows[feature_rows["robot_scope"] == "shape_sorter_interaction"].copy()
    interaction = interaction.reindex(interaction["robot_mean_minus_image_mean_in_image_sd"].abs().sort_values(ascending=False).index).head(8)
    valence = task_rows[task_rows["target"] == "valence"]
    arousal = task_rows[task_rows["target"] == "arousal"]
    rating_table = task_rows[task_rows["target"] == "valence"][["task_label", "post_task_valence_rating", "score_mean"]].merge(
        task_rows[task_rows["target"] == "arousal"][["task_label", "post_task_arousal_rating", "score_mean"]], on="task_label", suffixes=("_valence", "_arousal"))
    high_arousal = neighbor[neighbor["arousal_high_score"] >= 0.90]
    lines = ["# P19 Image-to-Robot Transfer Validity Audit", "", f"## Verdict: `{status}`", "", *[f"- {item}" for item in evidence], "",
             "This is a diagnostic comparison against the frozen P19 Image calibration domain. It does not infer emotion states, alter predictions, or establish clinical/scientific validity.",
             "", "## Calibration and data scope", "",
             "- Frozen models, scalers, selectors, hyperparameters, robot features, and primary inference outputs were read without modification.",
             "- Five Status-aligned robot tasks / 337 windows were audited. Shape Sorter Alone remains excluded.",
             "- Mean/SD z diagnostics are descriptive; feature distributions may be non-Gaussian, so threshold counts are not invalidity rules.",
             "", "## Strongest selected-feature shifts", ""]
    for row in shifted.itertuples(index=False):
        lines.append(f"- {row.target}/{row.feature}: robot mean shift {row.robot_mean_minus_image_mean_in_image_sd:.2f} Image SD; {row.robot_pct_outside_image_min_max:.1f}% outside Image min/max; robot 95th |z|={row.robot_z_p95_abs:.2f}.")
    lines.extend(["", "## Distance geometry", ""])
    for target in TARGETS:
        ref = references[target]["nearest"]
        current = task_rows[task_rows["target"] == target]
        lines.append(f"- {target.capitalize()} Image leave-self-out nearest distance: median {np.median(ref):.3f}, 95th percentile {percentile(ref, 95):.3f}, maximum {np.max(ref):.3f}.")
        for row in current.itertuples(index=False):
            lines.append(f"  - {row.task_label}: median robot nearest distance {row.nearest_image_distance_median:.3f}; {row.pct_nearest_distance_above_image_reference_p95:.1f}% above Image 95th percentile; {row.pct_nearest_distance_above_image_reference_max:.1f}% above Image maximum.")
    lines.extend(["", "## Arousal kNN geometry", "",
                  f"- Among {len(high_arousal)} robot windows with arousal HIGH probability >=0.90, {float(high_arousal['nearest_distance_above_image_reference_p95'].mean() * 100) if len(high_arousal) else np.nan:.1f}% are beyond the Image nearest-distance 95th percentile.",
                  "- The neighbour audit records exact 11-neighbour distances, labels, and distance-weighted vote fractions for every robot window; the reconstructed weighted fraction exactly matches the frozen classifier probability.",
                  "", "## Valence GaussianNB behavior", ""])
    for row in valence.itertuples(index=False):
        audit = gnb[gnb["task"] == row.task]
        driver = str(audit["dominant_feature"].mode().iloc[0])
        driver_fraction = float(audit["dominant_feature"].value_counts(normalize=True).iloc[0])
        near_zero = float(np.mean(audit["valence_high_score"] < 0.05) * 100)
        interpretation = "one feature commonly dominates" if driver_fraction >= 0.70 else "likelihood separation is distributed across features/windows"
        lines.append(f"- {row.task_label}: {near_zero:.1f}% scores <0.05; most common log-likelihood driver `{driver}` ({driver_fraction:.1%} of windows), so {interpretation}.")
    lines.extend(["", "## Arousal kNN geometry by task", ""])
    for row in arousal.itertuples(index=False):
        audit = neighbor[neighbor["task"] == row.task]
        high = audit[audit["arousal_high_score"] >= 0.90]
        far_high = float(np.mean(high["nearest_distance_above_image_reference_p95"]) * 100) if len(high) else np.nan
        lines.append(f"- {row.task_label}: mean distance-weighted HIGH neighbour vote {audit['distance_weighted_high_fraction'].mean():.3f}; {len(high)} windows score >=0.90, of which {far_high:.1f}% exceed the Image nearest-distance 95th percentile.")
    lines.extend(["", "## Shape Sorter Interaction reversal", ""])
    for row in interaction.itertuples(index=False):
        lines.append(f"- {row.target}/{row.feature}: Interaction robot mean is {row.robot_mean_minus_image_mean_in_image_sd:.2f} Image SD from the Image mean; task-level 95th |z|={row.robot_z_p95_abs:.2f}.")
    lines.extend(["", "## Post-task ratings: descriptive only", "", "| Task | Rating valence | Mean valence HIGH score | Rating arousal | Mean arousal HIGH score |", "|---|---:|---:|---:|---:|"])
    for row in rating_table.itertuples(index=False):
        lines.append(f"| {row.task_label} | {row.post_task_valence_rating:.0f} | {row.score_mean_valence:.3f} | {row.post_task_arousal_rating:.0f} | {row.score_mean_arousal:.3f} |")
    lines.extend(["", "No formal correlation or correctness claim is made from five post-task ratings: each rating summarizes a whole task, while the classifier outputs Image-calibrated window probabilities.",
                  "", "## Temporal score behavior", ""])
    for row in task_rows.itertuples(index=False):
        lines.append(f"- {row.target.capitalize()} / {row.task_label}: first={row.score_first:.3f}, last={row.score_last:.3f}, median={row.score_median:.3f}, IQR={row.score_iqr:.3f}, <0.05={row.pct_score_lt_0_05:.1f}%, >0.95={row.pct_score_gt_0_95:.1f}%.")
    lines.extend(["", "## Scientific limitation", "", state["image"]["valence"]["manifest"]["robot_preprocessing_provenance_limitation"],
                  "The audit result should guide further preprocessing/model investigation before strong direct Image-to-Robot interpretations."])
    (output_dir / "P19_transfer_validity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = load_frozen_inputs(args)
    if args.dry_run:
        print("Dry-run validation passed: frozen models, manifests, predictions, Image calibration data, and robot windows are compatible.")
        print(f"Robot windows: {len(state['robot'])}; tasks: {state['robot']['task'].nunique()}; excluded: {', '.join(EXCLUDED_TASKS)}")
        for target in TARGETS:
            print(f"{target}: {len(state['image'][target]['selected_features'])} selected frozen features")
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output directory: {args.output_dir}")
    feature_rows, z_by_task = feature_domain_rows(state)
    distances: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, dict[str, np.ndarray]] = {}
    transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for target in TARGETS:
        image_values = state["image"][target]["features"]
        robot_values = state["robot"][EEG_COLUMNS]
        selected_image = transform_selected(state["models"][target], image_values)
        selected_robot = transform_selected(state["models"][target], robot_values)
        references[target], distances[target] = distance_audit(selected_image, selected_robot, k=11)
        transformed[target] = (selected_image, selected_robot)
    neighbor = arousal_neighbor_audit(state, *transformed["arousal"], references["arousal"])
    gnb = valence_gnb_audit(state, transformed["valence"][1])
    summary = task_summary(state, z_by_task, distances, references, neighbor, gnb)
    diagnostic_frames = [feature_rows, z_by_task, summary, neighbor.drop(columns=["image_neighbor_indices", "image_neighbor_labels", "image_neighbor_distances"]), gnb]
    for frame in diagnostic_frames:
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise RuntimeError("Audit diagnostics contain NaN or infinity")
    args.output_dir.mkdir(parents=True)
    feature_rows.to_csv(args.output_dir / "P19_feature_domain_shift.csv", index=False)
    summary.to_csv(args.output_dir / "P19_task_domain_shift_summary.csv", index=False)
    neighbor.to_csv(args.output_dir / "P19_arousal_knn_neighbor_audit.csv", index=False)
    gnb.to_csv(args.output_dir / "P19_valence_gnb_audit.csv", index=False)
    plot_feature_distributions(state, args.output_dir)
    plot_z_scores(feature_rows, args.output_dir)
    plot_distances(state, distances, references, args.output_dir)
    plot_interaction_shift(state, args.output_dir)
    write_report(state, feature_rows, summary, neighbor, gnb, references, args.output_dir)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "participant": PARTICIPANT,
                "source_files": {"robot_features": str(args.robot_features), "inference_dir": str(args.inference_dir),
                                 "image_run": str(args.derived_root / PARTICIPANT / "Image_Experiment" / "runs" / "p19_final")},
                "read_only_frozen_artifacts": True, "software_versions": {"python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    (args.output_dir / "P19_domain_shift_audit_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"P19 transfer audit complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
