#!/usr/bin/env python3
"""Experimental P19 Image-to-Robot feature ablation with strict Image-only fitting.

Experiment B tests only the pre-specified removals documented in the completed
P19 domain-shift audit. Robot windows are never used for tuning, fitting,
scaling, feature selection, or choosing candidates. The original strict-transfer
artifacts are read as Experiment A references and are never modified.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, pairwise_distances
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.multimodal_image.modeling.train_classification import (  # noqa: E402
    load_participant_table, make_pipeline, prepare_modality_data, valid_stratified_splits,
)
from processing.multimodal_robot.audit_personalized_transfer import distance_audit, percentile  # noqa: E402
from processing.multimodal_robot.run_personalized_inference import (  # noqa: E402
    PARTICIPANT, ROBOT_FEATURE_PATH, TASK_LABELS, high_probabilities, prepare_robot_windows,
)

INFERENCE_DIR = Path("outputs/robot_personalized_inference/P19")
AUDIT_DIR = INFERENCE_DIR / "domain_shift_audit"
OUTPUT_DIR = INFERENCE_DIR / "feature_ablation"
TARGETS = ("valence", "arousal")
PLOT_TASKS = ("shape_sorter_observation", "shape_sorter_interaction", "sisyphus")
ORIGINAL_FEATURES = {
    "valence": ("eeg_se__PO8", "eeg_hc__PO8", "eeg_mf_hz__PO8", "eeg_mf_hz__Fz", "eeg_bp_delta__Fz"),
    "arousal": ("eeg_bp_beta__PO8", "eeg_bp_beta__Fz", "eeg_hc__Fz", "eeg_hm__Fz", "eeg_se__Fz", "eeg_hc__PO8", "eeg_mf_hz__Fz", "eeg_hm__PO8", "eeg_mf_hz__PO8", "eeg_bp_beta__Pz"),
}
CANDIDATES = {
    "V1_remove_delta_Fz": {"target": "valence", "classifier": "gnb", "remove": ("eeg_bp_delta__Fz",),
                            "rationale": "Pre-specified from completed audit: delta power at Fz dominated GaussianNB LOW-vs-HIGH likelihood in 95–100% of four shifted tasks."},
    "A1_remove_hc_Fz": {"target": "arousal", "classifier": "knn", "remove": ("eeg_hc__Fz",),
                         "rationale": "Pre-specified from completed audit: Hjorth complexity at Fz was the dominant nearest-Image distance contributor in the four problematic tasks."},
    "A2_remove_hc_and_se_Fz": {"target": "arousal", "classifier": "knn", "remove": ("eeg_hc__Fz", "eeg_se__Fz"),
                                "rationale": "A2 is pre-specified solely from existing audit: after Hjorth complexity at Fz, spectral entropy at Fz was the only repeated secondary nearest-neighbour distance contributor across Pick & Place, Observation, Stack, and Sisyphus (median fraction about 14–20%)."},
}
OUTPUT_NAMES = ("P19_feature_ablation_image_validation.csv", "P19_feature_ablation_robot_predictions.csv", "P19_feature_ablation_robot_summary.csv", "P19_feature_ablation_summary.md", "P19_feature_ablation_manifest.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default=PARTICIPANT, choices=(PARTICIPANT,))
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--robot-features", type=Path, default=ROBOT_FEATURE_PATH)
    parser.add_argument("--inference-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Replace only Experiment B files in --output-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Validate all fixed inputs and pre-specified candidates without fitting or writing.")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a non-zero integer")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_features(name: str) -> list[str]:
    candidate = CANDIDATES[name]
    features = [feature for feature in ORIGINAL_FEATURES[candidate["target"]] if feature not in candidate["remove"]]
    if len(features) != len(ORIGINAL_FEATURES[candidate["target"]]) - len(candidate["remove"]):
        raise ValueError(f"{name}: fixed ablation feature list is malformed")
    return features


def fixed_feature_pipeline(classifier: str, seed: int):
    """Use the original classifier grid but no SelectKBest search or replacement features."""
    pipeline, grid = make_pipeline(classifier, "all", seed)
    return pipeline, grid


def valid_grid(grid: list[dict[str, list[Any]]], classifier: str, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, list[Any]]]:
    if classifier != "knn":
        return grid
    minimum_train = min(len(train) for train, _ in splits)
    values = [value for value in grid[0]["classifier__n_neighbors"] if value <= minimum_train]
    if not values:
        raise ValueError("No valid kNN neighbor values remain for Image-only CV")
    return [{**grid[0], "classifier__n_neighbors": values}]


def nested_image_evaluation(features: pd.DataFrame, labels: pd.Series, classifier: str, seed: int, n_jobs: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Same individual nested StratifiedKFold design as the Image classification study."""
    folds = valid_stratified_splits(labels, 5, "P19 ablation outer Image CV")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer.split(features, labels), 1):
        x_train, y_train = features.iloc[train], labels.iloc[train].reset_index(drop=True)
        inner_count = valid_stratified_splits(y_train, min(3, folds), f"P19 ablation fold {fold} Image CV")
        inner = list(StratifiedKFold(n_splits=inner_count, shuffle=True, random_state=seed + fold).split(x_train, y_train))
        pipeline, grid = fixed_feature_pipeline(classifier, seed)
        search = GridSearchCV(pipeline, valid_grid(grid, classifier, inner), scoring="balanced_accuracy", cv=inner, n_jobs=n_jobs, refit=True, error_score="raise")
        search.fit(x_train, y_train)
        predicted = search.predict(features.iloc[test])
        tn, fp, fn, tp = confusion_matrix(labels.iloc[test], predicted, labels=[0, 1]).ravel()
        fold_rows.append({"fold": fold, "balanced_accuracy": float(balanced_accuracy_score(labels.iloc[test], predicted)),
                          "low_recall": float(tn / (tn + fp)) if tn + fp else 0.0, "high_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                          "best_inner_balanced_accuracy": float(search.best_score_), "best_parameters": json.dumps(search.best_params_, sort_keys=True)})
    frame = pd.DataFrame(fold_rows)
    return {"cv_folds": folds, "balanced_accuracy_mean": float(frame.balanced_accuracy.mean()), "balanced_accuracy_sd": float(frame.balanced_accuracy.std(ddof=1)),
            "low_recall_mean": float(frame.low_recall.mean()), "high_recall_mean": float(frame.high_recall.mean())}, fold_rows


def fit_final_image_pipeline(features: pd.DataFrame, labels: pd.Series, classifier: str, seed: int, n_jobs: int):
    """Tune and freeze the candidate using Image trials only, before Robot application."""
    folds = valid_stratified_splits(labels, 3, "P19 ablation final internal Image CV")
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, labels))
    pipeline, grid = fixed_feature_pipeline(classifier, seed)
    search = GridSearchCV(pipeline, valid_grid(grid, classifier, splits), scoring="balanced_accuracy", cv=splits, n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(features, labels)
    return search.best_estimator_, {"final_internal_cv_folds": folds, "final_internal_cv_balanced_accuracy": float(search.best_score_), "final_hyperparameters": search.best_params_}


def reference_image_validation(inference_dir: Path, target: str) -> float:
    manifest = json.loads((inference_dir / f"P19_{target}_image_calibration_manifest.json").read_text(encoding="utf-8"))
    return float(manifest["saved_classification_result_source"]["mean_nested_cv_balanced_accuracy"])


def robot_reliability(model, image: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    scaled_image = model.named_steps["scale"].transform(image)
    scaled_robot = model.named_steps["scale"].transform(robot)
    reference, distances = distance_audit(scaled_image, scaled_robot, k=11)
    p95, maximum = percentile(reference["nearest"], 95), float(np.max(reference["nearest"]))
    rows = []
    for index in range(len(robot)):
        nearest = float(distances["nearest"][index])
        status = "beyond_image_reference" if nearest > maximum else ("above_image_95pct" if nearest > p95 else "within_reference")
        rows.append({"nearest_image_distance": nearest, "image_reference_distance_percentile": float(np.mean(reference["nearest"] <= nearest) * 100),
                     "image_reference_nearest_distance_p95": p95, "image_reference_nearest_distance_max": maximum,
                     "nearest_distance_exceeds_image_p95": bool(nearest > p95), "nearest_distance_exceeds_image_max": bool(nearest > maximum),
                     "mean_11_image_neighbor_distance": float(distances["mean_k"][index]), "feature_domain_reliability": status})
    return pd.DataFrame(rows)


def original_reference_rows(inference_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(inference_dir / "P19_robot_window_predictions.csv")
    reliability = pd.read_csv(inference_dir / "domain_shift_audit/P19_window_domain_reliability.csv")
    rows = []
    reliability_columns = [
        "window_id", "nearest_image_distance", "image_reference_distance_percentile",
        "image_reference_nearest_distance_p95", "image_reference_nearest_distance_max",
        "nearest_distance_exceeds_image_p95", "nearest_distance_exceeds_image_max",
        "mean_11_image_neighbor_distance", "feature_domain_reliability",
    ]
    for target in TARGETS:
        joined = predictions[["task", "window_id", "window_index", "window_start_s", f"{target}_predicted_class", f"{target}_high_score"]].merge(
            reliability.loc[reliability.target == target, reliability_columns], on="window_id", validate="one_to_one")
        joined["candidate"] = "V0_original_strict_transfer" if target == "valence" else "A0_original_strict_transfer"
        joined["target"] = target
        joined = joined.rename(columns={f"{target}_predicted_class": "predicted_class", f"{target}_high_score": "high_class_score"})
        rows.append(joined[["candidate", "target", "task", "window_id", "window_index", "window_start_s", "predicted_class", "high_class_score", "nearest_image_distance", "image_reference_distance_percentile", "image_reference_nearest_distance_p95", "image_reference_nearest_distance_max", "nearest_distance_exceeds_image_p95", "nearest_distance_exceeds_image_max", "mean_11_image_neighbor_distance", "feature_domain_reliability"]])
    return pd.concat(rows, ignore_index=True)


def candidate_robot_rows(name: str, model, image: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    candidate = CANDIDATES[name]
    features = candidate_features(name)
    predictors = robot[features]
    labels, scores = high_probabilities(model, predictors)
    reliability = robot_reliability(model, image[features], predictors)
    base = robot[["task", "window_id", "window_index", "window_start_s"]].reset_index(drop=True).copy()
    base.insert(0, "candidate", name)
    base.insert(1, "target", candidate["target"])
    base["predicted_class"], base["high_class_score"] = labels, scores
    return pd.concat([base, reliability], axis=1)


def robot_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, target, task), part in predictions.groupby(["candidate", "target", "task"], sort=False):
        scores = part.high_class_score.to_numpy(float)
        rows.append({"participant": PARTICIPANT, "candidate": candidate, "target": target, "task": task, "task_label": TASK_LABELS[task], "window_count": len(part),
                     "high_probability_mean": float(np.mean(scores)), "high_probability_median": float(np.median(scores)),
                     "pct_score_lt_0_05": float(np.mean(scores < .05) * 100), "pct_score_gt_0_95": float(np.mean(scores > .95) * 100),
                     "nearest_image_distance_median": float(np.median(part.nearest_image_distance)),
                     "pct_above_image_reference_p95": float(np.mean(part.nearest_distance_exceeds_image_p95) * 100),
                     "pct_beyond_image_reference_max": float(np.mean(part.nearest_distance_exceeds_image_max) * 100)})
    return pd.DataFrame(rows)


def plot_comparisons(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    for task in PLOT_TASKS:
        figure, axes = plt.subplots(2, 1, figsize=(9.5, 5.8), sharex=True, constrained_layout=True)
        target_candidates = {"valence": ["V0_original_strict_transfer", "V1_remove_delta_Fz"], "arousal": ["A0_original_strict_transfer", "A1_remove_hc_Fz", "A2_remove_hc_and_se_Fz"]}
        for axis, target in zip(axes, TARGETS):
            part = predictions[(predictions.target == target) & (predictions.task == task)]
            for candidate in target_candidates[target]:
                data = part[part.candidate == candidate]
                if data.empty:
                    continue
                style = {"V0_original_strict_transfer": ("#707070", "Original strict transfer"), "A0_original_strict_transfer": ("#707070", "Original strict transfer"),
                         "V1_remove_delta_Fz": ("#1f77b4", "Experimental V1 ablation"), "A1_remove_hc_Fz": ("#d62728", "Experimental A1 ablation"), "A2_remove_hc_and_se_Fz": ("#2ca02c", "Experimental A2 ablation")}[candidate]
                axis.plot(data.window_start_s, data.high_class_score, marker="o", markersize=2.5, linewidth=1.1, color=style[0], label=style[1])
                far = data.feature_domain_reliability.eq("beyond_image_reference")
                if far.any(): axis.scatter(data.loc[far, "window_start_s"], data.loc[far, "high_class_score"], marker="x", s=22, color=style[0], zorder=4)
            axis.axhline(.5, color="black", linestyle="--", linewidth=1, label="0.5 classification threshold")
            axis.set(ylim=(-.03, 1.03), ylabel=f"High-{target} probability")
            axis.legend(loc="best", frameon=False, fontsize=7)
        axes[-1].set_xlabel("Task-relative EEG window start (s)")
        figure.suptitle(f"P19 {TASK_LABELS[task]}: Experiment A vs experimental feature ablations\nCrosses indicate candidate-specific distance beyond Image reference; scores are unchanged only for Experiment A")
        path = output_dir / f"P19_{task}_feature_ablation_comparison.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def write_report(validation: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# P19 Cross-Context Feature Ablation (Experiment B)", "", "Experiment B is separate from, and does not replace, the frozen strict-transfer result (Experiment A). All candidate fitting, scaling, hyperparameter selection, and validation use P19 Image trials only. Robot windows are applied only after each candidate has been frozen.", "",
             "## Pre-specified candidates", "", "- V1 removes only `eeg_bp_delta__Fz` from the original valence Top-5.", "- A1 removes only `eeg_hc__Fz` from the original arousal Top-10.", "- A2 removes `eeg_hc__Fz` and `eeg_se__Fz`; the latter was selected before ablation execution from the existing driver audit, not from Robot candidate outcomes.", "", "## Image-only validation", "", "| Candidate | Target | Features | Nested Image CV BA | Δ vs original saved nested-CV BA | LOW recall | HIGH recall | Final Image-only hyperparameters |", "|---|---|---|---:|---:|---:|---:|---|"]
    for row in validation.itertuples(index=False):
        lines.append(f"| {row.candidate} | {row.target} | {row.n_features} | {row.image_nested_cv_balanced_accuracy:.3f} | {row.delta_vs_original_image_nested_cv_ba:+.3f} | {row.image_nested_cv_low_recall:.3f} | {row.image_nested_cv_high_recall:.3f} | `{row.final_hyperparameters}` |")
    lines.extend(["", "## Robot application and interpretation boundary", "", "The robot summaries compare feature-domain distance and probability behavior, but do not choose a winner. Less extreme Robot probabilities do not demonstrate a better emotion model. Image validation and feature-domain diagnostics must both be considered. No Robot scaler, normalization, adaptation, or retraining was used.", "", "## Required caution", "", "If the retained representation remains largely beyond the Image nearest-neighbour reference, feature ablation alone is insufficient; matched preprocessing/provenance and a prospectively evaluated domain-adaptation study would still be required."])
    (output_dir / "P19_feature_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.inference_dir / "P19_robot_window_predictions.csv", args.inference_dir / "P19_valence_image_calibration_manifest.json", args.inference_dir / "P19_arousal_image_calibration_manifest.json", args.inference_dir / "domain_shift_audit/P19_window_domain_reliability.csv"):
        if not path.is_file(): raise FileNotFoundError(f"Required strict-transfer artifact missing: {path}")
    table, _ = load_participant_table(args.derived_root, PARTICIPANT)
    image = {target: prepare_modality_data(table, PARTICIPANT, target, "eeg")[:2] for target in TARGETS}
    _, robot = prepare_robot_windows(args.robot_features)
    for name in CANDIDATES:
        target, features = CANDIDATES[name]["target"], candidate_features(name)
        if not set(features).issubset(image[target][0].columns) or not set(features).issubset(robot.columns): raise ValueError(f"{name}: fixed features absent from Image or Robot data")
    if args.dry_run:
        print(f"Dry-run validation passed: {len(robot)} approved Robot windows and {len(CANDIDATES)} pre-specified candidates.")
        for name in CANDIDATES: print(f"{name}: {CANDIDATES[name]['target']} / {len(candidate_features(name))} fixed features / Image-only tuning")
        return 0
    existing = [args.output_dir / path for path in OUTPUT_NAMES if (args.output_dir / path).exists()]
    if existing and not args.overwrite: raise FileExistsError("Refusing to overwrite Experiment B outputs without --overwrite")
    input_hash = sha256(args.inference_dir / "P19_robot_window_predictions.csv")
    validation_rows, candidate_rows, fitted = [], [], {}
    for name, candidate in CANDIDATES.items():
        target, features = candidate["target"], candidate_features(name)
        x, y = image[target][0][features], image[target][1]
        evaluated, _ = nested_image_evaluation(x, y, candidate["classifier"], args.seed, args.n_jobs)
        model, final = fit_final_image_pipeline(x, y, candidate["classifier"], args.seed, args.n_jobs)
        reference_ba = reference_image_validation(args.inference_dir, target)
        validation_rows.append({"participant": PARTICIPANT, "candidate": name, "target": target, "classifier": candidate["classifier"], "features": ";".join(features), "n_features": len(features), "removed_features": ";".join(candidate["remove"]), "pre_specified_robot_diagnostic_rationale": candidate["rationale"], "image_nested_cv_folds": evaluated["cv_folds"], "image_nested_cv_balanced_accuracy": evaluated["balanced_accuracy_mean"], "image_nested_cv_balanced_accuracy_sd": evaluated["balanced_accuracy_sd"], "delta_vs_original_image_nested_cv_ba": evaluated["balanced_accuracy_mean"] - reference_ba, "image_nested_cv_low_recall": evaluated["low_recall_mean"], "image_nested_cv_high_recall": evaluated["high_recall_mean"], **final})
        candidate_rows.append(candidate_robot_rows(name, model, x, robot))
        fitted[name] = model
    predictions = pd.concat([original_reference_rows(args.inference_dir), *candidate_rows], ignore_index=True)
    validation, summary = pd.DataFrame(validation_rows), robot_summary(predictions)
    for frame in (validation, predictions, summary):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(float)
        if not np.isfinite(numeric).all(): raise RuntimeError("Ablation diagnostics contain NaN or infinity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output_dir / "P19_feature_ablation_image_validation.csv", index=False)
    predictions.to_csv(args.output_dir / "P19_feature_ablation_robot_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "P19_feature_ablation_robot_summary.csv", index=False)
    for name, model in fitted.items(): joblib.dump(model, args.output_dir / f"P19_{name}_image_only_frozen_model.joblib")
    plot_comparisons(predictions, args.output_dir)
    write_report(validation, summary, args.output_dir)
    manifest = {"experiment": "B_cross_context_feature_ablation", "participant": PARTICIPANT, "strict_transfer_artifacts_unchanged": True, "robot_data_used_only_after_image_only_freeze": True, "robot_data_not_used_for_tuning_scaling_selection_or_fitting": True, "candidates": {name: {**value, "fixed_features": candidate_features(name)} for name, value in CANDIDATES.items()}}
    (args.output_dir / "P19_feature_ablation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if sha256(args.inference_dir / "P19_robot_window_predictions.csv") != input_hash: raise RuntimeError("Original strict-transfer predictions changed during Experiment B")
    print(f"P19 Experiment B feature ablation complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
