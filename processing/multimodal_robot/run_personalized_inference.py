#!/usr/bin/env python3
"""Fit P19 Image-only calibration models and apply them to existing robot EEG windows.

This initial transfer pilot deliberately reads the already extracted robot feature
table.  It never regenerates robot features and never uses robot observations for
feature selection, scaling, hyperparameter selection, or classifier fitting.
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.multimodal_image.modeling.train_classification import (  # noqa: E402
    EEG_COLUMNS,
    TARGET_COLUMNS,
    label_ratings,
    load_participant_table,
    make_pipeline,
    prepare_modality_data,
    resolved_feature_count,
    valid_stratified_splits,
)


PARTICIPANT = "P19"
TASK_ORDER = (
    "pick_place",
    "shape_sorter_observation",
    "stack",
    "sisyphus",
    "shape_sorter_interaction",
    "shape_sorter_alone",
)
TASK_LABELS = {
    "pick_place": "Pick and Place",
    "shape_sorter_observation": "Shape Sorter Observation",
    "stack": "Stack",
    "sisyphus": "Sisyphus",
    "shape_sorter_interaction": "Shape Sorter Interaction",
    "shape_sorter_alone": "Shape Sorter Alone",
}
EXCLUDED_TASKS = {
    "shape_sorter_alone": (
        "Excluded from this initial pilot: no Status anchors, filename-fallback timing, "
        "BDF/header clock conflict, and no explicit usable task/video interval."
    )
}
TIE_RULE = [
    "For exact numerical ties in mean nested-CV balanced accuracy, prefer a single-modality model over a multimodal model.",
    "If modality and classifier family are also identical, prefer the smaller requested feature set.",
    "The selected configuration did not numerically outperform tied alternatives.",
]
ROBOT_PROVENANCE_LIMITATION = (
    "Initial P19 transfer-pilot limitation: the existing robot feature table has incomplete "
    "ICA provenance and its exact producing source version cannot currently be independently "
    "reconstructed. Robot preprocessing is not claimed to be proven perfectly equivalent to "
    "the Image preprocessing pipeline."
)
DEPLOYMENT_RECIPES = {
    "valence": {
        "modality": "eeg",
        "classifier": "gnb",
        "feature_count_request": "5",
        "tie_explanation": (
            "EEG/GaussianNB/Top-5 tied exactly in mean nested-CV balanced accuracy with "
            "EEG/GaussianNB/Top-10 and multimodal counterparts. The single-modality rule "
            "removes multimodal variants; the smaller-feature-set rule selects Top-5. This "
            "also matches stage_b_best_per_participant.csv."
        ),
    },
    "arousal": {
        "modality": "eeg",
        "classifier": "knn",
        "feature_count_request": "10",
        "tie_explanation": (
            "EEG/kNN/Top-10 is the pre-specified representative among exact mean-BA ties, "
            "which include EEG/SVM/Top-5 and multimodal variants. The single-modality rule "
            "removes multimodal variants; the requested representative retains EEG/kNN/Top-10. "
            "It did not numerically outperform the tied alternatives."
        ),
    },
}
ROBOT_FEATURE_PATH = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/features/eeg_window_features.csv")
CLASSIFICATION_ROOT = Path("outputs/image_classification/stage_b_individual_v1")
RATINGS_PATH = Path("data/Robot Ratings - Robot Ratings.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default=PARTICIPANT, choices=(PARTICIPANT,))
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--classification-root", type=Path, default=CLASSIFICATION_ROOT)
    parser.add_argument("--robot-features", type=Path, default=ROBOT_FEATURE_PATH)
    parser.add_argument("--ratings-file", type=Path, default=RATINGS_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robot_personalized_inference/P19"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and schemas without fitting or writing outputs.")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a non-zero integer")
    return args


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def read_saved_results(classification_root: Path, target: str, recipe: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover and verify the saved P19 winner and all exact mean-BA ties."""
    summary_path = classification_root / "results_summary.csv"
    stage_best_path = classification_root / "stage_b_best_per_participant.csv"
    if not summary_path.is_file() or not stage_best_path.is_file():
        raise FileNotFoundError(f"Saved P19 classification results are incomplete under {classification_root}")
    summary = pd.read_csv(summary_path)
    required = {"participant", "target", "modality", "classifier", "feature_count_request", "balanced_accuracy_mean"}
    if missing := sorted(required.difference(summary.columns)):
        raise ValueError(f"Saved results summary missing columns: {missing}")
    rows = summary[(summary["participant"] == PARTICIPANT) & (summary["target"] == target)].copy()
    if rows.empty:
        raise ValueError(f"No saved P19 results for target={target}")
    rows["feature_count_request"] = rows["feature_count_request"].astype(str)
    max_ba = float(rows["balanced_accuracy_mean"].max())
    tied = rows[np.isclose(rows["balanced_accuracy_mean"], max_ba, rtol=0.0, atol=1e-12)].copy()
    chosen = tied[(tied["modality"] == recipe["modality"])
                  & (tied["classifier"] == recipe["classifier"])
                  & (tied["feature_count_request"] == recipe["feature_count_request"])]
    if len(chosen) != 1:
        raise ValueError(
            f"Saved results do not support the approved {target} deployment recipe "
            f"{recipe['modality']}/{recipe['classifier']}/Top-{recipe['feature_count_request']}"
        )
    if target == "valence":
        stage_best = pd.read_csv(stage_best_path)
        row = stage_best[(stage_best["participant"] == PARTICIPANT) & (stage_best["target"] == target)]
        if len(row) != 1:
            raise ValueError("Saved stage-best table does not contain exactly one P19 valence row")
        row = row.iloc[0]
        if (str(row["modality"]), str(row["classifier"]), str(row["feature_count"])) != ("eeg", "gnb", "5"):
            raise ValueError("Saved stage-best P19 valence recipe no longer matches the approved tie resolution")
    columns = ["modality", "classifier", "feature_count_request", "feature_count_resolved", "balanced_accuracy_mean", "balanced_accuracy_sd"]
    return tied[columns].sort_values(columns[:3]).reset_index(drop=True), {
        "mean_nested_cv_balanced_accuracy": max_ba,
        "saved_summary_path": str(summary_path),
        "saved_stage_best_path": str(stage_best_path),
    }


def fit_image_calibration(derived_root: Path, target: str, recipe: dict[str, str], seed: int, n_jobs: int) -> dict[str, Any]:
    """Tune and refit one complete target-specific pipeline using Image trials only."""
    table, image_dataset = load_participant_table(derived_root, PARTICIPANT)
    features, labels, columns = prepare_modality_data(table, PARTICIPANT, target, recipe["modality"])
    if columns != EEG_COLUMNS:
        raise RuntimeError("Approved P19 deployment recipe unexpectedly resolved to non-EEG columns")
    feature_count = resolved_feature_count(recipe["feature_count_request"], len(columns))
    folds = valid_stratified_splits(labels, 3, f"P19 {target} final internal CV")
    internal_cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, labels))
    pipeline, parameter_grid = make_pipeline(recipe["classifier"], feature_count, seed)
    if recipe["classifier"] == "knn":
        minimum_training_rows = min(len(train) for train, _ in internal_cv)
        parameter_grid = [{**parameter_grid[0], "classifier__n_neighbors": [
            value for value in parameter_grid[0]["classifier__n_neighbors"] if value <= minimum_training_rows
        ]}]
    search = GridSearchCV(pipeline, parameter_grid, scoring="balanced_accuracy", cv=internal_cv,
                          n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(features, labels)
    fitted = search.best_estimator_
    selector = fitted.named_steps.get("selector")
    if selector is None:
        raise RuntimeError("Deployment recipe requires SelectKBest but fitted pipeline has no selector")
    selected_mask = selector.get_support()
    selected = pd.DataFrame({"feature": columns, "f_classif_score": selector.scores_, "selected": selected_mask})
    selected = selected[selected["selected"]].copy().sort_values("f_classif_score", ascending=False).reset_index(drop=True)
    if len(selected) != int(feature_count):
        raise RuntimeError(f"SelectKBest chose {len(selected)} features; expected {feature_count}")
    results = pd.DataFrame(search.cv_results_)
    cv_candidates = [
        {"rank": int(row.rank_test_score), "mean_balanced_accuracy": float(row.mean_test_score),
         "std_balanced_accuracy": float(row.std_test_score), "parameters": row.params}
        for row in results.itertuples(index=False)
    ]
    classifier = fitted.named_steps["classifier"]
    scaler = fitted.named_steps["scale"]
    return {
        "pipeline": fitted,
        "selected_features": selected,
        "input_columns": columns,
        "labels": labels,
        "image_dataset": image_dataset,
        "internal_cv_folds": folds,
        "best_parameters": search.best_params_,
        "best_internal_cv_balanced_accuracy": float(search.best_score_),
        "cv_candidates": cv_candidates,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": classifier.classes_.tolist(),
    }


def prepare_robot_windows(robot_features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the validated long robot feature table into strict Image EEG column order."""
    if not robot_features_path.is_file():
        raise FileNotFoundError(f"Robot EEG feature table does not exist: {robot_features_path}")
    long = pd.read_csv(robot_features_path)
    required = {"participant", "task", "window_id", "window_start_s", "window_end_s", "channel", "sample_count",
                "category", "actual_condition", "actual_speed", "segment_id", "alignment_method", "alignment_qc",
                "eeg_coverage", "video_coverage"}
    if missing := sorted(required.difference(long.columns)):
        raise ValueError(f"Robot EEG feature table missing required metadata columns: {missing}")
    if set(long["participant"].astype(str)) != {PARTICIPANT}:
        raise ValueError("Robot feature table is not exclusively P19")
    expected_families = {column.split("__", 1)[0] for column in EEG_COLUMNS}
    feature_columns = {column for column in long.columns if column.startswith("eeg_") and column != "eeg_coverage"}
    if feature_columns != expected_families:
        raise ValueError(
            "Robot feature definitions differ from the Image EEG feature schema; "
            f"missing={sorted(expected_families - feature_columns)}, extra={sorted(feature_columns - expected_families)}"
        )
    unknown_channels = sorted(set(long["channel"]).difference({column.rsplit("__", 1)[1] for column in EEG_COLUMNS}))
    if unknown_channels:
        raise ValueError(f"Robot feature table contains unexpected EEG channels: {unknown_channels}")
    duplicates = long.duplicated(["window_id", "channel"])
    if duplicates.any():
        raise ValueError(f"Robot feature table has {int(duplicates.sum())} duplicate window/channel rows")
    channel_counts = long.groupby("window_id")["channel"].agg(lambda values: set(values))
    expected_channels = {column.rsplit("__", 1)[1] for column in EEG_COLUMNS}
    invalid = channel_counts[channel_counts.apply(lambda values: values != expected_channels)]
    if not invalid.empty:
        raise ValueError(f"Robot windows with incomplete or unexpected channel sets: {invalid.index.tolist()[:10]}")
    metadata_columns = ["participant", "task", "category", "actual_condition", "actual_speed", "segment_id", "window_id",
                        "window_start_s", "window_end_s", "alignment_method", "alignment_qc", "eeg_coverage", "video_coverage"]
    metadata = long[metadata_columns].drop_duplicates()
    if metadata.duplicated("window_id").any():
        raise ValueError("Robot task metadata conflict within a window")
    wide = long.pivot(index="window_id", columns="channel", values=sorted(expected_families))
    wide.columns = [f"{family}__{channel}" for family, channel in wide.columns]
    wide = wide.reindex(columns=EEG_COLUMNS)
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError("Robot EEG predictors contain missing or non-finite values")
    windows = metadata.merge(wide.reset_index(), on="window_id", how="inner", validate="one_to_one")
    windows["task"] = pd.Categorical(windows["task"], categories=TASK_ORDER, ordered=True)
    windows = windows.sort_values(["task", "segment_id", "window_start_s", "window_id"]).reset_index(drop=True)
    windows["window_index"] = windows.groupby("task", observed=False).cumcount() + 1
    windows["observation_vs_interaction"] = windows["category"]
    included = windows[~windows["task"].isin(EXCLUDED_TASKS)].copy()
    expected_included = set(TASK_ORDER).difference(EXCLUDED_TASKS)
    if set(included["task"].astype(str)) != expected_included:
        raise ValueError(f"Included robot tasks differ from approved set: {sorted(set(included['task'].astype(str)))}")
    return windows, included


def high_probabilities(pipeline, predictors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return LOW/HIGH labels and the probability assigned to class HIGH (encoded 1)."""
    classifier = pipeline.named_steps["classifier"]
    classes = classifier.classes_
    high_indices = np.flatnonzero(classes == 1)
    if len(high_indices) != 1:
        raise RuntimeError(f"Fitted classifier classes do not contain exactly one HIGH=1 class: {classes.tolist()}")
    predicted = pipeline.predict(predictors)
    probabilities = pipeline.predict_proba(predictors)[:, high_indices[0]]
    return np.where(predicted == 1, "HIGH", "LOW"), probabilities


def read_post_task_ratings(ratings_path: Path) -> pd.DataFrame:
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Robot ratings file does not exist: {ratings_path}")
    ratings = pd.read_csv(ratings_path)
    required = {"Participant ID", "Robot Experiment", "Valence", "Arousal"}
    if missing := sorted(required.difference(ratings.columns)):
        raise ValueError(f"Robot ratings file missing columns: {missing}")
    ratings = ratings[ratings["Participant ID"].astype(str).str.upper() == PARTICIPANT].copy()
    ratings["task"] = ratings["Robot Experiment"].map({label: key for key, label in TASK_LABELS.items()})
    if ratings["task"].isna().any() or ratings.duplicated("task").any():
        raise ValueError("P19 post-task ratings do not map uniquely to the configured robot task names")
    return ratings[["task", "Valence", "Arousal"]].rename(columns={"Valence": "post_task_valence_rating", "Arousal": "post_task_arousal_rating"})


def task_summary(predictions: pd.DataFrame, all_windows: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        available = all_windows[all_windows["task"].astype(str) == task]
        rating = ratings[ratings["task"] == task]
        base = {"participant": PARTICIPANT, "task": task, "task_label": TASK_LABELS[task],
                "post_task_valence_rating": rating["post_task_valence_rating"].iloc[0] if len(rating) else np.nan,
                "post_task_arousal_rating": rating["post_task_arousal_rating"].iloc[0] if len(rating) else np.nan}
        if task in EXCLUDED_TASKS:
            rows.append({**base, "analysis_status": "SKIPPED", "exclusion_reason": EXCLUDED_TASKS[task], "source_window_count": len(available)})
            continue
        part = predictions[predictions["task"].astype(str) == task].sort_values("window_index")
        row = {**base, "analysis_status": "ANALYZED", "exclusion_reason": "", "source_window_count": len(available)}
        for target in ("valence", "arousal"):
            labels = part[f"{target}_predicted_class"]
            high = part[f"{target}_high_score"]
            transition = labels.ne(labels.shift())
            if len(transition):
                transition.iloc[0] = False
            row.update({
                f"{target}_valid_window_count": int(len(part)),
                f"{target}_high_count": int((labels == "HIGH").sum()),
                f"{target}_high_percentage": float((labels == "HIGH").mean() * 100.0),
                f"{target}_high_score_mean": float(high.mean()),
                f"{target}_high_score_median": float(high.median()),
                f"{target}_high_score_std": float(high.std(ddof=0)),
                f"{target}_high_score_min": float(high.min()),
                f"{target}_high_score_max": float(high.max()),
                f"{target}_low_to_high_transition_count": int(((labels == "HIGH") & transition).sum()),
                f"{target}_high_to_low_transition_count": int(((labels == "LOW") & transition).sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def save_task_figures(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for task in TASK_ORDER:
        if task in EXCLUDED_TASKS:
            continue
        part = predictions[predictions["task"].astype(str) == task].sort_values("window_index")
        figure, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
        for axis, target, label in zip(axes, ("valence", "arousal"), ("High-valence probability", "High-arousal probability")):
            axis.plot(part["window_start_s"], part[f"{target}_high_score"], marker="o", markersize=3, linewidth=1.2)
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 classification threshold")
            axis.set_ylim(-0.03, 1.03)
            axis.set_ylabel(label)
            axis.legend(loc="best", frameon=False)
        axes[-1].set_xlabel("Task-relative EEG window start (s)")
        figure.suptitle(f"P19 {TASK_LABELS[task]}: Image-calibrated robot EEG transfer pilot")
        path = output_dir / f"P19_{task}_high_class_probabilities.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def calibration_manifest(target: str, recipe: dict[str, str], saved_ties: pd.DataFrame, saved_info: dict[str, Any], fit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "participant": PARTICIPANT,
        "target": target,
        "image_run_path": str(args.derived_root / PARTICIPANT / "Image_Experiment" / "runs" / "p19_final"),
        "image_dataset_path": str(fit["image_dataset"]),
        "saved_classification_result_source": saved_info,
        "all_exact_mean_nested_cv_balanced_accuracy_ties": saved_ties.to_dict(orient="records"),
        "tie_resolution_rule": TIE_RULE,
        "tie_resolution_explanation": recipe["tie_explanation"],
        "chosen_deployment_recipe": {key: recipe[key] for key in ("modality", "classifier", "feature_count_request")},
        "chosen_configuration_did_not_numerically_outperform_tied_alternatives": True,
        "label_definition": "LOW: rating < 4; HIGH: rating >= 4",
        "usable_image_training_trials": int(len(fit["labels"])),
        "image_class_counts": {"LOW": int((fit["labels"] == 0).sum()), "HIGH": int((fit["labels"] == 1).sum())},
        "complete_input_feature_columns": fit["input_columns"],
        "selected_top_features": fit["selected_features"]["feature"].tolist(),
        "select_k_best_method": "sklearn.feature_selection.SelectKBest(score_func=f_classif)",
        "internal_cv_design": {"splitter": "StratifiedKFold", "n_splits": fit["internal_cv_folds"], "shuffle": True, "random_state": args.seed,
                               "scoring": "balanced_accuracy", "image_data_only": True},
        "internal_cv_candidates": fit["cv_candidates"],
        "internal_cv_result_used_to_choose_final_hyperparameters": fit["best_internal_cv_balanced_accuracy"],
        "final_classifier_hyperparameters": fit["best_parameters"],
        "scaler_parameters": {"mean": fit["scaler_mean"], "scale": fit["scaler_scale"]},
        "fitted_class_ordering": fit["classes"],
        "robot_data_used_during_fitting": False,
        "robot_preprocessing_provenance_limitation": ROBOT_PROVENANCE_LIMITATION,
        "software_versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                              "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "source_code_paths": ["processing/multimodal_robot/run_personalized_inference.py", "processing/multimodal_image/modeling/train_classification.py"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_summary(output_dir: Path, fits: dict[str, dict[str, Any]], ties: dict[str, pd.DataFrame], predictions: pd.DataFrame) -> None:
    lines = ["# P19 Image-to-Robot EEG Inference Pilot", "", "## Calibration", ""]
    for target in ("valence", "arousal"):
        recipe = DEPLOYMENT_RECIPES[target]
        fit = fits[target]
        lines.extend([
            f"- **{target.capitalize()}**: EEG -> StandardScaler -> SelectKBest(k={recipe['feature_count_request']}) -> {recipe['classifier']}. ",
            f"  Internal Image-only CV selected `{json.dumps(fit['best_parameters'], sort_keys=True)}` (BA {fit['best_internal_cv_balanced_accuracy']:.3f}).",
            f"  Image calibration trials: {len(fit['labels'])} (LOW={(fit['labels'] == 0).sum()}, HIGH={(fit['labels'] == 1).sum()}).",
            f"  Selected features: {', '.join(fit['selected_features']['feature'])}.",
            f"  Exact saved nested-CV ties: {len(ties[target])}; {recipe['tie_explanation']}",
        ])
    lines.extend([
        "", "## Robot application", "",
        f"- Analysed {len(predictions)} existing 2.0 s / 1.0 s-step EEG windows across five tasks.",
        "- Shape Sorter Alone was intentionally excluded: no Status anchors, filename-fallback timing, BDF/header clock conflict, and no explicit usable task/video interval.",
        "- `valence_high_score` and `arousal_high_score` are high-class probabilities: how strongly a window resembles the respective HIGH class learned from P19's Image Experiment. They are not continuous emotion levels.",
        "- Robot windows have no continuous emotion ground truth. Post-task ratings are descriptive only and were not used for fitting, tuning, or inference.",
        f"- {ROBOT_PROVENANCE_LIMITATION}",
    ])
    (output_dir / "P19_robot_inference_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    saved_ties: dict[str, pd.DataFrame] = {}
    saved_info: dict[str, dict[str, Any]] = {}
    for target, recipe in DEPLOYMENT_RECIPES.items():
        saved_ties[target], saved_info[target] = read_saved_results(args.classification_root, target, recipe)
    all_windows, robot_windows = prepare_robot_windows(args.robot_features)
    ratings = read_post_task_ratings(args.ratings_file)
    if args.dry_run:
        print("Dry-run validation passed.")
        for target, ties in saved_ties.items():
            recipe = DEPLOYMENT_RECIPES[target]
            print(f"{target}: approved {recipe['modality']}/{recipe['classifier']}/Top-{recipe['feature_count_request']}; exact saved ties={len(ties)}")
        print(f"Robot windows available for inference: {len(robot_windows)} across {robot_windows['task'].nunique()} tasks; Shape Sorter Alone excluded.")
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    fits = {target: fit_image_calibration(args.derived_root, target, recipe, args.seed, args.n_jobs)
            for target, recipe in DEPLOYMENT_RECIPES.items()}
    args.output_dir.mkdir(parents=True)
    try:
        predictions = robot_windows.copy()
        for target, fit in fits.items():
            robot_input_columns = [column for column in EEG_COLUMNS if column in predictions.columns]
            if robot_input_columns != fit["input_columns"] or len(robot_input_columns) != len(fit["input_columns"]):
                raise RuntimeError(f"Robot predictor ordering mismatch for {target}")
            predictors = predictions.loc[:, robot_input_columns]
            if predictors.shape[1] != len(fit["input_columns"]) or not np.isfinite(predictors.to_numpy(dtype=float)).all():
                raise RuntimeError(f"Robot predictors are incomplete or non-finite for {target}")
            labels, scores = high_probabilities(fit["pipeline"], predictors)
            predictions[f"{target}_predicted_class"] = labels
            predictions[f"{target}_high_score"] = scores
        metadata = ["participant", "task", "category", "observation_vs_interaction", "actual_condition", "actual_speed", "segment_id", "window_id", "window_index",
                    "window_start_s", "window_end_s", "alignment_method", "alignment_qc", "eeg_coverage", "video_coverage",
                    "valence_predicted_class", "valence_high_score", "arousal_predicted_class", "arousal_high_score"]
        predictions = predictions[metadata].merge(ratings, on="task", how="left", validate="many_to_one")
        predictions.to_csv(args.output_dir / "P19_robot_window_predictions.csv", index=False)
        task_summary(predictions, all_windows, ratings).to_csv(args.output_dir / "P19_robot_task_summary.csv", index=False)
        for target, fit in fits.items():
            fit["selected_features"].to_csv(args.output_dir / f"P19_{target}_selected_features.csv", index=False)
            manifest = calibration_manifest(target, DEPLOYMENT_RECIPES[target], saved_ties[target], saved_info[target], fit, args)
            (args.output_dir / f"P19_{target}_image_calibration_manifest.json").write_text(
                json.dumps(manifest, indent=2, default=_json_value) + "\n", encoding="utf-8"
            )
            joblib.dump(fit["pipeline"], args.output_dir / f"P19_{target}_image_calibration_model.joblib")
        save_task_figures(predictions, args.output_dir)
        write_summary(args.output_dir, fits, saved_ties, predictions)
    except Exception:
        # Outputs are deliberately retained for diagnosis only if an unexpected write failure occurs.
        # The normal overwrite guard prevents a later run from silently reusing this directory.
        raise
    print(f"P19 personalized inference complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
