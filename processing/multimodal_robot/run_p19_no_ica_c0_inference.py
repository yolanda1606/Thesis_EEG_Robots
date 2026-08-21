#!/usr/bin/env python3
"""Fit final P19 no-ICA Image models and apply them to existing C0 Robot windows.

This script never regenerates Image or Robot preprocessing/features. It fits
only on p19_no_ica Image trials, freezes the selected pipelines, and reads the
already generated Experiment C C0 window table for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "processing/multimodal_image"
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
if str(IMAGE_ROOT) not in sys.path: sys.path.insert(0, str(IMAGE_ROOT))

from modeling.train_classification import EEG_COLUMNS, make_pipeline, prepare_modality_data, valid_stratified_splits  # noqa: E402
from processing.multimodal_robot.audit_personalized_transfer import distance_audit, percentile  # noqa: E402
from processing.multimodal_robot.run_personalized_inference import PARTICIPANT, TASK_LABELS, high_probabilities  # noqa: E402


NO_ICA_RUN = Path("derived/P19/Image_Experiment/runs/p19_no_ica")
C0_FEATURES = Path("outputs/robot_personalized_inference/P19/preprocessing_harmonization/P19_C0_harmonized_window_features.csv")
COMPARISON_BEST = Path("outputs/image_classification/P19_ica_comparison/P19_ica_vs_no_ica_best_per_target.csv")
OUTPUT_DIR = Path("outputs/robot_personalized_inference/P19/no_ica_c0_inference")
RECIPES = {"valence": {"classifier": "gnb", "feature_count": 10}, "arousal": {"classifier": "knn", "feature_count": 10}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ica-run", type=Path, default=NO_ICA_RUN)
    parser.add_argument("--c0-features", type=Path, default=C0_FEATURES)
    parser.add_argument("--comparison-best", type=Path, default=COMPARISON_BEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without fitting, inference, or writing.")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1: parser.error("--n-jobs must be -1 or a non-zero integer")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_path = args.no_ica_run / "merged/p19_image_trial_dataset.csv"
    if not image_path.is_file() or not args.c0_features.is_file() or not args.comparison_best.is_file():
        raise FileNotFoundError("No-ICA Image table, C0 feature table, or no-ICA comparison best table is missing")
    image, c0, best = pd.read_csv(image_path), pd.read_csv(args.c0_features), pd.read_csv(args.comparison_best)
    if missing := sorted(set(EEG_COLUMNS).difference(image.columns) | set(EEG_COLUMNS).difference(c0.columns)):
        raise ValueError(f"Image/C0 schema lacks expected EEG columns: {missing}")
    required = {"participant", "task", "window_id", "window_index", "window_start_s", "window_end_s"}
    if missing := sorted(required.difference(c0.columns)): raise ValueError(f"C0 feature table missing metadata: {missing}")
    if c0.duplicated("window_id").any() or len(c0) != 337 or c0.task.nunique() != 5:
        raise ValueError("C0 table is not the approved 337-window/five-task representation")
    if c0[EEG_COLUMNS].isna().any().any() or not np.isfinite(c0[EEG_COLUMNS].to_numpy(float)).all():
        raise ValueError("C0 EEG predictors contain missing or non-finite values")
    for target, recipe in RECIPES.items():
        row = best[(best.representation == "no_ICA") & (best.target == target)]
        if len(row) != 1 or str(row.iloc[0].classifier) != recipe["classifier"] or int(row.iloc[0].feature_count_resolved) != recipe["feature_count"]:
            raise ValueError(f"Saved no-ICA comparison does not support approved {target} recipe")
    return image, c0.sort_values(["task", "window_index", "window_id"]).reset_index(drop=True), best


def fit_final(image: pd.DataFrame, target: str, seed: int, n_jobs: int):
    recipe = RECIPES[target]
    features, labels, columns = prepare_modality_data(image, PARTICIPANT, target, "eeg")
    if columns != EEG_COLUMNS: raise RuntimeError("No-ICA Image EEG input schema differs from the validated schema")
    folds = valid_stratified_splits(labels, 3, f"P19 no-ICA {target} final internal CV")
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, labels))
    pipeline, grid = make_pipeline(recipe["classifier"], recipe["feature_count"], seed)
    if recipe["classifier"] == "knn":
        minimum = min(len(train) for train, _ in splits)
        grid = [{**grid[0], "classifier__n_neighbors": [value for value in grid[0]["classifier__n_neighbors"] if value <= minimum]}]
    search = GridSearchCV(pipeline, grid, scoring="balanced_accuracy", cv=splits, n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(features, labels)
    model = search.best_estimator_; selector = model.named_steps["selector"]
    selected = [column for column, keep in zip(EEG_COLUMNS, selector.get_support()) if keep]
    if len(selected) != recipe["feature_count"]: raise RuntimeError("Final SelectKBest feature count differs from approved recipe")
    return model, features, labels, {"internal_cv_folds": folds, "internal_cv_balanced_accuracy": float(search.best_score_), "final_hyperparameters": search.best_params_, "selected_features": selected}


def infer(target: str, model, image: pd.DataFrame, c0: pd.DataFrame) -> pd.DataFrame:
    labels, scores = high_probabilities(model, c0[EEG_COLUMNS])
    scaled_image = model.named_steps["selector"].transform(model.named_steps["scale"].transform(image))
    scaled_c0_full = model.named_steps["scale"].transform(c0[EEG_COLUMNS])
    scaled_c0 = model.named_steps["selector"].transform(scaled_c0_full)
    reference, distance = distance_audit(scaled_image, scaled_c0, k=11)
    p95, maximum = percentile(reference["nearest"], 95), float(np.max(reference["nearest"]))
    selected = [column for column, keep in zip(EEG_COLUMNS, model.named_steps["selector"].get_support()) if keep]
    z = np.abs(pd.DataFrame(scaled_c0_full, columns=EEG_COLUMNS)[selected].to_numpy(float))
    output = c0[["participant", "task", "task_label", "window_id", "window_index", "window_start_s", "window_end_s"]].copy()
    output.insert(1, "target", target); output["predicted_class"] = labels; output["high_class_score"] = scores
    output["nearest_image_distance"] = distance["nearest"]
    output["image_reference_distance_percentile"] = [float(np.mean(reference["nearest"] <= value)*100) for value in distance["nearest"]]
    output["image_reference_nearest_distance_p95"] = p95; output["image_reference_nearest_distance_max"] = maximum
    output["nearest_distance_exceeds_image_p95"] = distance["nearest"] > p95; output["nearest_distance_exceeds_image_max"] = distance["nearest"] > maximum
    output["feature_domain_reliability"] = np.where(distance["nearest"] > maximum, "beyond_image_reference", np.where(distance["nearest"] > p95, "above_image_95pct", "within_reference"))
    output["selected_feature_abs_z_median"] = np.median(z, axis=1); output["selected_feature_abs_z_p95"] = np.percentile(z, 95, axis=1)
    return output


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, task), frame in predictions.groupby(["target", "task"], sort=False):
        scores = frame.high_class_score.to_numpy(float)
        rows.append({"participant": PARTICIPANT, "representation": "C0_reproducible_harmonized_no_ICA", "calibration": "P19_Image_no_ICA", "target": target, "task": task, "task_label": TASK_LABELS[task], "window_count": len(frame), "nearest_image_distance_median": float(np.median(frame.nearest_image_distance)), "pct_above_image_reference_p95": float(np.mean(frame.nearest_distance_exceeds_image_p95)*100), "pct_beyond_image_reference_max": float(np.mean(frame.nearest_distance_exceeds_image_max)*100), "selected_feature_abs_z_median": float(np.median(frame.selected_feature_abs_z_median)), "selected_feature_abs_z_p95": percentile(frame.selected_feature_abs_z_p95, 95), "high_score_mean": float(np.mean(scores)), "high_score_median": float(np.median(scores)), "pct_predicted_high": float(np.mean(frame.predicted_class.eq("HIGH"))*100)})
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, fits: dict, output_dir: Path) -> None:
    lines = ["# P19 Matched No-ICA Image-to-C0 Robot Inference", "", "P19 Image no-ICA models were selected from the completed Image-only comparison, tuned/refit only on all usable P19 no-ICA Image trials, then frozen before application to the existing C0 Robot feature table. No Image/Robot feature extraction was rerun and no Robot data was used for fitting, scaling, selection, or tuning.", "", "## Frozen no-ICA calibration", ""]
    for target in RECIPES:
        fit = fits[target]
        lines.append(f"- {target.capitalize()}: EEG -> StandardScaler -> SelectKBest(k=10) -> {RECIPES[target]['classifier']}; final Image-only internal CV BA={fit['internal_cv_balanced_accuracy']:.3f}; parameters `{fit['final_hyperparameters']}`; selected features: {', '.join(fit['selected_features'])}.")
    lines.extend(["", "## C0 Robot application", "", "| Target | Task | Median nearest Image distance | % beyond Image reference max | Mean HIGH score | % predicted HIGH |", "|---|---|---:|---:|---:|---:|"])
    for row in summary.itertuples(index=False): lines.append(f"| {row.target} | {row.task_label} | {row.nearest_image_distance_median:.3f} | {row.pct_beyond_image_reference_max:.1f} | {row.high_score_mean:.3f} | {row.pct_predicted_high:.1f} |")
    lines.extend(["", "## Interpretation boundary", "", "These are matched no-ICA Image-to-C0 Robot probabilities and feature-domain diagnostics, not Robot emotion ground truth or model confidence. They do not replace the canonical ICA-based transfer analysis."])
    (output_dir / "P19_no_ica_C0_inference_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv); image, c0, best = load_inputs(args)
    if args.dry_run:
        print("Dry-run validation passed: completed no-ICA Image comparison supports final Top-10 GNB/kNN recipes and C0 has 337 compatible windows.")
        return 0
    if args.output_dir.exists(): raise FileExistsError(f"Refusing to overwrite existing no-ICA/C0 inference output: {args.output_dir}")
    input_hashes = {str(path): sha256(path) for path in (args.no_ica_run / "merged/p19_image_trial_dataset.csv", args.c0_features, args.comparison_best)}
    fits, predictions = {}, []
    for target in RECIPES:
        model, image_features, labels, details = fit_final(image, target, args.seed, args.n_jobs)
        fits[target] = details; predictions.append(infer(target, model, image_features, c0)); fits[target]["model"] = model
    prediction_table = pd.concat(predictions, ignore_index=True); summary = summarize(prediction_table)
    for frame in (prediction_table, summary):
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(float)).all(): raise RuntimeError("No-ICA/C0 diagnostics contain NaN or infinity")
    args.output_dir.mkdir(parents=True)
    prediction_table.to_csv(args.output_dir / "P19_no_ica_C0_window_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "P19_no_ica_C0_task_summary.csv", index=False)
    for target, fit in fits.items():
        pd.DataFrame({"feature": fit["selected_features"]}).to_csv(args.output_dir / f"P19_no_ica_{target}_selected_features.csv", index=False)
        joblib.dump(fit["model"], args.output_dir / f"P19_no_ica_{target}_image_calibration_model.joblib")
    write_report(summary, fits, args.output_dir)
    manifest = {"participant": PARTICIPANT, "calibration_run": str(args.no_ica_run), "robot_representation": str(args.c0_features), "source_comparison_best": str(args.comparison_best), "recipes": {target: {key: value for key, value in fit.items() if key != "model"} for target, fit in fits.items()}, "robot_data_used_for_fit_scale_selection_or_tuning": False, "source_hashes": input_hashes}
    (args.output_dir / "P19_no_ica_C0_inference_manifest.json").write_text(json.dumps(manifest, indent=2, default=float)+"\n", encoding="utf-8")
    for text, before in input_hashes.items():
        if sha256(Path(text)) != before: raise RuntimeError(f"Read-only input changed during inference: {text}")
    print(f"P19 matched no-ICA/C0 inference complete: {args.output_dir}"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
