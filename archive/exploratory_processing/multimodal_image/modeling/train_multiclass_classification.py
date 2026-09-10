#!/usr/bin/env python3
"""Leakage-safe P19-only three-class Image Experiment classification pilot.

Ratings are independently labelled LOW=1--2, NEUTRAL=3--5, HIGH=6--7.
The script reads the immutable official P19 merged table and writes only to
``outputs/image_multiclass/<run-name>/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARTICIPANT = "P19"
CLASS_NAMES = ("LOW", "NEUTRAL", "HIGH")
CLASS_LABELS = (0, 1, 2)
TARGET_COLUMNS = {"valence": "valence_rating", "arousal": "arousal_rating"}
EEG_CHANNELS = ("C3", "C4", "Cz", "Fz", "Oz", "PO7", "PO8", "Pz")
EEG_FAMILIES = ("eeg_sd", "eeg_se", "eeg_hm", "eeg_hc", "eeg_mf_hz", "eeg_bp_delta", "eeg_se_delta", "eeg_bp_theta", "eeg_se_theta", "eeg_bp_alpha", "eeg_se_alpha", "eeg_bp_beta", "eeg_se_beta", "eeg_bp_gamma", "eeg_se_gamma")
EEG_COLUMNS = [f"{family}__{channel}" for family in EEG_FAMILIES for channel in EEG_CHANNELS]
FACE_COLUMNS = [f"video_{name}_norm_{stat}" for name in ("irisdo", "eso", "enso", "mnso", "mwo") for stat in ("mean", "std")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="p19_three_class_v1")
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/image_multiclass"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--target", choices=("valence", "arousal", "all"), default="all")
    parser.add_argument("--modality", choices=("eeg", "face", "multimodal", "all"), default="all")
    parser.add_argument("--append", action="store_true", help="Append a non-overlapping target/modality segment to an existing pilot run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a non-zero integer")
    if "/" in args.run_name or "\\" in args.run_name or args.run_name in {"", ".", ".."}:
        parser.error("--run-name must be a simple directory name")
    return args


def modality_columns(modality: str) -> list[str]:
    if modality == "eeg": return EEG_COLUMNS.copy()
    if modality == "face": return FACE_COLUMNS.copy()
    if modality == "multimodal": return EEG_COLUMNS + FACE_COLUMNS
    raise ValueError(f"Unknown modality: {modality}")


def label_ratings(ratings: pd.Series) -> pd.Series:
    """Map observed 1--7 ratings to LOW=0, NEUTRAL=1, HIGH=2."""
    values = pd.to_numeric(ratings, errors="raise")
    if (~values.isin(range(1, 8))).any():
        raise ValueError("Ratings must be integral values from 1 through 7 for three-class labelling")
    return pd.cut(values, bins=[0, 2, 5, 7], labels=CLASS_LABELS, include_lowest=True).astype(int)


def class_counts(labels: pd.Series) -> dict[str, int]:
    return {name: int((labels == label).sum()) for name, label in zip(CLASS_NAMES, CLASS_LABELS)}


def validate_nested_cv(labels: pd.Series, context: str) -> int:
    """Return outer folds, rejecting data unable to support nested stratification.

    At least three rows per class are required: this permits a stratified outer
    split and at least two examples per class in each outer-training partition
    for the inner StratifiedKFold.
    """
    counts = class_counts(labels)
    smallest = min(counts.values())
    if smallest < 3:
        raise ValueError(f"{context}: insufficient class counts for nested stratified CV: {counts}; need at least 3 in every class")
    return min(5, smallest)


def prepare_data(table: pd.DataFrame, target: str, modality: str) -> tuple[pd.DataFrame, pd.Series]:
    columns, target_column = modality_columns(modality), TARGET_COLUMNS[target]
    missing = sorted(set(columns + [target_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{PARTICIPANT} missing required {modality} columns: {missing}")
    numeric = table[columns].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[target_column], errors="coerce")
    usable = numeric.notna().all(axis=1) & ratings.notna()
    features = numeric.loc[usable].reset_index(drop=True)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"{PARTICIPANT} {modality} predictors contain infinity")
    return features, label_ratings(ratings.loc[usable]).reset_index(drop=True)


def make_pipeline(model: str, feature_count: int | str, seed: int) -> tuple[Pipeline, list[dict[str, list[Any]]]]:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if feature_count != "all":
        steps.append(("selector", SelectKBest(score_func=f_classif, k=int(feature_count))))
    if model == "knn":
        steps.append(("classifier", KNeighborsClassifier()))
        grid = [{"classifier__n_neighbors": [3, 5, 7, 11], "classifier__weights": ["uniform", "distance"]}]
    elif model == "svm":
        steps.append(("classifier", SVC(random_state=seed)))
        grid = [
            {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10]},
            {"classifier__kernel": ["rbf"], "classifier__C": [0.1, 1, 10], "classifier__gamma": ["scale", 0.01, 0.1]},
        ]
    elif model == "gnb":
        steps.append(("classifier", GaussianNB()))
        grid = [{"classifier__var_smoothing": [1e-11, 1e-9, 1e-7]}]
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline(steps), grid


def metric_row(y_true: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predicted, labels=CLASS_LABELS)
    recalls = recall_score(y_true, predicted, labels=CLASS_LABELS, average=None, zero_division=0)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "accuracy": accuracy_score(y_true, predicted),
        "macro_precision": precision_score(y_true, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, predicted, average="macro", zero_division=0),
        "recall_low": recalls[0], "recall_neutral": recalls[1], "recall_high": recalls[2],
        **{f"cm_{actual}_{predicted_label}": int(matrix[actual, predicted_label]) for actual in CLASS_LABELS for predicted_label in CLASS_LABELS},
    }


def selected_feature_rows(fitted: Pipeline, columns: list[str], common: dict[str, Any]) -> list[dict[str, Any]]:
    if "selector" not in fitted.named_steps:
        return [{**common, "feature": feature, "selected": True, "feature_score": np.nan} for feature in columns]
    selector: SelectKBest = fitted.named_steps["selector"]
    return [{**common, "feature": feature, "selected": bool(keep), "feature_score": float(score) if np.isfinite(score) else np.nan} for feature, keep, score in zip(columns, selector.get_support(), selector.scores_)]


def evaluate(x: pd.DataFrame, y: pd.Series, target: str, modality: str, model: str, request: str, seed: int, n_jobs: int) -> tuple[list[dict], list[dict], list[dict]]:
    outer_folds = validate_nested_cv(y, f"{PARTICIPANT} {target} {modality}")
    feature_count: int | str = "all" if request == "all" else min(int(request), x.shape[1])
    results: list[dict] = []; selected: list[dict] = []; parameters: list[dict] = []
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(x, y), 1):
        x_train, y_train = x.iloc[train], y.iloc[train].reset_index(drop=True)
        inner_folds = validate_nested_cv(y_train, f"{PARTICIPANT} {target} {modality} outer fold {fold} inner CV")
        inner_folds = min(3, inner_folds)
        inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold).split(x_train, y_train))
        pipeline, grid = make_pipeline(model, feature_count, seed)
        if model == "knn":
            minimum_training_rows = min(len(indices) for indices, _ in inner)
            grid = [{**grid[0], "classifier__n_neighbors": [value for value in grid[0]["classifier__n_neighbors"] if value <= minimum_training_rows]}]
        search = GridSearchCV(pipeline, grid, scoring="balanced_accuracy", cv=inner, n_jobs=n_jobs, refit=True, error_score="raise")
        search.fit(x_train, y_train)
        predicted = search.predict(x.iloc[test])
        common = {"mode": "individual", "participant": PARTICIPANT, "target": target, "modality": modality, "classifier": model, "feature_count_request": request, "feature_count_resolved": feature_count, "fold": fold}
        results.append({**common, "sample_size": len(y), "train_size": len(train), "test_size": len(test), **{f"{name.lower()}_count": count for name, count in class_counts(y).items()}, **metric_row(y.iloc[test], predicted)})
        selected.extend(selected_feature_rows(search.best_estimator_, list(x.columns), common))
        parameters.append({**common, "best_parameters": json.dumps(search.best_params_, sort_keys=True), "inner_best_balanced_accuracy": float(search.best_score_)})
    return results, selected, parameters


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "participant", "target", "modality", "classifier", "feature_count_request", "feature_count_resolved"]
    metrics = ["balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_low", "recall_neutral", "recall_high"]
    summary = results.groupby(keys, dropna=False).agg(folds=("fold", "size"), **{f"{metric}_mean": (metric, "mean") for metric in metrics}, balanced_accuracy_sd=("balanced_accuracy", "std")).reset_index()
    return summary.sort_values(["target", "balanced_accuracy_mean"], ascending=[True, False]).reset_index(drop=True)


def feature_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "participant", "target", "modality", "classifier", "feature_count_request", "feature_count_resolved", "feature"]
    return selected.groupby(keys, dropna=False).agg(selected_folds=("selected", "sum"), outer_folds=("fold", "nunique"), selection_frequency=("selected", "mean"), mean_feature_score=("feature_score", "mean")).reset_index().sort_values(["target", "selection_frequency"], ascending=[True, False])


def aggregated_confusions(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "participant", "target", "modality", "classifier", "feature_count_request", "feature_count_resolved"]
    cm_columns = [f"cm_{actual}_{predicted}" for actual in CLASS_LABELS for predicted in CLASS_LABELS]
    return results.groupby(keys, dropna=False)[cm_columns].sum().reset_index()


def plot_confusion(matrix: np.ndarray, target: str, description: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Trials")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class", title=f"P19 {target}: {description}")
    for row in range(3):
        for column in range(3): axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def write_figures(observed: dict[str, dict[str, int]], summary: pd.DataFrame, confusions: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(3); width = .35
    axis.bar(positions - width / 2, [observed["valence"][name] for name in CLASS_NAMES], width, label="Valence")
    axis.bar(positions + width / 2, [observed["arousal"][name] for name in CLASS_NAMES], width, label="Arousal")
    axis.set(xticks=positions, xticklabels=CLASS_NAMES, ylabel="P19 trials", title="P19 three-class label balance"); axis.legend(); fig.tight_layout(); fig.savefig(output / "p19_class_balance.png", dpi=160); plt.close(fig)
    comparison = summary.sort_values("balanced_accuracy_mean", ascending=False).groupby(["target", "modality", "classifier"], as_index=False).first()
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [f"{row.target}\n{row.modality}/{row.classifier}\n({row.feature_count_request})" for row in comparison.itertuples()]
    axis.bar(range(len(comparison)), comparison.balanced_accuracy_mean); axis.axhline(1 / 3, color="black", linestyle="--", label="Three-class chance BA (1/3)")
    axis.set(xticks=range(len(comparison)), xticklabels=labels, ylabel="Mean balanced accuracy", title="P19 multiclass BA by modality/model"); axis.tick_params(axis="x", rotation=70); axis.legend(); fig.tight_layout(); fig.savefig(output / "best_multiclass_ba_comparison.png", dpi=160); plt.close(fig)
    for target in TARGET_COLUMNS:
        best = summary[summary.target == target].iloc[0]
        match = confusions[(confusions.target == target) & (confusions.modality == best.modality) & (confusions.classifier == best.classifier) & (confusions.feature_count_request == best.feature_count_request)].iloc[0]
        matrix = np.array([[match[f"cm_{actual}_{predicted}"] for predicted in CLASS_LABELS] for actual in CLASS_LABELS])
        plot_confusion(matrix, target, f"best: {best.modality}/{best.classifier}/{best.feature_count_request}", output / f"best_{target}_confusion_matrix.png")


def git_commit() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None


def write_markdown(observed: dict[str, dict[str, int]], summary: pd.DataFrame, confusions: pd.DataFrame, output: Path) -> None:
    lines = ["# P19 three-class classification pilot", "", "Exploratory, leakage-safe nested-CV analysis. LOW=ratings 1--2, NEUTRAL=3--5, HIGH=6--7. Three-class balanced-accuracy chance is approximately 0.333; it is not numerically comparable to binary chance BA of 0.500.", "", "## Class counts", "", "| Target | LOW | NEUTRAL | HIGH |", "|---|---:|---:|---:|"]
    for target, counts in observed.items(): lines.append(f"| {target.title()} | {counts['LOW']} | {counts['NEUTRAL']} | {counts['HIGH']} |")
    lines += ["", "All classes met the pilot's minimum count requirement (three rows per class) for nested stratified CV after complete-feature filtering.", "", "## Best models", ""]
    for target in TARGET_COLUMNS:
        best = summary[summary.target == target].iloc[0]
        match = confusions[(confusions.target == target) & (confusions.modality == best.modality) & (confusions.classifier == best.classifier) & (confusions.feature_count_request == best.feature_count_request)].iloc[0]
        matrix = np.array([[match[f"cm_{actual}_{predicted}"] for predicted in CLASS_LABELS] for actual in CLASS_LABELS])
        low_neutral, neutral_high, low_high = matrix[0, 1] + matrix[1, 0], matrix[1, 2] + matrix[2, 1], matrix[0, 2] + matrix[2, 0]
        pairs = {"LOW vs NEUTRAL": low_neutral, "NEUTRAL vs HIGH": neutral_high, "LOW vs HIGH": low_high}
        most_common = max(pairs, key=pairs.get)
        lines += [f"### {target.title()}", "", f"Best configuration: **{best.modality} / {best.classifier} / {best.feature_count_request} features**. Mean multiclass BA: **{best.balanced_accuracy_mean:.3f}**; macro F1: **{best.macro_f1_mean:.3f}**. Per-class recall (fold mean): LOW={best.recall_low_mean:.3f}, NEUTRAL={best.recall_neutral_mean:.3f}, HIGH={best.recall_high_mean:.3f}.", "", f"Across out-of-fold predictions, the most frequent paired error was **{most_common}** ({int(pairs[most_common])} directional errors combined). This indicates whether NEUTRAL is mainly confused with an adjacent affect level.", ""]
    lines += ["## Context", "", "Existing P19 binary reference results were approximately BA=0.600 for valence and BA=0.683 for arousal. These are descriptive references only: a lower three-class BA does not by itself mean the three-class model is worse, because the chance baseline changes from 0.500 to about 0.333.", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path | None:
    derived_root = (PROJECT_ROOT / args.derived_root).resolve() if not args.derived_root.is_absolute() else args.derived_root
    output_root = (PROJECT_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    dataset = derived_root / PARTICIPANT / "Image_Experiment" / "runs" / "p19_final" / "merged" / "p19_image_trial_dataset.csv"
    if not dataset.is_file(): raise FileNotFoundError(f"P19 official merged dataset missing: {dataset}")
    table = pd.read_csv(dataset)
    observed: dict[str, dict[str, int]] = {}
    prepared: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]] = {}
    for target, column in TARGET_COLUMNS.items():
        ratings = pd.to_numeric(table[column], errors="coerce").dropna()
        observed[target] = class_counts(label_ratings(ratings))
        print(f"P19 {target} class counts (observed ratings): {observed[target]}", flush=True)
        for modality in ("eeg", "face", "multimodal"):
            x, y = prepare_data(table, target, modality)
            print(f"P19 {target} {modality} class counts (complete features): {class_counts(y)}", flush=True)
            validate_nested_cv(y, f"P19 {target} {modality}")
            prepared[(target, modality)] = (x, y)
    if args.dry_run: return None
    output = output_root / args.run_name
    if output.exists() and not args.append: raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    if args.append and not output.is_dir(): raise FileNotFoundError(f"Cannot append: output directory does not exist: {output}")
    output.mkdir(parents=True, exist_ok=args.append)
    def read_existing(filename: str) -> list[dict]:
        path = output / filename
        return pd.read_csv(path).to_dict("records") if args.append and path.is_file() else []
    all_results = read_existing("fold_results.csv"); all_selected = read_existing("selected_features_by_fold.csv"); all_parameters = read_existing("best_hyperparameters.csv")
    targets = list(TARGET_COLUMNS) if args.target == "all" else [args.target]
    modalities = ("eeg", "face", "multimodal") if args.modality == "all" else (args.modality,)
    for target in targets:
        for modality in modalities:
            x, y = prepared[(target, modality)]
            for model in ("knn", "svm", "gnb"):
                for request in ("all", "5", "10", "20"):
                    print(f"Training: {target} | {modality} | {model} | features={request}", flush=True)
                    results, selected, parameters = evaluate(x, y, target, modality, model, request, args.seed, args.n_jobs)
                    all_results.extend(results); all_selected.extend(selected); all_parameters.extend(parameters)
    results = pd.DataFrame(all_results); selected = pd.DataFrame(all_selected); parameters = pd.DataFrame(all_parameters)
    summary = summarize(results); frequency = feature_frequency(selected); confusions = aggregated_confusions(results)
    results.to_csv(output / "fold_results.csv", index=False); parameters.to_csv(output / "best_hyperparameters.csv", index=False); selected.to_csv(output / "selected_features_by_fold.csv", index=False); frequency.to_csv(output / "feature_selection_frequency.csv", index=False); summary.to_csv(output / "results_summary.csv", index=False); confusions.to_csv(output / "confusion_matrices.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "run_command": " ".join(sys.argv), "participant": PARTICIPANT, "mode": "individual", "targets": list(TARGET_COLUMNS), "modalities": ["eeg", "face", "multimodal"], "models": ["knn", "svm", "gnb"], "feature_counts": ["all", "5", "10", "20"], "random_state": args.seed, "outer_cv": "StratifiedKFold, up to 5 splits", "inner_cv": "StratifiedKFold, up to 3 splits", "scoring": "balanced_accuracy", "label_definition": "LOW: ratings 1-2; NEUTRAL: ratings 3-5; HIGH: ratings 6-7", "source_merged_table": str(dataset), "observed_class_counts": observed, "completed_target_modality_segments": sorted({f"{row['target']}:{row['modality']}" for row in all_results})}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if set(summary.target) == set(TARGET_COLUMNS):
        write_figures(observed, summary, confusions, output); write_markdown(observed, summary, confusions, output)
    return output


if __name__ == "__main__":
    try:
        destination = run(parse_args())
        if destination: print(f"Completed P19 three-class pilot: {destination}")
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
