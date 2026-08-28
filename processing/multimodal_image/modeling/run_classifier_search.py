#!/usr/bin/env python3
"""Focused leakage-safe search of individual no-ICA Image classifiers.

This script reads the existing merged trial tables and baseline summary.  It
does not rerun preprocessing or modify source data.  The participant loop is
intentionally serial; only each inner ``GridSearchCV`` uses CPU workers.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_classification as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_PARTICIPANTS = (
    "P36", "P22", "P06", "P17", "P40", "P19", "P09", "P13", "P15", "P39",
    "P16", "P18", "P35", "P04", "P44",
)
TARGETS = ("valence", "arousal")
MODALITIES = ("eeg", "face", "multimodal")
MODELS = ("gnb", "knn", "svm", "logreg", "extra_trees")
FEATURE_COUNTS = ("3", "5", "8", "10", "15", "20", "30", "all")
DEFAULT_SOURCE_RUN = "{participant_lower}_no_ica"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/image_classification/individual_optimized_top15"
DEFAULT_BASELINE = PROJECT_ROOT / "outputs/image_classification/no_ica/results_summary.csv"
CHECKPOINT_FILE = "checkpoint_state.json"


def parse_csv(value: str) -> list[str]:
    """Parse a non-empty comma-separated command-line option."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("option must contain at least one value")
    return values


def parse_participants(value: str) -> list[str]:
    """Validate an ordered, unique subset of the fixed development cohort."""
    participants = [participant.upper() for participant in parse_csv(value)]
    duplicates = sorted({participant for participant in participants if participants.count(participant) > 1})
    if duplicates:
        raise ValueError(f"--participants contains duplicate participant IDs: {duplicates}")
    outside_cohort = sorted(set(participants).difference(DEVELOPMENT_PARTICIPANTS))
    if outside_cohort:
        raise ValueError(
            "--participants contains IDs outside the fixed development cohort: "
            + ",".join(outside_cohort)
            + ". Allowed participants: "
            + ",".join(DEVELOPMENT_PARTICIPANTS)
        )
    return participants


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the focused-search command line without allowing cohort drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(DEVELOPMENT_PARTICIPANTS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4, help="Inner GridSearchCV workers (default: 4).")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        args.participants = parse_participants(args.participants)
        args.targets = parse_csv(args.targets)
        args.modalities = parse_csv(args.modalities)
    except ValueError as error:
        parser.error(str(error))
    invalid_targets, invalid_modalities = set(args.targets) - set(TARGETS), set(args.modalities) - set(MODALITIES)
    if invalid_targets:
        parser.error(f"Unknown --targets values: {sorted(invalid_targets)}")
    if invalid_modalities:
        parser.error(f"Unknown --modalities values: {sorted(invalid_modalities)}")
    if args.seed != 42:
        parser.error("This fixed experiment requires --seed 42")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer; use an explicit CPU count, not -1")
    return args


def resolved_feature_requests(modality: str) -> list[str]:
    """Return valid non-duplicate selection requests for a modality.

    ``all`` is retained because it means no selector.  Numeric values that
    resolve to the same number of predictors are retained only once.
    """
    available = len(baseline.modality_columns(modality))
    requests: list[str] = []
    seen_counts: set[int] = set()
    for request in FEATURE_COUNTS:
        if request == "all":
            requests.append(request)
            continue
        resolved = min(int(request), available)
        if resolved not in seen_counts:
            requests.append(request)
            seen_counts.add(resolved)
    return requests


def make_pipeline(model: str, feature_count: int | str, seed: int) -> tuple[Pipeline, list[dict[str, list[Any]]]]:
    """Create one leakage-safe pipeline and its model-specific grid."""
    steps: list[tuple[str, Any]] = []
    if model != "extra_trees":
        steps.append(("scale", StandardScaler()))
    if feature_count != "all":
        steps.append(("selector", SelectKBest(score_func=f_classif, k=int(feature_count))))
    if model == "gnb":
        classifier, grid = GaussianNB(), [{"classifier__var_smoothing": [1e-11, 1e-9, 1e-7]}]
    elif model == "knn":
        classifier = KNeighborsClassifier()
        grid = [{"classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21], "classifier__weights": ["uniform", "distance"], "classifier__metric": ["euclidean", "manhattan"]}]
    elif model == "svm":
        classifier = SVC(random_state=seed)
        grid = [
            {"classifier__kernel": ["linear"], "classifier__C": [0.01, 0.1, 1, 10, 100], "classifier__class_weight": [None, "balanced"]},
            {"classifier__kernel": ["rbf"], "classifier__C": [0.01, 0.1, 1, 10, 100], "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1], "classifier__class_weight": [None, "balanced"]},
        ]
    elif model == "logreg":
        classifier = LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, random_state=seed)
        grid = [{"classifier__C": [0.01, 0.1, 1, 10, 100], "classifier__class_weight": [None, "balanced"]}]
    elif model == "extra_trees":
        # GridSearchCV owns parallelism.  A second estimator worker pool would oversubscribe CPUs.
        classifier = ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        grid = [{"classifier__max_depth": [None, 5, 10], "classifier__min_samples_leaf": [1, 2, 4], "classifier__max_features": ["sqrt", "log2", None], "classifier__class_weight": [None, "balanced"]}]
    else:
        raise ValueError(f"Unknown classifier: {model}")
    steps.append(("classifier", classifier))
    return Pipeline(steps), grid


def filter_knn_grid(grid: list[dict[str, list[Any]]], inner_splits: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, list[Any]]]:
    """Remove kNN neighbor candidates invalid for every inner training fold."""
    minimum_train_rows = min(len(train) for train, _ in inner_splits)
    filtered = [value for value in grid[0]["classifier__n_neighbors"] if value <= minimum_train_rows]
    if not filtered:
        raise ValueError(f"No valid kNN n_neighbors values for smallest inner training partition ({minimum_train_rows} rows)")
    return [{**grid[0], "classifier__n_neighbors": filtered}]


def fit_outer_fold(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, *, model: str, feature_count: int | str, inner_splits: list[tuple[np.ndarray, np.ndarray]], seed: int, n_jobs: int) -> tuple[dict[str, Any], float, np.ndarray]:
    """Tune only on an outer-training partition, then predict its held-out fold."""
    pipeline, grid = make_pipeline(model, feature_count, seed)
    if model == "knn":
        grid = filter_knn_grid(grid, inner_splits)
    search = GridSearchCV(pipeline, grid, scoring="balanced_accuracy", cv=inner_splits, n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(x_train, y_train)
    return search.best_params_, float(search.best_score_), search.predict(x_test)


def evaluate_configuration(participant: str, table: pd.DataFrame, target: str, modality: str, model: str, request: str, seed: int, n_jobs: int) -> dict[str, Any]:
    """Evaluate one configuration with nested individual stratified CV."""
    x, y, _ = baseline.prepare_modality_data(table, participant, target, modality)
    folds = baseline.valid_stratified_splits(y, 5, f"{participant} {target} {modality}")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    feature_count: int | str = "all" if request == "all" else min(int(request), x.shape[1])
    balanced_scores: list[float] = []
    accuracy_scores: list[float] = []
    selected_params: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer.split(x, y), start=1):
        y_train, y_test = y.iloc[train].reset_index(drop=True), y.iloc[test].reset_index(drop=True)
        inner_fold_count = baseline.valid_stratified_splits(y_train, min(3, folds), f"{participant} outer fold {fold} inner CV")
        inner = list(StratifiedKFold(n_splits=inner_fold_count, shuffle=True, random_state=seed + fold).split(x.iloc[train], y_train))
        params, inner_score, predicted = fit_outer_fold(x.iloc[train], y_train, x.iloc[test], model=model, feature_count=feature_count, inner_splits=inner, seed=seed, n_jobs=n_jobs)
        balanced_scores.append(float(balanced_accuracy_score(y_test, predicted)))
        accuracy_scores.append(float(accuracy_score(y_test, predicted)))
        selected_params.append({"fold": fold, "inner_best_balanced_accuracy": inner_score, "parameters": params})
    return {
        "participant": participant, "target": target, "modality": modality, "classifier": model,
        "feature_count": feature_count, "feature_count_request": request, "outer_folds": folds,
        "mean_balanced_accuracy": float(np.mean(balanced_scores)), "std_balanced_accuracy": float(np.std(balanced_scores, ddof=0)),
        "mean_accuracy": float(np.mean(accuracy_scores)), "std_accuracy": float(np.std(accuracy_scores, ddof=0)),
        "best_hyperparameters": json.dumps(selected_params, sort_keys=True), "seed": seed,
        "gridsearch_n_jobs": n_jobs, "estimator_n_jobs": 1 if model == "extra_trees" else None,
    }


def load_baseline_best(path: Path, participants: list[str], targets: list[str]) -> pd.DataFrame:
    """Select the existing no-ICA best row for each fixed participant-target pair."""
    if not path.is_file():
        raise FileNotFoundError(f"Baseline summary missing: {path}")
    frame = pd.read_csv(path)
    required = {"mode", "participant", "target", "modality", "classifier", "feature_count_resolved", "balanced_accuracy_mean", "accuracy_mean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Baseline summary missing columns: {sorted(missing)}")
    frame["participant"] = frame["participant"].astype(str).str.upper()
    subset = frame[(frame["mode"] == "individual") & frame.participant.isin(participants) & frame.target.isin(targets)].copy()
    subset = subset.sort_values(["participant", "target", "balanced_accuracy_mean", "accuracy_mean", "classifier", "modality"], ascending=[True, True, False, False, True, True])
    best = subset.drop_duplicates(["participant", "target"], keep="first")
    expected = {(participant, target) for participant in participants for target in targets}
    observed = set(zip(best.participant, best.target))
    if expected != observed:
        raise ValueError(f"Baseline summary lacks participant-target pairs: {sorted(expected - observed)}")
    return best.rename(columns={"balanced_accuracy_mean": "previous_balanced_accuracy", "accuracy_mean": "previous_accuracy", "classifier": "previous_classifier", "modality": "previous_modality", "feature_count_resolved": "previous_feature_count"})[["participant", "target", "previous_balanced_accuracy", "previous_accuracy", "previous_classifier", "previous_modality", "previous_feature_count"]]


def choose_optimized_best(results: pd.DataFrame) -> pd.DataFrame:
    """Choose each participant-target winner by outer-fold balanced accuracy only."""
    ranked = results.sort_values(["participant", "target", "mean_balanced_accuracy", "mean_accuracy", "classifier", "modality", "feature_count_request"], ascending=[True, True, False, False, True, True, True])
    return ranked.drop_duplicates(["participant", "target"], keep="first").rename(columns={"mean_balanced_accuracy": "balanced_accuracy", "mean_accuracy": "accuracy"})


def baseline_comparison(previous: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    """Join existing baseline winners with new winners and calculate deltas."""
    columns = ["participant", "target", "modality", "classifier", "feature_count", "balanced_accuracy", "accuracy", "best_hyperparameters"]
    comparison = previous.merge(optimized[columns], on=["participant", "target"], validate="one_to_one")
    comparison = comparison.rename(columns={"modality": "optimized_modality", "classifier": "optimized_classifier", "feature_count": "optimized_feature_count", "best_hyperparameters": "optimized_hyperparameters", "balanced_accuracy": "optimized_balanced_accuracy", "accuracy": "optimized_accuracy"})
    comparison["BA_delta"] = comparison.optimized_balanced_accuracy - comparison.previous_balanced_accuracy
    comparison["accuracy_delta"] = comparison.optimized_accuracy - comparison.previous_accuracy
    return comparison


def aggregate_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Create compact overall and winner-stratified aggregate rows."""
    delta = comparison.BA_delta
    rows = [{"summary_type": "overall", "group": "all", "mean_previous_best_BA": comparison.previous_balanced_accuracy.mean(), "mean_optimized_best_BA": comparison.optimized_balanced_accuracy.mean(), "mean_BA_delta": delta.mean(), "median_BA_delta": delta.median(), "improved": int((delta > 0).sum()), "tied": int((delta == 0).sum()), "worsened": int((delta < 0).sum()), "count": len(comparison)}]
    for column, label in (("target", "target"), ("optimized_classifier", "winning_classifier"), ("optimized_modality", "winning_modality"), ("optimized_feature_count", "winning_feature_count")):
        for value, group in comparison.groupby(column, dropna=False):
            group_delta = group.BA_delta
            rows.append({"summary_type": label, "group": value, "mean_previous_best_BA": group.previous_balanced_accuracy.mean(), "mean_optimized_best_BA": group.optimized_balanced_accuracy.mean(), "mean_BA_delta": group_delta.mean(), "median_BA_delta": group_delta.median(), "improved": int((group_delta > 0).sum()), "tied": int((group_delta == 0).sum()), "worsened": int((group_delta < 0).sum()), "count": len(group)})
    return pd.DataFrame(rows)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace one small CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    """Atomically replace one JSON manifest or checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def configuration_key(participant: str, target: str, modality: str, classifier: str, request: str) -> str:
    """Return a stable checkpoint identity for a fully evaluated configuration."""
    return "|".join((participant, target, modality, classifier, request))


def settings(args: argparse.Namespace) -> dict[str, Any]:
    """Return resume-sensitive scientific and execution settings."""
    return {"participants": args.participants, "targets": args.targets, "modalities": args.modalities, "models": list(MODELS), "feature_counts": list(FEATURE_COUNTS), "seed": args.seed, "source_run": DEFAULT_SOURCE_RUN, "label_definition": "LOW: rating < 4; HIGH: rating >= 4", "outer_cv": "StratifiedKFold up to 5 folds, shuffled, random_state=42", "inner_cv": "StratifiedKFold up to 3 folds, shuffled, random_state=42+outer_fold", "selection_metric": "balanced_accuracy", "outer_experiment_loop": "serial", "gridsearch_n_jobs": args.n_jobs, "estimator_nested_parallelism": "disabled (ExtraTrees n_jobs=1)"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or safely resume the fixed individual classifier search."""
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    baseline_path = args.baseline_csv.resolve() if args.baseline_csv.is_absolute() else (PROJECT_ROOT / args.baseline_csv).resolve()
    derived_root = args.derived_root.resolve() if args.derived_root.is_absolute() else (PROJECT_ROOT / args.derived_root).resolve()
    run_settings = settings(args)
    manifest_path, checkpoint_path, results_path = output / "run_manifest.json", output / CHECKPOINT_FILE, output / "results_summary.csv"
    output_exists = output.exists()
    if output_exists and not args.resume:
        raise FileExistsError(f"Output directory already exists; use --resume only for the same interrupted search: {output}")
    completed: set[str] = set()
    if args.resume and output_exists:
        if not manifest_path.is_file() or not checkpoint_path.is_file() or not results_path.is_file():
            raise ValueError("Cannot resume: run manifest, checkpoint, or results_summary.csv is missing")
        if json.loads(manifest_path.read_text(encoding="utf-8")).get("settings") != run_settings:
            raise ValueError("Refusing resume: requested settings differ from the saved run")
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("settings") != run_settings:
            raise ValueError("Refusing resume: checkpoint settings differ from the saved run")
        completed = set(state.get("completed_configurations", []))
        results = pd.read_csv(results_path)
    else:
        output.mkdir(parents=True)
        results = pd.DataFrame()
    source_tables: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, str] = {}
    for participant in args.participants:
        table, path = baseline.load_participant_table(derived_root, participant, DEFAULT_SOURCE_RUN)
        source_tables[participant], source_paths[participant] = table, str(path)
    previous = load_baseline_best(baseline_path, args.participants, args.targets)
    if not (args.resume and output_exists):
        atomic_write_json({"created_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv), "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "baseline_csv": str(baseline_path), "source_merged_tables": source_paths, "settings": run_settings}, manifest_path)
    planned = [(participant, target, modality, model, request) for participant in args.participants for target in args.targets for modality in args.modalities for model in MODELS for request in resolved_feature_requests(modality)]
    print(f"CPU parallelism: GridSearchCV n_jobs={args.n_jobs}")
    print("Outer experiment loop: serial")
    print("Estimator nested parallelism: disabled")
    if args.progress and args.resume:
        print(f"Resuming: {len(completed)}/{len(planned)} configurations already completed", flush=True)
    for number, (participant, target, modality, model, request) in enumerate(planned, start=1):
        key = configuration_key(participant, target, modality, model, request)
        if key in completed:
            continue
        if args.progress:
            count = "all" if request == "all" else min(int(request), len(baseline.modality_columns(modality)))
            print(f"[{number}/{len(planned)}] {participant} | {target} | {modality} | {model} | k={count}", flush=True)
        row = evaluate_configuration(participant, source_tables[participant], target, modality, model, request, args.seed, args.n_jobs)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        completed.add(key)
        atomic_write_csv(results, results_path)
        atomic_write_json({"settings": run_settings, "completed_configurations": sorted(completed)}, checkpoint_path)
    optimized = choose_optimized_best(results)
    comparison = baseline_comparison(previous, optimized)
    best_columns = ["participant", "target", "modality", "classifier", "feature_count", "balanced_accuracy", "accuracy", "best_hyperparameters"]
    atomic_write_csv(optimized[best_columns], output / "best_per_participant_target.csv")
    atomic_write_csv(comparison, output / "baseline_vs_optimized.csv")
    atomic_write_csv(aggregate_summary(comparison), output / "aggregate_summary.csv")
    atomic_write_csv(optimized[["participant", "target", "classifier", "modality", "feature_count", "best_hyperparameters"]], output / "best_hyperparameters.csv")
    return {"output": output, "configurations": len(planned), "completed": len(completed)}


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
