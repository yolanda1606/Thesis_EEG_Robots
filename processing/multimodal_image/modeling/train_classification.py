#!/usr/bin/env python3
"""Leakage-safe experimental Image Experiment binary classification.

Reads participant Image merged tables and never changes them. By default it
uses the official ``p##_final`` runs. Results are written only below
``outputs/image_classification/<run-name>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNELS = ("C3", "C4", "Cz", "Fz", "Oz", "PO7", "PO8", "Pz")
EEG_FAMILIES = ("eeg_sd", "eeg_se", "eeg_hm", "eeg_hc", "eeg_mf_hz", "eeg_bp_delta", "eeg_se_delta", "eeg_bp_theta", "eeg_se_theta", "eeg_bp_alpha", "eeg_se_alpha", "eeg_bp_beta", "eeg_se_beta", "eeg_bp_gamma", "eeg_se_gamma")
EEG_COLUMNS = [f"{family}__{channel}" for family in EEG_FAMILIES for channel in EEG_CHANNELS]
FACE_COLUMNS = [f"video_{name}_norm_{stat}" for name in ("irisdo", "eso", "enso", "mnso", "mwo") for stat in ("mean", "std")]
TARGET_COLUMNS = {"valence": "valence_rating", "arousal": "arousal_rating"}
QC_GROUPS = ("Very clean", "Clean with minor EEG loss", "Review due to elevated EEG rejection", "Review due to missing ratings", "Special/incomplete case")
DEFAULT_SOURCE_RUN = "{participant_lower}_final"


def parse_csv_option(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("individual", "general"), required=True)
    parser.add_argument("--target", choices=("valence", "arousal", "all"), required=True)
    parser.add_argument("--modality", choices=("eeg", "face", "multimodal", "all"), required=True)
    parser.add_argument("--models", default="knn,svm,gnb")
    parser.add_argument("--feature-counts", default="all,5,10,20")
    parser.add_argument("--participants", help="Comma-separated participant IDs; overrides --qc-group.")
    parser.add_argument("--qc-group", help="Comma-separated groups from modeling_readiness.csv.")
    parser.add_argument("--readiness-csv", type=Path, default=Path("docs/meeting_14_08/modeling_readiness.csv"))
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument(
        "--source-run",
        default=DEFAULT_SOURCE_RUN,
        help=(
            "Run-name template beneath each participant's Image_Experiment/runs directory. "
            "Available fields: {participant} and {participant_lower}. "
            "Default: {participant_lower}_final."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/image_classification"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true", help="Append a non-overlapping configuration segment to an interrupted run.")
    parser.add_argument("--resume", action="store_true", help="Resume a matching interrupted run from its completed configuration checkpoints.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Inner GridSearchCV workers; -1 uses all logical CPUs.")
    parser.add_argument("--progress", action="store_true", help="Show concise experiment progress.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress diagnostics (implies --progress).")
    args = parser.parse_args(argv)
    args.models = parse_csv_option(args.models)
    args.feature_counts = parse_csv_option(args.feature_counts)
    invalid_models = set(args.models).difference({"knn", "svm", "gnb"})
    invalid_counts = set(args.feature_counts).difference({"all", "5", "10", "20"})
    if not args.models or invalid_models: parser.error(f"Invalid --models: {sorted(invalid_models)}")
    if not args.feature_counts or invalid_counts: parser.error(f"Invalid --feature-counts: {sorted(invalid_counts)}")
    if args.n_jobs == 0 or args.n_jobs < -1: parser.error("--n-jobs must be -1 or a non-zero integer")
    if args.append and args.resume: parser.error("--append and --resume cannot be used together")
    if "/" in args.run_name or "\\" in args.run_name or args.run_name in {"", ".", ".."}: parser.error("--run-name must be a simple directory name")
    try:
        example_source_run = args.source_run.format(participant="P01", participant_lower="p01")
    except (KeyError, ValueError) as error:
        parser.error(f"Invalid --source-run template: {error}")
    if "/" in example_source_run or "\\" in example_source_run or example_source_run in {"", ".", ".."}:
        parser.error("--source-run must resolve to a simple directory name")
    return args


class ProgressReporter:
    """Print flushed, opt-in terminal progress without affecting experiment results."""

    def __init__(self, progress: bool, verbose: bool) -> None:
        self.progress_enabled = progress or verbose
        self.verbose_enabled = verbose

    def progress(self, message: str) -> None:
        if self.progress_enabled:
            print(message, flush=True)

    def verbose(self, message: str) -> None:
        if self.verbose_enabled:
            print(message, flush=True)

    @staticmethod
    def candidate_count(grid: list[dict[str, list[Any]]]) -> int:
        return sum(int(np.prod([len(values) for values in candidate.values()])) for candidate in grid)


def label_ratings(ratings: pd.Series) -> pd.Series:
    """Convert observed ratings to LOW=0 (<4) and HIGH=1 (>=4)."""
    return (pd.to_numeric(ratings, errors="raise") >= 4.0).astype(int)


def modality_columns(modality: str) -> list[str]:
    if modality == "eeg": return EEG_COLUMNS.copy()
    if modality == "face": return FACE_COLUMNS.copy()
    if modality == "multimodal": return EEG_COLUMNS + FACE_COLUMNS
    raise ValueError(f"Unknown modality: {modality}")


def selected_dimensions(value: str, all_values: Iterable[str]) -> list[str]:
    return list(all_values) if value == "all" else [value]


def resolve_participants(readiness_path: Path, explicit: str | None, qc_groups: str | None) -> tuple[pd.DataFrame, list[str], str]:
    readiness = pd.read_csv(readiness_path)
    required = {"participant", "qc_group"}
    if missing := required.difference(readiness.columns): raise ValueError(f"Readiness CSV missing columns: {sorted(missing)}")
    readiness["participant"] = readiness.participant.astype(str).str.upper()
    if explicit:
        requested = [p.upper() for p in parse_csv_option(explicit)]
        unknown = sorted(set(requested).difference(readiness.participant))
        if unknown: raise ValueError(f"Participants absent from readiness CSV: {unknown}")
        return readiness.set_index("participant").loc[requested].reset_index(), requested, "explicit participants"
    groups = parse_csv_option(qc_groups) if qc_groups else list(QC_GROUPS)
    unknown_groups = sorted(set(groups).difference(QC_GROUPS))
    if unknown_groups: raise ValueError(f"Unknown QC groups: {unknown_groups}")
    resolved = readiness[readiness.qc_group.isin(groups)].copy().sort_values("participant")
    if resolved.empty: raise ValueError("No participants resolved from --qc-group")
    return resolved, resolved.participant.tolist(), "QC group"


def source_run_name(participant: str, source_run: str = DEFAULT_SOURCE_RUN) -> str:
    """Resolve one validated participant-specific source Image run name."""
    try:
        run_name = source_run.format(participant=participant.upper(), participant_lower=participant.lower())
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid source-run template {source_run!r}: {error}") from error
    if "/" in run_name or "\\" in run_name or run_name in {"", ".", ".."}:
        raise ValueError(f"Source run must be a simple directory name, got {run_name!r}")
    return run_name


def final_run_path(derived_root: Path, participant: str, source_run: str = DEFAULT_SOURCE_RUN) -> Path:
    """Return the selected participant Image run path (defaulting to final)."""
    return derived_root / participant / "Image_Experiment" / "runs" / source_run_name(participant, source_run)


def load_participant_table(derived_root: Path, participant: str, source_run: str = DEFAULT_SOURCE_RUN) -> tuple[pd.DataFrame, Path]:
    run = final_run_path(derived_root, participant, source_run)
    manifest_path = run / "manifest" / "run_manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError(f"Image source-run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("resolved_configuration", manifest.get("configuration", {}))
    recorded = str(config.get("participant", "")).upper()
    if recorded != participant.upper(): raise ValueError(f"Participant mismatch: requested {participant}, manifest records {recorded or 'none'}")
    dataset = run / "merged" / f"{participant.lower()}_image_trial_dataset.csv"
    if not dataset.is_file(): raise FileNotFoundError(f"Merged source-run trial dataset missing: {dataset}")
    table = pd.read_csv(dataset)
    return table, dataset.resolve()


def prepare_modality_data(table: pd.DataFrame, participant: str, target: str, modality: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    columns, target_column = modality_columns(modality), TARGET_COLUMNS[target]
    required = set(columns + [target_column])
    if missing := sorted(required.difference(table.columns)): raise ValueError(f"{participant} missing required {modality} columns: {missing}")
    numeric = table[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.notna().all(axis=1) & pd.to_numeric(table[target_column], errors="coerce").notna()
    features = numeric.loc[usable].reset_index(drop=True)
    if not np.isfinite(features.to_numpy()).all(): raise ValueError(f"{participant} {modality} predictors contain infinity")
    labels = label_ratings(table.loc[usable, target_column]).reset_index(drop=True)
    return features, labels, columns


def resolved_feature_count(request: str, available: int) -> int | str:
    return "all" if request == "all" else min(int(request), available)


def make_pipeline(model: str, feature_count: int | str, seed: int) -> tuple[Pipeline, list[dict[str, list[Any]]]]:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if feature_count != "all": steps.append(("selector", SelectKBest(score_func=f_classif, k=int(feature_count))))
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
    else: raise ValueError(f"Unknown model: {model}")
    return Pipeline(steps), grid


def metric_row(y_true: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"balanced_accuracy": balanced_accuracy_score(y_true, predicted), "accuracy": accuracy_score(y_true, predicted), "precision": precision_score(y_true, predicted, pos_label=1, zero_division=0), "recall": recall_score(y_true, predicted, pos_label=1, zero_division=0), "f1": f1_score(y_true, predicted, pos_label=1, zero_division=0), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def selected_feature_rows(fitted: Pipeline, columns: list[str], common: dict[str, Any]) -> list[dict[str, Any]]:
    if "selector" not in fitted.named_steps:
        return [{**common, "feature": name, "selected": True, "feature_score": np.nan} for name in columns]
    selector: SelectKBest = fitted.named_steps["selector"]
    selected = selector.get_support()
    return [{**common, "feature": name, "selected": bool(keep), "feature_score": float(score) if np.isfinite(score) else np.nan} for name, keep, score in zip(columns, selected, selector.scores_)]


def valid_stratified_splits(labels: pd.Series, requested: int, context: str) -> int:
    smallest = int(labels.value_counts().min())
    folds = min(requested, smallest)
    if folds < 2: raise ValueError(f"{context}: insufficient minority-class rows ({smallest}) for stratified CV")
    return folds


def fit_outer_fold(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, *, model: str, feature_count: int | str, inner_splits: list[tuple[np.ndarray, np.ndarray]], seed: int, n_jobs: int = 1) -> tuple[Pipeline, dict[str, Any], float, np.ndarray]:
    pipeline, grid = make_pipeline(model, feature_count, seed)
    # kNN candidates larger than an inner training partition are removed rather than failing a valid experiment.
    min_inner_train = min(len(train) for train, _ in inner_splits)
    if model == "knn": grid = [{**grid[0], "classifier__n_neighbors": [k for k in grid[0]["classifier__n_neighbors"] if k <= min_inner_train]}]
    search = GridSearchCV(pipeline, grid, scoring="balanced_accuracy", cv=inner_splits, n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_), search.predict(x_test)


def evaluate_individual(participant: str, x: pd.DataFrame, y: pd.Series, target: str, modality: str, model: str, request: str, seed: int, reporter: ProgressReporter | None = None, n_jobs: int = 1) -> tuple[list[dict], list[dict], list[dict]]:
    reporter = reporter or ProgressReporter(False, False)
    folds = valid_stratified_splits(y, 5, f"{participant} {target} {modality}")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    feature_count = resolved_feature_count(request, x.shape[1]); results=[]; selected=[]; params=[]
    for fold, (train, test) in enumerate(outer.split(x, y), 1):
        context = f"participant={participant} | target={target} | modality={modality} | model={model} | features={request} | outer fold {fold}/{folds}"
        started = time.perf_counter()
        try:
            reporter.progress(f"  outer fold {fold}/{folds} ...")
            y_train, y_test = y.iloc[train].reset_index(drop=True), y.iloc[test].reset_index(drop=True)
            inner_folds = valid_stratified_splits(y_train, min(3, folds), f"{participant} fold {fold} inner CV")
            inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold).split(x.iloc[train], y_train))
            _, grid = make_pipeline(model, feature_count, seed)
            reporter.verbose(f"    train rows: {len(train)}; test rows: {len(test)}; predictors: {x.shape[1]}")
            reporter.verbose(f"    tuning {model} ({ProgressReporter.candidate_count(grid)} candidates)...")
            fitted, best, inner_score, predicted = fit_outer_fold(x.iloc[train], y_train, x.iloc[test], model=model, feature_count=feature_count, inner_splits=inner, seed=seed, n_jobs=n_jobs)
            metrics = metric_row(y_test, predicted)
            reporter.verbose("    inner tuning complete")
            reporter.verbose(f"    best params: {json.dumps(best, sort_keys=True)}")
            reporter.verbose(f"    balanced accuracy: {metrics['balanced_accuracy']:.3f}; elapsed: {time.perf_counter() - started:.1f} s")
            common = {"mode":"individual", "participant":participant, "held_out_participant":participant, "target":target, "modality":modality, "classifier":model, "feature_count_request":request, "feature_count_resolved":feature_count, "fold":fold}
            results.append({**common, "training_participants":participant, "sample_size":len(y), "train_size":len(train), "test_size":len(test), "low_count":int((y==0).sum()), "high_count":int((y==1).sum()), **metrics})
            selected.extend(selected_feature_rows(fitted, list(x.columns), common))
            params.append({**common, "best_parameters":json.dumps(best, sort_keys=True), "inner_best_balanced_accuracy":inner_score})
        except Exception:
            print(f"ERROR during {context}", file=sys.stderr, flush=True)
            raise
    return results, selected, params


def general_outer_splits(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))


def grouped_inner_splits(x: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = groups.nunique()
    if n_groups < 2: raise ValueError("General-mode inner CV needs at least two training participants")
    splitter = StratifiedGroupKFold(n_splits=min(5, n_groups), shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y, groups))
    for train, test in splits:
        if set(groups.iloc[train]).intersection(groups.iloc[test]): raise RuntimeError("Grouped inner CV mixed participant groups")
    return splits


def evaluate_general(data: pd.DataFrame, target: str, modality: str, model: str, request: str, seed: int, reporter: ProgressReporter | None = None, n_jobs: int = 1) -> tuple[list[dict], list[dict], list[dict]]:
    reporter = reporter or ProgressReporter(False, False)
    columns = modality_columns(modality); feature_count=resolved_feature_count(request, len(columns)); results=[]; selected=[]; params=[]
    outer_splits = general_outer_splits(data.participant)
    for fold, (train, test) in enumerate(outer_splits, 1):
        train_data, test_data = data.iloc[train], data.iloc[test]
        held = str(test_data.participant.iloc[0]); common={"mode":"general", "participant":"COHORT", "held_out_participant":held, "target":target, "modality":modality, "classifier":model, "feature_count_request":request, "feature_count_resolved":feature_count, "fold":fold}
        context = f"target={target} | modality={modality} | model={model} | features={request} | outer fold {fold}/{len(outer_splits)} | held out={held}"
        started = time.perf_counter()
        try:
            reporter.progress(f"  outer fold {fold}/{len(outer_splits)} | held out: {held} ...")
            x_train, y_train = train_data[columns], train_data.label
            x_test, y_test = test_data[columns], test_data.label
            inner = grouped_inner_splits(x_train, y_train, train_data.participant.reset_index(drop=True), seed + fold)
            _, grid = make_pipeline(model, feature_count, seed)
            reporter.verbose(f"    train rows: {len(train)}; test rows: {len(test)}; training participants: {', '.join(sorted(train_data.participant.unique()))}")
            reporter.verbose(f"    tuning {model} ({ProgressReporter.candidate_count(grid)} candidates)...")
            fitted, best, inner_score, predicted = fit_outer_fold(x_train, y_train, x_test, model=model, feature_count=feature_count, inner_splits=inner, seed=seed, n_jobs=n_jobs)
            metrics = metric_row(y_test, predicted)
            reporter.verbose("    inner tuning complete")
            reporter.verbose(f"    best params: {json.dumps(best, sort_keys=True)}")
            reporter.verbose(f"    balanced accuracy: {metrics['balanced_accuracy']:.3f}; elapsed: {time.perf_counter() - started:.1f} s")
            results.append({**common, "training_participants":";".join(sorted(train_data.participant.unique())), "sample_size":len(data), "train_size":len(train), "test_size":len(test), "low_count":int((data.label==0).sum()), "high_count":int((data.label==1).sum()), **metrics})
            selected.extend(selected_feature_rows(fitted, columns, common)); params.append({**common, "best_parameters":json.dumps(best, sort_keys=True), "inner_best_balanced_accuracy":inner_score})
        except Exception:
            print(f"ERROR during {context}", file=sys.stderr, flush=True)
            raise
    return results, selected, params


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys=["mode","participant","target","modality","classifier","feature_count_request","feature_count_resolved"]
    stats=results.groupby(keys, dropna=False).agg(folds=("fold","size"), balanced_accuracy_mean=("balanced_accuracy","mean"), balanced_accuracy_sd=("balanced_accuracy","std"), balanced_accuracy_median=("balanced_accuracy","median"), balanced_accuracy_min=("balanced_accuracy","min"), balanced_accuracy_max=("balanced_accuracy","max"), accuracy_mean=("accuracy","mean"), precision_mean=("precision","mean"), recall_mean=("recall","mean"), f1_mean=("f1","mean")).reset_index()
    return stats


def feature_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    keys=["mode","target","modality","classifier","feature_count_request","feature_count_resolved","feature"]
    return selected.groupby(keys, dropna=False).agg(selected_folds=("selected","sum"), outer_folds=("fold","nunique"), selection_frequency=("selected","mean"), mean_feature_score=("feature_score","mean")).reset_index().sort_values(["mode","target","modality","classifier","feature_count_request","selection_frequency"], ascending=[True,True,True,True,True,False])


def write_plots(summary: pd.DataFrame, frequency: pd.DataFrame, out: Path) -> None:
    if summary.empty: return
    fig, ax=plt.subplots(figsize=(10,5)); labels=[f"{r.modality}/{r.classifier}/{r.feature_count_request}" for r in summary.itertuples()]; ax.bar(range(len(summary)), summary.balanced_accuracy_mean); ax.axhline(.5, color="black", ls="--"); ax.set(xticks=range(len(summary)), xticklabels=labels, ylabel="Balanced accuracy", title="Balanced accuracy summary"); ax.tick_params(axis="x", rotation=65); fig.tight_layout(); fig.savefig(out/"balanced_accuracy_summary.png",dpi=160); plt.close(fig)
    fig, ax=plt.subplots(figsize=(8,5)); grouped=summary.groupby(["modality","classifier"]).balanced_accuracy_mean.mean().unstack(fill_value=np.nan); grouped.plot(kind="bar",ax=ax); ax.axhline(.5,color="black",ls="--"); ax.set(ylabel="Mean balanced accuracy",title="Modality/model comparison"); fig.tight_layout(); fig.savefig(out/"modality_model_comparison.png",dpi=160); plt.close(fig)
    top=frequency.sort_values("selection_frequency",ascending=False).head(15); fig, ax=plt.subplots(figsize=(10,5)); ax.barh(top.feature,top.selection_frequency); ax.set(xlabel="Outer-fold selection frequency",title="Most consistently selected features"); fig.tight_layout(); fig.savefig(out/"top_selected_features.png",dpi=160); plt.close(fig)


def git_commit() -> str | None:
    try: return subprocess.check_output(["git","rev-parse","HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None


CHECKPOINT_STATE_FILE = "checkpoint_state.json"
OUTPUT_FILES = {"results": "fold_results.csv", "selected": "selected_features_by_fold.csv", "parameters": "best_hyperparameters.csv", "frequency": "feature_selection_frequency.csv", "summary": "results_summary.csv", "confusion": "confusion_matrices.csv"}


def atomic_write_text(path: Path, content: str) -> None:
    """Durably replace one artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try: os.unlink(temporary_name)
        except FileNotFoundError: pass
        raise


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def configuration_key(mode: str, participant: str | None, target: str, modality: str, model: str, request: str) -> str:
    identity = {"mode": mode, "target": target, "modality": modality, "model": model, "feature_count_request": request}
    if mode == "individual": identity["participant"] = participant
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def scientific_settings(args: argparse.Namespace, participants: list[str], targets: list[str], modalities: list[str], source: str) -> dict[str, Any]:
    return {"mode": args.mode, "participants": participants, "cohort_source": source, "source_run": args.source_run, "target": targets, "modalities": modalities, "models": args.models, "feature_counts": args.feature_counts, "seed": args.seed, "cv_strategy": "StratifiedKFold nested CV" if args.mode == "individual" else "LeaveOneGroupOut outer CV; StratifiedGroupKFold inner CV", "hyperparameter_grids": {"knn": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}, "svm": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "rbf_gamma": ["scale", 0.01, 0.1]}, "gnb": {"var_smoothing": [1e-11, 1e-9, 1e-7]}}, "label_definition": "LOW: rating < 4; HIGH: rating >= 4"}


def checkpoint_outputs(out: Path, results: pd.DataFrame, selected: pd.DataFrame, parameters: pd.DataFrame, completed_keys: set[str], settings: dict[str, Any]) -> None:
    """Refresh all CSV artifacts and write the completion marker last."""
    frequency = feature_frequency(selected)
    summary = summarize(results)
    confusion = results[["mode", "participant", "held_out_participant", "target", "modality", "classifier", "feature_count_request", "fold", "tn", "fp", "fn", "tp"]]
    for frame, name in ((results, "results"), (selected, "selected"), (parameters, "parameters"), (frequency, "frequency"), (summary, "summary"), (confusion, "confusion")):
        atomic_write_csv(frame, out / OUTPUT_FILES[name])
    atomic_write_text(out / CHECKPOINT_STATE_FILE, json.dumps({"scientific_settings": settings, "completed_configurations": sorted(completed_keys)}, indent=2) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    reporter = ProgressReporter(args.progress, args.verbose)
    readiness_path=(PROJECT_ROOT/args.readiness_csv).resolve() if not args.readiness_csv.is_absolute() else args.readiness_csv
    derived_root=(PROJECT_ROOT/args.derived_root).resolve() if not args.derived_root.is_absolute() else args.derived_root
    output_root=(PROJECT_ROOT/args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    resolved, participants, source = resolve_participants(readiness_path, args.participants, args.qc_group)
    source_tables={}; tables={}
    for participant in participants: tables[participant], source_tables[participant]=load_participant_table(derived_root,participant,args.source_run)
    targets=selected_dimensions(args.target,TARGET_COLUMNS); modalities=selected_dimensions(args.modality,("eeg","face","multimodal"))
    prepared=[]
    for participant, table in tables.items():
        for target in targets:
            for modality in modalities:
                x,y,columns=prepare_modality_data(table,participant,target,modality)
                prepared.append({"participant":participant,"target":target,"modality":modality,"rows":len(y),"low":int((y==0).sum()),"high":int((y==1).sum()),"predictors":";".join(columns)})
    reporter.progress("Resolved participants ("+source+"): "+", ".join(participants))
    for row in prepared: reporter.verbose(f"{row['participant']} {row['target']} {row['modality']}: usable={row['rows']}, low={row['low']}, high={row['high']}, predictors={row['predictors']}")
    if args.dry_run: return {"dry_run":True,"participants":participants,"prepared":prepared}
    out=output_root/args.run_name
    if out.exists() and not args.append and not args.resume: raise FileExistsError(f"Refusing to overwrite existing run directory: {out}")
    if args.append and not out.is_dir(): raise FileNotFoundError(f"Cannot append: run directory does not exist: {out}")
    if args.resume and not out.is_dir(): raise FileNotFoundError(f"Cannot resume: run directory does not exist: {out}")
    out.mkdir(parents=True, exist_ok=args.append or args.resume)
    reporter.progress(f"Output: {out}")
    settings=scientific_settings(args,participants,targets,modalities,source)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"git_commit":git_commit(),"python":sys.executable,"python_version":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"n_jobs":args.n_jobs,"resolved_participants":participants,"cohort_source":source,"source_merged_tables":{p:str(v) for p,v in source_tables.items()},**settings,"scientific_settings":settings}
    if not args.append:
        if args.resume:
            existing=json.loads((out/"run_manifest.json").read_text(encoding="utf-8")) if (out/"run_manifest.json").is_file() else None
            state=json.loads((out/CHECKPOINT_STATE_FILE).read_text(encoding="utf-8")) if (out/CHECKPOINT_STATE_FILE).is_file() else None
            if not existing or not state or existing.get("scientific_settings") != settings or state.get("scientific_settings") != settings: raise ValueError("Refusing resume: saved manifest/checkpoint scientific settings do not exactly match the requested run")
            completed_keys=set(state.get("completed_configurations",[]))
            def load(name: str) -> pd.DataFrame:
                frame = pd.read_csv(out/OUTPUT_FILES[name]) if (out/OUTPUT_FILES[name]).is_file() else pd.DataFrame()
                if frame.empty: return frame
                keys = frame.apply(lambda row: configuration_key(str(row["mode"]), str(row["participant"]) if str(row["mode"]) == "individual" else None, str(row["target"]), str(row["modality"]), str(row["classifier"]), str(row["feature_count_request"])), axis=1)
                return frame.loc[keys.isin(completed_keys)].copy()
            results,selected,parameters=load("results"),load("selected"),load("parameters")
        else:
            atomic_write_text(out/"run_manifest.json",json.dumps(manifest,indent=2)+"\n"); atomic_write_csv(resolved,out/"resolved_participants.csv")
            completed_keys=set(); results=pd.DataFrame(); selected=pd.DataFrame(); parameters=pd.DataFrame()
    else:
        def load_append(name: str) -> pd.DataFrame: return pd.read_csv(out/OUTPUT_FILES[name]) if (out/OUTPUT_FILES[name]).is_file() else pd.DataFrame()
        results,selected,parameters=load_append("results"),load_append("selected"),load_append("parameters")
        completed_keys=set() if results.empty else set(results.apply(lambda row: configuration_key(str(row["mode"]), str(row["participant"]) if str(row["mode"]) == "individual" else None, str(row["target"]), str(row["modality"]), str(row["classifier"]), str(row["feature_count_request"])), axis=1))
    configurations = len(targets) * len(modalities) * len(args.models) * len(args.feature_counts) * (len(participants) if args.mode == "individual" else 1)
    completed = 0
    if args.resume:
        reporter.progress(f"Resuming run: {args.run_name}")
        reporter.progress(f"Completed configurations found: {len(completed_keys)}")
        reporter.progress(f"Remaining configurations: {configurations-len(completed_keys)}")
    for target in targets:
        for modality in modalities:
            if args.mode=="general":
                chunks=[]
                for participant, table in tables.items():
                    x,y,columns=prepare_modality_data(table,participant,target,modality); chunks.append(x.assign(participant=participant,label=y.to_numpy()))
                cohort=pd.concat(chunks,ignore_index=True)
            for model in args.models:
                for request in args.feature_counts:
                    if args.mode=="individual":
                        for participant_index, (participant, table) in enumerate(tables.items(), 1):
                            completed += 1; key=configuration_key(args.mode,participant,target,modality,model,request)
                            if key in completed_keys: continue
                            configuration_started = time.perf_counter()
                            reporter.progress(f"Participant {participant} ({participant_index}/{len(participants)})")
                            x,y,_=prepare_modality_data(table,participant,target,modality)
                            actual = resolved_feature_count(request, x.shape[1])
                            reporter.progress(f"[{completed}/{configurations}] {target} | {modality} | {model} | features={actual}")
                            reporter.verbose(f"  usable rows: {len(y)}; predictors: {x.shape[1]}; requested features: {request}; resolved features: {actual}")
                            a,b,c=evaluate_individual(participant,x,y,target,modality,model,request,args.seed,reporter,n_jobs=args.n_jobs)
                            results=pd.concat([results,pd.DataFrame(a)],ignore_index=True); selected=pd.concat([selected,pd.DataFrame(b)],ignore_index=True); parameters=pd.concat([parameters,pd.DataFrame(c)],ignore_index=True)
                            completed_keys.add(key); checkpoint_outputs(out,results,selected,parameters,completed_keys,settings); reporter.verbose("  checkpoint saved")
                            reporter.progress(f"  completed in {time.perf_counter() - configuration_started:.1f} s; mean BA: {pd.DataFrame(a).balanced_accuracy.mean():.3f}")
                    else:
                        completed += 1; key=configuration_key(args.mode,None,target,modality,model,request)
                        if key in completed_keys: continue
                        configuration_started = time.perf_counter()
                        actual = resolved_feature_count(request, len(modality_columns(modality)))
                        reporter.progress(f"[{completed}/{configurations}] {target} | {modality} | {model} | features={actual}")
                        reporter.verbose(f"  usable rows: {len(cohort)}; predictors: {len(modality_columns(modality))}; requested features: {request}; resolved features: {actual}")
                        a,b,c=evaluate_general(cohort,target,modality,model,request,args.seed,reporter,n_jobs=args.n_jobs)
                        results=pd.concat([results,pd.DataFrame(a)],ignore_index=True); selected=pd.concat([selected,pd.DataFrame(b)],ignore_index=True); parameters=pd.concat([parameters,pd.DataFrame(c)],ignore_index=True)
                        completed_keys.add(key); checkpoint_outputs(out,results,selected,parameters,completed_keys,settings); reporter.verbose("  checkpoint saved")
                        reporter.progress(f"  completed in {time.perf_counter() - configuration_started:.1f} s; mean BA: {pd.DataFrame(a).balanced_accuracy.mean():.3f}")
    frequency=feature_frequency(selected); summary=summarize(results)
    checkpoint_outputs(out,results,selected,parameters,completed_keys,settings)
    atomic_write_text(out/"README.md","# Image classification experiment\n\nExperimental, leakage-safe binary classification. `all` means **No feature selection**. See `run_manifest.json` for the resolved cohort and design.\n")
    write_plots(summary,frequency,out)
    reporter.progress(f"Completed successfully\nConfigurations: {configurations}\nOuter folds evaluated: {len(results)}\nTotal elapsed time: {time.perf_counter() - started:.1f} s\nOutput: {out}")
    return {"dry_run":False,"output":out,"summary":summary,"prepared":prepared}


if __name__ == "__main__":
    try: run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
