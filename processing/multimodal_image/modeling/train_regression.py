#!/usr/bin/env python3
"""Leakage-safe continuous-rating Image Experiment regression."""
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

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EEG_CHANNELS = ("C3", "C4", "Cz", "Fz", "Oz", "PO7", "PO8", "Pz")
EEG_FAMILIES = ("eeg_sd", "eeg_se", "eeg_hm", "eeg_hc", "eeg_mf_hz", "eeg_bp_delta", "eeg_se_delta", "eeg_bp_theta", "eeg_se_theta", "eeg_bp_alpha", "eeg_se_alpha", "eeg_bp_beta", "eeg_se_beta", "eeg_bp_gamma", "eeg_se_gamma")
EEG_COLUMNS = [f"{family}__{channel}" for family in EEG_FAMILIES for channel in EEG_CHANNELS]
FACE_COLUMNS = [f"video_{name}_norm_{stat}" for name in ("irisdo", "eso", "enso", "mnso", "mwo") for stat in ("mean", "std")]
TARGET_COLUMNS = {"valence": "valence_rating", "arousal": "arousal_rating"}
QC_GROUPS = ("Very clean", "Clean with minor EEG loss", "Review due to elevated EEG rejection", "Review due to missing ratings", "Special/incomplete case")
STAGE_A_PARTICIPANTS = ("P10", "P11", "P19", "P22", "P25", "P32", "P40", "P42", "P43", "P45")


def parse_csv_option(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("individual", "general"), required=True)
    parser.add_argument("--target", choices=("valence", "arousal", "all"), required=True)
    parser.add_argument("--modality", choices=("eeg", "face", "multimodal", "all"), required=True)
    parser.add_argument("--models", default="knn,svr,ridge")
    parser.add_argument("--feature-counts", default="all,5,10,20")
    parser.add_argument("--participants")
    parser.add_argument("--qc-group")
    parser.add_argument("--readiness-csv", type=Path, default=Path("docs/meeting_14_08/modeling_readiness.csv"))
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/image_regression"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume a matching interrupted run from completed configuration checkpoints.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Inner GridSearchCV workers; -1 uses all logical CPUs.")
    parser.add_argument("--progress", action="store_true", help="Show concise experiment progress.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress diagnostics (implies --progress).")
    args = parser.parse_args(argv)
    args.models, args.feature_counts = parse_csv_option(args.models), parse_csv_option(args.feature_counts)
    if not args.models or set(args.models).difference({"knn", "svr", "ridge"}): parser.error("Invalid --models")
    if not args.feature_counts or set(args.feature_counts).difference({"all", "5", "10", "20"}): parser.error("Invalid --feature-counts")
    if args.n_jobs == 0 or args.n_jobs < -1: parser.error("--n-jobs must be -1 or a non-zero integer")
    if args.append and args.resume: parser.error("--append and --resume cannot be used together")
    if "/" in args.run_name or "\\" in args.run_name or args.run_name in {"", ".", ".."}: parser.error("--run-name must be a simple directory name")
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


def modality_columns(modality: str) -> list[str]:
    if modality == "eeg": return EEG_COLUMNS.copy()
    if modality == "face": return FACE_COLUMNS.copy()
    if modality == "multimodal": return EEG_COLUMNS + FACE_COLUMNS
    raise ValueError(f"Unknown modality: {modality}")


def selected_dimensions(value: str, options: Iterable[str]) -> list[str]:
    return list(options) if value == "all" else [value]


def resolve_participants(readiness_path: Path, explicit: str | None, qc_groups: str | None) -> tuple[pd.DataFrame, list[str], str]:
    readiness = pd.read_csv(readiness_path)
    if missing := {"participant", "qc_group"}.difference(readiness.columns): raise ValueError(f"Readiness CSV missing columns: {sorted(missing)}")
    readiness["participant"] = readiness.participant.astype(str).str.upper()
    if explicit:
        requested = [value.upper() for value in parse_csv_option(explicit)]
        if unknown := sorted(set(requested).difference(readiness.participant)): raise ValueError(f"Participants absent from readiness CSV: {unknown}")
        return readiness.set_index("participant").loc[requested].reset_index(), requested, "explicit participants"
    groups = parse_csv_option(qc_groups) if qc_groups else list(QC_GROUPS)
    if unknown := sorted(set(groups).difference(QC_GROUPS)): raise ValueError(f"Unknown QC groups: {unknown}")
    resolved = readiness[readiness.qc_group.isin(groups)].copy().sort_values("participant")
    if resolved.empty: raise ValueError("No participants resolved from --qc-group")
    return resolved, resolved.participant.tolist(), "QC group"


def load_participant_table(derived_root: Path, participant: str) -> tuple[pd.DataFrame, Path]:
    run = derived_root / participant / "Image_Experiment" / "runs" / f"{participant.lower()}_final"
    manifest_path = run / "manifest" / "run_manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError(f"Official final-run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("resolved_configuration", manifest.get("configuration", {}))
    if str(config.get("participant", "")).upper() != participant.upper(): raise ValueError(f"Participant mismatch for {participant}")
    dataset = run / "merged" / f"{participant.lower()}_image_trial_dataset.csv"
    if not dataset.is_file(): raise FileNotFoundError(f"Merged final trial dataset missing: {dataset}")
    return pd.read_csv(dataset), dataset.resolve()


def prepare_modality_data(table: pd.DataFrame, participant: str, target: str, modality: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    columns, target_column = modality_columns(modality), TARGET_COLUMNS[target]
    if missing := sorted(set(columns + [target_column]).difference(table.columns)): raise ValueError(f"{participant} missing required {modality} columns: {missing}")
    features = table[columns].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[target_column], errors="coerce")
    usable = features.notna().all(axis=1) & ratings.notna()
    features, ratings = features.loc[usable].reset_index(drop=True), ratings.loc[usable].reset_index(drop=True)
    if not np.isfinite(features.to_numpy()).all(): raise ValueError(f"{participant} {modality} predictors contain infinity")
    return features, ratings, columns


def resolved_feature_count(request: str, available: int) -> int | str:
    return "all" if request == "all" else min(int(request), available)


def make_pipeline(model: str, count: int | str, seed: int) -> tuple[Pipeline, list[dict[str, list[Any]]]]:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if count != "all": steps.append(("selector", SelectKBest(f_regression, k=int(count))))
    if model == "knn":
        steps.append(("regressor", KNeighborsRegressor()))
        grid = [{"regressor__n_neighbors": [3, 5, 7, 11], "regressor__weights": ["uniform", "distance"]}]
    elif model == "svr":
        steps.append(("regressor", SVR()))
        grid = [{"regressor__kernel": ["linear"], "regressor__C": [0.1, 1, 10], "regressor__epsilon": [0.1, 0.25]}, {"regressor__kernel": ["rbf"], "regressor__C": [0.1, 1, 10], "regressor__gamma": ["scale", 0.01, 0.1], "regressor__epsilon": [0.1, 0.25]}]
    elif model == "ridge":
        steps.append(("regressor", Ridge(random_state=seed)))
        grid = [{"regressor__alpha": [0.01, 0.1, 1, 10, 100]}]
    else: raise ValueError(f"Unknown model: {model}")
    return Pipeline(steps), grid


def metric_row(y_true: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, predicted))), "mae": float(mean_absolute_error(y_true, predicted)), "r2": float(r2_score(y_true, predicted)), "explained_variance": float(explained_variance_score(y_true, predicted))}


def selected_feature_rows(fitted: Pipeline, columns: list[str], common: dict[str, Any]) -> list[dict[str, Any]]:
    if "selector" not in fitted.named_steps: return [{**common, "feature": name, "selected": True, "feature_score": np.nan} for name in columns]
    selector: SelectKBest = fitted.named_steps["selector"]
    return [{**common, "feature": name, "selected": bool(keep), "feature_score": float(score) if np.isfinite(score) else np.nan} for name, keep, score in zip(columns, selector.get_support(), selector.scores_)]


def valid_kfold_splits(rows: int, requested: int, context: str) -> int:
    if rows < 6: raise ValueError(f"{context}: at least 6 usable rows are required for nested regression CV")
    return min(requested, rows // 2)


def fit_outer_fold(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, model: str, count: int | str, inner: list[tuple[np.ndarray, np.ndarray]], seed: int, n_jobs: int = 1) -> tuple[Pipeline, dict[str, Any], float, np.ndarray]:
    pipeline, grid = make_pipeline(model, count, seed)
    if model == "knn":
        allowed = [value for value in grid[0]["regressor__n_neighbors"] if value <= min(len(train) for train, _ in inner)]
        if not allowed: raise ValueError("No valid kNN neighbor values remain for inner folds")
        grid = [{**grid[0], "regressor__n_neighbors": allowed}]
    search = GridSearchCV(pipeline, grid, scoring="neg_root_mean_squared_error", cv=inner, n_jobs=n_jobs, error_score="raise")
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, float(-search.best_score_), search.predict(x_test)


def evaluate_individual(participant: str, x: pd.DataFrame, y: pd.Series, target: str, modality: str, model: str, request: str, seed: int, reporter: ProgressReporter | None = None, n_jobs: int = 1) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    reporter = reporter or ProgressReporter(False, False)
    folds, count = valid_kfold_splits(len(y), 5, participant), resolved_feature_count(request, x.shape[1])
    rows: list[dict] = []; selected: list[dict] = []; parameters: list[dict] = []; predictions: list[dict] = []
    for fold, (train, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(x), 1):
        context = f"participant={participant} | target={target} | modality={modality} | model={model} | features={request} | outer fold {fold}/{folds}"
        started = time.perf_counter()
        try:
            reporter.progress(f"  outer fold {fold}/{folds} ...")
            inner_count = valid_kfold_splits(len(train), min(3, folds), f"{participant} fold {fold}")
            inner = list(KFold(inner_count, shuffle=True, random_state=seed + fold).split(x.iloc[train]))
            _, grid = make_pipeline(model, count, seed)
            reporter.verbose(f"    train rows: {len(train)}; test rows: {len(test)}; predictors: {x.shape[1]}")
            reporter.verbose(f"    tuning {model} ({ProgressReporter.candidate_count(grid)} candidates)...")
            fitted, best, inner_rmse, predicted = fit_outer_fold(x.iloc[train], y.iloc[train], x.iloc[test], model, count, inner, seed, n_jobs=n_jobs)
            metrics = metric_row(y.iloc[test], predicted)
            reporter.verbose("    inner tuning complete")
            reporter.verbose(f"    best params: {json.dumps(best, sort_keys=True)}")
            reporter.verbose(f"    RMSE: {metrics['rmse']:.2f}; R2: {metrics['r2']:.2f}; elapsed: {time.perf_counter() - started:.1f} s")
            common = {"mode": "individual", "participant": participant, "held_out_participant": participant, "target": target, "modality": modality, "regressor": model, "feature_count_request": request, "feature_count_resolved": count, "fold": fold}
            rows.append({**common, "training_participants": participant, "sample_size": len(y), "train_size": len(train), "test_size": len(test), **metrics})
            selected += selected_feature_rows(fitted, list(x.columns), common)
            parameters.append({**common, "best_parameters": json.dumps(best, sort_keys=True), "inner_best_rmse": inner_rmse})
            predictions += [{**common, "true_rating": float(actual), "predicted_rating": float(estimate)} for actual, estimate in zip(y.iloc[test], predicted)]
        except Exception:
            print(f"ERROR during {context}", file=sys.stderr, flush=True)
            raise
    return rows, selected, parameters, predictions


def grouped_inner_splits(x: pd.DataFrame, groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups.nunique() < 2: raise ValueError("General-mode inner CV needs at least two training participants")
    splits = list(GroupKFold(min(5, groups.nunique())).split(x, groups=groups))
    if any(set(groups.iloc[train]).intersection(groups.iloc[test]) for train, test in splits): raise RuntimeError("Grouped inner CV mixed participant groups")
    return splits


def evaluate_general(data: pd.DataFrame, target: str, modality: str, model: str, request: str, seed: int, reporter: ProgressReporter | None = None, n_jobs: int = 1) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    reporter = reporter or ProgressReporter(False, False)
    columns, count = modality_columns(modality), resolved_feature_count(request, len(modality_columns(modality)))
    rows: list[dict] = []; selected: list[dict] = []; parameters: list[dict] = []; predictions: list[dict] = []
    outer_splits = list(LeaveOneGroupOut().split(data, groups=data.participant))
    for fold, (train, test) in enumerate(outer_splits, 1):
        training, heldout = data.iloc[train], data.iloc[test]
        held = str(heldout.participant.iloc[0])
        common = {"mode": "general", "participant": "COHORT", "held_out_participant": held, "target": target, "modality": modality, "regressor": model, "feature_count_request": request, "feature_count_resolved": count, "fold": fold}
        context = f"target={target} | modality={modality} | model={model} | features={request} | outer fold {fold}/{len(outer_splits)} | held out={held}"
        started = time.perf_counter()
        try:
            reporter.progress(f"  outer fold {fold}/{len(outer_splits)} | held out: {held} ...")
            inner = grouped_inner_splits(training[columns], training.participant.reset_index(drop=True))
            _, grid = make_pipeline(model, count, seed)
            reporter.verbose(f"    train rows: {len(train)}; test rows: {len(test)}; training participants: {', '.join(sorted(training.participant.unique()))}")
            reporter.verbose(f"    tuning {model} ({ProgressReporter.candidate_count(grid)} candidates)...")
            fitted, best, inner_rmse, predicted = fit_outer_fold(training[columns], training.rating, heldout[columns], model, count, inner, seed, n_jobs=n_jobs)
            metrics = metric_row(heldout.rating, predicted)
            reporter.verbose("    inner tuning complete")
            reporter.verbose(f"    best params: {json.dumps(best, sort_keys=True)}")
            reporter.verbose(f"    RMSE: {metrics['rmse']:.2f}; R2: {metrics['r2']:.2f}; elapsed: {time.perf_counter() - started:.1f} s")
            rows.append({**common, "training_participants": ";".join(sorted(training.participant.unique())), "sample_size": len(data), "train_size": len(train), "test_size": len(test), **metrics})
            selected += selected_feature_rows(fitted, columns, common)
            parameters.append({**common, "best_parameters": json.dumps(best, sort_keys=True), "inner_best_rmse": inner_rmse})
            predictions += [{**common, "true_rating": float(actual), "predicted_rating": float(estimate)} for actual, estimate in zip(heldout.rating, predicted)]
        except Exception:
            print(f"ERROR during {context}", file=sys.stderr, flush=True)
            raise
    return rows, selected, parameters, predictions


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "participant", "target", "modality", "regressor", "feature_count_request", "feature_count_resolved"]
    aggregations: dict[str, tuple[str, str]] = {"folds": ("fold", "size")}
    for metric in ("rmse", "mae", "r2", "explained_variance"):
        for statistic in ("mean", "std", "median", "min", "max"): aggregations[f"{metric}_{statistic}"] = (metric, statistic)
    return results.groupby(keys, dropna=False).agg(**aggregations).reset_index()


def feature_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "target", "modality", "regressor", "feature_count_request", "feature_count_resolved", "feature"]
    return selected.groupby(keys, dropna=False).agg(selected_folds=("selected", "sum"), outer_folds=("fold", "nunique"), selection_frequency=("selected", "mean"), mean_feature_score=("feature_score", "mean")).reset_index()


CHECKPOINT_STATE_FILE = "checkpoint_state.json"
OUTPUT_FILES = {"results": "fold_results.csv", "predictions": "predictions.csv", "parameters": "best_hyperparameters.csv", "selected": "selected_features_by_fold.csv", "frequency": "feature_selection_frequency.csv", "summary": "results_summary.csv"}


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
    identity: dict[str, Any] = {"mode": mode, "target": target, "modality": modality, "model": model, "feature_count_request": request}
    if mode == "individual": identity["participant"] = participant
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def scientific_settings(args: argparse.Namespace, participants: list[str], targets: list[str], modalities: list[str], source: str) -> dict[str, Any]:
    """Settings that must agree before outputs can be scientifically resumed."""
    return {"mode": args.mode, "participants": participants, "cohort_source": source, "target": targets, "modalities": modalities, "models": args.models, "feature_counts": args.feature_counts, "seed": args.seed, "cv_strategy": "Nested shuffled KFold" if args.mode == "individual" else "LeaveOneGroupOut outer CV; GroupKFold inner CV", "hyperparameter_grids": {"knn": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}, "svr": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "rbf_gamma": ["scale", 0.01, 0.1], "epsilon": [0.1, 0.25]}, "ridge": {"alpha": [0.01, 0.1, 1, 10, 100]}}, "feature_selection": "SelectKBest(f_regression) inside Pipeline", "target_definition": "Original continuous ratings; primary predictions are unconstrained.", "prediction_clipping": "unconstrained"}


def checkpoint_outputs(out: Path, results: pd.DataFrame, selected: pd.DataFrame, parameters: pd.DataFrame, predictions: pd.DataFrame, completed_keys: set[str], settings: dict[str, Any]) -> None:
    """Refresh scientific outputs, then write completion state last."""
    frequency, summary = feature_frequency(selected), summarize(results)
    for frame, name in ((results, "results"), (predictions, "predictions"), (parameters, "parameters"), (selected, "selected"), (frequency, "frequency"), (summary, "summary")):
        atomic_write_csv(frame, out / OUTPUT_FILES[name])
    atomic_write_text(out / CHECKPOINT_STATE_FILE, json.dumps({"scientific_settings": settings, "completed_configurations": sorted(completed_keys)}, indent=2) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter(); reporter = ProgressReporter(args.progress, args.verbose)
    readiness = (PROJECT_ROOT / args.readiness_csv).resolve() if not args.readiness_csv.is_absolute() else args.readiness_csv
    derived = (PROJECT_ROOT / args.derived_root).resolve() if not args.derived_root.is_absolute() else args.derived_root
    root = (PROJECT_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    resolved, participants, source = resolve_participants(readiness, args.participants, args.qc_group)
    tables: dict[str, pd.DataFrame] = {}; sources: dict[str, Path] = {}
    for participant in participants: tables[participant], sources[participant] = load_participant_table(derived, participant)
    targets, modalities = selected_dimensions(args.target, TARGET_COLUMNS), selected_dimensions(args.modality, ("eeg", "face", "multimodal"))
    prepared = [{"participant": participant, "target": target, "modality": modality, "rows": len(prepare_modality_data(table, participant, target, modality)[1]), "predictors": ";".join(modality_columns(modality))} for participant, table in tables.items() for target in targets for modality in modalities]
    reporter.progress("Resolved participants (" + source + "): " + ", ".join(participants))
    for item in prepared: reporter.verbose(f"{item['participant']} {item['target']} {item['modality']}: usable={item['rows']}, predictors={len(item['predictors'].split(';'))}")
    if args.dry_run: return {"dry_run": True, "participants": participants, "prepared": prepared}
    out, settings = root / args.run_name, scientific_settings(args, participants, targets, modalities, source)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(), "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "n_jobs": args.n_jobs, "resolved_participants": participants, "cohort_source": source, "stage_a_participants": list(STAGE_A_PARTICIPANTS), "source_merged_tables": {key: str(value) for key, value in sources.items()}, "inner_selection_metric": "neg_root_mean_squared_error; sklearn negates RMSE because greater scores are better", **settings, "scientific_settings": settings}
    if out.exists() and not args.append and not args.resume: raise FileExistsError(f"Refusing to overwrite existing run directory: {out}")
    if (args.append or args.resume) and not out.is_dir(): raise FileNotFoundError(f"Cannot {'resume' if args.resume else 'append'}: run directory does not exist: {out}")
    out.mkdir(parents=True, exist_ok=args.append or args.resume); reporter.progress(f"Output: {out}")
    if args.resume:
        existing = json.loads((out / "run_manifest.json").read_text(encoding="utf-8")) if (out / "run_manifest.json").is_file() else None
        state = json.loads((out / CHECKPOINT_STATE_FILE).read_text(encoding="utf-8")) if (out / CHECKPOINT_STATE_FILE).is_file() else None
        if not existing or not state or existing.get("scientific_settings") != settings or state.get("scientific_settings") != settings: raise ValueError("Refusing resume: saved manifest/checkpoint scientific settings do not exactly match the requested run")
        completed_keys = set(state.get("completed_configurations", []))
        def load(name: str) -> pd.DataFrame:
            path = out / OUTPUT_FILES[name]
            frame = pd.read_csv(path) if path.is_file() else pd.DataFrame()
            if frame.empty: return frame
            keys = frame.apply(lambda row: configuration_key(str(row["mode"]), str(row["participant"]) if str(row["mode"]) == "individual" else None, str(row["target"]), str(row["modality"]), str(row["regressor"]), str(row["feature_count_request"])), axis=1)
            return frame.loc[keys.isin(completed_keys)].copy()
        result_frame, selected_frame, parameter_frame, prediction_frame = load("results"), load("selected"), load("parameters"), load("predictions")
    elif args.append:
        def load_append(name: str) -> pd.DataFrame:
            path = out / OUTPUT_FILES[name]; return pd.read_csv(path) if path.is_file() else pd.DataFrame()
        result_frame, selected_frame, parameter_frame, prediction_frame = load_append("results"), load_append("selected"), load_append("parameters"), load_append("predictions")
        completed_keys = set() if result_frame.empty else set(result_frame.apply(lambda row: configuration_key(str(row["mode"]), str(row["participant"]) if str(row["mode"]) == "individual" else None, str(row["target"]), str(row["modality"]), str(row["regressor"]), str(row["feature_count_request"])), axis=1))
    else:
        atomic_write_text(out / "run_manifest.json", json.dumps(manifest, indent=2) + "\n"); atomic_write_csv(resolved, out / "resolved_participants.csv")
        completed_keys = set(); result_frame = pd.DataFrame(); selected_frame = pd.DataFrame(); parameter_frame = pd.DataFrame(); prediction_frame = pd.DataFrame()
    configurations = len(targets) * len(modalities) * len(args.models) * len(args.feature_counts) * (len(participants) if args.mode == "individual" else 1)
    if args.resume:
        reporter.progress(f"Resuming run: {args.run_name}"); reporter.progress(f"Completed configurations found: {len(completed_keys)}"); reporter.progress(f"Remaining configurations: {configurations - len(completed_keys)}")
    completed = 0
    for target in targets:
        for modality in modalities:
            cohort = pd.concat([prepare_modality_data(table, participant, target, modality)[0].assign(participant=participant, rating=prepare_modality_data(table, participant, target, modality)[1].to_numpy()) for participant, table in tables.items()], ignore_index=True) if args.mode == "general" else None
            for model in args.models:
                for request in args.feature_counts:
                    for participant_index, (participant, table) in enumerate(tables.items(), 1) if args.mode == "individual" else [(1, (None, None))]:
                        completed += 1; key = configuration_key(args.mode, participant, target, modality, model, request)
                        if key in completed_keys: continue
                        configuration_started = time.perf_counter()
                        if args.mode == "individual":
                            reporter.progress(f"Participant {participant} ({participant_index}/{len(participants)})"); x, y, _ = prepare_modality_data(table, participant, target, modality); actual = resolved_feature_count(request, x.shape[1])
                        else:
                            x = y = None; actual = resolved_feature_count(request, len(modality_columns(modality)))
                        reporter.progress(f"[{completed}/{configurations}] {target} | {modality} | {model} | features={actual}")
                        reporter.verbose(f"  usable rows: {len(y) if y is not None else len(cohort)}; requested features: {request}; resolved features: {actual}")
                        if args.mode == "individual": a, b, c, d = evaluate_individual(participant, x, y, target, modality, model, request, args.seed, reporter, n_jobs=args.n_jobs)
                        else: a, b, c, d = evaluate_general(cohort, target, modality, model, request, args.seed, reporter, n_jobs=args.n_jobs)
                        result_frame = pd.concat([result_frame, pd.DataFrame(a)], ignore_index=True); selected_frame = pd.concat([selected_frame, pd.DataFrame(b)], ignore_index=True); parameter_frame = pd.concat([parameter_frame, pd.DataFrame(c)], ignore_index=True); prediction_frame = pd.concat([prediction_frame, pd.DataFrame(d)], ignore_index=True)
                        completed_keys.add(key); checkpoint_outputs(out, result_frame, selected_frame, parameter_frame, prediction_frame, completed_keys, settings); reporter.verbose("  checkpoint saved")
                        reporter.progress(f"  completed in {time.perf_counter() - configuration_started:.1f} s; mean RMSE: {pd.DataFrame(a).rmse.mean():.2f}")
    checkpoint_outputs(out, result_frame, selected_frame, parameter_frame, prediction_frame, completed_keys, settings)
    atomic_write_text(out / "README.md", "# Image regression experiment\n\nContinuous unconstrained-rating prediction. Inner selection uses neg_root_mean_squared_error, which sklearn negates for its greater-is-better convention.\n\nSee results_summary.csv for fold-aggregated RMSE, MAE, R2, and explained-variance metrics.\n")
    reporter.progress(f"Completed successfully\nConfigurations: {configurations}\nOuter folds evaluated: {len(result_frame)}\nTotal elapsed time: {time.perf_counter() - started:.1f} s\nOutput: {out}")
    return {"dry_run": False, "output": out, "summary": summarize(result_frame), "prepared": prepared}


if __name__ == "__main__":
    try: run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
