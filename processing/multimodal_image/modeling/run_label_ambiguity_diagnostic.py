#!/usr/bin/env python3
"""Diagnose the effect of removing rating-4 midpoint trials before nested CV.

This script only reads existing no-ICA merged trial tables.  It never runs
preprocessing, feature extraction, video extraction, or modifies derived data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import run_classifier_search as classifier_search


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARTICIPANTS = ("P36", "P22", "P06", "P17", "P40")
TARGETS = ("valence", "arousal")
MODALITIES = ("eeg", "face", "multimodal")
CONDITIONS = ("standard", "exclude_midpoint")
MODELS = ("svm", "knn", "logreg", "gnb")
DEFAULT_SOURCE_RUN = "{participant_lower}_no_ica"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/image_classification/label_ambiguity_top5"
CHECKPOINT_FILE = "checkpoint_state.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-design label ambiguity diagnostic command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4, help="Inner GridSearchCV workers (default: 4).")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Report rating distributions without CV or classifier fitting.")
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        args.participants = classifier_search.parse_participants(args.participants)
        args.targets = classifier_search.parse_csv(args.targets)
        args.modalities = classifier_search.parse_csv(args.modalities)
    except ValueError as error:
        parser.error(str(error))
    if set(args.targets).difference(TARGETS):
        parser.error(f"Unknown --targets values: {sorted(set(args.targets).difference(TARGETS))}")
    if set(args.modalities).difference(MODALITIES):
        parser.error(f"Unknown --modalities values: {sorted(set(args.modalities).difference(MODALITIES))}")
    if args.seed != 42:
        parser.error("This fixed diagnostic requires --seed 42")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer; use an explicit CPU count, not -1")
    return args


def label_condition(ratings: pd.Series, condition: str) -> tuple[pd.Series, pd.Series]:
    """Return retained-row mask and LOW=0/HIGH=1 labels for one condition."""
    numeric = pd.to_numeric(ratings, errors="raise")
    if condition == "standard":
        retained = numeric.notna()
        labels = (numeric.loc[retained] >= 4.0).astype(int)
    elif condition == "exclude_midpoint":
        retained = numeric.notna() & (numeric != 4.0)
        retained_values = numeric.loc[retained]
        unsupported = retained_values[(retained_values > 3.0) & (retained_values < 5.0)]
        if not unsupported.empty:
            raise ValueError("exclude_midpoint requires observed ratings to be <=3, 4, or >=5; found values between 3 and 5")
        labels = (retained_values >= 5.0).astype(int)
    else:
        raise ValueError(f"Unknown label condition: {condition}")
    return retained, labels.reset_index(drop=True)


def trial_count_row(participant: str, target: str, ratings: pd.Series, condition: str) -> dict[str, Any]:
    """Report label counts without modifying ratings or imputing missing values."""
    numeric = pd.to_numeric(ratings, errors="coerce")
    observed = numeric.dropna()
    retained, labels = label_condition(observed, condition)
    n_original = len(observed)
    n_trials = int(retained.sum())
    return {"participant": participant, "target": target, "label_condition": condition, "total_original_trials": n_original, "n_trials": n_trials, "n_low": int((labels == 0).sum()), "n_high": int((labels == 1).sum()), "n_dropped_midpoint": int((observed == 4.0).sum()) if condition == "exclude_midpoint" else 0, "retained_fraction": float(n_trials / n_original) if n_original else 0.0, "class_balance_high_fraction": float((labels == 1).mean()) if n_trials else np.nan}


def extreme_rule_counts(ratings: pd.Series, low_upper: float, high_lower: float) -> dict[str, Any]:
    """Count retained LOW/HIGH trials for an audit-only prospective rule."""
    observed = pd.to_numeric(ratings, errors="coerce").dropna()
    low = int((observed <= low_upper).sum())
    high = int((observed >= high_lower).sum())
    retained = low + high
    total = len(observed)
    return {"n_low": low, "n_high": high, "n_dropped": total - retained, "retained_fraction": float(retained / total) if total else 0.0}


def rating_distribution_row(participant: str, target: str, ratings: pd.Series) -> dict[str, Any]:
    """Return all requested audit-only rating and future-rule counts."""
    observed = pd.to_numeric(ratings, errors="coerce").dropna()
    counts = observed.value_counts()
    standard = extreme_rule_counts(observed, 3, 4)
    midpoint = extreme_rule_counts(observed, 3, 5)
    moderate = extreme_rule_counts(observed, 2, 5)
    strong = extreme_rule_counts(observed, 2, 6)
    return {
        "participant": participant, "target": target, "n_total": len(observed),
        **{f"n_rating_{rating}": int(counts.get(rating, 0)) for rating in range(1, 8)},
        "standard_n_low": standard["n_low"], "standard_n_high": standard["n_high"],
        "standard_retained_fraction": standard["retained_fraction"],
        "exclude_midpoint_n_low": midpoint["n_low"], "exclude_midpoint_n_high": midpoint["n_high"],
        "n_dropped_midpoint": midpoint["n_dropped"], "retained_fraction": midpoint["retained_fraction"],
        "moderate_extremes_n_low": moderate["n_low"], "moderate_extremes_n_high": moderate["n_high"],
        "moderate_extremes_n_dropped": moderate["n_dropped"], "moderate_extremes_retained_fraction": moderate["retained_fraction"],
        "strong_extremes_n_low": strong["n_low"], "strong_extremes_n_high": strong["n_high"],
        "strong_extremes_n_dropped": strong["n_dropped"], "strong_extremes_retained_fraction": strong["retained_fraction"],
    }


def print_audit_summary(audit: pd.DataFrame) -> None:
    """Print compact target-level totals for the audit-only report."""
    for target, group in audit.groupby("target", sort=True):
        total = int(group.n_total.sum())
        rating_four_fraction = float(group.n_rating_4.sum() / total) if total else 0.0
        print(
            f"Audit {target}: total={total}; rating_4_fraction={rating_four_fraction:.3f}; "
            f"standard retained={group.standard_retained_fraction.mean():.3f} low/high={int(group.standard_n_low.sum())}/{int(group.standard_n_high.sum())}; "
            f"exclude_midpoint retained={group.retained_fraction.mean():.3f} low/high={int(group.exclude_midpoint_n_low.sum())}/{int(group.exclude_midpoint_n_high.sum())}; "
            f"moderate_extremes retained={group.moderate_extremes_retained_fraction.mean():.3f} low/high={int(group.moderate_extremes_n_low.sum())}/{int(group.moderate_extremes_n_high.sum())}; "
            f"strong_extremes retained={group.strong_extremes_retained_fraction.mean():.3f} low/high={int(group.strong_extremes_n_low.sum())}/{int(group.strong_extremes_n_high.sum())}",
            flush=True,
        )


def prepare_condition_data(table: pd.DataFrame, participant: str, target: str, modality: str, condition: str) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare one modality after applying the requested label condition."""
    columns = classifier_search.baseline.modality_columns(modality)
    target_column = classifier_search.baseline.TARGET_COLUMNS[target]
    missing = sorted(set(columns + [target_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{participant} missing required {modality} columns: {missing}")
    features = table[columns].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[target_column], errors="coerce")
    complete_features = features.notna().all(axis=1)
    retained_ratings, _ = label_condition(ratings, condition)
    usable = complete_features & retained_ratings
    x = features.loc[usable].reset_index(drop=True)
    _, y = label_condition(ratings.loc[usable], condition)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(f"{participant} {modality} predictors contain infinity")
    return x, y


def condition_status(x: pd.DataFrame, y: pd.Series, context: str) -> tuple[str, str | None]:
    """Return ready or insufficient_data before any nested-CV fitting begins."""
    if len(y) == 0:
        return "insufficient_data", f"{context}: no retained rows"
    counts = y.value_counts()
    if len(counts) < 2:
        return "insufficient_data", f"{context}: only one retained class"
    if int(counts.min()) < 2:
        return "insufficient_data", f"{context}: fewer than two minority-class rows"
    return "ready", None


def evaluate_configuration(participant: str, table: pd.DataFrame, target: str, condition: str, modality: str, model: str, request: str, seed: int, n_jobs: int) -> dict[str, Any]:
    """Run leakage-safe nested individual CV after condition filtering."""
    x, y = prepare_condition_data(table, participant, target, modality, condition)
    status, reason = condition_status(x, y, f"{participant} {target} {condition} {modality}")
    if status != "ready":
        raise ValueError(reason)
    folds = classifier_search.baseline.valid_stratified_splits(y, 5, f"{participant} {target} {condition} {modality}")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    feature_count: int | str = "all" if request == "all" else min(int(request), x.shape[1])
    balanced_scores: list[float] = []
    accuracy_scores: list[float] = []
    selected_parameters: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer.split(x, y), start=1):
        y_train, y_test = y.iloc[train].reset_index(drop=True), y.iloc[test].reset_index(drop=True)
        inner_count = classifier_search.baseline.valid_stratified_splits(y_train, min(3, folds), f"{participant} outer fold {fold} inner CV")
        inner = list(StratifiedKFold(n_splits=inner_count, shuffle=True, random_state=seed + fold).split(x.iloc[train], y_train))
        parameters, inner_score, predicted = classifier_search.fit_outer_fold(x.iloc[train], y_train, x.iloc[test], model=model, feature_count=feature_count, inner_splits=inner, seed=seed, n_jobs=n_jobs)
        balanced_scores.append(float(balanced_accuracy_score(y_test, predicted)))
        accuracy_scores.append(float(accuracy_score(y_test, predicted)))
        selected_parameters.append({"fold": fold, "inner_best_balanced_accuracy": inner_score, "parameters": parameters})
    counts = trial_count_row(participant, target, table[classifier_search.baseline.TARGET_COLUMNS[target]], condition)
    return {**counts, "status": "complete", "modality": modality, "classifier": model, "feature_count": feature_count, "feature_count_request": request, "outer_folds": folds, "mean_balanced_accuracy": float(np.mean(balanced_scores)), "std_balanced_accuracy": float(np.std(balanced_scores, ddof=0)), "mean_accuracy": float(np.mean(accuracy_scores)), "std_accuracy": float(np.std(accuracy_scores, ddof=0)), "best_hyperparameters": json.dumps(selected_parameters, sort_keys=True), "seed": seed, "gridsearch_n_jobs": n_jobs}


def configuration_key(participant: str, target: str, condition: str, modality: str, classifier: str, request: str) -> str:
    """Return a stable identity for one completed diagnostic configuration."""
    return "|".join((participant, target, condition, modality, classifier, request))


def completed_result_keys(results: pd.DataFrame) -> set[str]:
    """Validate result identity uniqueness before a resume operation."""
    required = {"participant", "target", "label_condition", "modality", "classifier", "feature_count_request"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Cannot resume: results_summary.csv missing configuration identity columns: {sorted(missing)}")
    keys = results.apply(lambda row: configuration_key(str(row.participant), str(row.target), str(row.label_condition), str(row.modality), str(row.classifier), str(row.feature_count_request)), axis=1)
    if keys.duplicated().any():
        raise ValueError("Cannot resume: results_summary.csv contains duplicate completed configurations")
    return set(keys)


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    """Choose the best completed configuration per participant-target-condition."""
    if results.empty:
        return pd.DataFrame(columns=["participant", "target", "label_condition", "mean_balanced_accuracy", "mean_accuracy", "classifier", "modality", "feature_count"])
    ranked = results.sort_values(["participant", "target", "label_condition", "mean_balanced_accuracy", "mean_accuracy", "classifier", "modality", "feature_count_request"], ascending=[True, True, True, False, False, True, True, True])
    return ranked.drop_duplicates(["participant", "target", "label_condition"], keep="first")


def comparison_frame(best: pd.DataFrame, trial_counts: pd.DataFrame) -> pd.DataFrame:
    """Pivot standard and midpoint-excluded winners into within-pair deltas."""
    standard = best[best.label_condition == "standard"].copy()
    filtered = best[best.label_condition == "exclude_midpoint"].copy()
    standard = standard.rename(columns={"mean_balanced_accuracy": "standard_balanced_accuracy", "mean_accuracy": "standard_accuracy", "classifier": "standard_classifier", "modality": "standard_modality", "feature_count": "standard_feature_count"})
    filtered = filtered.rename(columns={"mean_balanced_accuracy": "exclude_midpoint_balanced_accuracy", "mean_accuracy": "exclude_midpoint_accuracy", "classifier": "exclude_midpoint_classifier", "modality": "exclude_midpoint_modality", "feature_count": "exclude_midpoint_feature_count"})
    columns = ["participant", "target", "standard_balanced_accuracy", "standard_accuracy", "standard_classifier", "standard_modality", "standard_feature_count"]
    filtered_columns = ["participant", "target", "exclude_midpoint_balanced_accuracy", "exclude_midpoint_accuracy", "exclude_midpoint_classifier", "exclude_midpoint_modality", "exclude_midpoint_feature_count"]
    comparison = standard[columns].merge(filtered[filtered_columns], on=["participant", "target"], how="outer", validate="one_to_one")
    comparison["BA_delta"] = comparison.exclude_midpoint_balanced_accuracy - comparison.standard_balanced_accuracy
    comparison["accuracy_delta"] = comparison.exclude_midpoint_accuracy - comparison.standard_accuracy
    retained = trial_counts[trial_counts.label_condition == "exclude_midpoint"][["participant", "target", "n_trials", "n_low", "n_high", "n_dropped_midpoint", "retained_fraction"]]
    return comparison.merge(retained, on=["participant", "target"], how="left", validate="one_to_one")


def aggregate_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize within-participant midpoint-removal differences."""
    rows = []
    for label, group in [("overall", comparison), *[(str(target), subset) for target, subset in comparison.groupby("target")]]:
        delta = group.BA_delta.dropna()
        rows.append({"summary_type": label, "count": len(group), "mean_standard_BA": group.standard_balanced_accuracy.mean(), "mean_exclude_midpoint_BA": group.exclude_midpoint_balanced_accuracy.mean(), "mean_BA_delta": delta.mean(), "median_BA_delta": delta.median(), "improved": int((delta > 0).sum()), "tied": int((delta == 0).sum()), "worsened": int((delta < 0).sum()), "mean_retained_fraction": group.retained_fraction.mean()})
    return pd.DataFrame(rows)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one compact CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    """Atomically write a manifest or checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or safely resume the label ambiguity diagnostic."""
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    derived_root = args.derived_root.resolve() if args.derived_root.is_absolute() else (PROJECT_ROOT / args.derived_root).resolve()
    output_exists = output.exists()
    if output_exists and not args.resume and not args.audit_only:
        raise FileExistsError(f"Output directory already exists; use --resume only for the same interrupted diagnostic: {output}")
    tables: dict[str, pd.DataFrame] = {}
    for participant in args.participants:
        table, _ = classifier_search.baseline.load_participant_table(derived_root, participant, DEFAULT_SOURCE_RUN)
        tables[participant] = table
    if args.audit_only:
        audit = pd.DataFrame([
            rating_distribution_row(participant, target, tables[participant][classifier_search.baseline.TARGET_COLUMNS[target]])
            for participant in args.participants for target in args.targets
        ])
        atomic_write_csv(audit, output / "rating_distribution_audit.csv")
        print_audit_summary(audit)
        return {"output": output, "audit_only": True, "rows": len(audit)}
    settings = {"targets": args.targets, "modalities": args.modalities, "conditions": list(CONDITIONS), "models": list(MODELS), "feature_requests": list(classifier_search.FEATURE_COUNTS), "hyperparameter_grids": {model: classifier_search.make_pipeline(model, 3, args.seed)[1] for model in MODELS}, "source_run": DEFAULT_SOURCE_RUN, "seed": args.seed, "label_definitions": {"standard": "LOW: rating < 4; HIGH: rating >= 4", "exclude_midpoint": "LOW: rating <= 3; HIGH: rating >= 5; DROP rating == 4"}, "outer_cv": "StratifiedKFold up to 5 folds, shuffled, random_state=42", "inner_cv": "StratifiedKFold up to 3 folds, shuffled, random_state=42+outer_fold", "outer_experiment_loop": "serial", "output_directory": str(output)}
    manifest_path, checkpoint_path, results_path = output / "run_manifest.json", output / CHECKPOINT_FILE, output / "results_summary.csv"
    if args.resume and output_exists:
        if not manifest_path.is_file() or not checkpoint_path.is_file() or not results_path.is_file():
            raise ValueError("Cannot resume: run manifest, checkpoint, or results_summary.csv is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if manifest.get("settings") != settings or state.get("settings") != settings:
            raise ValueError("Refusing resume: requested scientific settings differ from the saved run")
        results = pd.read_csv(results_path)
        completed = completed_result_keys(results)
        if completed != set(state.get("completed_configurations", [])):
            raise ValueError("Cannot resume: checkpoint completed configurations do not match results_summary.csv")
    else:
        output.mkdir(parents=True, exist_ok=True)
        results, completed = pd.DataFrame(), set()
    if not (args.resume and output_exists):
        atomic_write_json({"command": " ".join(sys.argv), "requested_participants": args.participants, "settings": settings}, manifest_path)
    trial_rows = []
    for participant in args.participants:
        for target in args.targets:
            ratings = tables[participant][classifier_search.baseline.TARGET_COLUMNS[target]]
            for condition in CONDITIONS:
                row = trial_count_row(participant, target, ratings, condition)
                _, labels = label_condition(pd.to_numeric(ratings, errors="coerce").dropna(), condition)
                row["status"], row["insufficient_reason"] = condition_status(
                    pd.DataFrame(index=range(len(labels))), labels, f"{participant} {target} {condition}"
                )
                trial_rows.append(row)
    trial_counts = pd.DataFrame(trial_rows)
    atomic_write_csv(trial_counts, output / "trial_counts.csv")
    planned = [(participant, target, condition, modality, model, request) for participant in args.participants for target in args.targets for condition in CONDITIONS for modality in args.modalities for model in MODELS for request in classifier_search.resolved_feature_requests(modality)]
    print(f"CPU parallelism: GridSearchCV n_jobs={args.n_jobs}")
    print("Outer experiment loop: serial")
    print("Estimator nested parallelism: disabled")
    for number, (participant, target, condition, modality, model, request) in enumerate(planned, start=1):
        key = configuration_key(participant, target, condition, modality, model, request)
        if key in completed:
            continue
        x, y = prepare_condition_data(tables[participant], participant, target, modality, condition)
        status, reason = condition_status(x, y, f"{participant} {target} {condition} {modality}")
        if status != "ready":
            if args.progress:
                print(f"[{number}/{len(planned)}] {participant} | {target} | {condition} | {modality} | insufficient_data: {reason}", flush=True)
            continue
        if args.progress:
            count = "all" if request == "all" else min(int(request), x.shape[1])
            print(f"[{number}/{len(planned)}] {participant} | {target} | {condition} | {modality} | {model} | k={count}", flush=True)
        row = evaluate_configuration(participant, tables[participant], target, condition, modality, model, request, args.seed, args.n_jobs)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        completed.add(key)
        atomic_write_csv(results, results_path)
        atomic_write_json({"settings": settings, "completed_configurations": sorted(completed)}, checkpoint_path)
    requested = results[results.participant.astype(str).str.upper().isin(args.participants) & results.target.isin(args.targets)].copy() if not results.empty else results
    best = choose_best(requested)
    insufficient = trial_counts[trial_counts.status == "insufficient_data"]
    if not insufficient.empty:
        placeholders = insufficient.assign(
            modality=np.nan, classifier=np.nan, feature_count=np.nan,
            mean_balanced_accuracy=np.nan, mean_accuracy=np.nan,
        )[["participant", "target", "label_condition", "modality", "classifier", "feature_count", "mean_balanced_accuracy", "mean_accuracy", "status"]]
        best = pd.concat([best, placeholders], ignore_index=True)
    comparison = comparison_frame(best, trial_counts)
    atomic_write_csv(best, output / "best_per_participant_target_condition.csv")
    atomic_write_csv(comparison, output / "standard_vs_exclude_midpoint.csv")
    atomic_write_csv(aggregate_summary(comparison), output / "aggregate_summary.csv")
    return {"output": output, "configurations": len(planned), "completed": len(completed)}


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
