#!/usr/bin/env python3
"""Leakage-safe individual EEG feature-family search on no-ICA trial tables.

The script only selects existing merged EEG columns.  It does not perform EEG
preprocessing, ICA, video extraction, or feature extraction.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import run_classifier_search as classifier_search


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARTICIPANTS = ("P36", "P22", "P06", "P17", "P40")
TARGETS = ("valence", "arousal")
MODELS = ("knn", "svm", "logreg", "gnb")
FEATURE_REQUESTS = ("3", "5", "8", "10", "15", "20", "all")
CHANNEL_SUBSETS = {
    "all_channels": ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"),
    "midline": ("Fz", "Cz", "Pz", "Oz"),
    "posterior": ("Pz", "PO7", "Oz", "PO8"),
    "frontal_central": ("Fz", "C3", "Cz", "C4"),
}
FAMILY_CHANNEL_PLANS = (
    ("all_eeg", ("all_channels",)),
    ("hjorth", ("all_channels", "midline")),
    ("entropy_all_bands", ("all_channels",)),
    ("entropy_beta_gamma", ("all_channels", "midline", "posterior")),
    ("bandpower_all", ("all_channels",)),
    ("bandpower_beta_gamma", ("all_channels", "midline", "posterior")),
    ("paper_compact", ("all_channels", "midline", "posterior", "frontal_central")),
    ("time_statistical", ("all_channels",)),
)
DEFAULT_SOURCE_RUN = "{participant_lower}_no_ica"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/image_classification/feature_family_top5"
DEFAULT_ORIGINAL_BASELINE = PROJECT_ROOT / "outputs/image_classification/no_ica/results_summary.csv"
DEFAULT_OPTIMIZED_BASELINE = PROJECT_ROOT / "outputs/image_classification/individual_optimized_top15/best_per_participant_target.csv"
CHECKPOINT_FILE = "checkpoint_state.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a fixed-design feature-family experiment command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4, help="Inner GridSearchCV workers (default: 4).")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--original-baseline-csv", type=Path, default=DEFAULT_ORIGINAL_BASELINE)
    parser.add_argument("--optimized-baseline-csv", type=Path, default=DEFAULT_OPTIMIZED_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        args.participants = classifier_search.parse_participants(args.participants)
        args.targets = classifier_search.parse_csv(args.targets)
    except ValueError as error:
        parser.error(str(error))
    invalid_targets = set(args.targets).difference(TARGETS)
    if invalid_targets:
        parser.error(f"Unknown --targets values: {sorted(invalid_targets)}")
    if args.seed != 42:
        parser.error("This fixed experiment requires --seed 42")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer; use an explicit CPU count, not -1")
    return args


def parse_eeg_column(column: str) -> tuple[str, str, str | None] | None:
    """Parse an existing wide EEG column into feature type, channel, and band."""
    if "__" not in column:
        return None
    feature_type, channel = column.split("__", 1)
    if not feature_type.startswith("eeg_") or not channel:
        return None
    band = None
    for prefix in ("eeg_bp_", "eeg_se_"):
        if feature_type.startswith(prefix):
            band = feature_type.removeprefix(prefix)
            break
    return feature_type, channel, band


def feature_definition_frame(columns: list[str]) -> pd.DataFrame:
    """Describe every actual EEG predictor available in a merged table."""
    rows = []
    for column in columns:
        parsed = parse_eeg_column(column)
        if parsed is None:
            continue
        feature_type, channel, band = parsed
        rows.append({"feature_column": column, "parsed_feature_type": feature_type, "parsed_channel": channel, "parsed_frequency_band": band or ""})
    return pd.DataFrame(rows)


def build_feature_families(columns: list[str]) -> dict[str, list[str]]:
    """Build requested families strictly from documented, observed column types."""
    definitions = feature_definition_frame(columns)
    if definitions.empty:
        raise ValueError("No EEG columns using the expected eeg_<type>__<channel> naming scheme were found")
    observed_types = set(definitions.parsed_feature_type)
    required_types = {
        "eeg_sd", "eeg_hm", "eeg_hc", "eeg_mf_hz",
        *(f"eeg_bp_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma")),
        *(f"eeg_se_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma")),
    }
    missing = sorted(required_types - observed_types)
    if missing:
        raise ValueError(f"Cannot construct requested feature families; required observed EEG types are missing: {missing}")
    family_types = {
        "all_eeg": observed_types,
        "hjorth": {"eeg_hm", "eeg_hc"},
        "entropy_all_bands": {f"eeg_se_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma")},
        "entropy_beta_gamma": {"eeg_se_beta", "eeg_se_gamma"},
        "bandpower_all": {f"eeg_bp_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma")},
        "bandpower_beta_gamma": {"eeg_bp_beta", "eeg_bp_gamma"},
        "paper_compact": {"eeg_hm", "eeg_hc", "eeg_se_beta", "eeg_se_gamma", "eeg_bp_beta", "eeg_bp_gamma", "eeg_mf_hz", "eeg_sd"},
        # eeg_se is spectral (not time-domain) entropy, so it is deliberately excluded.
        "time_statistical": {"eeg_sd"},
    }
    families = {
        name: definitions.loc[definitions.parsed_feature_type.isin(types), "feature_column"].tolist()
        for name, types in family_types.items()
    }
    empty = sorted(name for name, family_columns in families.items() if not family_columns)
    if empty:
        raise ValueError(f"Cannot construct requested feature families from observed columns: {empty}")
    return families


def subset_family_columns(family_columns: list[str], subset: str) -> list[str]:
    """Select one declared channel subset without reordering source columns."""
    if subset not in CHANNEL_SUBSETS:
        raise ValueError(f"Unknown channel subset: {subset}")
    allowed = set(CHANNEL_SUBSETS[subset])
    columns = [column for column in family_columns if parse_eeg_column(column) and parse_eeg_column(column)[1] in allowed]
    if not columns:
        raise ValueError(f"Channel subset {subset} contains no columns for this feature family")
    return columns


def feature_count_requests(available: int) -> list[str]:
    """Return valid, unique SelectKBest requests; ``all`` replaces k=N."""
    if available < 1:
        raise ValueError("Feature family must contain at least one column")
    requests: list[str] = []
    seen: set[int] = set()
    for request in FEATURE_REQUESTS:
        if request == "all":
            requests.append(request)
            continue
        resolved = min(int(request), available)
        if resolved != available and resolved not in seen:
            requests.append(request)
            seen.add(resolved)
    return requests


def prepare_family_data(table: pd.DataFrame, participant: str, target: str, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return finite existing family predictors and unchanged binary labels."""
    target_column = classifier_search.baseline.TARGET_COLUMNS[target]
    missing = sorted(set(columns + [target_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{participant} missing requested family columns: {missing}")
    numeric = table[columns].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[target_column], errors="coerce")
    usable = numeric.notna().all(axis=1) & ratings.notna()
    features = numeric.loc[usable].reset_index(drop=True)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"{participant} family predictors contain infinity")
    return features, classifier_search.baseline.label_ratings(ratings.loc[usable]).reset_index(drop=True)


def evaluate_configuration(participant: str, table: pd.DataFrame, target: str, family: str, channel_subset: str, columns: list[str], classifier: str, request: str, seed: int, n_jobs: int) -> dict[str, Any]:
    """Evaluate one feature representation with nested individual CV."""
    x, y = prepare_family_data(table, participant, target, columns)
    folds = classifier_search.baseline.valid_stratified_splits(y, 5, f"{participant} {target} {family}")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    feature_count: int | str = "all" if request == "all" else min(int(request), x.shape[1])
    balanced_scores: list[float] = []
    accuracy_scores: list[float] = []
    selected_parameters: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer.split(x, y), start=1):
        y_train, y_test = y.iloc[train].reset_index(drop=True), y.iloc[test].reset_index(drop=True)
        inner_count = classifier_search.baseline.valid_stratified_splits(y_train, min(3, folds), f"{participant} outer fold {fold} inner CV")
        inner = list(StratifiedKFold(n_splits=inner_count, shuffle=True, random_state=seed + fold).split(x.iloc[train], y_train))
        parameters, inner_score, predicted = classifier_search.fit_outer_fold(x.iloc[train], y_train, x.iloc[test], model=classifier, feature_count=feature_count, inner_splits=inner, seed=seed, n_jobs=n_jobs)
        balanced_scores.append(float(balanced_accuracy_score(y_test, predicted)))
        accuracy_scores.append(float(accuracy_score(y_test, predicted)))
        selected_parameters.append({"fold": fold, "inner_best_balanced_accuracy": inner_score, "parameters": parameters})
    return {"participant": participant, "target": target, "feature_family": family, "channel_subset": channel_subset, "classifier": classifier, "feature_count": feature_count, "feature_count_request": request, "available_feature_count": len(columns), "outer_folds": folds, "mean_balanced_accuracy": float(np.mean(balanced_scores)), "std_balanced_accuracy": float(np.std(balanced_scores, ddof=0)), "mean_accuracy": float(np.mean(accuracy_scores)), "std_accuracy": float(np.std(accuracy_scores, ddof=0)), "best_hyperparameters": json.dumps(selected_parameters, sort_keys=True), "seed": seed, "gridsearch_n_jobs": n_jobs}


def load_optimized_baseline(path: Path, participants: list[str], targets: list[str]) -> pd.DataFrame:
    """Load one existing optimized-classifier result per requested pair."""
    if not path.is_file():
        raise FileNotFoundError(f"Optimized classifier baseline missing: {path}")
    frame = pd.read_csv(path)
    required = {"participant", "target", "modality", "classifier", "feature_count", "balanced_accuracy", "accuracy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Optimized baseline missing columns: {sorted(missing)}")
    frame["participant"] = frame.participant.astype(str).str.upper()
    subset = frame[frame.participant.isin(participants) & frame.target.isin(targets)].copy()
    expected = {(participant, target) for participant in participants for target in targets}
    observed = set(zip(subset.participant, subset.target))
    if expected != observed or subset.duplicated(["participant", "target"]).any():
        raise ValueError("Optimized baseline must contain exactly one row for every requested participant-target pair")
    return subset.rename(columns={"balanced_accuracy": "optimized_classifier_balanced_accuracy", "accuracy": "optimized_classifier_accuracy", "modality": "optimized_classifier_modality", "classifier": "optimized_classifier", "feature_count": "optimized_classifier_feature_count"})[["participant", "target", "optimized_classifier_balanced_accuracy", "optimized_classifier_accuracy", "optimized_classifier_modality", "optimized_classifier", "optimized_classifier_feature_count"]]


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    """Select outer-CV winners by balanced accuracy, with deterministic ties."""
    ranked = results.sort_values(["participant", "target", "mean_balanced_accuracy", "mean_accuracy", "feature_family", "channel_subset", "classifier", "feature_count_request"], ascending=[True, True, False, False, True, True, True, True])
    return ranked.drop_duplicates(["participant", "target"], keep="first").rename(columns={"mean_balanced_accuracy": "feature_family_balanced_accuracy", "mean_accuracy": "feature_family_accuracy", "feature_family": "winning_feature_family", "channel_subset": "winning_channel_subset", "classifier": "winning_classifier", "feature_count": "winning_feature_count", "best_hyperparameters": "winning_hyperparameters"})


def comparison_frame(original: pd.DataFrame, optimized: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    """Compare each feature-family winner with both pre-existing references."""
    winner_columns = ["participant", "target", "winning_feature_family", "winning_channel_subset", "winning_classifier", "winning_feature_count", "feature_family_balanced_accuracy", "feature_family_accuracy", "winning_hyperparameters"]
    comparison = original.merge(optimized, on=["participant", "target"], validate="one_to_one").merge(winners[winner_columns], on=["participant", "target"], validate="one_to_one")
    comparison["BA_delta_vs_original"] = comparison.feature_family_balanced_accuracy - comparison.previous_balanced_accuracy
    comparison["accuracy_delta_vs_original"] = comparison.feature_family_accuracy - comparison.previous_accuracy
    comparison["BA_delta_vs_optimized_classifier"] = comparison.feature_family_balanced_accuracy - comparison.optimized_classifier_balanced_accuracy
    comparison["accuracy_delta_vs_optimized_classifier"] = comparison.feature_family_accuracy - comparison.optimized_classifier_accuracy
    return comparison


def aggregate_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize comparison results overall and by winning representation fields."""
    rows = []
    for field, label in ((None, "overall"), ("target", "target"), ("winning_feature_family", "feature_family"), ("winning_channel_subset", "channel_subset"), ("winning_classifier", "classifier"), ("winning_feature_count", "feature_count")):
        groups = [("all", comparison)] if field is None else comparison.groupby(field, dropna=False)
        for value, group in groups:
            original_delta, optimized_delta = group.BA_delta_vs_original, group.BA_delta_vs_optimized_classifier
            rows.append({"summary_type": label, "group": value, "count": len(group), "mean_original_BA": group.previous_balanced_accuracy.mean(), "mean_optimized_classifier_BA": group.optimized_classifier_balanced_accuracy.mean(), "mean_feature_family_BA": group.feature_family_balanced_accuracy.mean(), "mean_BA_delta_vs_original": original_delta.mean(), "mean_BA_delta_vs_optimized_classifier": optimized_delta.mean(), "improved_vs_original": int((original_delta > 0).sum()), "tied_vs_original": int((original_delta == 0).sum()), "worsened_vs_original": int((original_delta < 0).sum()), "improved_vs_optimized_classifier": int((optimized_delta > 0).sum()), "tied_vs_optimized_classifier": int((optimized_delta == 0).sum()), "worsened_vs_optimized_classifier": int((optimized_delta < 0).sum())})
    return pd.DataFrame(rows)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a compact CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    """Atomically write a manifest or resume checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def configuration_key(participant: str, target: str, family: str, channel_subset: str, classifier: str, request: str) -> str:
    """Return a stable identity for one completed feature-family configuration."""
    return "|".join((participant, target, family, channel_subset, classifier, request))


def hyperparameter_grids(seed: int) -> dict[str, list[dict[str, list[Any]]]]:
    """Record the reused classifier grids as resume-sensitive scientific settings."""
    return {model: classifier_search.make_pipeline(model, 3, seed)[1] for model in MODELS}


def scientific_settings(args: argparse.Namespace, representation_plan: list[tuple[str, str, list[str]]], output: Path) -> dict[str, Any]:
    """Return settings whose changes would alter the scientific experiment.

    Participant membership is intentionally excluded: a run may be extended to
    more allowed development participants or summarized for a subset later.
    """
    return {
        "targets": args.targets,
        "models": list(MODELS),
        "hyperparameter_grids": hyperparameter_grids(args.seed),
        "feature_requests": list(FEATURE_REQUESTS),
        "channel_subsets": {name: list(channels) for name, channels in CHANNEL_SUBSETS.items()},
        "source_run": DEFAULT_SOURCE_RUN,
        "output_directory": str(output),
        "seed": args.seed,
        "label_definition": "LOW: rating < 4; HIGH: rating >= 4",
        "selection_metric": "balanced_accuracy",
        "outer_cv": "StratifiedKFold up to 5 folds, shuffled, random_state=42",
        "inner_cv": "StratifiedKFold up to 3 folds, shuffled, random_state=42+outer_fold",
        "outer_experiment_loop": "serial",
        "estimator_nested_parallelism": "disabled",
        "feature_representation_plan": {f"{family}|{subset}": columns for family, subset, columns in representation_plan},
    }


def saved_settings_match(saved: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare scientific settings, accepting only participant/n-jobs legacy variance."""
    saved = dict(saved)
    # Earlier runs recorded these execution/cohort fields in ``settings``.
    saved.pop("participants", None)
    saved.pop("gridsearch_n_jobs", None)
    legacy_missing = {"hyperparameter_grids", "channel_subsets", "output_directory", "label_definition"}
    missing = set(expected).difference(saved)
    if missing.difference(legacy_missing):
        return False
    comparable_expected = {key: expected.get(key) for key in saved}
    return saved == comparable_expected


def completed_result_keys(results: pd.DataFrame) -> set[str]:
    """Validate persisted result identities and return their completed keys."""
    required = {"participant", "target", "feature_family", "channel_subset", "classifier", "feature_count_request"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Cannot resume: results_summary.csv missing configuration identity columns: {sorted(missing)}")
    keys = results.apply(lambda row: configuration_key(str(row.participant), str(row.target), str(row.feature_family), str(row.channel_subset), str(row.classifier), str(row.feature_count_request)), axis=1)
    if keys.duplicated().any():
        duplicates = keys[keys.duplicated()].unique().tolist()
        raise ValueError(f"Cannot resume: results_summary.csv contains duplicate completed configurations: {duplicates[:3]}")
    return set(keys)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume the focused feature-representation experiment."""
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    derived_root = args.derived_root.resolve() if args.derived_root.is_absolute() else (PROJECT_ROOT / args.derived_root).resolve()
    original_path = args.original_baseline_csv.resolve() if args.original_baseline_csv.is_absolute() else (PROJECT_ROOT / args.original_baseline_csv).resolve()
    optimized_path = args.optimized_baseline_csv.resolve() if args.optimized_baseline_csv.is_absolute() else (PROJECT_ROOT / args.optimized_baseline_csv).resolve()
    output_exists = output.exists()
    if output_exists and not args.resume:
        raise FileExistsError(f"Output directory already exists; use --resume only for the same interrupted search: {output}")
    tables: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, str] = {}
    for participant in args.participants:
        table, path = classifier_search.baseline.load_participant_table(derived_root, participant, DEFAULT_SOURCE_RUN)
        tables[participant], source_paths[participant] = table, str(path)
    eeg_columns = [column for column in tables[args.participants[0]].columns if parse_eeg_column(column)]
    for participant, table in tables.items():
        participant_columns = [column for column in table.columns if parse_eeg_column(column)]
        if participant_columns != eeg_columns:
            raise ValueError(f"EEG feature schema differs for {participant}; feature-family experiment requires an identical schema")
    families = build_feature_families(eeg_columns)
    definitions = feature_definition_frame(eeg_columns)
    representation_plan = [(family, subset, subset_family_columns(families[family], subset)) for family, subsets in FAMILY_CHANNEL_PLANS for subset in subsets]
    settings = scientific_settings(args, representation_plan, output)
    manifest_path, checkpoint_path, results_path = output / "run_manifest.json", output / CHECKPOINT_FILE, output / "results_summary.csv"
    completed: set[str] = set()
    if args.resume and output_exists:
        if not manifest_path.is_file() or not checkpoint_path.is_file() or not results_path.is_file():
            raise ValueError("Cannot resume: run manifest, checkpoint, or results_summary.csv is missing")
        if not saved_settings_match(json.loads(manifest_path.read_text(encoding="utf-8")).get("settings", {}), settings):
            raise ValueError("Refusing resume: requested settings differ from the saved run")
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not saved_settings_match(state.get("settings", {}), settings):
            raise ValueError("Refusing resume: checkpoint settings differ from the saved run")
        results = pd.read_csv(results_path)
        completed = completed_result_keys(results)
        checkpoint_completed = set(state.get("completed_configurations", []))
        if checkpoint_completed != completed:
            raise ValueError("Cannot resume: checkpoint completed configurations do not match results_summary.csv")
    else:
        output.mkdir(parents=True, exist_ok=True)
        results = pd.DataFrame()
        atomic_write_json({"created_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv), "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "requested_participants": args.participants, "source_merged_tables": source_paths, "original_baseline_csv": str(original_path), "optimized_baseline_csv": str(optimized_path), "settings": settings}, manifest_path)
    original = classifier_search.load_baseline_best(original_path, args.participants, args.targets)
    optimized = load_optimized_baseline(optimized_path, args.participants, args.targets)
    definition_rows = []
    for family, subset, columns in representation_plan:
        subset_definitions = definitions[definitions.feature_column.isin(columns)].copy()
        subset_definitions.insert(0, "channel_subset", subset)
        subset_definitions.insert(0, "feature_family", family)
        definition_rows.append(subset_definitions)
    atomic_write_csv(pd.concat(definition_rows, ignore_index=True), output / "feature_family_definition.csv")
    planned = [(participant, target, family, subset, model, request, columns) for participant in args.participants for target in args.targets for family, subset, columns in representation_plan for model in MODELS for request in feature_count_requests(len(columns))]
    print(f"Detected EEG naming scheme: eeg_<feature_type>__<channel>; {len(eeg_columns)} predictors")
    for feature_type, group in definitions.groupby("parsed_feature_type", sort=True):
        print(f"Detected feature type: {feature_type} | channels={','.join(group.parsed_channel.tolist())}")
    for family, subset, columns in representation_plan:
        print(f"Feature mapping: {family} | {subset} | {len(columns)} columns")
    print(f"CPU parallelism: GridSearchCV n_jobs={args.n_jobs}")
    print("Outer experiment loop: serial")
    print("Estimator nested parallelism: disabled")
    for number, (participant, target, family, subset, model, request, columns) in enumerate(planned, start=1):
        key = configuration_key(participant, target, family, subset, model, request)
        if key in completed:
            continue
        if args.progress:
            resolved = "all" if request == "all" else min(int(request), len(columns))
            print(f"[{number}/{len(planned)}] {participant} | {target} | {family} | {subset} | {model} | k={resolved}", flush=True)
        row = evaluate_configuration(participant, tables[participant], target, family, subset, columns, model, request, args.seed, args.n_jobs)
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        completed.add(key)
        atomic_write_csv(results, results_path)
        atomic_write_json({"settings": settings, "completed_configurations": sorted(completed)}, checkpoint_path)
    requested_results = results[results.participant.astype(str).str.upper().isin(args.participants) & results.target.isin(args.targets)].copy()
    winners = choose_best(requested_results)
    comparison = comparison_frame(original, optimized, winners)
    winner_columns = ["participant", "target", "winning_feature_family", "winning_channel_subset", "winning_classifier", "winning_feature_count", "feature_family_balanced_accuracy", "feature_family_accuracy", "winning_hyperparameters"]
    atomic_write_csv(winners[winner_columns], output / "best_per_participant_target.csv")
    atomic_write_csv(comparison, output / "baseline_comparison.csv")
    atomic_write_csv(aggregate_summary(comparison), output / "aggregate_summary.csv")
    return {"output": output, "configurations": len(planned), "feature_families": families, "representation_plan": representation_plan}


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
