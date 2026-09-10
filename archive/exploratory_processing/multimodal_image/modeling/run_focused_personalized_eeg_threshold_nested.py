#!/usr/bin/env python3
"""Nested probability-threshold evaluation for the frozen focused EEG search.

This evaluation keeps the focused classifier grids and EEG representations
unchanged.  Candidate selection, hyperparameter tuning, and threshold choice
use only each outer fold's training partition.  It is deliberately an
evaluation-only run: it does not create frozen transfer models.
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
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import run_focused_personalized_binary_search as focused
import train_classification as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FROZEN_RESULTS = PROJECT_ROOT / "outputs/image_classification/focused_personalized_binary_v1"
DIAGNOSTIC = PROJECT_ROOT / "outputs/image_classification/focused_personalized_weak_participant_diagnostic.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/image_classification/focused_personalized_eeg_threshold_nested_v1"
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.30, 0.701, 0.05))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--participant", type=str.upper, choices=focused.PARTICIPANTS,
                        help="Optionally evaluate one frozen participant, e.g. P01.")
    parser.add_argument("--target", choices=focused.TARGETS,
                        help="Optionally evaluate only valence or arousal.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Print selected candidate, threshold, and both outer-fold BAs.")
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("The frozen design requires --seed 42")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer")
    return args


def candidate_plan() -> list[dict[str, Any]]:
    """Return exactly the completed focused EEG candidate space (2,070 rows)."""
    return focused.planned_configurations(list(focused.PARTICIPANTS), list(focused.TARGETS))


def candidate_identity(config: dict[str, Any] | pd.Series) -> tuple[str, str, str, str]:
    return (str(config["classifier"]), str(config["feature_family"]), str(config["channel_subset"]), str(config["feature_count_request"]))


def format_elapsed(seconds: float) -> str:
    """Format a monotonic elapsed duration for terminal progress output."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_candidate_progress(
    participant: str, participant_index: int, participant_total: int, target: str,
    outer_fold: int, outer_folds: int, candidate_index: int, candidate_total: int,
    config: dict[str, Any], elapsed_seconds: float,
) -> str:
    """Return one compact, immediately printable candidate-progress line."""
    return (
        f"{participant} [{participant_index}/{participant_total}] | {target} | "
        f"fold {outer_fold}/{outer_folds} | candidate {candidate_index}/{candidate_total} | "
        f"{config['classifier']} | {config['feature_family']} | k={config['feature_count_request']} | "
        f"elapsed {format_elapsed(elapsed_seconds)}"
    )


def probability_pipeline(model: str, feature_count: int | str, seed: int):
    """Use the frozen pipeline with probability output enabled for SVM only.

    ``SVC(probability=True)`` changes no classifier/search hyperparameter and
    performs its probability calibration only on the training partition passed
    to ``fit``.  It is necessary because fixed and selected thresholds operate
    on P(HIGH), not decision scores.
    """
    pipeline = focused.make_pipeline(model, feature_count, seed)
    if model == "svm":
        pipeline.named_steps["classifier"].set_params(probability=True)
    return pipeline


def tune_and_probability(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, model: str,
    feature_count: int | str, seed: int, cv: list[tuple[np.ndarray, np.ndarray]], n_jobs: int,
) -> tuple[Any, dict[str, Any], float, np.ndarray]:
    """Tune one frozen candidate using only its supplied training partition."""
    search = GridSearchCV(
        probability_pipeline(model, feature_count, seed), focused.model_grid(model),
        scoring="balanced_accuracy", cv=cv, n_jobs=n_jobs, refit=True, error_score="raise",
    )
    search.fit(x_train, y_train)
    probability = search.predict_proba(x_test)[:, 1]
    return search.best_estimator_, search.best_params_, float(search.best_score_), probability


def threshold_predictions(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probability) >= threshold).astype(int)


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Select BA; ties use nearest 0.50 and then the lower threshold."""
    scored = [(threshold, float(balanced_accuracy_score(y, threshold_predictions(probability, threshold)))) for threshold in THRESHOLDS]
    best_score = max(score for _, score in scored)
    tied = [threshold for threshold, score in scored if np.isclose(score, best_score, rtol=0, atol=1e-12)]
    selected = min(tied, key=lambda threshold: (abs(threshold - 0.50), threshold))
    return selected, best_score


def sensitivity_specificity(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return positive-class sensitivity and negative-class specificity."""
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return float(tp / (tp + fn)), float(tn / (tn + fp))


def select_candidate_with_oof_probabilities(
    configs: list[dict[str, Any]], x_by_candidate: dict[tuple[str, str, str, str], pd.DataFrame],
    y: pd.Series, selection_splits: list[tuple[np.ndarray, np.ndarray]], seed: int, n_jobs: int,
    common: dict[str, Any], progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Return the fixed-0.50 inner winner and its fully OOF probabilities.

    Each selection-validation probability comes from a model whose candidate
    hyperparameters were tuned using a further CV split of selection training
    data.  Candidate selection therefore cannot see the outer test fold.
    """
    records: list[dict[str, Any]] = []
    oof: dict[tuple[str, str, str, str], np.ndarray] = {}
    oof_folds: dict[tuple[str, str, str, str], list[float]] = {}
    for candidate_index, config in enumerate(configs, start=1):
        if progress_callback is not None:
            progress_callback(candidate_index, len(configs), config)
        identity = candidate_identity(config)
        oof[identity] = np.full(len(y), np.nan)
        oof_folds[identity] = []
        x = x_by_candidate[identity]
        count: int | str = "all" if config["feature_count_request"] == "all" else int(config["feature_count_request"])
        for selection_fold, (train, validation) in enumerate(selection_splits, start=1):
            y_train = y.iloc[train].reset_index(drop=True)
            tuning_count = baseline.valid_stratified_splits(y_train, min(3, len(selection_splits)), "threshold selection tuning CV")
            tuning_splits = list(StratifiedKFold(n_splits=tuning_count, shuffle=True, random_state=seed + 100 * common["outer_fold"] + selection_fold).split(x.iloc[train], y_train))
            _, parameters, tuning_ba, probability = tune_and_probability(
                x.iloc[train], y_train, x.iloc[validation], config["classifier"], count, seed,
                tuning_splits, n_jobs,
            )
            oof[identity][validation] = probability
            oof_folds[identity].append(float(balanced_accuracy_score(y.iloc[validation], threshold_predictions(probability, 0.50))))
            records.append({**common, **config, "selection_fold": selection_fold,
                            "tuning_best_parameters": json.dumps(parameters, sort_keys=True),
                            "tuning_inner_balanced_accuracy": tuning_ba,
                            "selection_validation_ba_fixed_050": oof_folds[identity][-1]})
    ranking: list[dict[str, Any]] = []
    for config in configs:
        identity = candidate_identity(config)
        probability = oof[identity]
        if np.isnan(probability).any():
            raise RuntimeError("Inner OOF probability construction was incomplete")
        values = oof_folds[identity]
        ranking.append({**config, "selection_oof_ba_fixed_050": float(balanced_accuracy_score(y, threshold_predictions(probability, .50))),
                        "selection_oof_fold_ba_sd_fixed_050": float(np.std(values, ddof=1)),
                        "selection_oof_probability": probability})
    ranked = sorted(ranking, key=lambda row: (
        -row["selection_oof_ba_fixed_050"], row["selection_oof_fold_ba_sd_fixed_050"],
        focused.complexity(pd.Series({**row, "feature_count_resolved": "all" if row["feature_count_request"] == "all" else int(row["feature_count_request"])})),
    ))
    winner = ranked[0]
    for record in records:
        record["selected_candidate"] = candidate_identity(record) == candidate_identity(winner)
    return winner, winner.pop("selection_oof_probability"), records


def prepare_outer_data(table: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    columns = focused.feature_columns(config["feature_family"], config["channel_subset"])
    x, y, _ = focused.prepare_family_data(table, config["participant"], config["target"], columns)
    return x, y


def evaluate_participant_target(
    participant: str, target: str, table: pd.DataFrame, seed: int, n_jobs: int,
    participant_index: int = 1, participant_total: int = 1, progress: bool = False, verbose: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate both thresholds on identical untouched outer-fold probabilities."""
    configs = [config for config in candidate_plan() if config["participant"] == participant and config["target"] == target]
    first_x, y = prepare_outer_data(table, configs[0])
    outer_count = baseline.valid_stratified_splits(y, 5, f"{participant} {target} threshold outer CV")
    outer = list(StratifiedKFold(n_splits=outer_count, shuffle=True, random_state=seed).split(first_x, y))
    x_by_candidate = {candidate_identity(config): prepare_outer_data(table, config)[0] for config in configs}
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for outer_fold, (train, test) in enumerate(outer, start=1):
        fold_started = time.monotonic()
        y_train, y_test = y.iloc[train].reset_index(drop=True), y.iloc[test].to_numpy(dtype=int)
        selection_count = baseline.valid_stratified_splits(y_train, min(3, outer_count), f"{participant} {target} threshold selection CV")
        selection_splits = list(StratifiedKFold(n_splits=selection_count, shuffle=True, random_state=seed + outer_fold).split(first_x.iloc[train], y_train))
        common = {"participant": participant, "target": target, "outer_fold": outer_fold, "outer_folds": outer_count,
                  "outer_train_size": len(train), "outer_test_size": len(test), "low_count": int((y == 0).sum()), "high_count": int((y == 1).sum())}
        def report_candidate(candidate_index: int, candidate_total: int, config: dict[str, Any]) -> None:
            if progress:
                print(format_candidate_progress(
                    participant, participant_index, participant_total, target, outer_fold, outer_count,
                    candidate_index, candidate_total, config, time.monotonic() - fold_started,
                ), flush=True)
        winner, oof_probability, rows = select_candidate_with_oof_probabilities(
            configs, {key: value.iloc[train].reset_index(drop=True) for key, value in x_by_candidate.items()},
            y_train, selection_splits, seed, n_jobs, common, report_candidate if progress else None,
        )
        threshold, threshold_oof_ba = choose_threshold(y_train.to_numpy(dtype=int), oof_probability)
        count: int | str = "all" if winner["feature_count_request"] == "all" else int(winner["feature_count_request"])
        final_cv = list(StratifiedKFold(n_splits=selection_count, shuffle=True, random_state=seed + outer_fold).split(x_by_candidate[candidate_identity(winner)].iloc[train], y_train))
        fitted, final_parameters, final_inner_ba, outer_probability = tune_and_probability(
            x_by_candidate[candidate_identity(winner)].iloc[train], y_train,
            x_by_candidate[candidate_identity(winner)].iloc[test], winner["classifier"], count, seed, final_cv, n_jobs,
        )
        fixed_prediction = threshold_predictions(outer_probability, .50)
        optimized_prediction = threshold_predictions(outer_probability, threshold)
        fixed_ba = float(balanced_accuracy_score(y_test, fixed_prediction))
        optimized_ba = float(balanced_accuracy_score(y_test, optimized_prediction))
        fixed_sensitivity, fixed_specificity = sensitivity_specificity(y_test, fixed_prediction)
        optimized_sensitivity, optimized_specificity = sensitivity_specificity(y_test, optimized_prediction)
        if verbose:
            print(
                f"{participant} [{participant_index}/{participant_total}] | {target} | fold {outer_fold}/{outer_count} | "
                f"selected {winner['classifier']} | {winner['feature_family']} | k={winner['feature_count_request']} | "
                f"threshold {threshold:.2f} | BA 0.50={fixed_ba:.3f} | BA optimized={optimized_ba:.3f} | "
                f"elapsed {format_elapsed(time.monotonic() - fold_started)}",
                flush=True,
            )
        fold_rows.append({**common, **{key: winner[key] for key in ("classifier", "feature_family", "channel_subset", "feature_count_request", "analysis_tier", "available_feature_count")},
                          "selected_threshold": threshold, "selection_oof_ba_fixed_050": winner["selection_oof_ba_fixed_050"],
                          "selection_oof_fold_ba_sd_fixed_050": winner["selection_oof_fold_ba_sd_fixed_050"], "selection_oof_ba_at_selected_threshold": threshold_oof_ba,
                          "final_refit_best_parameters": json.dumps(final_parameters, sort_keys=True), "final_refit_inner_balanced_accuracy": final_inner_ba,
                          "fixed_threshold_ba": fixed_ba, "optimized_threshold_ba": optimized_ba,
                          "fixed_threshold_sensitivity": fixed_sensitivity, "fixed_threshold_specificity": fixed_specificity,
                          "optimized_threshold_sensitivity": optimized_sensitivity, "optimized_threshold_specificity": optimized_specificity})
        for source_index, probability, fixed, optimized, truth in zip(test, outer_probability, fixed_prediction, optimized_prediction, y_test):
            prediction_rows.append({**common, "selected_classifier": winner["classifier"], "selected_feature_family": winner["feature_family"],
                                    "selected_channel_subset": winner["channel_subset"], "selected_feature_count_request": winner["feature_count_request"],
                                    "selected_threshold": threshold, "source_trial_index": int(source_index), "true_label": int(truth), "probability_high": float(probability),
                                    "fixed_threshold_prediction": int(fixed), "optimized_threshold_prediction": int(optimized)})
        selection_rows.extend(rows)
    return fold_rows, prediction_rows, selection_rows


def summarize_participants(folds: pd.DataFrame) -> pd.DataFrame:
    return folds.groupby(["participant", "target"], as_index=False).agg(
        outer_folds=("outer_fold", "size"), low_count=("low_count", "first"), high_count=("high_count", "first"),
        mean_ba_fixed_050=("fixed_threshold_ba", "mean"), mean_ba_optimized_threshold=("optimized_threshold_ba", "mean"),
        fold_sd_fixed_050=("fixed_threshold_ba", "std"), fold_sd_optimized_threshold=("optimized_threshold_ba", "std"),
        median_selected_threshold=("selected_threshold", "median"),
    ).assign(ba_change_optimized_minus_fixed=lambda frame: frame.mean_ba_optimized_threshold - frame.mean_ba_fixed_050)


def cohort_rows(participants: pd.DataFrame, diagnostic: pd.DataFrame) -> pd.DataFrame:
    merged = participants.merge(diagnostic[["participant", "target", "severe_class_imbalance", "threshold_optimization_plausible", "diagnostic_classification"]], on=["participant", "target"], how="left", validate="one_to_one")
    groups = [("all", pd.Series(True, index=merged.index)), ("severe_imbalance", merged.severe_class_imbalance), ("non_severe_imbalance", ~merged.severe_class_imbalance),
              ("threshold_plausible", merged.threshold_optimization_plausible), ("likely_weak_no_separable_signal", merged.diagnostic_classification.eq("likely weak/no separable signal"))]
    rows = []
    for target, frame in merged.groupby("target", sort=True):
        for name, mask in groups:
            group = frame.loc[mask.loc[frame.index]]
            if group.empty:
                continue
            change = group.ba_change_optimized_minus_fixed
            rows.append({"target": target, "stratum": name, "participants": len(group),
                         "mean_participant_ba_fixed_050": group.mean_ba_fixed_050.mean(), "mean_participant_ba_optimized": group.mean_ba_optimized_threshold.mean(),
                         "median_participant_ba_fixed_050": group.mean_ba_fixed_050.median(), "median_participant_ba_optimized": group.mean_ba_optimized_threshold.median(),
                         "fixed_ge_060_count": int((group.mean_ba_fixed_050 >= .60).sum()), "fixed_ge_060_fraction": float((group.mean_ba_fixed_050 >= .60).mean()),
                         "optimized_ge_060_count": int((group.mean_ba_optimized_threshold >= .60).sum()), "optimized_ge_060_fraction": float((group.mean_ba_optimized_threshold >= .60).mean()),
                         "fixed_ge_065_count": int((group.mean_ba_fixed_050 >= .65).sum()), "fixed_ge_065_fraction": float((group.mean_ba_fixed_050 >= .65).mean()),
                         "optimized_ge_065_count": int((group.mean_ba_optimized_threshold >= .65).sum()), "optimized_ge_065_fraction": float((group.mean_ba_optimized_threshold >= .65).mean()),
                         "median_fold_sd_fixed_050": group.fold_sd_fixed_050.median(), "median_fold_sd_optimized": group.fold_sd_optimized_threshold.median(),
                         "improved_count": int((change > 1e-12).sum()), "unchanged_count": int(np.isclose(change, 0, atol=1e-12).sum()), "worsened_count": int((change < -1e-12).sum()),
                         "improved_by_at_least_003_count": int((change >= .03).sum())})
    return pd.DataFrame(rows)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def dry_run_summary(participant: str | None = None, target: str | None = None) -> dict[str, Any]:
    completed = pd.read_csv(FROZEN_RESULTS / "complete_search_results.csv")
    if participant is not None:
        completed = completed.loc[completed.participant.eq(participant)]
    if target is not None:
        completed = completed.loc[completed.target.eq(target)]
    outer_by_target = completed.groupby(["participant", "target"])["outer_folds"].first().groupby("target").sum().to_dict()
    # Per representation, three tuning folds require 57 estimator fits
    # (LogReg 6x3, SVM 12x3, GNB 1x3), plus 3 GridSearch refits.
    valence_folds = int(outer_by_target.get("valence", 0))
    arousal_folds = int(outer_by_target.get("arousal", 0))
    selection_fit_count = int(valence_folds * 3 * 9 * (57 + 3) + arousal_folds * 3 * 6 * (57 + 3))
    outer_folds = int(sum(outer_by_target.values()))
    candidate_count = len([config for config in candidate_plan() if (participant is None or config["participant"] == participant) and (target is None or config["target"] == target)])
    return {"candidate_configurations": candidate_count, "outer_folds": outer_folds,
            "outer_folds_by_target": {key: int(value) for key, value in outer_by_target.items()},
            "selection_candidate_fold_evaluations": int(valence_folds * 3 * 27 + arousal_folds * 3 * 18),
            "selection_gridsearch_estimator_fits": selection_fit_count,
            "final_refit_estimator_fits_range": [outer_folds * 4, outer_folds * 37],
            "thresholds": list(THRESHOLDS), "output_dir": str(DEFAULT_OUTPUT)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    preview = dry_run_summary(args.participant, args.target)
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    preview["output_dir"] = str(output)
    if args.dry_run:
        return {"dry_run": True, **preview}
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if not DIAGNOSTIC.exists():
        raise FileNotFoundError(f"Required read-only diagnostic is missing: {DIAGNOSTIC}")
    frozen_manifest = json.loads((FROZEN_RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    diagnostic = pd.read_csv(DIAGNOSTIC)
    output.mkdir(parents=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "python_version": sys.version,
                "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "frozen_focused_results": str(FROZEN_RESULTS), "frozen_source_merged_tables": frozen_manifest["source_merged_tables"],
                "diagnostic": str(DIAGNOSTIC), "seed": args.seed, "thresholds": list(THRESHOLDS), "participant_restriction": args.participant, "target_restriction": args.target,
                "leakage_control": "outer test untouched; selection OOF probabilities use tuning CV within each selection-training split; threshold selected only from selected candidate OOF probabilities",
                "candidate_selection": "maximum selection-OOF BA at fixed 0.50; tie: lower selection-fold BA SD; then focused complexity", "probability_condition": "SVC probability=True; calibration fit only within training partitions"}
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True); handle.write("\n")
    folds: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []; selection: list[dict[str, Any]] = []
    participant_items = [(participant, path) for participant, path in frozen_manifest["source_merged_tables"].items() if args.participant is None or participant == args.participant]
    targets = (args.target,) if args.target is not None else focused.TARGETS
    for index, (participant, path) in enumerate(participant_items, start=1):
        table = pd.read_csv(path)
        for target in targets:
            evaluated = evaluate_participant_target(
                participant, target, table, args.seed, args.n_jobs, index, len(participant_items), args.progress, args.verbose,
            )
            folds.extend(evaluated[0]); predictions.extend(evaluated[1]); selection.extend(evaluated[2])
    fold_frame = pd.DataFrame(folds); participant_frame = summarize_participants(fold_frame); cohort = cohort_rows(participant_frame, diagnostic)
    atomic_csv(fold_frame, output / "outer_fold_results.csv")
    atomic_csv(pd.DataFrame(predictions), output / "outer_fold_probabilities.csv")
    atomic_csv(pd.DataFrame(selection), output / "inner_selection_candidate_results.csv")
    atomic_csv(participant_frame, output / "participant_threshold_summary.csv")
    atomic_csv(cohort, output / "cohort_threshold_comparison.csv")
    return {"dry_run": False, "output": output, **preview}


if __name__ == "__main__":
    try:
        print(run(parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
