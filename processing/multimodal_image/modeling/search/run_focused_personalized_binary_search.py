#!/usr/bin/env python3
"""Frozen, focused nested-CV personalized Image binary-classifier search.

This implements the design approved from the v3 exploratory audit.  It reads
existing no-ICA merged Image tables, never modifies ``data/``, and writes a new
versioned result directory only when not invoked with ``--dry-run``.
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

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from processing.multimodal_image.modeling.core import train_classification as baseline
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/image_classification/focused_personalized_binary_v1"
DEFAULT_EXTENSION_OUTPUT = DEFAULT_OUTPUT / "face_multimodal_extension_v1"
DEFAULT_SOURCE_RUN = "{participant_lower}_no_ica"
TARGETS = ("valence", "arousal")
MODALITIES = ("eeg", "face", "multimodal")
PARTICIPANTS = tuple(f"P{number:02d}" for number in range(1, 47))
POSSIBLE_EXCLUSION = {"P03", "P13", "P28"}
REVIEW = {"P01", "P02", "P04", "P05", "P06", "P07", "P08", "P09", "P12", "P20", "P21", "P27", "P29", "P35", "P36", "P37", "P39"}
LABEL_DEFINITION = "LOW: rating < 4; HIGH: rating >= 4"
COMPETITIVE_DELTA = 0.02


def parse_participants(value: str) -> list[str]:
    """Validate an ordered subset of the frozen 46-person cohort."""
    participants = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not participants or len(participants) != len(set(participants)):
        raise ValueError("--participants must be a non-empty unique list")
    unknown = sorted(set(participants).difference(PARTICIPANTS))
    if unknown:
        raise ValueError(f"Participants outside the frozen cohort: {unknown}")
    return participants


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(PARTICIPANTS))
    parser.add_argument("--targets", default="valence,arousal")
    parser.add_argument("--modalities", default="eeg", help="Comma-separated: eeg,face,multimodal. Default preserves the completed EEG design.")
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eeg-results-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Immutable completed EEG result directory used when building all-modality files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.participants = parse_participants(args.participants)
    except ValueError as error:
        parser.error(str(error))
    args.targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if not args.targets or set(args.targets).difference(TARGETS):
        parser.error("--targets must contain valence and/or arousal")
    args.modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    if not args.modalities or len(args.modalities) != len(set(args.modalities)) or set(args.modalities).difference(MODALITIES):
        parser.error("--modalities must contain unique values from eeg,face,multimodal")
    if "eeg" in args.modalities and len(args.modalities) > 1:
        parser.error("The completed EEG branch is immutable; run only --modalities eeg or a face/multimodal extension.")
    if args.modalities != ["eeg"] and args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_EXTENSION_OUTPUT
    if args.seed != 42:
        parser.error("The frozen design requires --seed 42")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer")
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume cannot be combined")
    return args


def qc_status(participant: str) -> str:
    if participant in POSSIBLE_EXCLUSION:
        return "POSSIBLE EXCLUSION"
    if participant in REVIEW:
        return "REVIEW"
    return "KEEP"


def feature_columns(family: str, subset: str, modality: str = "eeg") -> list[str]:
    """Return the audit-frozen EEG representation and declared channel subset."""
    if modality == "face":
        if family != "face_geometry" or subset != "all_face":
            raise ValueError("Face supports only the frozen face_geometry/all_face representation")
        return baseline.FACE_COLUMNS.copy()
    channels = {
        "all_channels": baseline.EEG_CHANNELS,
        "frontal_central": ("Fz", "C3", "Cz", "C4"),
    }
    if subset not in channels:
        raise ValueError(f"Unknown frozen channel subset: {subset}")
    types = {
        "all_eeg": set(baseline.EEG_FAMILIES),
        "hjorth": {"eeg_hm", "eeg_hc"},
        "time_statistical": {"eeg_sd"},
        "paper_compact": {"eeg_sd", "eeg_hm", "eeg_hc", "eeg_mf_hz", "eeg_bp_beta", "eeg_se_beta", "eeg_bp_gamma", "eeg_se_gamma"},
    }
    if modality == "multimodal":
        multimodal = {
            "all_eeg_plus_face": ("all_eeg", "all_channels"),
            "paper_compact_all_channels_plus_face": ("paper_compact", "all_channels"),
        }
        if family not in multimodal or subset != "all_channels":
            raise ValueError(f"Unknown frozen multimodal representation: {family}/{subset}")
        eeg_family, eeg_subset = multimodal[family]
        return feature_columns(eeg_family, eeg_subset, "eeg") + baseline.FACE_COLUMNS.copy()
    if modality != "eeg" or family not in types:
        raise ValueError(f"Unknown frozen feature family: {family}")
    allowed = set(channels[subset])
    return [column for column in baseline.EEG_COLUMNS if column.split("__", 1)[0] in types[family] and column.split("__", 1)[1] in allowed]


def representation_plan(target: str) -> list[dict[str, Any]]:
    """Return the approved target-specific primary and secondary grid."""
    primary = [
        ("all_eeg", "all_channels", "8", "primary"),
        ("all_eeg", "all_channels", "10", "primary"),
    ]
    if target == "valence":
        primary += [
            ("hjorth", "all_channels", "8", "primary"),
            ("hjorth", "all_channels", "10", "primary"),
            ("time_statistical", "all_channels", "all", "primary"),
        ]
    secondary = [
        ("paper_compact", "all_channels", "8", "secondary"),
        ("paper_compact", "all_channels", "10", "secondary"),
        ("paper_compact", "frontal_central", "8", "secondary"),
        ("paper_compact", "frontal_central", "10", "secondary"),
    ]
    return [
        {"feature_family": family, "channel_subset": subset, "feature_count_request": request,
         "analysis_tier": tier, "available_feature_count": len(feature_columns(family, subset))}
        for family, subset, request, tier in primary + secondary
    ]


def modality_representation_plan(target: str, modality: str) -> list[dict[str, Any]]:
    """Return the approved focused representation plan for one modality."""
    if modality == "eeg":
        return [{**row, "modality": "eeg"} for row in representation_plan(target)]
    if modality == "face":
        plan = [("face_geometry", "all_face", "5", "primary"), ("face_geometry", "all_face", "all", "primary")]
    elif modality == "multimodal":
        plan = [
            ("all_eeg_plus_face", "all_channels", "5", "primary"),
            ("all_eeg_plus_face", "all_channels", "10", "primary"),
            ("all_eeg_plus_face", "all_channels", "20", "primary"),
        ]
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return [{"modality": modality, "feature_family": family, "channel_subset": subset,
             "feature_count_request": request, "analysis_tier": tier,
             "available_feature_count": len(feature_columns(family, subset, modality))}
            for family, subset, request, tier in plan]


def model_grid(model: str, modality: str = "eeg") -> list[dict[str, list[Any]]]:
    """Return the frozen EEG grid or the approved Face/Multimodal SVM grid."""
    if model == "logreg":
        return [{"classifier__C": [0.01, 0.1, 1], "classifier__class_weight": [None, "balanced"]}]
    if model == "svm":
        if modality in {"face", "multimodal"}:
            return [
                {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10], "classifier__class_weight": [None, "balanced"]},
                {"classifier__kernel": ["rbf"], "classifier__C": [1, 10], "classifier__gamma": ["scale", 0.01], "classifier__class_weight": [None, "balanced"]},
            ]
        return [
            {"classifier__kernel": ["linear"], "classifier__C": [1, 10], "classifier__class_weight": [None, "balanced"]},
            {"classifier__kernel": ["rbf"], "classifier__C": [1, 10], "classifier__gamma": [0.001, 0.01], "classifier__class_weight": [None, "balanced"]},
        ]
    if model == "gnb":
        return [{"classifier__var_smoothing": [1e-11]}]
    raise ValueError(f"Unknown frozen classifier: {model}")


def make_pipeline(model: str, feature_count: int | str, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if feature_count != "all":
        steps.append(("selector", SelectKBest(score_func=f_classif, k=int(feature_count))))
    if model == "logreg":
        classifier: Any = LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, random_state=seed)
    elif model == "svm":
        classifier = SVC(random_state=seed)
    elif model == "gnb":
        classifier = GaussianNB()
    else:
        raise ValueError(f"Unknown frozen classifier: {model}")
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def planned_configurations(participants: list[str], targets: list[str], modalities: list[str] | None = None) -> list[dict[str, Any]]:
    """Materialize the exact frozen configuration identities without training."""
    modalities = modalities or ["eeg"]
    return [
        {"participant": participant, "target": target, "classifier": classifier, **representation}
        for participant in participants for target in targets for modality in modalities
        for representation in modality_representation_plan(target, modality) for classifier in ("logreg", "svm", "gnb")
    ]


def configuration_key(row: dict[str, Any]) -> str:
    return "|".join(str(row[name]) for name in ("participant", "target", "modality", "classifier", "feature_family", "channel_subset", "feature_count_request"))


def prepare_family_data(table: pd.DataFrame, participant: str, target: str, columns: list[str]) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    target_column = baseline.TARGET_COLUMNS[target]
    required = set(columns + [target_column])
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{participant} missing frozen feature columns: {missing}")
    numeric = table[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.notna().all(axis=1) & pd.to_numeric(table[target_column], errors="coerce").notna()
    features = numeric.loc[usable].reset_index(drop=True)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"{participant} {target}: non-finite predictors")
    labels = baseline.label_ratings(table.loc[usable, target_column]).reset_index(drop=True)
    return features, labels, np.flatnonzero(usable.to_numpy())


def selected_names(fitted: Pipeline, columns: list[str]) -> list[str]:
    if "selector" not in fitted.named_steps:
        return columns
    support = fitted.named_steps["selector"].get_support()
    return [name for name, keep in zip(columns, support) if keep]


def fit_fold(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, model: str, modality: str, feature_count: int | str, seed: int, inner: list[tuple[np.ndarray, np.ndarray]], n_jobs: int) -> tuple[Pipeline, dict[str, Any], float, np.ndarray]:
    search = GridSearchCV(make_pipeline(model, feature_count, seed), model_grid(model, modality), scoring="balanced_accuracy", cv=inner, n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_), search.predict(x_test)


def evaluate_configuration(config: dict[str, Any], table: pd.DataFrame, seed: int, n_jobs: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the same individual outer/inner stratified nested CV as prior searches."""
    columns = feature_columns(config["feature_family"], config["channel_subset"], config.get("modality", "eeg"))
    x, y, source_indices = prepare_family_data(table, config["participant"], config["target"], columns)
    folds = baseline.valid_stratified_splits(y, 5, f"{config['participant']} {config['target']} focused search")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    count: int | str = "all" if config["feature_count_request"] == "all" else int(config["feature_count_request"])
    fold_rows: list[dict[str, Any]] = []; parameter_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []; prediction_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(outer.split(x, y), start=1):
        y_train, y_test = y.iloc[train].reset_index(drop=True), y.iloc[test].reset_index(drop=True)
        inner_folds = baseline.valid_stratified_splits(y_train, min(3, folds), f"{config['participant']} fold {fold} inner CV")
        inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold).split(x.iloc[train], y_train))
        fitted, params, inner_ba, predicted = fit_fold(x.iloc[train], y_train, x.iloc[test], config["classifier"], config.get("modality", "eeg"), count, seed, inner, n_jobs)
        ba = float(balanced_accuracy_score(y_test, predicted))
        common = {**config, "fold": fold, "feature_count_resolved": count, "outer_folds": folds}
        fold_rows.append({**common, "sample_size": len(y), "train_size": len(train), "test_size": len(test), "low_count": int((y == 0).sum()), "high_count": int((y == 1).sum()), "balanced_accuracy": ba, "accuracy": float(accuracy_score(y_test, predicted))})
        names = selected_names(fitted, columns)
        parameter_rows.append({**common, "best_parameters": json.dumps(params, sort_keys=True), "inner_best_balanced_accuracy": inner_ba, "selected_feature_names": json.dumps(names)})
        feature_rows.extend({**common, "feature": name, "selected": name in names} for name in columns)
        prediction_rows.extend({**common, "source_trial_index": int(source_indices[index]), "true_label": int(y.iloc[index]), "predicted_label": int(prediction)} for index, prediction in zip(test, predicted))
    return fold_rows, parameter_rows, feature_rows, prediction_rows


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "target", "modality", "classifier", "feature_family", "channel_subset", "feature_count_request", "analysis_tier", "available_feature_count", "feature_count_resolved"]
    return folds.groupby(keys, dropna=False).agg(outer_folds=("fold", "size"), mean_outer_cv_balanced_accuracy=("balanced_accuracy", "mean"), fold_ba_sd=("balanced_accuracy", "std"), outer_fold_ba_values=("balanced_accuracy", lambda values: json.dumps([float(value) for value in values])), mean_accuracy=("accuracy", "mean"), low_count=("low_count", "first"), high_count=("high_count", "first"), sample_size=("sample_size", "first")).reset_index()


def complexity(row: pd.Series) -> tuple[int, int, int]:
    return (int(row.feature_count_resolved) if row.feature_count_resolved != "all" else int(row.available_feature_count), int(row.available_feature_count), {"gnb": 0, "logreg": 1, "svm": 2}[row.classifier])


def ranked_top3(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return absolute BA top-3 and competitive, diversity-aware transfer top-3."""
    if "modality" not in summary:
        summary = with_modality(summary, "eeg")
    absolute: list[pd.DataFrame] = []; transfer: list[dict[str, Any]] = []
    for (_, _, _), group in summary.groupby(["participant", "target", "modality"], sort=True):
        ranked = group.sort_values(["mean_outer_cv_balanced_accuracy", "fold_ba_sd"], ascending=[False, True], kind="stable").copy()
        ranked["absolute_rank"] = range(1, len(ranked) + 1)
        absolute.append(ranked.head(3))
        best = ranked.iloc[0]; pool = ranked[ranked.mean_outer_cv_balanced_accuracy >= best.mean_outer_cv_balanced_accuracy - COMPETITIVE_DELTA]
        chosen: list[pd.Series] = []
        for _, candidate in pool.iterrows():
            if len(chosen) == 3: break
            classifiers = {row.classifier for row in chosen}; representations = {(row.feature_family, row.channel_subset) for row in chosen}
            if not chosen:
                reason = "best"
            elif candidate.classifier not in classifiers:
                reason = "diverse_classifier"
            elif (candidate.feature_family, candidate.channel_subset) not in representations:
                reason = "diverse_feature_family"
            else:
                continue
            transfer.append({**candidate.to_dict(), "transfer_rank": len(chosen) + 1, "delta_ba_from_best": candidate.mean_outer_cv_balanced_accuracy - best.mean_outer_cv_balanced_accuracy, "selection_reason": reason})
            chosen.append(candidate)
        for _, candidate in ranked.iterrows():
            if len(chosen) == 3: break
            if any(candidate.name == selected.name for selected in chosen): continue
            transfer.append({**candidate.to_dict(), "transfer_rank": len(chosen) + 1, "delta_ba_from_best": candidate.mean_outer_cv_balanced_accuracy - best.mean_outer_cv_balanced_accuracy, "selection_reason": "next_best_stable"})
            chosen.append(candidate)
    return pd.concat(absolute, ignore_index=True), pd.DataFrame(transfer)


def ranked_image_domain_top3_across_modalities(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select three artifact-backed Image-domain models across all modalities.

    Candidates must already have fitted-model provenance.  Within each
    participant-target, the absolute BA winner is selected first.  Remaining
    competitive candidates (within ``COMPETITIVE_DELTA`` BA) are scanned in BA
    descending, fold-SD ascending order and retained when they add classifier,
    modality, or representation diversity.  Remaining ranks are next-best
    stable candidates.  No model fitting or refitting occurs here.
    """
    required = {"participant", "target", "modality", "classifier", "feature_family", "channel_subset",
                "mean_outer_cv_balanced_accuracy", "fold_ba_sd", "final_model_path"}
    if missing := required.difference(candidates.columns):
        raise ValueError(f"Image-domain candidates missing required columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for (_, _), group in candidates.groupby(["participant", "target"], sort=True):
        ranked = group.sort_values(["mean_outer_cv_balanced_accuracy", "fold_ba_sd"], ascending=[False, True], kind="stable").copy()
        if len(ranked) < 3:
            raise ValueError("Each participant-target requires at least three artifact-backed candidates")
        ranked["absolute_rank"] = range(1, len(ranked) + 1)
        best = ranked.iloc[0]
        competitive = ranked.loc[ranked.mean_outer_cv_balanced_accuracy >= best.mean_outer_cv_balanced_accuracy - COMPETITIVE_DELTA]
        selected: list[pd.Series] = []
        for _, candidate in competitive.iterrows():
            if len(selected) == 3:
                break
            if not selected:
                reason = "best"
            else:
                classifiers = {choice.classifier for choice in selected}
                modalities = {choice.modality for choice in selected}
                representations = {(choice.feature_family, choice.channel_subset) for choice in selected}
                if candidate.classifier not in classifiers:
                    reason = "diverse_classifier"
                elif candidate.modality not in modalities:
                    reason = "diverse_modality"
                elif (candidate.feature_family, candidate.channel_subset) not in representations:
                    reason = "diverse_feature_representation"
                else:
                    continue
            rows.append({**candidate.to_dict(), "transfer_rank": len(selected) + 1,
                         "delta_ba_from_best": candidate.mean_outer_cv_balanced_accuracy - best.mean_outer_cv_balanced_accuracy,
                         "selection_reason": reason})
            selected.append(candidate)
        for _, candidate in ranked.iterrows():
            if len(selected) == 3:
                break
            if any(candidate.name == choice.name for choice in selected):
                continue
            rows.append({**candidate.to_dict(), "transfer_rank": len(selected) + 1,
                         "delta_ba_from_best": candidate.mean_outer_cv_balanced_accuracy - best.mean_outer_cv_balanced_accuracy,
                         "selection_reason": "next_best_stable"})
            selected.append(candidate)
    output = pd.DataFrame(rows).sort_values(["participant", "target", "transfer_rank"], kind="stable").reset_index(drop=True)
    if len(output) != candidates[["participant", "target"]].drop_duplicates().shape[0] * 3:
        raise RuntimeError("Image-domain top-3 selection did not produce exactly three rows per participant-target")
    return output


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
    os.replace(temporary, path)


def settings(args: argparse.Namespace) -> dict[str, Any]:
    return {"participants": args.participants, "targets": args.targets, "modalities": args.modalities, "source_run": DEFAULT_SOURCE_RUN, "seed": args.seed, "label_definition": LABEL_DEFINITION, "outer_cv": "StratifiedKFold up to 5 folds, shuffled, random_state=42", "inner_cv": "StratifiedKFold up to 3 folds, shuffled, random_state=42+outer_fold", "selection_metric": "balanced_accuracy", "competitive_delta_ba": COMPETITIVE_DELTA, "models": ["logreg", "svm", "gnb"], "model_grids": {modality: {model: model_grid(model, modality) for model in ("logreg", "svm", "gnb")} for modality in args.modalities}, "feature_plan": {modality: {target: modality_representation_plan(target, modality) for target in TARGETS} for modality in args.modalities}}


def cohort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_values = {"valence": (0.576, 0.578, 14, 3), "arousal": (0.568, 0.566, 10, 4)}
    rows = []
    for (target, modality), group in summary.groupby(["target", "modality"], sort=True):
        best = group.sort_values(["mean_outer_cv_balanced_accuracy", "fold_ba_sd"], ascending=[False, True]).groupby("participant").head(1)
        mean, median, count60, count65 = baseline_values[target]
        rows.append({"target": target, "modality": modality, "participants": best.participant.nunique(), "mean_participant_best_ba": best.mean_outer_cv_balanced_accuracy.mean(), "median_participant_best_ba": best.mean_outer_cv_balanced_accuracy.median(), "median_candidate_ba": group.mean_outer_cv_balanced_accuracy.median(), "best_ba_ge_060_count": int((best.mean_outer_cv_balanced_accuracy >= .60).sum()), "best_ba_ge_060_fraction": (best.mean_outer_cv_balanced_accuracy >= .60).mean(), "best_ba_ge_065_count": int((best.mean_outer_cv_balanced_accuracy >= .65).sum()), "best_ba_ge_065_fraction": (best.mean_outer_cv_balanced_accuracy >= .65).mean(), "maximum_ba": best.mean_outer_cv_balanced_accuracy.max(), "median_best_fold_sd": best.fold_ba_sd.median(), "frozen_baseline_mean_best_ba": mean if modality == "eeg" else np.nan, "frozen_baseline_median_best_ba": median if modality == "eeg" else np.nan, "frozen_baseline_ge_060_count": count60 if modality == "eeg" else np.nan, "frozen_baseline_ge_065_count": count65 if modality == "eeg" else np.nan})
    return pd.DataFrame(rows)


def export_transfer_models(transfer: pd.DataFrame, parameter_rows: pd.DataFrame, tables: dict[str, pd.DataFrame], output: Path, seed: int, n_jobs: int, image_domain_only: bool = False) -> pd.DataFrame:
    """Refit each frozen transfer choice on all Image rows using inner CV only.

    The final refit is not an additional performance estimate.  Outer-CV values
    remain the reported performance evidence; this model is the reproducible
    Image-trained artifact later used for frozen Robot transfer.
    """
    enriched: list[dict[str, Any]] = []
    key_columns = ["participant", "target", "modality", "classifier", "feature_family", "channel_subset", "feature_count_request"]
    for row in transfer.to_dict("records"):
        config = {key: row[key] for key in key_columns}
        columns = feature_columns(config["feature_family"], config["channel_subset"], config["modality"])
        x, y, _ = prepare_family_data(tables[config["participant"]], config["participant"], config["target"], columns)
        count: int | str = "all" if config["feature_count_request"] == "all" else int(config["feature_count_request"])
        inner_folds = baseline.valid_stratified_splits(y, min(3, 5), f"{config['participant']} final transfer refit")
        inner = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(x, y))
        search = GridSearchCV(make_pipeline(config["classifier"], count, seed), model_grid(config["classifier"], config["modality"]), scoring="balanced_accuracy", cv=inner, n_jobs=n_jobs, refit=True, error_score="raise")
        search.fit(x, y)
        fitted = search.best_estimator_; names = selected_names(fitted, columns)
        model_name = f"{config['participant']}_{config['target']}_{config['modality']}_rank{int(row['transfer_rank'])}.joblib"
        model_path = output / ("final_image_domain_models" if image_domain_only else "final_transfer_models") / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(fitted, model_path)
        mask = pd.Series(True, index=parameter_rows.index)
        for key in key_columns:
            mask &= parameter_rows[key].astype(str).eq(str(config[key]))
        fold_metadata = parameter_rows.loc[mask, ["fold", "best_parameters", "selected_feature_names", "inner_best_balanced_accuracy"]].to_dict("records")
        scaler = fitted.named_steps["scale"]
        selector = fitted.named_steps.get("selector")
        inference = {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "input_feature_names": columns, "selected_feature_names": names, "selector_scores": selector.scores_.tolist() if selector is not None else None}
        enriched.append({**row, "label_definition": LABEL_DEFINITION, "preprocessing_condition": "no_ica source run {participant_lower}_no_ica", "final_refit_best_parameters": json.dumps(search.best_params_, sort_keys=True), "final_refit_inner_balanced_accuracy": float(search.best_score_), "actual_selected_feature_names": json.dumps(names), "outer_fold_selection_metadata": json.dumps(fold_metadata, sort_keys=True), "inference_preprocessing_parameters": json.dumps(inference), "final_model_path": str(model_path), "model_use": "image_domain_only" if image_domain_only else "robot_transfer_candidate", "intended_for_robot_transfer": not image_domain_only})
    return pd.DataFrame(enriched)


def with_modality(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Add an explicit modality without modifying the immutable source CSV."""
    output = frame.copy()
    output["modality"] = modality
    return output


def participant_modality_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Select participant-best configurations independently within each modality."""
    ranked = summary.sort_values(["participant", "target", "modality", "mean_outer_cv_balanced_accuracy", "fold_ba_sd"], ascending=[True, True, True, False, True], kind="stable")
    return ranked.groupby(["participant", "target", "modality"], as_index=False).first()


def write_combined_all_modality_outputs(eeg_root: Path, extension_root: Path, destination: Path) -> None:
    """Build additive modality CSVs without opening any immutable EEG CSV for writing."""
    files = {
        "configuration_summary_all_modalities.csv": ("complete_search_results.csv", "complete_search_results.csv"),
        "fold_results_all_modalities.csv": ("fold_results.csv", "fold_results.csv"),
        "predictions_all_modalities.csv": ("predictions_by_fold.csv", "predictions_by_fold.csv"),
        "selected_features_all_modalities.csv": ("selected_features_by_fold.csv", "selected_features_by_fold.csv"),
        "selected_hyperparameters_all_modalities.csv": ("selected_hyperparameters_by_fold.csv", "selected_hyperparameters_by_fold.csv"),
        "top3_models_per_participant_target_modality.csv": ("top3_models_per_participant_target.csv", "top3_models_per_participant_target_modality.csv"),
    }
    for combined_name, (eeg_name, extension_name) in files.items():
        eeg = with_modality(pd.read_csv(eeg_root / eeg_name), "eeg")
        extension = pd.read_csv(extension_root / extension_name)
        if "modality" not in extension:
            raise ValueError(f"Extension file lacks modality: {extension_name}")
        atomic_csv(pd.concat([eeg, extension], ignore_index=True, sort=False), destination / combined_name)
    combined = pd.read_csv(destination / "configuration_summary_all_modalities.csv")
    participant = participant_modality_summary(combined)
    atomic_csv(participant, destination / "participant_summary_all_modalities.csv")
    atomic_csv(cohort_summary(combined), destination / "modality_comparison_summary.csv")
    pivot = participant.pivot(index=["participant", "target"], columns="modality", values="mean_outer_cv_balanced_accuracy").reset_index()
    for modality in MODALITIES:
        if modality not in pivot:
            pivot[modality] = np.nan
    for left, right in (("eeg", "face"), ("eeg", "multimodal"), ("face", "multimodal")):
        pivot[f"{left}_minus_{right}_ba"] = pivot[left] - pivot[right]
    atomic_csv(pivot, destination / "participant_modality_comparisons.csv")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = planned_configurations(args.participants, args.targets, args.modalities)
    counts = {"eeg": {"valence": 27, "arousal": 18}, "face": {"valence": 6, "arousal": 6}, "multimodal": {"valence": 9, "arousal": 9}}
    expected = sum(counts[modality][target] for modality in args.modalities for target in args.targets) * len(args.participants)
    if len(configs) != expected: raise RuntimeError("Frozen plan configuration count mismatch")
    derived_root = args.derived_root if args.derived_root.is_absolute() else PROJECT_ROOT / args.derived_root
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    preview = pd.DataFrame(configs); preview.insert(1, "participant_qc_status", preview.participant.map(qc_status))
    if args.dry_run:
        return {"dry_run": True, "configurations": len(configs), "by_target": preview.groupby("target").size().to_dict(), "by_modality": preview.groupby("modality").size().to_dict(), "plan": preview}
    if output.exists() and not args.resume: raise FileExistsError(f"Refusing to overwrite output: {output}")
    if args.resume: raise NotImplementedError("Resume will be enabled with the first full-cohort execution; do not begin that run before review.")
    output.mkdir(parents=True)
    tables = {}; sources = {}
    for participant in args.participants:
        tables[participant], path = baseline.load_participant_table(derived_root, participant, DEFAULT_SOURCE_RUN); sources[participant] = str(path)
    run_settings = settings(args)
    atomic_json({"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "source_merged_tables": sources, "settings": run_settings, "top3_rule": "BA descending; fold SD ascending; diversity within delta BA <= 0.02; then next-best stable"}, output / "run_manifest.json")
    atomic_csv(preview.drop_duplicates("participant")[["participant", "participant_qc_status"]], output / "participant_qc_status.csv")
    folds: list[dict[str, Any]] = []; params: list[dict[str, Any]] = []; features: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        if args.progress: print(f"[{index}/{len(configs)}] {configuration_key(config)}", flush=True)
        rows = evaluate_configuration(config, tables[config["participant"]], args.seed, args.n_jobs)
        folds.extend(rows[0]); params.extend(rows[1]); features.extend(rows[2]); predictions.extend(rows[3])
    fold_frame = pd.DataFrame(folds); parameter_frame = pd.DataFrame(params); summary = summarize(fold_frame); absolute, transfer = ranked_top3(summary)
    image_domain_only = set(args.modalities).difference({"eeg"}) != set()
    transfer = export_transfer_models(transfer, parameter_frame, tables, output, args.seed, args.n_jobs, image_domain_only=image_domain_only)
    top3_name = "top3_models_per_participant_target_modality.csv" if image_domain_only else "top3_models_per_participant_target.csv"
    atomic_csv(fold_frame, output / "fold_results.csv"); atomic_csv(pd.DataFrame(params), output / "selected_hyperparameters_by_fold.csv"); atomic_csv(pd.DataFrame(features), output / "selected_features_by_fold.csv"); atomic_csv(pd.DataFrame(predictions), output / "predictions_by_fold.csv"); atomic_csv(summary, output / "complete_search_results.csv"); atomic_csv(absolute, output / "absolute_top3_by_ba.csv"); atomic_csv(transfer, output / top3_name); atomic_csv(cohort_summary(summary), output / "cohort_summary_vs_frozen_baseline.csv")
    if set(args.modalities) == {"face", "multimodal"}:
        eeg_root = args.eeg_results_dir if args.eeg_results_dir.is_absolute() else PROJECT_ROOT / args.eeg_results_dir
        required = ["complete_search_results.csv", "fold_results.csv", "predictions_by_fold.csv", "selected_features_by_fold.csv", "selected_hyperparameters_by_fold.csv", "top3_models_per_participant_target.csv"]
        missing = [name for name in required if not (eeg_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Immutable EEG result files missing: {missing}")
        write_combined_all_modality_outputs(eeg_root, output, eeg_root)
    return {"dry_run": False, "output": output, "configurations": len(configs)}


if __name__ == "__main__":
    try: print(run(parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, NotImplementedError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
