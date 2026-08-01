#!/usr/bin/env python3
"""Compare within- and cross-participant image-rating classifiers."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/four_participant_model_matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("four_participant_model_comparison")
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
RANDOM_SEED = 42
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
CLASSIFICATION_THRESHOLD = 4.0

PARTICIPANT_RUNS = {
    "P01": "p01_image_initial_v2",
    "P15": "p15_image_initial_v2",
    "P18": "p18_image_initial_v1",
    "P27": "p27_image_initial_v1",
}
PARTICIPANT_DATASETS = {
    participant: PROJECT_ROOT
    / "derived"
    / participant
    / "Image_Experiment"
    / "runs"
    / run_name
    / "merged"
    / "p01_image_trial_dataset.csv"
    for participant, run_name in PARTICIPANT_RUNS.items()
}

OUTPUT_DIR = PROJECT_ROOT / "derived" / "Image_Experiment" / "modeling"
RESULTS_PATH = OUTPUT_DIR / "four_participant_model_results.csv"
REPORT_PATH = OUTPUT_DIR / "four_participant_model_report.md"
WITHIN_PLOT_PATH = OUTPUT_DIR / "plots" / "within_participant_balanced_accuracy.png"
CROSS_PLOT_PATH = OUTPUT_DIR / "plots" / "cross_participant_balanced_accuracy.png"

EEG_CHANNELS = ("C3", "C4", "Cz", "Fz", "Oz", "PO7", "PO8", "Pz")
EEG_FEATURE_FAMILIES = (
    "eeg_sd",
    "eeg_se",
    "eeg_hm",
    "eeg_hc",
    "eeg_mf_hz",
    "eeg_bp_delta",
    "eeg_se_delta",
    "eeg_bp_theta",
    "eeg_se_theta",
    "eeg_bp_alpha",
    "eeg_se_alpha",
    "eeg_bp_beta",
    "eeg_se_beta",
    "eeg_bp_gamma",
    "eeg_se_gamma",
)
EEG_COLUMNS = tuple(
    f"{family}__{channel}" for family in EEG_FEATURE_FAMILIES for channel in EEG_CHANNELS
)
FACE_COLUMNS = (
    "video_irisdo_norm_mean",
    "video_irisdo_norm_std",
    "video_eso_norm_mean",
    "video_eso_norm_std",
    "video_enso_norm_mean",
    "video_enso_norm_std",
    "video_mnso_norm_mean",
    "video_mnso_norm_std",
    "video_mwo_norm_mean",
    "video_mwo_norm_std",
)
TARGETS = {
    "valence": "valence_rating",
    "arousal": "arousal_rating",
}
MODALITY_COLUMNS = {
    "EEG": list(EEG_COLUMNS),
    "Face": list(FACE_COLUMNS),
    "Multimodal": list(EEG_COLUMNS) + list(FACE_COLUMNS),
}
TOP_N_GRIDS = {
    "EEG": [1, 2, 3, 5, 10, 20, 40, 80, 120],
    "Face": [1, 2, 3, 5, 10],
    "Multimodal": [1, 2, 3, 5, 10, 20, 40, 80, 130],
}
EXPECTED_AUDIT = {
    "P01": {
        "rows": 120,
        "valid_eeg": 119,
        "valid_face": 120,
        "missing_valence": 0,
        "missing_arousal": 2,
        "valence": (119, 54, 65),
        "arousal": (117, 47, 70),
    },
    "P15": {
        "rows": 120,
        "valid_eeg": 120,
        "valid_face": 120,
        "missing_valence": 0,
        "missing_arousal": 0,
        "valence": (120, 40, 80),
        "arousal": (120, 26, 94),
    },
    "P18": {
        "rows": 120,
        "valid_eeg": 119,
        "valid_face": 120,
        "missing_valence": 0,
        "missing_arousal": 0,
        "valence": (119, 57, 62),
        "arousal": (119, 62, 57),
    },
    "P27": {
        "rows": 120,
        "valid_eeg": 117,
        "valid_face": 120,
        "missing_valence": 0,
        "missing_arousal": 0,
        "valence": (117, 38, 79),
        "arousal": (117, 29, 88),
    },
}

RESULT_COLUMNS = (
    "status",
    "evaluation",
    "participant",
    "training_participants",
    "target",
    "target_column",
    "target_definition",
    "modality",
    "model",
    "repeat",
    "fold",
    "outer_split_index",
    "sample_size",
    "train_size",
    "test_size",
    "train_low_count",
    "train_high_count",
    "test_low_count",
    "test_high_count",
    "available_predictor_count",
    "selected_top_n",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "accuracy",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "inner_best_balanced_accuracy",
    "selected_features",
    "feature_ranks",
    "selected_parameters",
    "random_seed",
)
KEY_COLUMNS = (
    "evaluation",
    "participant",
    "target",
    "modality",
    "model",
    "repeat",
    "fold",
)
COMPLETION_COLUMNS = (
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "accuracy",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "selected_top_n",
    "selected_features",
    "feature_ranks",
    "selected_parameters",
)


def parse_arguments() -> argparse.Namespace:
    """Parse an explicit analysis mode and overwrite safety option."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--within-only", action="store_true", help="Run only within-participant evaluation.")
    mode.add_argument("--cross-only", action="store_true", help="Run only leave-one-participant-out evaluation.")
    mode.add_argument("--all", action="store_true", help="Run both evaluations.")
    mode.add_argument(
        "--dry-validate",
        action="store_true",
        help="Validate inputs and planned outputs without creating files or fitting models.",
    )
    parser.add_argument(
        "--overwrite-summaries",
        action="store_true",
        help="Explicitly allow replacement of existing report and requested plot files; never replaces CSV rows.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def load_participant_data() -> dict[str, pd.DataFrame]:
    """Load and validate the four approved participant trial tables."""
    frames: dict[str, pd.DataFrame] = {}
    required = set(EEG_COLUMNS).union(FACE_COLUMNS).union(TARGETS.values())
    for participant, path in PARTICIPANT_DATASETS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing merged dataset for {participant}: {path}")
        frame = pd.read_csv(path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{participant} is missing required columns: {missing}")
        frames[participant] = frame
    return frames


def participant_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the exact input and complete-case counts used by modeling."""
    valid_eeg = frame[list(EEG_COLUMNS)].notna().all(axis=1)
    valid_face = frame[list(FACE_COLUMNS)].notna().all(axis=1)
    fair = valid_eeg & valid_face
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "valid_eeg": int(valid_eeg.sum()),
        "valid_face": int(valid_face.sum()),
        "missing_valence": int(frame[TARGETS["valence"]].isna().sum()),
        "missing_arousal": int(frame[TARGETS["arousal"]].isna().sum()),
    }
    for target, column in TARGETS.items():
        usable = fair & frame[column].notna()
        labels = (frame.loc[usable, column] >= CLASSIFICATION_THRESHOLD).astype(int)
        result[target] = (int(usable.sum()), int((labels == 0).sum()), int((labels == 1).sum()))
    return result


def validate_inputs(frames: dict[str, pd.DataFrame]) -> None:
    """Fail if predictor or sample counts differ from the approved audit."""
    if len(EEG_COLUMNS) != 120 or len(set(EEG_COLUMNS)) != 120:
        raise ValueError("Expected exactly 120 unique channel-specific EEG predictors")
    if len(FACE_COLUMNS) != 10 or len(set(FACE_COLUMNS)) != 10:
        raise ValueError("Expected exactly 10 unique facial predictors")
    if len(MODALITY_COLUMNS["Multimodal"]) != 130:
        raise ValueError("Expected exactly 130 multimodal predictors")

    for participant, frame in frames.items():
        audit = participant_audit(frame)
        if audit != EXPECTED_AUDIT[participant]:
            raise ValueError(
                f"{participant} counts differ from the approved audit. "
                f"Expected {EXPECTED_AUDIT[participant]}, observed {audit}"
            )
        numeric = frame[list(EEG_COLUMNS) + list(FACE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        finite_or_missing = np.isfinite(numeric.to_numpy()) | numeric.isna().to_numpy()
        if not finite_or_missing.all():
            raise ValueError(f"{participant} contains infinite predictor values")


def print_dry_validation(frames: dict[str, pd.DataFrame]) -> None:
    """Print validated paths and counts without creating outputs."""
    print("DRY VALIDATION PASSED: no models fitted and no files created")
    print("Predictors: EEG=120, Face=10, Multimodal=130")
    for participant, frame in frames.items():
        audit = participant_audit(frame)
        print(f"{participant}: {PARTICIPANT_DATASETS[participant]}")
        print(
            "  rows={rows}, valid_eeg={valid_eeg}, valid_face={valid_face}, "
            "missing_valence={missing_valence}, missing_arousal={missing_arousal}".format(**audit)
        )
        for target in TARGETS:
            count, low, high = audit[target]
            print(f"  {target}: usable={count}, low={low}, high={high}")
    print("Planned outputs:")
    for path in (RESULTS_PATH, REPORT_PATH, WITHIN_PLOT_PATH, CROSS_PLOT_PATH):
        state = "exists" if path.exists() else "absent"
        print(f"  {path} [{state}]")


def model_definitions(seed: int) -> dict[str, Pipeline]:
    """Return the three fixed classifiers with leakage-safe pipelines."""
    return {
        "Logistic elastic-net": Pipeline(
            steps=[
                ("selector", SelectKBest(score_func=f_classif)),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=1.0,
                        l1_ratio=0.5,
                        class_weight="balanced",
                        max_iter=10_000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("selector", SelectKBest(score_func=f_classif)),
                ("scale", StandardScaler()),
                ("classifier", GaussianNB(var_smoothing=1e-9)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("selector", SelectKBest(score_func=f_classif)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        criterion="gini",
                        max_depth=None,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def fair_target_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Use identical complete-case trials for all modality comparisons."""
    predictors = list(EEG_COLUMNS) + list(FACE_COLUMNS)
    valid = frame[target_column].notna() & frame[predictors].notna().all(axis=1)
    return frame.loc[valid].reset_index(drop=True)


def ranked_feature_metadata(selector: SelectKBest, columns: list[str]) -> tuple[list[str], dict[str, int]]:
    """Return selected names and stable descending training-fold ranks."""
    scores = np.asarray(selector.scores_, dtype=float)
    sortable = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(-sortable, kind="stable")
    ranks = np.empty(len(columns), dtype=int)
    ranks[order] = np.arange(1, len(columns) + 1)
    selected = [feature for feature, keep in zip(columns, selector.get_support()) if keep]
    return selected, {feature: int(rank) for feature, rank in zip(columns, ranks)}


def high_class_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return high-class probabilities for ROC AUC."""
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    class_index = list(model.classes_).index(1)
    return probabilities[:, class_index]


def result_key(
    evaluation: str,
    participant: str,
    target: str,
    modality: str,
    model: str,
    repeat: int,
    fold: int,
) -> tuple[Any, ...]:
    """Create the stable key used for safe resumability."""
    return evaluation, participant, target, modality, model, repeat, fold


def load_completed_keys() -> set[tuple[Any, ...]]:
    """Load only structurally complete, unique result rows as resumable keys."""
    if not RESULTS_PATH.exists():
        return set()
    existing = pd.read_csv(RESULTS_PATH)
    missing_columns = sorted(set(RESULT_COLUMNS).difference(existing.columns))
    if missing_columns:
        raise ValueError(f"Existing result CSV has an incompatible schema: missing {missing_columns}")
    if existing.duplicated(list(KEY_COLUMNS)).any():
        duplicates = existing.loc[existing.duplicated(list(KEY_COLUMNS), keep=False), list(KEY_COLUMNS)]
        raise ValueError(f"Existing result CSV has duplicate evaluation keys:\n{duplicates.to_string(index=False)}")

    completed: set[tuple[Any, ...]] = set()
    for _, row in existing.iterrows():
        key = tuple(row[column] for column in KEY_COLUMNS)
        complete = row["status"] == "complete" and row[list(COMPLETION_COLUMNS)].notna().all()
        if not complete:
            raise ValueError(
                "Existing result CSV contains an incomplete row. It will not be skipped or overwritten: "
                f"{key}"
            )
        for json_column in ("selected_features", "feature_ranks", "selected_parameters"):
            try:
                json.loads(row[json_column])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid {json_column} for completed key {key}") from exc
        completed.add(key)
    return completed


def append_completed_result(row: dict[str, Any]) -> None:
    """Append one completed evaluation without replacing any existing row."""
    missing = sorted(set(RESULT_COLUMNS).difference(row))
    if missing:
        raise ValueError(f"Cannot save incomplete result; missing fields: {missing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        RESULTS_PATH,
        mode="a",
        header=write_header,
        index=False,
    )


def fitted_result(
    *,
    evaluation: str,
    participant: str,
    training_participants: list[str],
    target: str,
    target_column: str,
    modality: str,
    model_name: str,
    repeat: int,
    fold: int,
    outer_split_index: int,
    sample_size: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    inner_cv: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    """Tune Top-N on training data and score one untouched outer test set."""
    estimator = model_definitions(seed)[model_name]
    search = GridSearchCV(
        estimator=estimator,
        param_grid={"selector__k": TOP_N_GRIDS[modality]},
        scoring="balanced_accuracy",
        cv=list(inner_cv),
        n_jobs=-1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    fitted = search.best_estimator_
    predictions = fitted.predict(x_test)
    scores = high_class_scores(fitted, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    selected, ranks = ranked_feature_metadata(fitted.named_steps["selector"], list(x_train.columns))
    selected_parameters = {
        "top_n": int(search.best_params_["selector__k"]),
        "classifier": model_name,
    }
    return {
        "status": "complete",
        "evaluation": evaluation,
        "participant": participant,
        "training_participants": json.dumps(training_participants),
        "target": target,
        "target_column": target_column,
        "target_definition": "low_rating_lt_4__high_rating_ge_4",
        "modality": modality,
        "model": model_name,
        "repeat": repeat,
        "fold": fold,
        "outer_split_index": outer_split_index,
        "sample_size": sample_size,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_low_count": int((y_train == 0).sum()),
        "train_high_count": int((y_train == 1).sum()),
        "test_low_count": int((y_test == 0).sum()),
        "test_high_count": int((y_test == 1).sum()),
        "available_predictor_count": int(x_train.shape[1]),
        "selected_top_n": int(search.best_params_["selector__k"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "inner_best_balanced_accuracy": float(search.best_score_),
        "selected_features": json.dumps(selected),
        "feature_ranks": json.dumps(ranks, sort_keys=True),
        "selected_parameters": json.dumps(selected_parameters, sort_keys=True),
        "random_seed": seed,
    }


def run_within(
    frames: dict[str, pd.DataFrame],
    completed: set[tuple[Any, ...]],
    seed: int,
) -> None:
    """Run repeated nested CV separately within each participant."""
    model_names = list(model_definitions(seed))
    for participant, source in frames.items():
        for target, target_column in TARGETS.items():
            valid = fair_target_frame(source, target_column)
            labels = (valid[target_column] >= CLASSIFICATION_THRESHOLD).astype(int)
            outer = RepeatedStratifiedKFold(
                n_splits=OUTER_FOLDS,
                n_repeats=OUTER_REPEATS,
                random_state=seed,
            )
            outer_splits = list(outer.split(np.arange(len(valid)), labels))
            for split_index, (train_indices, test_indices) in enumerate(outer_splits):
                repeat = split_index // OUTER_FOLDS + 1
                fold = split_index % OUTER_FOLDS + 1
                y_train = labels.iloc[train_indices].reset_index(drop=True)
                y_test = labels.iloc[test_indices].reset_index(drop=True)
                inner = StratifiedKFold(
                    n_splits=INNER_FOLDS,
                    shuffle=True,
                    random_state=seed + split_index,
                )
                inner_splits = list(inner.split(np.arange(len(y_train)), y_train))
                LOGGER.info(
                    "Within %s %s: repeat %d/%d, fold %d/%d",
                    participant,
                    target,
                    repeat,
                    OUTER_REPEATS,
                    fold,
                    OUTER_FOLDS,
                )
                for modality, columns in MODALITY_COLUMNS.items():
                    x_train = valid.iloc[train_indices][columns].reset_index(drop=True)
                    x_test = valid.iloc[test_indices][columns].reset_index(drop=True)
                    for model_name in model_names:
                        key = result_key("within", participant, target, modality, model_name, repeat, fold)
                        if key in completed:
                            continue
                        row = fitted_result(
                            evaluation="within",
                            participant=participant,
                            training_participants=[participant],
                            target=target,
                            target_column=target_column,
                            modality=modality,
                            model_name=model_name,
                            repeat=repeat,
                            fold=fold,
                            outer_split_index=split_index,
                            sample_size=len(valid),
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            inner_cv=inner_splits,
                            seed=seed,
                        )
                        append_completed_result(row)
                        completed.add(key)


def run_cross(
    frames: dict[str, pd.DataFrame],
    completed: set[tuple[Any, ...]],
    seed: int,
) -> None:
    """Run leave-one-participant-out testing with grouped inner selection."""
    model_names = list(model_definitions(seed))
    fair_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for participant, frame in frames.items():
        for target, target_column in TARGETS.items():
            fair = fair_target_frame(frame, target_column).copy()
            fair["__participant_group"] = participant
            fair_frames[(participant, target)] = fair

    for heldout in frames:
        training_participants = [participant for participant in frames if participant != heldout]
        for target, target_column in TARGETS.items():
            train = pd.concat(
                [fair_frames[(participant, target)] for participant in training_participants],
                ignore_index=True,
            )
            test = fair_frames[(heldout, target)].reset_index(drop=True)
            y_train = (train[target_column] >= CLASSIFICATION_THRESHOLD).astype(int)
            y_test = (test[target_column] >= CLASSIFICATION_THRESHOLD).astype(int)
            groups = train["__participant_group"].to_numpy()
            inner_splits = list(LeaveOneGroupOut().split(np.arange(len(train)), y_train, groups))
            LOGGER.info(
                "Cross %s %s: train=%s, test=%s",
                heldout,
                target,
                training_participants,
                heldout,
            )
            for modality, columns in MODALITY_COLUMNS.items():
                x_train = train[columns].reset_index(drop=True)
                x_test = test[columns].reset_index(drop=True)
                for model_name in model_names:
                    key = result_key("cross", heldout, target, modality, model_name, 0, 0)
                    if key in completed:
                        continue
                    row = fitted_result(
                        evaluation="cross",
                        participant=heldout,
                        training_participants=training_participants,
                        target=target,
                        target_column=target_column,
                        modality=modality,
                        model_name=model_name,
                        repeat=0,
                        fold=0,
                        outer_split_index=0,
                        sample_size=len(train) + len(test),
                        x_train=x_train,
                        y_train=y_train.reset_index(drop=True),
                        x_test=x_test,
                        y_test=y_test.reset_index(drop=True),
                        inner_cv=inner_splits,
                        seed=seed,
                    )
                    append_completed_result(row)
                    completed.add(key)


def requested_summary_paths(args: argparse.Namespace) -> list[Path]:
    """Return only the report and plots requested by the selected mode."""
    paths = [REPORT_PATH]
    if args.within_only or args.all:
        paths.append(WITHIN_PLOT_PATH)
    if args.cross_only or args.all:
        paths.append(CROSS_PLOT_PATH)
    return paths


def enforce_summary_overwrite_policy(args: argparse.Namespace) -> None:
    """Refuse to replace report or plots without an explicit CLI acknowledgement."""
    collisions = [path for path in requested_summary_paths(args) if path.exists()]
    if collisions and not args.overwrite_summaries:
        rendered = "\n".join(f"  {path}" for path in collisions)
        raise FileExistsError(
            "The following report/plot files already exist and will not be overwritten:\n"
            f"{rendered}\nObtain approval before rerunning with --overwrite-summaries."
        )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a small dataframe without an optional tabulate dependency."""
    if frame.empty:
        return "No completed results are available."
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def feature_stability_table(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-feature frequencies across completed test fits."""
    rows: list[dict[str, Any]] = []
    group_columns = ["evaluation", "target", "modality", "model"]
    for keys, group in results.groupby(group_columns, sort=False):
        counts: dict[str, int] = {}
        for encoded in group["selected_features"]:
            for feature in json.loads(encoded):
                counts[feature] = counts.get(feature, 0) + 1
        denominator = len(group)
        for feature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "feature": feature,
                    "selection_frequency": count / denominator,
                }
            )
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame) -> None:
    """Write one combined report from all currently completed evaluations."""
    summary = (
        results.groupby(["evaluation", "participant", "target", "modality", "model"], sort=False)
        .agg(
            evaluations=("balanced_accuracy", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            selected_top_n_median=("selected_top_n", "median"),
        )
        .reset_index()
    )
    stability = feature_stability_table(results)
    lines = [
        "# Four-participant model comparison",
        "",
        "This exploratory analysis predicts participant-specific low/high ratings using a theoretical midpoint of 4.",
        "All feature ranking, Top-N selection, scaling, and fitting occur inside the applicable training folds.",
        "",
        "The cross-participant analysis measures generalization to an unseen participant on the same stimulus set; it does not measure generalization to unseen images.",
        "",
        "## Completed evaluations",
        "",
        f"Fold-level or held-out test rows: {len(results)}.",
        "",
        markdown_table(
            summary,
            [
                "evaluation",
                "participant",
                "target",
                "modality",
                "model",
                "evaluations",
                "balanced_accuracy_mean",
                "balanced_accuracy_sd",
                "macro_f1_mean",
                "roc_auc_mean",
                "accuracy_mean",
                "selected_top_n_median",
            ],
        ),
        "",
        "## Feature-selection stability",
        "",
        "Selection frequency is the fraction of completed fits in the stated evaluation/target/modality/model group.",
        "",
        markdown_table(
            stability,
            ["evaluation", "target", "modality", "model", "feature", "selection_frequency"],
        ),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_within(results: pd.DataFrame) -> None:
    """Plot within-participant balanced accuracy in one eight-panel figure."""
    within = results.loc[results["evaluation"] == "within"]
    if within.empty:
        return
    participants = list(PARTICIPANT_RUNS)
    model_names = list(model_definitions(RANDOM_SEED))
    colors = {"EEG": "#4C78A8", "Face": "#F58518", "Multimodal": "#54A24B"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    x = np.arange(len(model_names))
    offsets = {"EEG": -0.22, "Face": 0.0, "Multimodal": 0.22}
    for row_index, target in enumerate(TARGETS):
        for column_index, participant in enumerate(participants):
            axis = axes[row_index, column_index]
            subset = within.loc[(within["target"] == target) & (within["participant"] == participant)]
            for modality in MODALITY_COLUMNS:
                stats = subset.loc[subset["modality"] == modality].groupby("model")["balanced_accuracy"].agg(["mean", "std"])
                means = [stats.loc[model, "mean"] for model in model_names]
                errors = [stats.loc[model, "std"] for model in model_names]
                axis.errorbar(x + offsets[modality], means, yerr=errors, marker="o", capsize=3, color=colors[modality], label=modality)
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
            axis.set_title(f"{participant} {target}")
            axis.set_xticks(x, ["Elastic-net", "Gaussian NB", "Random Forest"], rotation=25, ha="right")
            axis.set_ylim(0.25, 0.9)
            if column_index == 0:
                axis.set_ylabel("Balanced accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Within-participant nested cross-validation", y=1.02)
    fig.tight_layout()
    WITHIN_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(WITHIN_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cross(results: pd.DataFrame) -> None:
    """Plot held-out-participant balanced accuracy by target and modality."""
    cross = results.loc[results["evaluation"] == "cross"]
    if cross.empty:
        return
    participants = list(PARTICIPANT_RUNS)
    model_names = list(model_definitions(RANDOM_SEED))
    colors = {"Logistic elastic-net": "#4C78A8", "Gaussian Naive Bayes": "#F58518", "Random Forest": "#54A24B"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    x = np.arange(len(participants))
    for row_index, target in enumerate(TARGETS):
        for column_index, modality in enumerate(MODALITY_COLUMNS):
            axis = axes[row_index, column_index]
            subset = cross.loc[(cross["target"] == target) & (cross["modality"] == modality)]
            for model_name in model_names:
                model_rows = subset.loc[subset["model"] == model_name].set_index("participant")
                values = [model_rows.loc[participant, "balanced_accuracy"] for participant in participants]
                axis.plot(x, values, marker="o", color=colors[model_name], label=model_name)
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
            axis.set_title(f"{target} — {modality}")
            axis.set_xticks(x, participants)
            axis.set_ylim(0.25, 0.9)
            if column_index == 0:
                axis.set_ylabel("Balanced accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Generalization to an unseen participant on the same stimuli", y=1.02)
    fig.tight_layout()
    CROSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CROSS_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Validate inputs, run requested evaluations, and write approved summaries."""
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    frames = load_participant_data()
    validate_inputs(frames)
    if args.dry_validate:
        print_dry_validation(frames)
        return 0

    enforce_summary_overwrite_policy(args)
    completed = load_completed_keys()
    LOGGER.info("Loaded %d genuinely completed evaluation keys", len(completed))
    if args.within_only or args.all:
        run_within(frames, completed, args.seed)
    if args.cross_only or args.all:
        run_cross(frames, completed, args.seed)

    results = pd.read_csv(RESULTS_PATH)
    write_report(results)
    if args.within_only or args.all:
        plot_within(results)
    if args.cross_only or args.all:
        plot_cross(results)
    LOGGER.info("Completed requested analysis; results: %s", RESULTS_PATH)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
