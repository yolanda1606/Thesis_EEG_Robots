"""Run compact-feature regression and binary-classification baselines."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/baseline_regression_matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42
OUTER_FOLDS = 5
OUTER_REPEATS = 5
INNER_FOLDS = 4
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTIVE_RUNS = {
    "P01": "p01_image_initial_v2",
    "P15": "p15_image_initial_v2",
    "P18": "p18_image_initial_v1",
    "P27": "p27_image_initial_v1",
}
LEGACY_P01_OUTPUT_DIR = PROJECT_ROOT / "derived" / "P01" / "Image_Experiment" / "modeling"

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
COMPACT_EEG_COLUMNS = tuple(f"{family}_channel_mean" for family in EEG_FEATURE_FAMILIES)
CHANNEL_SPECIFIC_EEG_COLUMNS = tuple(
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
TARGETS = ("valence_rating", "arousal_rating")
CLASSIFICATION_THRESHOLD = 4.0
OUTPUT_FILENAMES = (
    "baseline_regression_results.csv",
    "baseline_regression_report.md",
    "plots/baseline_regression_rmse.png",
    "plots/baseline_classification_balanced_accuracy.png",
)
FEATURE_SELECTION_OUTPUT_FILENAMES = (
    "feature_selection_results.csv",
    "feature_selection_report.md",
    "plots/feature_selection_balanced_accuracy.png",
)
FACE_TOP_N_GRID = tuple(range(1, 11))
EEG_TOP_N_GRID = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 120)


@dataclass(frozen=True)
class AnalysisContext:
    """Resolved participant input, output, and naming information."""

    participant: str
    run_name: str | None
    trials: Path
    output_dir: Path
    participant_aware: bool


def parse_arguments() -> argparse.Namespace:
    """Parse legacy explicit paths or participant-aware run selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, help="Legacy merged trial-level dataset CSV.")
    parser.add_argument("--output-dir", type=Path, help="Legacy directory for final baseline artifacts.")
    parser.add_argument("--participant", help="Participant identifier, for example P15.")
    parser.add_argument("--run-name", help="Active Image Experiment run name for --participant.")
    parser.add_argument(
        "--all-participants",
        action="store_true",
        help="Run every registered active participant, skipping P01 when its historical baseline exists.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for repeated and inner folds.")
    parser.add_argument(
        "--feature-selection-only",
        action="store_true",
        help="Run only the approved face and channel-specific EEG feature-selection analysis.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Resolve participants, validate manifests, inputs, and output collisions without fitting models.",
    )
    return parser.parse_args()


def baseline_output_filenames(context: AnalysisContext) -> tuple[str, ...]:
    """Return legacy or participant-aware baseline artifact names."""
    if not context.participant_aware:
        return OUTPUT_FILENAMES
    prefix = context.participant.lower()
    return (
        f"{prefix}_baseline_regression_results.csv",
        f"{prefix}_baseline_regression_report.md",
        f"plots/{prefix}_baseline_regression_rmse.png",
        f"plots/{prefix}_baseline_classification_balanced_accuracy.png",
    )


def feature_selection_output_filenames(context: AnalysisContext) -> tuple[str, ...]:
    """Return legacy or participant-aware feature-selection artifact names."""
    if not context.participant_aware:
        return FEATURE_SELECTION_OUTPUT_FILENAMES
    prefix = context.participant.lower()
    return (
        f"{prefix}_feature_selection_results.csv",
        f"{prefix}_feature_selection_report.md",
        f"plots/{prefix}_feature_selection_balanced_accuracy.png",
    )


def historical_p01_baseline_exists() -> bool:
    """Detect the approved historical P01 baseline without changing it."""
    return any((LEGACY_P01_OUTPUT_DIR / filename).exists() for filename in OUTPUT_FILENAMES)


def resolve_participant_context(participant: str, run_name: str) -> AnalysisContext:
    """Resolve one active run and verify its manifest participant identity."""
    normalized = participant.upper()
    run_dir = PROJECT_ROOT / "derived" / normalized / "Image_Experiment" / "runs" / run_name
    manifest_path = run_dir / "manifest" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest was not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_configuration", manifest.get("configuration"))
    if not isinstance(resolved, dict) or "participant" not in resolved:
        raise ValueError(f"Run manifest does not contain a resolved participant: {manifest_path}")
    manifest_participant = str(resolved["participant"]).upper()
    if manifest_participant != normalized:
        raise ValueError(
            f"Participant mismatch: CLI requested {normalized}, but {manifest_path} records {manifest_participant}"
        )

    merged_dir = run_dir / "merged"
    preferred = merged_dir / f"{normalized.lower()}_image_trial_dataset.csv"
    historical = merged_dir / "p01_image_trial_dataset.csv"
    if preferred.is_file():
        trials = preferred
    elif historical.is_file():
        trials = historical
    else:
        raise FileNotFoundError(
            f"No merged trial dataset found. Checked participant-aware {preferred} and historical {historical}"
        )
    return AnalysisContext(
        participant=normalized,
        run_name=run_name,
        trials=trials.resolve(),
        output_dir=(run_dir / "modeling").resolve(),
        participant_aware=True,
    )


def resolve_analysis_contexts(args: argparse.Namespace) -> tuple[list[AnalysisContext], list[str]]:
    """Resolve legacy, single-participant, or registered batch execution."""
    if args.all_participants:
        if any((args.trials, args.output_dir, args.participant, args.run_name)):
            raise ValueError(
                "--all-participants cannot be combined with --trials, --output-dir, --participant, or --run-name"
            )
        contexts: list[AnalysisContext] = []
        skipped: list[str] = []
        for participant, run_name in ACTIVE_RUNS.items():
            if participant == "P01" and historical_p01_baseline_exists():
                skipped.append(
                    f"P01: existing historical baseline outputs found in {LEGACY_P01_OUTPUT_DIR}; not rerunning"
                )
                continue
            contexts.append(resolve_participant_context(participant, run_name))
        return contexts, skipped

    participant_mode = args.participant is not None or args.run_name is not None
    legacy_mode = args.trials is not None or args.output_dir is not None
    if participant_mode and legacy_mode:
        raise ValueError("Participant-aware mode cannot be combined with legacy --trials or --output-dir")
    if participant_mode:
        if not args.participant or not args.run_name:
            raise ValueError("--participant and --run-name must be supplied together")
        return [resolve_participant_context(args.participant, args.run_name)], []
    if not args.trials or not args.output_dir:
        raise ValueError(
            "Use both legacy --trials and --output-dir, or use --participant with --run-name, or use --all-participants"
        )
    trials = args.trials.resolve(strict=True)
    return [
        AnalysisContext(
            participant="P01",
            run_name=None,
            trials=trials,
            output_dir=args.output_dir.resolve(),
            participant_aware=False,
        )
    ], []


def add_compact_eeg_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    """Average each EEG family across all eight channels, requiring complete data."""
    result = frame.copy()
    missing_columns: list[str] = []
    for family, compact_column in zip(EEG_FEATURE_FAMILIES, COMPACT_EEG_COLUMNS):
        source_columns = [f"{family}__{channel}" for channel in EEG_CHANNELS]
        missing_columns.extend(column for column in source_columns if column not in result.columns)
        if not missing_columns:
            complete_channels = result[source_columns].notna().all(axis=1)
            result[compact_column] = result[source_columns].mean(axis=1).where(complete_channels)
    if missing_columns:
        raise ValueError(f"Merged trial table is missing EEG columns: {sorted(set(missing_columns))}")
    return result


def validate_input_columns(frame: pd.DataFrame) -> None:
    """Fail clearly if a required target or facial predictor is unavailable."""
    required = set(TARGETS).union(FACE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Merged trial table is missing required columns: {missing}")


def model_definitions(seed: int) -> dict[str, tuple[Any, dict[str, list[Any]] | None]]:
    """Return the approved models and small inner parameter grids."""
    return {
        "Dummy mean": (DummyRegressor(strategy="mean"), None),
        "Ridge": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("regressor", Ridge()),
                ]
            ),
            {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "ElasticNet": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("regressor", ElasticNet(max_iter=10_000)),
                ]
            ),
            {
                "regressor__alpha": [0.01, 0.1, 1.0],
                "regressor__l1_ratio": [0.1, 0.5, 0.9],
            },
        ),
        "kNN": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("regressor", KNeighborsRegressor(weights="uniform", p=2)),
                ]
            ),
            {"regressor__n_neighbors": [3, 5, 7, 9]},
        ),
        "SVR-RBF": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("regressor", SVR(kernel="rbf", gamma="scale")),
                ]
            ),
            {
                "regressor__C": [0.1, 1.0, 10.0],
                "regressor__epsilon": [0.1, 0.2],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=100,
                criterion="squared_error",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "max_depth": [None, 5],
                "min_samples_leaf": [1, 3],
                "max_features": [1.0, "sqrt"],
            },
        ),
        "Extra Trees": (
            ExtraTreesRegressor(
                n_estimators=100,
                criterion="squared_error",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "max_depth": [None, 5],
                "min_samples_leaf": [1, 3],
                "max_features": [1.0, "sqrt"],
            },
        ),
    }


def classification_model_definitions(seed: int) -> dict[str, tuple[Any, dict[str, list[Any]] | None]]:
    """Return the approved classifiers and small inner parameter grids."""
    return {
        "Dummy most frequent": (DummyClassifier(strategy="most_frequent"), None),
        "Logistic L2": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="l2",
                            solver="liblinear",
                            max_iter=10_000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "Logistic elastic-net": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            max_iter=10_000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__l1_ratio": [0.1, 0.5, 0.9],
            },
        ),
        "kNN classifier": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("classifier", KNeighborsClassifier(weights="uniform", p=2)),
                ]
            ),
            {"classifier__n_neighbors": [3, 5, 7, 9]},
        ),
        "SVM-RBF": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("classifier", SVC(kernel="rbf", gamma="scale")),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "Gaussian NB": (
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("classifier", GaussianNB()),
                ]
            ),
            {"classifier__var_smoothing": [1e-11, 1e-9, 1e-7]},
        ),
        "Random Forest classifier": (
            RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "max_depth": [None, 5],
                "min_samples_leaf": [1, 3],
                "max_features": [1.0, "sqrt"],
            },
        ),
        "Extra Trees classifier": (
            ExtraTreesClassifier(
                n_estimators=100,
                criterion="gini",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "max_depth": [None, 5],
                "min_samples_leaf": [1, 3],
                "max_features": [1.0, "sqrt"],
            },
        ),
    }


def feature_selection_model_definitions(
    top_n: int,
    seed: int,
    include_random_forest: bool,
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Return leakage-safe SelectKBest pipelines for one fixed Top-N value."""
    definitions: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "Logistic elastic-net": (
            Pipeline(
                steps=[
                    ("selector", SelectKBest(score_func=f_classif, k=top_n)),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            max_iter=10_000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__l1_ratio": [0.1, 0.5, 0.9],
            },
        )
    }
    if include_random_forest:
        definitions["Random Forest classifier"] = (
            Pipeline(
                steps=[
                    ("selector", SelectKBest(score_func=f_classif, k=top_n)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=100,
                            criterion="gini",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__max_depth": [None, 5],
                "classifier__min_samples_leaf": [1, 3],
                "classifier__max_features": [1.0, "sqrt"],
            },
        )
    return definitions


def evaluate_target(frame: pd.DataFrame, target: str, seed: int) -> list[dict[str, Any]]:
    """Evaluate every modality and model on identical repeated outer folds."""
    all_predictors = list(COMPACT_EEG_COLUMNS) + list(FACE_COLUMNS)
    valid = frame.loc[frame[target].notna() & frame[all_predictors].notna().all(axis=1)].reset_index(drop=True)
    modality_columns = {
        "EEG only": list(COMPACT_EEG_COLUMNS),
        "Face only": list(FACE_COLUMNS),
        "Multimodal": all_predictors,
    }
    outer_cv = RepeatedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=seed)
    outer_splits = list(outer_cv.split(np.arange(len(valid))))
    definitions = model_definitions(seed)
    rows: list[dict[str, Any]] = []

    for split_index, (train_indices, test_indices) in enumerate(outer_splits):
        repeat_number = split_index // OUTER_FOLDS + 1
        fold_number = split_index % OUTER_FOLDS + 1
        inner_cv = list(
            KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + split_index).split(
                np.arange(len(train_indices))
            )
        )
        y_train = valid.loc[train_indices, target]
        y_test = valid.loc[test_indices, target]

        for modality, predictors in modality_columns.items():
            x_train = valid.loc[train_indices, predictors]
            x_test = valid.loc[test_indices, predictors]
            for model_name, (estimator, parameter_grid) in definitions.items():
                inner_best_rmse = float("nan")
                if parameter_grid is None:
                    fitted_model = clone(estimator).fit(x_train, y_train)
                    selected_parameters = {"strategy": "mean"}
                else:
                    search = GridSearchCV(
                        estimator=clone(estimator),
                        param_grid=parameter_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=inner_cv,
                        n_jobs=-1,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(x_train, y_train)
                    fitted_model = search.best_estimator_
                    selected_parameters = {
                        key.removeprefix("regressor__"): value for key, value in search.best_params_.items()
                    }
                    inner_best_rmse = float(-search.best_score_)

                predictions = fitted_model.predict(x_test)
                rows.append(
                    {
                        "task_type": "regression",
                        "target": target,
                        "target_definition": "continuous_rating",
                        "modality": modality,
                        "model": model_name,
                        "repeat": repeat_number,
                        "fold": fold_number,
                        "outer_split_index": split_index,
                        "sample_size": len(valid),
                        "train_size": len(train_indices),
                        "test_size": len(test_indices),
                        "predictor_count": len(predictors),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                        "mae": float(mean_absolute_error(y_test, predictions)),
                        "r2": float(r2_score(y_test, predictions)),
                        "inner_best_rmse": inner_best_rmse,
                        "selected_parameters": json.dumps(selected_parameters, sort_keys=True),
                        "random_seed": seed,
                    }
                )
    return rows


def classification_scores(fitted_model: Any, predictors: pd.DataFrame) -> np.ndarray:
    """Return continuous high-class scores suitable for ROC AUC."""
    if hasattr(fitted_model, "decision_function"):
        return np.asarray(fitted_model.decision_function(predictors), dtype=float)
    probabilities = np.asarray(fitted_model.predict_proba(predictors), dtype=float)
    high_class_index = list(fitted_model.classes_).index(1)
    return probabilities[:, high_class_index]


def evaluate_classification_target(frame: pd.DataFrame, target: str, seed: int) -> list[dict[str, Any]]:
    """Evaluate binary classifiers on identical repeated stratified outer folds."""
    all_predictors = list(COMPACT_EEG_COLUMNS) + list(FACE_COLUMNS)
    valid = frame.loc[frame[target].notna() & frame[all_predictors].notna().all(axis=1)].reset_index(drop=True)
    labels = (valid[target] >= CLASSIFICATION_THRESHOLD).astype(int)
    modality_columns = {
        "EEG only": list(COMPACT_EEG_COLUMNS),
        "Face only": list(FACE_COLUMNS),
        "Multimodal": all_predictors,
    }
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=OUTER_REPEATS,
        random_state=seed,
    )
    outer_splits = list(outer_cv.split(np.arange(len(valid)), labels))
    definitions = classification_model_definitions(seed)
    rows: list[dict[str, Any]] = []

    for split_index, (train_indices, test_indices) in enumerate(outer_splits):
        repeat_number = split_index // OUTER_FOLDS + 1
        fold_number = split_index % OUTER_FOLDS + 1
        y_train = labels.iloc[train_indices]
        y_test = labels.iloc[test_indices]
        inner_cv = list(
            StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=seed + split_index,
            ).split(np.arange(len(train_indices)), y_train)
        )

        for modality, predictors in modality_columns.items():
            x_train = valid.loc[train_indices, predictors]
            x_test = valid.loc[test_indices, predictors]
            for model_name, (estimator, parameter_grid) in definitions.items():
                inner_best_balanced_accuracy = float("nan")
                if parameter_grid is None:
                    fitted_model = clone(estimator).fit(x_train, y_train)
                    selected_parameters = {"strategy": "most_frequent"}
                else:
                    search = GridSearchCV(
                        estimator=clone(estimator),
                        param_grid=parameter_grid,
                        scoring="balanced_accuracy",
                        cv=inner_cv,
                        n_jobs=-1,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(x_train, y_train)
                    fitted_model = search.best_estimator_
                    selected_parameters = {
                        key.removeprefix("classifier__"): value for key, value in search.best_params_.items()
                    }
                    inner_best_balanced_accuracy = float(search.best_score_)

                predictions = fitted_model.predict(x_test)
                high_class_scores = classification_scores(fitted_model, x_test)
                true_negative, false_positive, false_negative, true_positive = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1],
                ).ravel()
                rows.append(
                    {
                        "task_type": "classification",
                        "target": target,
                        "target_definition": "low_rating_lt_4__high_rating_ge_4",
                        "modality": modality,
                        "model": model_name,
                        "repeat": repeat_number,
                        "fold": fold_number,
                        "outer_split_index": split_index,
                        "sample_size": len(valid),
                        "low_class_count": int((labels == 0).sum()),
                        "high_class_count": int((labels == 1).sum()),
                        "train_size": len(train_indices),
                        "test_size": len(test_indices),
                        "predictor_count": len(predictors),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                        "roc_auc": float(roc_auc_score(y_test, high_class_scores)),
                        "accuracy": float(accuracy_score(y_test, predictions)),
                        "true_negative": int(true_negative),
                        "false_positive": int(false_positive),
                        "false_negative": int(false_negative),
                        "true_positive": int(true_positive),
                        "inner_best_balanced_accuracy": inner_best_balanced_accuracy,
                        "selected_parameters": json.dumps(selected_parameters, sort_keys=True),
                        "random_seed": seed,
                    }
                )
    return rows


def ranked_feature_metadata(selector: SelectKBest, feature_columns: list[str]) -> tuple[list[str], dict[str, int]]:
    """Return selected feature names and deterministic descending f-score ranks."""
    scores = np.asarray(selector.scores_, dtype=float)
    sortable_scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(-sortable_scores, kind="stable")
    ranks = np.empty(len(feature_columns), dtype=int)
    ranks[order] = np.arange(1, len(feature_columns) + 1)
    selected = [feature for feature, keep in zip(feature_columns, selector.get_support()) if keep]
    rank_map = {feature: int(rank) for feature, rank in zip(feature_columns, ranks)}
    return selected, rank_map


def evaluate_feature_selection_task(
    frame: pd.DataFrame,
    target: str,
    modality: str,
    feature_columns: list[str],
    top_n_grid: tuple[int, ...],
    include_random_forest: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate fixed Top-N values with selection and tuning inside each training fold."""
    fair_predictors = list(CHANNEL_SPECIFIC_EEG_COLUMNS) + list(FACE_COLUMNS)
    valid = frame.loc[frame[target].notna() & frame[fair_predictors].notna().all(axis=1)].reset_index(drop=True)
    labels = (valid[target] >= CLASSIFICATION_THRESHOLD).astype(int)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=OUTER_REPEATS,
        random_state=seed,
    )
    outer_splits = list(outer_cv.split(np.arange(len(valid)), labels))
    rows: list[dict[str, Any]] = []

    for split_index, (train_indices, test_indices) in enumerate(outer_splits):
        repeat_number = split_index // OUTER_FOLDS + 1
        fold_number = split_index % OUTER_FOLDS + 1
        y_train = labels.iloc[train_indices]
        y_test = labels.iloc[test_indices]
        x_train = valid.loc[train_indices, feature_columns]
        x_test = valid.loc[test_indices, feature_columns]
        inner_cv = list(
            StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=seed + split_index,
            ).split(np.arange(len(train_indices)), y_train)
        )

        for top_n in top_n_grid:
            definitions = feature_selection_model_definitions(top_n, seed, include_random_forest)
            for model_name, (estimator, parameter_grid) in definitions.items():
                search = GridSearchCV(
                    estimator=clone(estimator),
                    param_grid=parameter_grid,
                    scoring="balanced_accuracy",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(x_train, y_train)
                fitted_model = search.best_estimator_
                predictions = fitted_model.predict(x_test)
                high_class_scores = classification_scores(fitted_model, x_test)
                true_negative, false_positive, false_negative, true_positive = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1],
                ).ravel()
                selected_features, feature_ranks = ranked_feature_metadata(
                    fitted_model.named_steps["selector"],
                    feature_columns,
                )
                selected_parameters = {
                    key.removeprefix("classifier__"): value for key, value in search.best_params_.items()
                }
                rows.append(
                    {
                        "task_type": "feature_selection",
                        "target": target,
                        "target_definition": "low_rating_lt_4__high_rating_ge_4",
                        "modality": modality,
                        "model": model_name,
                        "top_n": top_n,
                        "repeat": repeat_number,
                        "fold": fold_number,
                        "outer_split_index": split_index,
                        "sample_size": len(valid),
                        "low_class_count": int((labels == 0).sum()),
                        "high_class_count": int((labels == 1).sum()),
                        "train_size": len(train_indices),
                        "test_size": len(test_indices),
                        "available_predictor_count": len(feature_columns),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                        "roc_auc": float(roc_auc_score(y_test, high_class_scores)),
                        "accuracy": float(accuracy_score(y_test, predictions)),
                        "true_negative": int(true_negative),
                        "false_positive": int(false_positive),
                        "false_negative": int(false_negative),
                        "true_positive": int(true_positive),
                        "inner_best_balanced_accuracy": float(search.best_score_),
                        "selected_parameters": json.dumps(selected_parameters, sort_keys=True),
                        "selected_features": json.dumps(selected_features),
                        "feature_ranks": json.dumps(feature_ranks, sort_keys=True),
                        "random_seed": seed,
                    }
                )
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize outer-fold metrics without discarding fold-level results."""
    summary = (
        results.groupby(["target", "modality", "model"], sort=False)
        .agg(
            folds=("rmse", "size"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
        )
        .reset_index()
    )
    return summary


def summarize_classification_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize classification metrics and repeated-fold confusion totals."""
    return (
        results.groupby(["target", "modality", "model"], sort=False)
        .agg(
            folds=("balanced_accuracy", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            true_negative=("true_negative", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
            true_positive=("true_positive", "sum"),
        )
        .reset_index()
    )


def markdown_summary_table(summary: pd.DataFrame) -> str:
    """Format all model summaries as a Markdown table."""
    lines = [
        "| Target | Modality | Model | Folds | RMSE mean ± SD | MAE mean ± SD | R² mean ± SD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.modality} | {row.model} | {row.folds} | "
            f"{row.rmse_mean:.3f} ± {row.rmse_sd:.3f} | {row.mae_mean:.3f} ± {row.mae_sd:.3f} | "
            f"{row.r2_mean:.3f} ± {row.r2_sd:.3f} |"
        )
    return "\n".join(lines)


def classification_summary_table(summary: pd.DataFrame) -> str:
    """Format classification metrics and confusion totals as Markdown."""
    lines = [
        "| Target | Modality | Model | Balanced accuracy mean ± SD | Macro F1 mean ± SD | ROC AUC mean ± SD | Accuracy mean ± SD | TN | FP | FN | TP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.modality} | {row.model} | "
            f"{row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_sd:.3f} | "
            f"{row.macro_f1_mean:.3f} ± {row.macro_f1_sd:.3f} | "
            f"{row.roc_auc_mean:.3f} ± {row.roc_auc_sd:.3f} | "
            f"{row.accuracy_mean:.3f} ± {row.accuracy_sd:.3f} | "
            f"{row.true_negative} | {row.false_positive} | {row.false_negative} | {row.true_positive} |"
        )
    return "\n".join(lines)


def hyperparameter_frequency_table(results: pd.DataFrame, dummy_model: str) -> str:
    """Report the most frequently selected parameters for every tuned comparison."""
    tuned = results.loc[results["model"] != dummy_model]
    counts = (
        tuned.groupby(["target", "modality", "model", "selected_parameters"], sort=False)
        .size()
        .rename("selection_count")
        .reset_index()
    )
    most_frequent = counts.loc[
        counts.groupby(["target", "modality", "model"])["selection_count"].idxmax()
    ].sort_values(["target", "modality", "model"])
    lines = [
        "| Target | Modality | Model | Most frequent parameters | Selected folds |",
        "|---|---|---|---|---:|",
    ]
    for row in most_frequent.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.modality} | {row.model} | `{row.selected_parameters}` | "
            f"{row.selection_count}/{OUTER_FOLDS * OUTER_REPEATS} |"
        )
    return "\n".join(lines)


def comparison_text(summary: pd.DataFrame) -> str:
    """Answer the requested baseline questions using descriptive outer-fold means."""
    paragraphs: list[str] = []
    for target in TARGETS:
        target_rows = summary.loc[summary["target"] == target]
        answers: list[str] = []
        best_by_modality: dict[str, pd.Series] = {}
        for modality in ("EEG only", "Face only", "Multimodal"):
            modality_rows = target_rows.loc[target_rows["modality"] == modality]
            dummy_rmse = float(modality_rows.loc[modality_rows["model"] == "Dummy mean", "rmse_mean"].iloc[0])
            candidates = modality_rows.loc[modality_rows["model"] != "Dummy mean"]
            best = candidates.loc[candidates["rmse_mean"].idxmin()]
            best_by_modality[modality] = best
            difference = dummy_rmse - float(best["rmse_mean"])
            result_wording = "lower" if difference > 0 else "not lower"
            answers.append(
                f"{modality} best model was {best['model']} (RMSE {best['rmse_mean']:.3f}), which was "
                f"{result_wording} than the dummy RMSE of {dummy_rmse:.3f} (difference {difference:+.3f})."
            )
        multimodal = best_by_modality["Multimodal"]
        unimodal_best = min((best_by_modality["EEG only"], best_by_modality["Face only"]), key=lambda row: row["rmse_mean"])
        multimodal_difference = float(unimodal_best["rmse_mean"] - multimodal["rmse_mean"])
        answers.append(
            f"The best multimodal RMSE differed from the best unimodal result ({unimodal_best['modality']}, "
            f"{unimodal_best['model']}) by {multimodal_difference:+.3f}; positive values favor multimodal."
        )
        paragraphs.append(f"### {target}\n\n" + " ".join(answers))

    predictive = summary.loc[summary["model"] != "Dummy mean"]
    best_targets = predictive.loc[predictive.groupby("target")["rmse_mean"].idxmin()].set_index("target")
    valence_rmse = float(best_targets.loc["valence_rating", "rmse_mean"])
    arousal_rmse = float(best_targets.loc["arousal_rating", "rmse_mean"])
    easier = "valence" if valence_rmse < arousal_rmse else "arousal"
    paragraphs.append(
        "### Target comparison\n\n"
        f"The lowest mean predictive-model RMSE was {valence_rmse:.3f} for valence and {arousal_rmse:.3f} "
        f"for arousal. On this common 1–7 scale, {easier} had the lower error in this baseline."
    )
    most_variable = summary.loc[summary["rmse_sd"].idxmax()]
    paragraphs.append(
        "### Fold variability\n\n"
        f"RMSE varied across all 25 outer test folds. The largest RMSE standard deviation was "
        f"{most_variable['rmse_sd']:.3f} for {most_variable['target']}, {most_variable['modality']}, "
        f"{most_variable['model']} (range {most_variable['rmse_min']:.3f}–{most_variable['rmse_max']:.3f}). "
        "Small mean differences should therefore not be treated as evidence of a stable advantage."
    )
    return "\n\n".join(paragraphs)


def classification_comparison_text(summary: pd.DataFrame) -> str:
    """Compare the best classifiers with the most-frequent dummy baseline."""
    paragraphs: list[str] = []
    best_by_target: dict[str, pd.Series] = {}
    for target in TARGETS:
        target_rows = summary.loc[summary["target"] == target]
        answers: list[str] = []
        for modality in ("EEG only", "Face only", "Multimodal"):
            modality_rows = target_rows.loc[target_rows["modality"] == modality]
            dummy_score = float(
                modality_rows.loc[
                    modality_rows["model"] == "Dummy most frequent",
                    "balanced_accuracy_mean",
                ].iloc[0]
            )
            candidates = modality_rows.loc[modality_rows["model"] != "Dummy most frequent"]
            best = candidates.loc[candidates["balanced_accuracy_mean"].idxmax()]
            difference = float(best["balanced_accuracy_mean"] - dummy_score)
            result_wording = "higher" if difference > 0 else "not higher"
            answers.append(
                f"{modality} best classifier was {best['model']} (balanced accuracy "
                f"{best['balanced_accuracy_mean']:.3f}), which was {result_wording} than the dummy "
                f"score of {dummy_score:.3f} (difference {difference:+.3f})."
            )
        best_by_target[target] = target_rows.loc[
            target_rows.loc[target_rows["model"] != "Dummy most frequent", "balanced_accuracy_mean"].idxmax()
        ]
        paragraphs.append(f"### {target}\n\n" + " ".join(answers))

    valence_best = best_by_target["valence_rating"]
    arousal_best = best_by_target["arousal_rating"]
    paragraphs.append(
        "### Classification target comparison\n\n"
        f"The highest mean balanced accuracy was {valence_best['balanced_accuracy_mean']:.3f} for valence "
        f"({valence_best['modality']}, {valence_best['model']}) and "
        f"{arousal_best['balanced_accuracy_mean']:.3f} for arousal "
        f"({arousal_best['modality']}, {arousal_best['model']})."
    )
    most_variable = summary.loc[summary["balanced_accuracy_sd"].idxmax()]
    paragraphs.append(
        "### Classification fold variability\n\n"
        f"The largest balanced-accuracy standard deviation was "
        f"{most_variable['balanced_accuracy_sd']:.3f} for {most_variable['target']}, "
        f"{most_variable['modality']}, {most_variable['model']} (range "
        f"{most_variable['balanced_accuracy_min']:.3f}–{most_variable['balanced_accuracy_max']:.3f}). "
        "Small improvements over 0.5 should not be treated as stable classification evidence."
    )
    return "\n\n".join(paragraphs)


def actual_exclusion_text(frame: pd.DataFrame) -> str:
    """Describe target missingness and fair-comparison exclusions from this dataset."""
    predictors = list(COMPACT_EEG_COLUMNS) + list(FACE_COLUMNS)
    lines: list[str] = []
    for target in TARGETS:
        missing_rating = frame[target].isna()
        valid = frame[target].notna() & frame[predictors].notna().all(axis=1)
        excluded = ~valid
        if "trigger" in frame.columns:
            missing_labels = [str(int(value)) for value in frame.loc[missing_rating, "trigger"]]
            excluded_labels = [str(int(value)) for value in frame.loc[excluded, "trigger"]]
            missing_text = ", ".join(missing_labels) if missing_labels else "none"
            excluded_text = ", ".join(excluded_labels) if excluded_labels else "none"
            lines.append(
                f"- {target}: {int(missing_rating.sum())} missing rating(s), triggers {missing_text}; "
                f"{int(excluded.sum())} fair-comparison exclusion(s), triggers {excluded_text}."
            )
        else:
            lines.append(
                f"- {target}: {int(missing_rating.sum())} missing rating(s); "
                f"{int(excluded.sum())} fair-comparison exclusion(s)."
            )
    return "\n".join(lines)


def build_report(
    regression_results: pd.DataFrame,
    regression_summary: pd.DataFrame,
    classification_results: pd.DataFrame,
    classification_summary: pd.DataFrame,
    participant: str,
    exclusion_text: str,
) -> str:
    """Build the combined continuous-regression and binary-classification report."""
    sample_sizes = regression_results.groupby("target")["sample_size"].first().to_dict()
    class_counts = (
        classification_results.groupby("target")[["low_class_count", "high_class_count"]]
        .first()
        .astype(int)
        .to_dict("index")
    )
    eeg_lines = "\n".join(f"- `{column}`" for column in COMPACT_EEG_COLUMNS)
    face_lines = "\n".join(f"- `{column}`" for column in FACE_COLUMNS)
    return f"""# {participant} compact-feature regression and binary-classification baselines

## Scope

This exploratory individual-participant analysis predicts {participant}'s own valence
and arousal ratings as continuous outcomes and as midpoint-defined binary
classes. OASIS ratings were not read, used as targets, or used as predictors.
Rows lacking complete EEG, face, or target data were excluded consistently
across all three modalities.

- Valence sample size: {sample_sizes['valence_rating']}
- Arousal sample size: {sample_sizes['arousal_rating']}
- Regression outer validation: {OUTER_FOLDS}-fold repeated KFold with {OUTER_REPEATS} repeats ({OUTER_FOLDS * OUTER_REPEATS} test folds), seed {int(regression_results['random_seed'].iloc[0])}
- Classification outer validation: {OUTER_FOLDS}-fold repeated stratified KFold with {OUTER_REPEATS} repeats ({OUTER_FOLDS * OUTER_REPEATS} test folds), seed {int(classification_results['random_seed'].iloc[0])}
- Regression inner selection: {INNER_FOLDS}-fold shuffled KFold using training data only and RMSE
- Classification inner selection: {INNER_FOLDS}-fold shuffled stratified KFold using training data only and balanced accuracy
- No imputation and no feature selection

Scaling was fitted inside each inner/outer training pipeline for models that
use it. Tree ensembles were not scaled because tree splits are invariant to
feature scale. The three modalities used identical outer folds for each target
and task.

## Binary target definition

- Low class (`0`): participant rating below 4.
- High class (`1`): participant rating equal to or above 4.
- Valence fair-subset counts: {class_counts['valence_rating']['low_class_count']} low and {class_counts['valence_rating']['high_class_count']} high.
- Arousal fair-subset counts: {class_counts['arousal_rating']['low_class_count']} low and {class_counts['arousal_rating']['high_class_count']} high.
- Participant-specific medians and four-quadrant labels were not used.

## Actual missing and excluded trials

{exclusion_text}

## Compact EEG predictors

Each feature is the mean across all eight channels (`C3`, `C4`, `Cz`, `Fz`,
`Oz`, `PO7`, `PO8`, and `Pz`), with all eight values required:

{eeg_lines}

## Facial predictors

{face_lines}

## Regression models and tuning

- Dummy: training-fold mean.
- Ridge: standardized predictors with `alpha` selected from 0.1, 1.0, 10.0,
  and 100.0.
- ElasticNet: standardized predictors with `alpha` selected from 0.01, 0.1,
  and 1.0 and `l1_ratio` from 0.1, 0.5, and 0.9; `max_iter=10000`.
- kNN: standardized predictors, uniform Euclidean distance weights, with
  `n_neighbors` selected from 3, 5, 7, and 9 inside each outer training fold.
- SVR: standardized predictors and RBF kernel with `gamma="scale"`; `C` selected
  from 0.1, 1.0, and 10.0 and `epsilon` from 0.1 and 0.2 inside each outer
  training fold.
- Random Forest: 100 trees with squared-error splits; `max_depth` selected from
  no limit and 5, `min_samples_leaf` from 1 and 3, and `max_features` from 1.0
  and `"sqrt"`.
- Extra Trees: the same small tree-count, criterion, depth, leaf-size, and
  feature-fraction grid as Random Forest.

## Regression outer-fold results

{markdown_summary_table(regression_summary)}

## Regression hyperparameter selections

The table reports the modal inner-CV choice across the 25 outer folds. Complete
fold-level selections remain in `baseline_regression_results.csv`.

{hyperparameter_frequency_table(regression_results, 'Dummy mean')}

## Regression comparisons

{comparison_text(regression_summary)}

## Classification models and tuning

- Dummy: most frequent training-fold class.
- L2 logistic regression: standardized predictors with `C` selected from 0.1,
  1.0, and 10.0 using the liblinear solver.
- Elastic-net logistic regression: standardized predictors with `C` selected
  from 0.1, 1.0, and 10.0 and `l1_ratio` from 0.1, 0.5, and 0.9 using saga.
- kNN: standardized predictors with `n_neighbors` selected from 3, 5, 7, and 9.
- RBF SVM: standardized predictors with `C` selected from 0.1, 1.0, and 10.0
  and fixed `gamma="scale"`.
- Gaussian Naive Bayes: standardized predictors with `var_smoothing` selected
  from 1e-11, 1e-9, and 1e-7.
- Random Forest classifier: 100 trees with Gini splits; `max_depth` selected
  from no limit and 5, `min_samples_leaf` from 1 and 3, and `max_features` from
  1.0 and `"sqrt"`.
- Extra Trees classifier: the same small tree-count, criterion, depth,
  leaf-size, and feature-fraction grid as Random Forest.

## Classification outer-fold results

Balanced accuracy is the primary classification metric. Confusion values are
sums across all 25 outer test folds, so every participant trial contributes
once per repeat rather than representing unique-trial counts.

{classification_summary_table(classification_summary)}

## Classification hyperparameter selections

{hyperparameter_frequency_table(classification_results, 'Dummy most frequent')}

## Classification comparisons

{classification_comparison_text(classification_summary)}

## Regression versus classification

RMSE and balanced accuracy measure different prediction tasks and cannot be
compared numerically. Classification provides stronger evidence than regression
only if its non-dummy models show a clear, stable improvement over balanced
accuracy 0.5 while regression models fail to improve meaningfully over their
training-mean dummy. Both tasks remain exploratory for this participant.

## Interpretation limits

These results describe one participant and a small number of trials. Model and
modality differences are exploratory. A model is not considered successful
because of one favorable fold or a small mean difference. Negative mean R² and
classification scores close to 0.5 both indicate weak out-of-sample evidence.
The explicit dummy rows provide direct baselines under identical outer folds.
"""


def build_rmse_figure(summary: pd.DataFrame, participant: str) -> plt.Figure:
    """Create readable target panels with model rows and modality-coded RMSE."""
    modalities = ("EEG only", "Face only", "Multimodal")
    models = ("Dummy mean", "Ridge", "ElasticNet", "kNN", "SVR-RBF", "Random Forest", "Extra Trees")
    colors = ("#4c78a8", "#f58518", "#54a24b")
    markers = ("o", "s", "^")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
    y_positions = np.arange(len(models), dtype=float)
    offsets = (-0.20, 0.0, 0.20)
    for axis, target in zip(axes, TARGETS):
        target_summary = summary.loc[summary["target"] == target]
        for modality, color, marker, offset in zip(modalities, colors, markers, offsets):
            modality_rows = target_summary.loc[target_summary["modality"] == modality].set_index("model").loc[list(models)]
            axis.errorbar(
                modality_rows["rmse_mean"],
                y_positions + offset,
                xerr=modality_rows["rmse_sd"],
                fmt=marker,
                markersize=5,
                capsize=3,
                linewidth=1.2,
                label=modality,
                color=color,
            )
        axis.set_title(target.replace("_rating", "").capitalize())
        axis.set_yticks(y_positions, models)
        axis.set_xlabel("Outer-fold RMSE (mean ± SD)")
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    axes[0].set_ylabel("Model")
    axes[1].legend(title="Modality", loc="lower right")
    figure.suptitle(f"{participant} baseline regression across repeated folds")
    return figure


def build_classification_figure(summary: pd.DataFrame, participant: str) -> plt.Figure:
    """Create target panels for balanced accuracy with fold variability."""
    modalities = ("EEG only", "Face only", "Multimodal")
    models = (
        "Dummy most frequent",
        "Logistic L2",
        "Logistic elastic-net",
        "kNN classifier",
        "SVM-RBF",
        "Gaussian NB",
        "Random Forest classifier",
        "Extra Trees classifier",
    )
    colors = ("#4c78a8", "#f58518", "#54a24b")
    markers = ("o", "s", "^")
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, constrained_layout=True)
    y_positions = np.arange(len(models), dtype=float)
    offsets = (-0.20, 0.0, 0.20)
    for axis, target in zip(axes, TARGETS):
        target_summary = summary.loc[summary["target"] == target]
        for modality, color, marker, offset in zip(modalities, colors, markers, offsets):
            modality_rows = target_summary.loc[target_summary["modality"] == modality].set_index("model").loc[list(models)]
            axis.errorbar(
                modality_rows["balanced_accuracy_mean"],
                y_positions + offset,
                xerr=modality_rows["balanced_accuracy_sd"],
                fmt=marker,
                markersize=5,
                capsize=3,
                linewidth=1.2,
                label=modality,
                color=color,
            )
        axis.axvline(0.5, color="#666666", linestyle="--", linewidth=1.0, label="Dummy level")
        axis.set_title(target.replace("_rating", "").capitalize())
        axis.set_yticks(y_positions, models)
        axis.set_xlabel("Outer-fold balanced accuracy (mean ± SD)")
        axis.set_xlim(0.25, 0.8)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    axes[0].set_ylabel("Classifier")
    handles, labels = axes[1].get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    axes[1].legend(unique_legend.values(), unique_legend.keys(), title="Comparison", loc="lower right")
    figure.suptitle(f"{participant} binary classification across repeated stratified folds")
    return figure


def summarize_feature_selection_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance for every target, model, and fixed Top-N value."""
    return (
        results.groupby(["target", "modality", "model", "top_n"], sort=False)
        .agg(
            folds=("balanced_accuracy", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        )
        .reset_index()
    )


def average_rank_and_frequency(
    rank_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Combine training-fold average ranks with best-Top-N selection counts."""
    rank_totals = {feature: 0.0 for feature in feature_columns}
    for value in rank_rows["feature_ranks"]:
        ranks = json.loads(value)
        for feature in feature_columns:
            rank_totals[feature] += float(ranks[feature])
    selection_counts = {feature: 0 for feature in feature_columns}
    for value in selection_rows["selected_features"]:
        for feature in json.loads(value):
            selection_counts[feature] += 1
    denominator = len(rank_rows)
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "average_rank": [rank_totals[feature] / denominator for feature in feature_columns],
            "selection_count": [selection_counts[feature] for feature in feature_columns],
        }
    ).sort_values(["selection_count", "average_rank", "feature"], ascending=[False, True, True])


def feature_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    """Format feature stability statistics as Markdown."""
    displayed = frame if limit is None else frame.head(limit)
    lines = [
        "| Feature | Average training-fold rank | Selected folds |",
        "|---|---:|---:|",
    ]
    for row in displayed.itertuples(index=False):
        lines.append(f"| `{row.feature}` | {row.average_rank:.2f} | {row.selection_count}/25 |")
    return "\n".join(lines)


def feature_selection_performance_table(summary: pd.DataFrame) -> str:
    """Format Top-N performance summaries as Markdown."""
    lines = [
        "| Target | Modality | Model | Top N | Balanced accuracy mean ± SD | Macro F1 | ROC AUC | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.modality} | {row.model} | {row.top_n} | "
            f"{row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_sd:.3f} | "
            f"{row.macro_f1_mean:.3f} | {row.roc_auc_mean:.3f} | {row.accuracy_mean:.3f} |"
        )
    return "\n".join(lines)


def baseline_balanced_accuracy(
    baseline_results: pd.DataFrame,
    target: str,
    modality: str,
    model: str,
) -> float:
    """Read one preserved compact/full baseline balanced-accuracy mean."""
    rows = baseline_results.loc[
        (baseline_results["task_type"] == "classification")
        & (baseline_results["target"] == target)
        & (baseline_results["modality"] == modality)
        & (baseline_results["model"] == model),
        "balanced_accuracy",
    ]
    if len(rows) != OUTER_FOLDS * OUTER_REPEATS:
        raise ValueError(
            f"Expected 25 preserved baseline rows for {target}, {modality}, {model}; found {len(rows)}."
        )
    return float(rows.mean())


def channel_and_family_summary(selection_rows: pd.DataFrame) -> tuple[str, str]:
    """Summarize selected EEG occurrences by channel and feature family."""
    channels: dict[str, int] = {channel: 0 for channel in EEG_CHANNELS}
    families: dict[str, int] = {family: 0 for family in EEG_FEATURE_FAMILIES}
    for value in selection_rows["selected_features"]:
        for feature in json.loads(value):
            family, channel = feature.rsplit("__", 1)
            channels[channel] += 1
            families[family] += 1
    channel_text = ", ".join(
        f"{name}={count}" for name, count in sorted(channels.items(), key=lambda item: (-item[1], item[0]))
    )
    family_text = ", ".join(
        f"{name}={count}" for name, count in sorted(families.items(), key=lambda item: (-item[1], item[0]))
    )
    return channel_text, family_text


def build_feature_selection_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_results: pd.DataFrame,
    participant: str,
) -> str:
    """Build the approved face and channel-specific EEG feature-selection report."""
    best = summary.loc[
        summary.groupby(["target", "modality", "model"])["balanced_accuracy_mean"].idxmax()
    ].sort_values(["modality", "target", "model"])
    best_lines = [
        "| Target | Modality | Model | Best Top N | Balanced accuracy | Macro F1 | ROC AUC | Previous baseline | Difference |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    comparison_paragraphs: list[str] = []
    stability_sections: list[str] = []

    face_best = best.loc[
        (best["target"] == "arousal_rating")
        & (best["modality"] == "Face only")
        & (best["model"] == "Logistic elastic-net")
    ].iloc[0]
    face_baseline = baseline_balanced_accuracy(
        baseline_results,
        "arousal_rating",
        "Face only",
        "Logistic elastic-net",
    )
    best_lines.append(
        f"| arousal_rating | Face only | Logistic elastic-net | {int(face_best['top_n'])} | "
        f"{face_best['balanced_accuracy_mean']:.3f} | {face_best['macro_f1_mean']:.3f} | "
        f"{face_best['roc_auc_mean']:.3f} | {face_baseline:.3f} | "
        f"{face_best['balanced_accuracy_mean'] - face_baseline:+.3f} |"
    )
    face_rank_rows = results.loc[
        (results["target"] == "arousal_rating")
        & (results["modality"] == "Face only")
        & (results["model"] == "Logistic elastic-net")
        & (results["top_n"] == max(FACE_TOP_N_GRID))
    ]
    face_selection_rows = results.loc[
        (results["target"] == "arousal_rating")
        & (results["modality"] == "Face only")
        & (results["model"] == "Logistic elastic-net")
        & (results["top_n"] == int(face_best["top_n"]))
    ]
    face_features = average_rank_and_frequency(face_rank_rows, face_selection_rows, list(FACE_COLUMNS))
    stability_sections.append(
        f"## Facial-feature stability\n\n"
        f"Selection counts refer to the best Top N ({int(face_best['top_n'])}) across 25 outer training folds. "
        "Average ranks use one full ranking per outer training fold.\n\n"
        f"{feature_table(face_features)}"
    )

    for row in best.loc[best["modality"] == "Channel-specific EEG"].itertuples(index=False):
        baseline = baseline_balanced_accuracy(baseline_results, row.target, "EEG only", row.model)
        best_lines.append(
            f"| {row.target} | Channel-specific EEG | {row.model} | {int(row.top_n)} | "
            f"{row.balanced_accuracy_mean:.3f} | {row.macro_f1_mean:.3f} | {row.roc_auc_mean:.3f} | "
            f"{baseline:.3f} | {row.balanced_accuracy_mean - baseline:+.3f} |"
        )
        rank_rows = results.loc[
            (results["target"] == row.target)
            & (results["modality"] == "Channel-specific EEG")
            & (results["model"] == "Logistic elastic-net")
            & (results["top_n"] == max(EEG_TOP_N_GRID))
        ]
        selection_rows = results.loc[
            (results["target"] == row.target)
            & (results["modality"] == "Channel-specific EEG")
            & (results["model"] == row.model)
            & (results["top_n"] == int(row.top_n))
        ]
        eeg_features = average_rank_and_frequency(
            rank_rows,
            selection_rows,
            list(CHANNEL_SPECIFIC_EEG_COLUMNS),
        )
        channel_text, family_text = channel_and_family_summary(selection_rows)
        stability_sections.append(
            f"## EEG stability: {row.target}, {row.model}\n\n"
            f"Best Top N: {int(row.top_n)}. The 15 most consistently selected channel-feature columns are shown; "
            "all fold selections and ranks remain in the CSV.\n\n"
            f"{feature_table(eeg_features, limit=15)}\n\n"
            f"Channel occurrence totals at the best Top N: {channel_text}.\n\n"
            f"Feature-family occurrence totals at the best Top N: {family_text}."
        )

    for row in best.itertuples(index=False):
        full_top_n = max(FACE_TOP_N_GRID) if row.modality == "Face only" else max(EEG_TOP_N_GRID)
        full_row = summary.loc[
            (summary["target"] == row.target)
            & (summary["modality"] == row.modality)
            & (summary["model"] == row.model)
            & (summary["top_n"] == full_top_n)
        ].iloc[0]
        change = float(row.balanced_accuracy_mean - full_row["balanced_accuracy_mean"])
        peak_text = (
            "performance declined after the smaller peak"
            if int(row.top_n) < full_top_n and change > 0
            else "the full feature set matched or exceeded the smaller subsets"
        )
        comparison_paragraphs.append(
            f"- {row.target}, {row.modality}, {row.model}: best Top N={int(row.top_n)} "
            f"(balanced accuracy {row.balanced_accuracy_mean:.3f}); full Top N={full_top_n} "
            f"({full_row['balanced_accuracy_mean']:.3f}); difference {change:+.3f}, so {peak_text}."
        )

    return f"""# {participant} leakage-safe feature-selection exploration

## Scope and validation

This analysis is limited to face-only arousal classification and channel-specific
EEG classification of valence and arousal. Binary labels use low `< 4` and high
`>= 4`. Every fixed Top-N value uses 5-fold repeated stratified KFold with five
repeats and seed 42. `SelectKBest(f_classif)`, optional scaling, inner model
tuning, and model fitting occur inside training folds only. No quadrant labels
or full-data feature ranking were used.

The face analysis uses 10 trial-level predictors with Top N from 1 through 10.
The EEG analysis preserves all 120 channel-specific predictors and tests Top N
values {list(EEG_TOP_N_GRID)}. The preserved baseline CSV is read only.

## Best configurations and baseline comparisons

Previous baseline means use the full 10 facial predictors or the 15 EEG
channel-mean predictors, as appropriate.

{chr(10).join(best_lines)}

## Top-N performance

{feature_selection_performance_table(summary)}

## Peak-versus-full-set patterns

{chr(10).join(comparison_paragraphs)}

{chr(10).join(stability_sections)}

## Interpretation limits

Top N was compared using the same repeated outer validation results, so the
best observed Top N is exploratory rather than an independently confirmed
choice. Selection frequency and average rank describe training-fold stability,
not causal importance. Small balanced-accuracy differences relative to fold
variability should not be treated as evidence of a robust model improvement.
"""


def build_feature_selection_figure(summary: pd.DataFrame, participant: str) -> plt.Figure:
    """Create the approved three-panel Top-N balanced-accuracy plot."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panel_definitions = (
        (axes[0], "arousal_rating", "Face only", FACE_TOP_N_GRID, ("Logistic elastic-net",), "Face arousal"),
        (
            axes[1],
            "valence_rating",
            "Channel-specific EEG",
            EEG_TOP_N_GRID,
            ("Logistic elastic-net", "Random Forest classifier"),
            "EEG valence",
        ),
        (
            axes[2],
            "arousal_rating",
            "Channel-specific EEG",
            EEG_TOP_N_GRID,
            ("Logistic elastic-net", "Random Forest classifier"),
            "EEG arousal",
        ),
    )
    colors = {"Logistic elastic-net": "#4c78a8", "Random Forest classifier": "#f58518"}
    for axis, target, modality, top_n_grid, models, title in panel_definitions:
        x_positions = np.arange(len(top_n_grid))
        for model in models:
            rows = (
                summary.loc[
                    (summary["target"] == target)
                    & (summary["modality"] == modality)
                    & (summary["model"] == model)
                ]
                .set_index("top_n")
                .loc[list(top_n_grid)]
            )
            axis.errorbar(
                x_positions,
                rows["balanced_accuracy_mean"],
                yerr=rows["balanced_accuracy_sd"],
                marker="o",
                capsize=3,
                linewidth=1.3,
                color=colors[model],
                label=model,
            )
        axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0, label="Dummy balanced accuracy")
        axis.set_xticks(x_positions, [str(value) for value in top_n_grid], rotation=45 if len(top_n_grid) > 10 else 0)
        axis.set_title(title)
        axis.set_xlabel("Top N selected predictors")
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(0.25, 0.8)
    axes[0].set_ylabel("Outer-fold balanced accuracy (mean ± SD)")
    handles, labels = axes[2].get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    axes[2].legend(unique_legend.values(), unique_legend.keys(), loc="lower right")
    figure.suptitle(f"{participant} leakage-safe feature-selection comparison")
    return figure


def run_feature_selection(context: AnalysisContext, seed: int, frame: pd.DataFrame) -> None:
    """Run only the approved feature-selection tasks and write three new artifacts."""
    output_filenames = feature_selection_output_filenames(context)
    expected_outputs = [context.output_dir / filename for filename in output_filenames]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite existing feature-selection outputs: {paths}")

    missing_eeg = sorted(set(CHANNEL_SPECIFIC_EEG_COLUMNS).difference(frame.columns))
    if missing_eeg:
        raise ValueError(f"Merged trial table is missing channel-specific EEG columns: {missing_eeg}")
    baseline_path = context.output_dir / baseline_output_filenames(context)[0]
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Preserved baseline comparison CSV was not found: {baseline_path}")
    baseline_results = pd.read_csv(baseline_path)

    rows: list[dict[str, Any]] = []
    rows.extend(
        evaluate_feature_selection_task(
            frame=frame,
            target="arousal_rating",
            modality="Face only",
            feature_columns=list(FACE_COLUMNS),
            top_n_grid=FACE_TOP_N_GRID,
            include_random_forest=False,
            seed=seed,
        )
    )
    for target in TARGETS:
        rows.extend(
            evaluate_feature_selection_task(
                frame=frame,
                target=target,
                modality="Channel-specific EEG",
                feature_columns=list(CHANNEL_SPECIFIC_EEG_COLUMNS),
                top_n_grid=EEG_TOP_N_GRID,
                include_random_forest=True,
                seed=seed,
            )
        )
    results = pd.DataFrame(rows)
    results.insert(0, "participant", context.participant)
    summary = summarize_feature_selection_results(results)
    report = build_feature_selection_report(results, summary, baseline_results, context.participant)
    figure = build_feature_selection_figure(summary, context.participant)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = context.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    results.to_csv(context.output_dir / output_filenames[0], index=False)
    (context.output_dir / output_filenames[1]).write_text(report, encoding="utf-8")
    figure.savefig(context.output_dir / output_filenames[2], dpi=200)
    plt.close(figure)

    best = summary.loc[
        summary.groupby(["target", "modality", "model"])["balanced_accuracy_mean"].idxmax()
    ].sort_values(["modality", "target", "model"])
    print(f"Wrote {len(results)} feature-selection fold rows for {context.participant} to {context.output_dir}")
    print(
        best[
            [
                "target",
                "modality",
                "model",
                "top_n",
                "balanced_accuracy_mean",
                "balanced_accuracy_sd",
                "macro_f1_mean",
                "roc_auc_mean",
            ]
        ].to_string(index=False)
    )


def prepare_context(
    context: AnalysisContext,
    feature_selection_only: bool,
) -> tuple[pd.DataFrame, list[Path]]:
    """Validate one input and all expected output collisions before fitting."""
    frame = pd.read_csv(context.trials)
    validate_input_columns(frame)
    frame = add_compact_eeg_predictors(frame)
    output_filenames = (
        feature_selection_output_filenames(context)
        if feature_selection_only
        else baseline_output_filenames(context)
    )
    expected_outputs = [context.output_dir / filename for filename in output_filenames]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {paths}")
    return frame, expected_outputs


def run_baseline(context: AnalysisContext, seed: int, frame: pd.DataFrame) -> None:
    """Run the unchanged baseline logic for one resolved participant."""
    output_filenames = baseline_output_filenames(context)

    regression_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        regression_rows.extend(evaluate_target(frame, target, seed))
    regression_results = pd.DataFrame(regression_rows)
    regression_results.insert(0, "participant", context.participant)
    regression_summary = summarize_results(regression_results)

    classification_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        classification_rows.extend(evaluate_classification_target(frame, target, seed))
    classification_results = pd.DataFrame(classification_rows)
    classification_results.insert(0, "participant", context.participant)
    classification_summary = summarize_classification_results(classification_results)

    combined_results = pd.concat([regression_results, classification_results], ignore_index=True, sort=False)
    report = build_report(
        regression_results,
        regression_summary,
        classification_results,
        classification_summary,
        context.participant,
        actual_exclusion_text(frame),
    )
    regression_figure = build_rmse_figure(regression_summary, context.participant)
    classification_figure = build_classification_figure(classification_summary, context.participant)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = context.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    combined_results.to_csv(context.output_dir / output_filenames[0], index=False)
    (context.output_dir / output_filenames[1]).write_text(report, encoding="utf-8")
    regression_figure.savefig(context.output_dir / output_filenames[2], dpi=200)
    classification_figure.savefig(context.output_dir / output_filenames[3], dpi=200)
    plt.close(regression_figure)
    plt.close(classification_figure)

    print(f"Wrote {len(combined_results)} fold-level rows for {context.participant} to {context.output_dir}")
    print(f"Regression rows: {len(regression_results)}; classification rows: {len(classification_results)}")
    print(f"Sample sizes: {regression_results.groupby('target')['sample_size'].first().to_dict()}")
    print("Regression summary:")
    print(
        regression_summary[
            ["target", "modality", "model", "rmse_mean", "rmse_sd", "mae_mean", "r2_mean"]
        ].to_string(index=False)
    )
    print("Classification summary:")
    print(
        classification_summary[
            [
                "target",
                "modality",
                "model",
                "balanced_accuracy_mean",
                "balanced_accuracy_sd",
                "macro_f1_mean",
                "roc_auc_mean",
                "accuracy_mean",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Resolve, preflight, and run the approved participant baselines."""
    args = parse_arguments()
    contexts, skipped = resolve_analysis_contexts(args)
    for reason in skipped:
        print(f"SKIP: {reason}")
    if not contexts:
        print("No participants remain to process after approved skips.")
        return

    prepared: list[tuple[AnalysisContext, pd.DataFrame, list[Path]]] = []
    for context in contexts:
        frame, expected_outputs = prepare_context(context, args.feature_selection_only)
        prepared.append((context, frame, expected_outputs))
        source_kind = "participant-aware" if context.trials.name.startswith(context.participant.lower()) else "historical fallback"
        print(
            f"PREFLIGHT {context.participant}: run={context.run_name or 'legacy'}, "
            f"trials={context.trials} ({source_kind}), rows={len(frame)}, output={context.output_dir}"
        )
        for path in expected_outputs:
            print(f"  planned: {path}")

    if args.preflight_only:
        print("Preflight complete; no models were fitted and no outputs were created.")
        return

    for context, frame, _ in prepared:
        if args.feature_selection_only:
            run_feature_selection(context, args.seed, frame)
        else:
            run_baseline(context, args.seed, frame)


if __name__ == "__main__":
    main()
