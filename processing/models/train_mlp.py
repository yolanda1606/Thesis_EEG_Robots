"""Experiment orchestration for the EEG-only MLP work."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Re-export these established helpers for existing callers while keeping this
# module responsible only for experiment orchestration.
from processing.models.mlp_data import (  # noqa: F401
    DEFAULT_NO_ICA_SOURCE_RUN, EEG_INPUT_CONFIGURATIONS, EEGScaler,
    fit_eeg_scaler, get_eeg_input_configuration, load_general_eeg_data,
    load_general_eeg_rating_data, load_general_modality_rating_data, load_participant_eeg_data,
    make_grouped_validation_split, modality_rating_columns,
    make_loso_split, prepare_eeg_data, prepare_eeg_rating_data,
    participant_zscore_features, select_eeg_input_features, transform_eeg_features,
)
from processing.models.mlp_model import (  # noqa: F401
    TinyMLP, build_balanced_binary_loss, build_binary_loss, build_optimizer,
    build_regression_loss, build_small_mlp, evaluate_binary,
    evaluate_rating_classification, evaluate_regression, fit_with_early_stopping,
    predict_binary, prediction_diagnostics, train_one_epoch, train_one_step,
)
from processing.models.multimodal_mlp_model import (
    EEG_INPUT_DIM, FACE_INPUT_DIM, build_branched_multimodal_mlp,
)


def _fit_binary_experiment(x_train, y_train, x_validation, y_validation, x_test, y_test,
                           seed: int, hidden_dims: tuple[int, int], batch_size: int,
                           learning_rate: float = .001):
    """Fit one binary MLP with train-only scaling and validation stopping."""
    scaler = fit_eeg_scaler(x_train)
    tensors = {name: torch.tensor(transform_eeg_features(features, scaler), dtype=torch.float32)
               for name, features in {"train": x_train, "validation": x_validation, "test": x_test}.items()}
    labels = {"train": torch.tensor(y_train, dtype=torch.float32),
              "validation": torch.tensor(y_validation, dtype=torch.float32),
              "test": torch.tensor(y_test, dtype=torch.float32)}
    model = build_small_mlp(tensors["train"].shape[1], hidden_dims=hidden_dims)
    fit_result = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate), build_binary_loss(),
        tensors["train"], labels["train"], tensors["validation"], labels["validation"],
        max_epochs=200, batch_size=batch_size, patience=20, seed=seed,
    )
    metrics = {name: evaluate_binary(model, tensors[name], labels[name]) for name in tensors}
    diagnostics = {name: prediction_diagnostics(model, tensors[name], labels[name]) for name in tensors}
    return scaler, model, fit_result, metrics, diagnostics


def run_single_participant_experiment(derived_root: Path, participant: str, target: str, seed: int = 42,
                                      balanced_loss: bool = False, hidden_dims: tuple[int, int] = (16, 8),
                                      learning_rate: float = .001) -> dict[str, object]:
    """Run one leakage-safe participant EEG train/validation/test experiment."""
    x, y, _ = load_participant_eeg_data(derived_root, participant, target)
    if set(np.unique(y)) != {0, 1}: raise ValueError(f"{participant} {target} requires both LOW and HIGH labels")
    indices = np.arange(len(y))
    try:
        train_validation, test = train_test_split(indices, test_size=.20, stratify=y, random_state=seed)
        train, validation = train_test_split(train_validation, test_size=.25, stratify=y[train_validation], random_state=seed)
    except ValueError as error:
        raise ValueError(f"{participant} {target} cannot form stratified 60/20/20 train/validation/test splits: {error}") from error
    splits = {"train": train, "val": validation, "test": test}
    split_labels = {name: y[index] for name, index in splits.items()}
    if any(set(np.unique(labels)) != {0, 1} for labels in split_labels.values()):
        raise ValueError(f"{participant} {target} requires both LOW and HIGH labels in every split")
    scaler = fit_eeg_scaler(x[train])
    features = {name: torch.tensor(transform_eeg_features(x[index], scaler), dtype=torch.float32) for name, index in splits.items()}
    labels = {name: torch.tensor(value, dtype=torch.float32) for name, value in split_labels.items()}
    model = build_small_mlp(features["train"].shape[1], hidden_dims)
    loss_fn = build_balanced_binary_loss(labels["train"]) if balanced_loss else build_binary_loss()
    fit = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate), loss_fn, features["train"], labels["train"],
        features["val"], labels["val"], max_epochs=200, batch_size=16, patience=20, seed=seed,
    )
    metrics = {name: evaluate_binary(model, features[name], labels[name]) for name in splits}
    diagnostics = {name: prediction_diagnostics(model, features[name], labels[name]) for name in splits}
    counts = {name: {"low": int((value == 0).sum()), "high": int((value == 1).sum())} for name, value in split_labels.items()}
    return {"participant": participant, "target": target, "loss": "balanced_bce" if balanced_loss else "ordinary_bce", "architecture": (features["train"].shape[1], *hidden_dims, 1), "learning_rate": learning_rate, "n_total": len(y), "n_train": len(train), "n_val": len(validation), "n_test": len(test), "class_counts": counts, "diagnostics": diagnostics, "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"], "train_balanced_accuracy": metrics["train"]["balanced_accuracy"], "train_accuracy": metrics["train"]["accuracy"], "val_balanced_accuracy": metrics["val"]["balanced_accuracy"], "val_accuracy": metrics["val"]["accuracy"], "test_balanced_accuracy": metrics["test"]["balanced_accuracy"], "test_accuracy": metrics["test"]["accuracy"]}


def run_general_loso_fold_experiment(derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
                                     held_out_participant: str, seed: int = 42,
                                     input_configuration: str = "all_features_all_channels") -> dict[str, object]:
    """Train/evaluate one general binary EEG MLP LOSO fold."""
    x, y, groups, feature_names = load_general_eeg_data(derived_root, participants, target)
    if input_configuration == "all_features_all_channels":
        # Preserve the prior general experiment exactly, including support for
        # compact synthetic schemas used by the focused orchestration tests.
        selected_names = feature_names
    else:
        x, selected_names = select_eeg_input_features(x, feature_names, input_configuration)
    x_outer, y_outer, groups_outer, x_test, y_test, groups_test = make_loso_split(x, y, groups, held_out_participant)
    x_train, y_train, groups_train, x_val, y_val, groups_val = make_grouped_validation_split(x_outer, y_outer, groups_outer, held_out_participant, seed)
    _, _, fit, metrics, diagnostics = _fit_binary_experiment(x_train, y_train, x_val, y_val, x_test, y_test, seed, (32, 16), 32)
    configuration = get_eeg_input_configuration(input_configuration)
    return {"target": target, "held_out_participant": str(held_out_participant).upper(), "input_configuration": configuration.name, "selected_channels": configuration.channels, "selected_feature_families": configuration.feature_families, "feature_names": selected_names, "input_dim": len(selected_names), "architecture": (len(selected_names), 32, 16, 1), "learning_rate": .001, "groups": {"train": sorted(set(groups_train)), "validation": sorted(set(groups_val)), "test": sorted(set(groups_test))}, "class_counts": {"train": {"low": int((y_train == 0).sum()), "high": int((y_train == 1).sum())}, "validation": {"low": int((y_val == 0).sum()), "high": int((y_val == 1).sum())}, "test": {"low": int((y_test == 0).sum()), "high": int((y_test == 1).sum())}}, "n_train": len(y_train), "n_validation": len(y_val), "n_test": len(y_test), "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"], "metrics": metrics, "diagnostics": diagnostics}


def run_general_input_selection_comparison(derived_root: Path, participants: list[str] | tuple[str, ...],
                                           target: str = "valence", held_out_participant: str = "P15", seed: int = 42) -> list[dict[str, object]]:
    """Run the fixed general LOSO experiment once per named input configuration.

    Returned rows are ranked solely by validation balanced accuracy.
    """
    results = [run_general_loso_fold_experiment(derived_root, participants, target, held_out_participant, seed, name)
               for name in EEG_INPUT_CONFIGURATIONS]
    return sorted(results, key=lambda result: result["metrics"]["validation"]["balanced_accuracy"], reverse=True)


def run_overfitting_sanity_check(derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
                                 held_out_participant: str, seed: int = 42, epochs: int = 1000) -> dict[str, object]:
    """Attempt to memorize 32 LOW and 32 HIGH rows from inner training only."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0: raise ValueError("epochs must be a positive integer")
    x, y, groups, _ = load_general_eeg_data(derived_root, participants, target)
    x_outer, y_outer, groups_outer, _, _, _ = make_loso_split(x, y, groups, held_out_participant)
    x_train, y_train, groups_train, _, _, _ = make_grouped_validation_split(x_outer, y_outer, groups_outer, held_out_participant, seed)
    low, high = np.flatnonzero(y_train == 0), np.flatnonzero(y_train == 1)
    if len(low) < 32 or len(high) < 32: raise ValueError("Inner training data needs at least 32 LOW and 32 HIGH samples for the sanity check")
    rng = np.random.default_rng(seed); chosen = np.concatenate((rng.choice(low, 32, False), rng.choice(high, 32, False))); chosen = chosen[rng.permutation(len(chosen))]
    x_subset, y_subset, groups_subset = x_train[chosen], y_train[chosen], groups_train[chosen]
    scaler = fit_eeg_scaler(x_subset); x_tensor = torch.tensor(transform_eeg_features(x_subset, scaler), dtype=torch.float32); y_tensor = torch.tensor(y_subset, dtype=torch.float32)
    model = build_small_mlp(x_tensor.shape[1], (32, 16)); loss = build_binary_loss(); optimizer = build_optimizer(model)
    with torch.no_grad(): initial = float(loss(model(x_tensor), y_tensor.unsqueeze(1)))
    history = []
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, optimizer, loss, x_tensor, y_tensor, 16, True, seed + epoch - 1)
        if epoch % 50 == 0 or epoch == epochs:
            with torch.no_grad(): current = float(loss(model(x_tensor), y_tensor.unsqueeze(1)))
            history.append({"epoch": epoch, "training_loss": current, "training_balanced_accuracy": evaluate_binary(model, x_tensor, y_tensor)["balanced_accuracy"]})
    with torch.no_grad(): final = float(loss(model(x_tensor), y_tensor.unsqueeze(1)))
    metrics = evaluate_binary(model, x_tensor, y_tensor)
    return {"target": target, "held_out_participant": str(held_out_participant).upper(), "subset_size": len(y_subset), "class_counts": {"low": int((y_subset == 0).sum()), "high": int((y_subset == 1).sum())}, "subset_groups": sorted(set(groups_subset)), "initial_training_loss": initial, "final_training_loss": final, "training_balanced_accuracy": metrics["balanced_accuracy"], "training_accuracy": metrics["accuracy"], "diagnostics": prediction_diagnostics(model, x_tensor, y_tensor), "history": history}


def run_general_loso_regression_fold(derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
                                    held_out_participant: str, seed: int = 42,
                                    input_configuration: str = "all_features_all_channels") -> dict[str, object]:
    """Train one general LOSO EEG MLP for original-rating regression."""
    x, ratings, groups, feature_names = load_general_eeg_rating_data(derived_root, participants, target)
    if input_configuration == "all_features_all_channels":
        # Retain the established default regression path, including compact
        # synthetic schemas used in pre-existing focused tests.
        selected_names = feature_names
    else:
        x, selected_names = select_eeg_input_features(x, feature_names, input_configuration)
    binary = (ratings >= 4.).astype(int); row_ids = np.arange(len(ratings), dtype=float)
    x_outer, y_outer, groups_outer, x_test, _, _ = make_loso_split(np.column_stack((x, row_ids)), binary, groups, held_out_participant)
    x_train, _, groups_train, x_val, _, groups_val = make_grouped_validation_split(x_outer, y_outer, groups_outer, held_out_participant, seed)
    train_ratings, val_ratings, test_ratings = ratings[x_train[:, -1].astype(int)], ratings[x_val[:, -1].astype(int)], ratings[x_test[:, -1].astype(int)]
    x_train, x_val, x_test = x_train[:, :-1], x_val[:, :-1], x_test[:, :-1]
    scaler = fit_eeg_scaler(x_train); features = {name: torch.tensor(transform_eeg_features(value, scaler), dtype=torch.float32) for name, value in {"train": x_train, "validation": x_val, "test": x_test}.items()}; labels = {"train": torch.tensor(train_ratings, dtype=torch.float32), "validation": torch.tensor(val_ratings, dtype=torch.float32), "test": torch.tensor(test_ratings, dtype=torch.float32)}
    model = build_small_mlp(features["train"].shape[1], (32, 16)); fit = fit_with_early_stopping(model, build_optimizer(model), build_regression_loss(), features["train"], labels["train"], features["validation"], labels["validation"], 200, 32, 20, seed)
    regression = {name: evaluate_regression(model, features[name], labels[name]) for name in features}; classification = {name: evaluate_rating_classification(model, features[name], labels[name]) for name in ("validation", "test")}
    configuration = get_eeg_input_configuration(input_configuration)
    return {"target": target, "held_out_participant": str(held_out_participant).upper(), "input_configuration": configuration.name, "selected_channels": configuration.channels, "selected_feature_families": configuration.feature_families, "feature_names": selected_names, "input_dim": len(selected_names), "architecture": (features["train"].shape[1], 32, 16, 1), "learning_rate": .001, "n_train": len(train_ratings), "n_validation": len(val_ratings), "n_test": len(test_ratings), "groups": {"train": sorted(set(groups_train)), "validation": sorted(set(groups_val)), "test": [str(held_out_participant).upper()]}, "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"], "regression": regression, "classification": classification}


def run_general_regression_input_selection_comparison(
    derived_root: Path,
    participants: list[str] | tuple[str, ...],
    target: str = "valence",
    held_out_participant: str = "P15",
    seed: int = 42,
) -> list[dict[str, object]]:
    """Compare fixed general-rating MLP inputs, ranked only by validation RMSE."""
    results = [
        run_general_loso_regression_fold(
            derived_root, participants, target, held_out_participant, seed, name
        )
        for name in EEG_INPUT_CONFIGURATIONS
    ]
    return sorted(results, key=lambda result: result["regression"]["validation"]["rmse"])


def _regression_dummy_metrics(train_ratings: np.ndarray, evaluation_ratings: np.ndarray) -> dict[str, float]:
    """Evaluate the inner-training-mean rating baseline on an untouched split."""
    mean_rating = float(np.mean(train_ratings))
    residuals = np.asarray(evaluation_ratings, dtype=float) - mean_rating
    return {
        "mean_training_rating": mean_rating,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def _make_training_progress_reporter(label: str, enabled: bool):
    """Return concise epoch progress reporting without affecting optimization."""
    if not enabled:
        return None

    def report(epoch: int, training_loss: float, validation_loss: float, improved: bool) -> None:
        if improved or epoch % 10 == 0:
            suffix = " best" if improved else ""
            print(
                f"{label}: epoch {epoch}; train MSE={training_loss:.6f}; "
                f"validation MSE={validation_loss:.6f}{suffix}", flush=True
            )
    return report


def _load_fixed_modality_partitions(
    derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
    held_out_participant: str, modality: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, list[str]], dict[str, np.ndarray]]:
    """Load modality rows into the reference EEG-determined participant partitions."""
    reference_x, reference_ratings, reference_groups, _ = load_general_eeg_rating_data(
        derived_root, participants, target
    )
    reference_binary = (reference_ratings >= 4.0).astype(int)
    outer_x, outer_y, outer_groups, _, _, _ = make_loso_split(
        reference_x, reference_binary, reference_groups, held_out_participant
    )
    _, _, training_groups, _, _, validation_groups = make_grouped_validation_split(
        outer_x, outer_y, outer_groups, held_out_participant, seed
    )
    train_participants, validation_participants = set(training_groups), set(validation_groups)
    test_participant = str(held_out_participant).upper()
    x, ratings, groups, feature_names = load_general_modality_rating_data(
        derived_root, participants, target, modality, eeg_configuration="midline_bandpower"
    )
    train_mask = np.isin(groups, list(train_participants))
    validation_mask = np.isin(groups, list(validation_participants))
    test_mask = np.char.upper(groups.astype(str)) == test_participant
    if not (train_mask.any() and validation_mask.any() and test_mask.any()):
        raise ValueError(f"{modality} has no usable rows in one or more fixed split partitions")
    if np.any(train_mask & validation_mask) or np.any(train_mask & test_mask) or np.any(validation_mask & test_mask):
        raise RuntimeError("Modality split masks overlap")
    return (
        x[train_mask], ratings[train_mask], x[validation_mask], ratings[validation_mask],
        x[test_mask], ratings[test_mask], feature_names,
        {"train": sorted(train_participants), "validation": sorted(validation_participants), "test": [test_participant]},
        {"train": groups[train_mask], "validation": groups[validation_mask], "test": groups[test_mask]},
    )


def run_general_modality_regression_experiment(
    derived_root: Path,
    participants: list[str] | tuple[str, ...],
    target: str,
    held_out_participant: str,
    modality: str,
    seed: int = 42,
    progress: bool = False,
) -> dict[str, object]:
    """Run the fixed general rating MLP for EEG, face, or their concatenation.

    Reference EEG rows determine the LOSO and grouped validation participant
    assignments. Each modality then retains only its own usable rows within
    those fixed participant sets, preventing missing predictors from changing
    the selected participants or leaking test information.
    """
    x_train, y_train, x_validation, y_validation, x_test, y_test, feature_names, groups, _ = _load_fixed_modality_partitions(
        derived_root, participants, target, held_out_participant, modality, seed
    )
    scaler = fit_eeg_scaler(x_train)
    tensors = {
        name: torch.tensor(transform_eeg_features(features, scaler), dtype=torch.float32)
        for name, features in {"train": x_train, "validation": x_validation, "test": x_test}.items()
    }
    targets = {
        "train": torch.tensor(y_train, dtype=torch.float32),
        "validation": torch.tensor(y_validation, dtype=torch.float32),
        "test": torch.tensor(y_test, dtype=torch.float32),
    }
    model = build_small_mlp(tensors["train"].shape[1], hidden_dims=(32, 16))
    fit = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate=.001), build_regression_loss(),
        tensors["train"], targets["train"], tensors["validation"], targets["validation"],
        max_epochs=200, batch_size=32, patience=20, seed=seed,
        progress_callback=_make_training_progress_reporter(f"{held_out_participant} {modality}", progress),
    )
    regression = {name: evaluate_regression(model, tensors[name], targets[name]) for name in tensors}
    classification = {name: evaluate_rating_classification(model, tensors[name], targets[name]) for name in ("validation", "test")}
    dummy_validation, dummy_test = _regression_dummy_metrics(y_train, y_validation), _regression_dummy_metrics(y_train, y_test)
    return {
        "modality": modality, "feature_names": feature_names, "input_dim": len(feature_names),
        "n_train": len(y_train), "n_validation": len(y_validation), "n_test": len(y_test), "groups": groups,
        "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_mse": fit["best_validation_loss"],
        "regression": regression, "classification": classification,
        "dummy": {"training_mean_rating": dummy_validation["mean_training_rating"], "validation": dummy_validation, "test": dummy_test},
        "beats_dummy_validation_rmse": regression["validation"]["rmse"] < dummy_validation["rmse"],
        "beats_dummy_test_rmse": regression["test"]["rmse"] < dummy_test["rmse"],
    }


def run_general_branched_multimodal_regression_experiment(
    derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
    held_out_participant: str, seed: int = 42, progress: bool = False,
) -> dict[str, object]:
    """Run the fixed branched EEG-plus-face rating MLP on aligned usable rows."""
    x_train, y_train, x_validation, y_validation, x_test, y_test, feature_names, groups, _ = _load_fixed_modality_partitions(
        derived_root, participants, target, held_out_participant, "multimodal", seed
    )
    expected_dim = EEG_INPUT_DIM + FACE_INPUT_DIM
    if x_train.shape[1] != expected_dim or len(feature_names) != expected_dim:
        raise ValueError(f"Branched multimodal regression requires {expected_dim} EEG-first features")

    eeg_scaler = fit_eeg_scaler(x_train[:, :EEG_INPUT_DIM])
    face_scaler = fit_eeg_scaler(x_train[:, EEG_INPUT_DIM:])
    tensors = {}
    for name, features in {"train": x_train, "validation": x_validation, "test": x_test}.items():
        scaled_eeg = transform_eeg_features(features[:, :EEG_INPUT_DIM], eeg_scaler)
        scaled_face = transform_eeg_features(features[:, EEG_INPUT_DIM:], face_scaler)
        tensors[name] = torch.tensor(np.concatenate((scaled_eeg, scaled_face), axis=1), dtype=torch.float32)
    targets = {"train": torch.tensor(y_train, dtype=torch.float32), "validation": torch.tensor(y_validation, dtype=torch.float32), "test": torch.tensor(y_test, dtype=torch.float32)}
    model = build_branched_multimodal_mlp()
    fit = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate=.001), build_regression_loss(),
        tensors["train"], targets["train"], tensors["validation"], targets["validation"],
        max_epochs=200, batch_size=32, patience=20, seed=seed,
        progress_callback=_make_training_progress_reporter(f"{held_out_participant} branched-multimodal", progress),
    )
    regression = {name: evaluate_regression(model, tensors[name], targets[name]) for name in tensors}
    classification = {name: evaluate_rating_classification(model, tensors[name], targets[name]) for name in ("validation", "test")}
    dummy_validation, dummy_test = _regression_dummy_metrics(y_train, y_validation), _regression_dummy_metrics(y_train, y_test)
    return {
        "modality": "branched-multimodal", "feature_names": feature_names, "input_dim": expected_dim,
        "eeg_input_dim": EEG_INPUT_DIM, "face_input_dim": FACE_INPUT_DIM,
        "n_train": len(y_train), "n_validation": len(y_validation), "n_test": len(y_test), "groups": groups,
        "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_mse": fit["best_validation_loss"],
        "regression": regression, "classification": classification,
        "dummy": {"training_mean_rating": dummy_validation["mean_training_rating"], "validation": dummy_validation, "test": dummy_test},
        "beats_dummy_validation_rmse": regression["validation"]["rmse"] < dummy_validation["rmse"],
        "beats_dummy_test_rmse": regression["test"]["rmse"] < dummy_test["rmse"],
    }


def run_general_eeg_normalization_experiment(
    derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
    held_out_participant: str, normalization_mode: str, seed: int = 42, progress: bool = False,
) -> dict[str, object]:
    """Compare pooled train scaling with participant-wise transductive EEG normalization."""
    if normalization_mode == "global_train_scaler":
        result = run_general_modality_regression_experiment(
            derived_root, participants, target, held_out_participant, "eeg", seed, progress
        )
        result["normalization_mode"] = normalization_mode
        result["normalization_description"] = "Pooled scaler fit on inner-training rows only."
        return result
    if normalization_mode != "participant_zscore":
        raise ValueError("normalization_mode must be global_train_scaler or participant_zscore")

    x_train, y_train, x_validation, y_validation, x_test, y_test, feature_names, groups, row_groups = _load_fixed_modality_partitions(
        derived_root, participants, target, held_out_participant, "eeg", seed
    )
    if x_train.shape[1] != 20:
        raise ValueError("participant_zscore experiment requires 20 midline-bandpower EEG inputs")
    # No second global scaler: each partition is normalized only from its own
    # participants' unlabeled feature rows.
    tensors = {
        name: torch.tensor(participant_zscore_features(features, row_groups[name]), dtype=torch.float32)
        for name, features in {"train": x_train, "validation": x_validation, "test": x_test}.items()
    }
    targets = {"train": torch.tensor(y_train, dtype=torch.float32), "validation": torch.tensor(y_validation, dtype=torch.float32), "test": torch.tensor(y_test, dtype=torch.float32)}
    model = build_small_mlp(20, hidden_dims=(32, 16))
    fit = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate=.001), build_regression_loss(),
        tensors["train"], targets["train"], tensors["validation"], targets["validation"],
        max_epochs=200, batch_size=32, patience=20, seed=seed,
        progress_callback=_make_training_progress_reporter(f"{held_out_participant} eeg participant_zscore", progress),
    )
    regression = {name: evaluate_regression(model, tensors[name], targets[name]) for name in tensors}
    classification = {name: evaluate_rating_classification(model, tensors[name], targets[name]) for name in ("validation", "test")}
    dummy_validation, dummy_test = _regression_dummy_metrics(y_train, y_validation), _regression_dummy_metrics(y_train, y_test)
    return {
        "modality": "eeg", "normalization_mode": normalization_mode,
        "normalization_description": "Participant-wise transductive feature-only z-scoring; no additional global scaler.",
        "feature_names": feature_names, "input_dim": 20,
        "n_train": len(y_train), "n_validation": len(y_validation), "n_test": len(y_test), "groups": groups,
        "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_mse": fit["best_validation_loss"],
        "regression": regression, "classification": classification,
        "dummy": {"training_mean_rating": dummy_validation["mean_training_rating"], "validation": dummy_validation, "test": dummy_test},
        "beats_dummy_validation_rmse": regression["validation"]["rmse"] < dummy_validation["rmse"],
        "beats_dummy_test_rmse": regression["test"]["rmse"] < dummy_test["rmse"],
    }
def run_general_modality_regression_comparison(
    derived_root: Path, participants: list[str] | tuple[str, ...], target: str = "valence",
    held_out_participant: str = "P15", modalities: tuple[str, ...] = ("eeg", "face", "multimodal"),
    seed: int = 42, progress: bool = False,
) -> list[dict[str, object]]:
    """Run requested modalities and rank only by validation RMSE."""
    allowed = {"eeg", "face", "multimodal", "branched-multimodal"}
    if not modalities or set(modalities).difference(allowed):
        raise ValueError("modalities must be a non-empty subset of eeg, face, multimodal")
    results = [
        run_general_branched_multimodal_regression_experiment(
            derived_root, participants, target, held_out_participant, seed, progress
        ) if modality == "branched-multimodal" else run_general_modality_regression_experiment(
            derived_root, participants, target, held_out_participant, modality, seed, progress
        ) for modality in modalities
    ]
    return sorted(results, key=lambda result: result["regression"]["validation"]["rmse"])


def _print_modality_regression_comparison(results: list[dict[str, object]]) -> None:
    """Print the requested no-file terminal summary."""
    print("modality  dim  best_epoch  val_rmse  val_mae  test_rmse  test_mae  val_ba  test_ba")
    for result in results:
        validation = result["regression"]["validation"]
        test = result["regression"]["test"]
        print(
            f"{result['modality']:10} {result['input_dim']:>3} {result['best_epoch']:>11} "
            f"{validation['rmse']:>9.4f} {validation['mae']:>8.4f} "
            f"{test['rmse']:>10.4f} {test['mae']:>9.4f} "
            f"{result['classification']['validation']['balanced_accuracy']:>6.4f} "
            f"{result['classification']['test']['balanced_accuracy']:>7.4f}"
        )
        dummy = result["dummy"]
        print(
            f"  rows train/validation/test={result['n_train']}/{result['n_validation']}/{result['n_test']}; "
            f"dummy mean={dummy['training_mean_rating']:.6f}; "
            f"beats dummy RMSE: validation={result['beats_dummy_validation_rmse']}, "
            f"test={result['beats_dummy_test_rmse']}"
        )
        print(f"  features ({result['input_dim']}): {', '.join(result['feature_names'])}")
        print(f"  best validation MSE={result['best_validation_mse']:.6f}")
        for split in ("train", "validation", "test"):
            metrics = result["regression"][split]
            print(
                f"  {split}: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
                f"prediction mean/min/median/max={metrics['mean_predicted_rating']:.6f}/"
                f"{metrics['min_predicted_rating']:.6f}/{metrics['median_predicted_rating']:.6f}/"
                f"{metrics['max_predicted_rating']:.6f}"
            )
        for split in ("validation", "test"):
            diagnostics = result["classification"][split]
            print(
                f"  {split} LOW/HIGH: BA={diagnostics['balanced_accuracy']:.6f}, "
                f"accuracy={diagnostics['accuracy']:.6f}, TN/FP/FN/TP="
                f"{diagnostics['tn']}/{diagnostics['fp']}/{diagnostics['fn']}/{diagnostics['tp']}, "
                f"predicted HIGH fraction={diagnostics['predicted_high_fraction']:.6f}"
            )
        print(
            f"  dummy validation RMSE/MAE={dummy['validation']['rmse']:.6f}/{dummy['validation']['mae']:.6f}; "
            f"dummy test RMSE/MAE={dummy['test']['rmse']:.6f}/{dummy['test']['mae']:.6f}"
        )


def main(argv: list[str] | None = None) -> None:
    """Run one explicitly requested terminal-only MLP experiment."""
    parser = argparse.ArgumentParser(description="EEG/face MLP experiment runner")
    parser.add_argument("--experiment", choices=("general-modality-regression", "general-branched-multimodal-regression"), required=True)
    parser.add_argument("--participant", required=True, help="Outer held-out participant, e.g. P15")
    parser.add_argument("--target", choices=("valence", "arousal"), required=True)
    parser.add_argument("--modalities", help="Comma-separated: eeg,face,multimodal,branched-multimodal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true", help="Print epoch-level training progress")
    args = parser.parse_args(argv)
    cohort = [f"P{index:02d}" for index in range(1, 47)]
    if args.experiment == "general-branched-multimodal-regression":
        results = [run_general_branched_multimodal_regression_experiment(
            Path("derived"), cohort, args.target, args.participant, args.seed, args.progress
        )]
    else:
        if not args.modalities:
            parser.error("--modalities is required for general-modality-regression")
        modalities = tuple(item.strip() for item in args.modalities.split(",") if item.strip())
        results = run_general_modality_regression_comparison(
            Path("derived"), cohort, args.target, args.participant, modalities, args.seed, args.progress
        )
    _print_modality_regression_comparison(results)


if __name__ == "__main__":
    main()
