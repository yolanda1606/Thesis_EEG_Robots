"""One-fold general LOSO orchestration for the saved-epoch EEG CNN."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from processing.models.cnn_model import CNN_ARCHITECTURES, build_named_eeg_cnn
from processing.models.mlp_data import (
    load_general_eeg_epoch_data, make_grouped_validation_split, make_loso_split,
    participant_channel_zscore_epochs,
)
from processing.models.mlp_model import (
    build_optimizer, build_regression_loss, evaluate_rating_classification,
    evaluate_regression, fit_with_early_stopping, predict_regression,
)
from processing.models.train_mlp import _make_training_progress_reporter, _regression_dummy_metrics


def run_general_eeg_cnn_fold(
    derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
    held_out_participant: str, seed: int = 42, progress: bool = False, architecture: str = "baseline",
) -> dict[str, object]:
    """Train one participant-wise-transductively normalized no-ICA EEG CNN fold."""
    loaded_epochs = load_general_eeg_epoch_data(derived_root, participants, target, include_triggers=True)
    if len(loaded_epochs) == 6:
        epochs, ratings, groups, channels, sampling_hz, triggers = loaded_epochs
    else:  # Compatibility with synthetic test fixtures lacking trigger IDs.
        epochs, ratings, groups, channels, sampling_hz = loaded_epochs
        triggers = np.full(len(ratings), np.nan)
    binary_labels = (ratings >= 4.0).astype(int)
    row_ids = np.arange(len(ratings), dtype=float).reshape(-1, 1)
    outer_rows, outer_labels, outer_groups, test_rows, _, test_groups = make_loso_split(
        row_ids, binary_labels, groups, held_out_participant
    )
    train_rows, _, train_groups, validation_rows, _, validation_groups = make_grouped_validation_split(
        outer_rows, outer_labels, outer_groups, held_out_participant, seed
    )
    train_indices, validation_indices, test_indices = (
        train_rows[:, 0].astype(int), validation_rows[:, 0].astype(int), test_rows[:, 0].astype(int)
    )
    raw_epochs = {"train": epochs[train_indices], "validation": epochs[validation_indices], "test": epochs[test_indices]}
    row_groups = {"train": train_groups, "validation": validation_groups, "test": test_groups}
    # Participant/channel statistics span available trials and time samples;
    # no ratings are used, and validation/test use only their own features.
    tensors = {name: torch.tensor(participant_channel_zscore_epochs(raw_epochs[name], row_groups[name]), dtype=torch.float32)
               for name in raw_epochs}
    targets = {"train": torch.tensor(ratings[train_indices], dtype=torch.float32),
               "validation": torch.tensor(ratings[validation_indices], dtype=torch.float32),
               "test": torch.tensor(ratings[test_indices], dtype=torch.float32)}
    model = build_named_eeg_cnn(len(channels), architecture)
    fit = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate=.001), build_regression_loss(),
        tensors["train"], targets["train"], tensors["validation"], targets["validation"],
        max_epochs=200, batch_size=32, patience=20, seed=seed,
        progress_callback=_make_training_progress_reporter(f"{held_out_participant} EEG CNN {architecture}", progress),
    )
    regression = {name: evaluate_regression(model, tensors[name], targets[name]) for name in tensors}
    classification = {name: evaluate_rating_classification(model, tensors[name], targets[name]) for name in ("validation", "test")}
    predicted_test_ratings = predict_regression(model, tensors["test"]).numpy()
    dummy_validation = _regression_dummy_metrics(ratings[train_indices], ratings[validation_indices])
    dummy_test = _regression_dummy_metrics(ratings[train_indices], ratings[test_indices])
    return {
        "held_out_participant": str(held_out_participant).upper(), "architecture": architecture, "channels": channels,
        "sampling_hz": sampling_hz, "epoch_samples": epochs.shape[2],
        "normalization": "participant-wise transductive channel z-score over participant trials and time; no labels",
        "n_train": len(train_indices), "n_validation": len(validation_indices), "n_test": len(test_indices),
        "groups": {"train": sorted(set(train_groups)), "validation": sorted(set(validation_groups)), "test": sorted(set(test_groups))},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_mse": fit["best_validation_loss"],
        "regression": regression, "classification": classification,
        "dummy": {"training_mean_rating": dummy_validation["mean_training_rating"], "validation": dummy_validation, "test": dummy_test},
        "validation_rmse_delta": regression["validation"]["rmse"] - dummy_validation["rmse"],
        "test_rmse_delta": regression["test"]["rmse"] - dummy_test["rmse"],
        "test_predictions": [
            {
                "held_out_participant": str(held_out_participant).upper(),
                "trigger": None if not np.isfinite(triggers[index]) else int(triggers[index]),
                "true_rating": float(ratings[index]), "predicted_rating": float(predicted),
                "true_binary_LOW_HIGH": int(ratings[index] >= 4.0),
                "predicted_binary_LOW_HIGH": int(predicted >= 4.0),
            }
            for index, predicted in zip(test_indices, predicted_test_ratings)
        ],
    }


def print_cnn_result(result: dict[str, object]) -> None:
    """Print the one-fold terminal-only CNN report."""
    print(f"held-out={result['held_out_participant']}; rows train/validation/test={result['n_train']}/{result['n_validation']}/{result['n_test']}")
    print(f"channels={result['channels']}; samples={result['epoch_samples']}; sampling_hz={result['sampling_hz']}; parameters={result['parameter_count']}")
    print(f"normalization={result['normalization']}")
    print(f"best epoch={result['best_epoch']}; best validation MSE={result['best_validation_mse']:.6f}")
    for split in ("train", "validation", "test"):
        metrics = result["regression"][split]
        print(f"{split}: RMSE={metrics['rmse']:.6f}; MAE={metrics['mae']:.6f}; mean/min/median/max prediction={metrics['mean_predicted_rating']:.6f}/{metrics['min_predicted_rating']:.6f}/{metrics['median_predicted_rating']:.6f}/{metrics['max_predicted_rating']:.6f}")
    for split in ("validation", "test"):
        metrics = result["classification"][split]
        print(f"{split} LOW/HIGH: BA={metrics['balanced_accuracy']:.6f}; accuracy={metrics['accuracy']:.6f}; TN/FP/FN/TP={metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}")
    dummy = result["dummy"]
    print(f"dummy mean={dummy['training_mean_rating']:.6f}; validation RMSE/MAE={dummy['validation']['rmse']:.6f}/{dummy['validation']['mae']:.6f}; test RMSE/MAE={dummy['test']['rmse']:.6f}/{dummy['test']['mae']:.6f}")
    print(f"validation/test RMSE delta vs dummy={result['validation_rmse_delta']:.6f}/{result['test_rmse_delta']:.6f}")


def main(argv: list[str] | None = None) -> None:
    """Run one explicitly requested no-output CNN LOSO fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--target", choices=("valence", "arousal"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=tuple(CNN_ARCHITECTURES), default="baseline")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    result = run_general_eeg_cnn_fold(
        Path("derived"), [f"P{index:02d}" for index in range(1, 47)], args.target,
        args.participant, args.seed, args.progress, args.architecture,
    )
    print_cnn_result(result)


if __name__ == "__main__":
    main()
