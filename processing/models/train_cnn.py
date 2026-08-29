"""General LOSO regression for saved no-ICA EEG CNN conditions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from processing.models.cnn_model import CNN_ARCHITECTURES, LateFusionEEGCNNRegressor, build_late_fusion_eeg_cnn, build_named_eeg_cnn
from processing.models.mlp_data import FACE_FEATURE_NAMES, load_general_eeg_epoch_data, load_general_multimodal_epoch_data, make_grouped_validation_split, make_loso_split, participant_channel_zscore_epochs
from processing.models.mlp_model import build_optimizer, build_regression_loss, evaluate_rating_classification, evaluate_regression, fit_with_early_stopping, predict_regression
from processing.models.train_mlp import _regression_dummy_metrics


ALL_PARTICIPANTS = tuple(f"P{index:02d}" for index in range(1, 47))
FULL_EPOCH_WINDOW = "full"
POSTSTIM_EPOCH_WINDOW = "poststim"
TRAINING_HISTORY_COLUMNS = ("held_out_participant", "epoch", "train_mse", "validation_mse", "is_best_validation", "epochs_without_improvement", "learning_rate")


def crop_epoch_window(epochs: np.ndarray, times: np.ndarray, epoch_window: str) -> tuple[np.ndarray, np.ndarray]:
    """Select the requested window using saved MNE timestamps, not indices."""
    epoch_array, epoch_times = np.asarray(epochs), np.asarray(times, dtype=float)
    if epoch_array.ndim != 3 or epoch_times.ndim != 1 or epoch_array.shape[2] != len(epoch_times):
        raise ValueError("epochs and times must have matching (trials, channels, time) dimensions")
    if epoch_window == FULL_EPOCH_WINDOW:
        return epoch_array, epoch_times
    if epoch_window != POSTSTIM_EPOCH_WINDOW:
        raise ValueError(f"Unknown epoch window: {epoch_window}")
    selected = (epoch_times >= 0.0) & (epoch_times <= 2.0)
    if not selected.any():
        raise ValueError("Saved epochs contain no timestamps in the requested 0.0..2.0 s window")
    return epoch_array[:, :, selected], epoch_times[selected]


class _LateFusionTrainingAdapter(torch.nn.Module):
    """Present separate EEG/face branches as one 2-D tensor to shared fit code."""

    def __init__(self, model: LateFusionEEGCNNRegressor, epoch_shape: tuple[int, int]) -> None:
        super().__init__()
        self.model = model
        self.epoch_shape = epoch_shape
        self.eeg_width = int(np.prod(epoch_shape))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        matrix = torch.as_tensor(features, dtype=torch.float32)
        expected = self.eeg_width + len(FACE_FEATURE_NAMES)
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise ValueError(f"Expected flattened EEG plus face matrix with {expected} columns")
        return self.model(matrix[:, :self.eeg_width].reshape(-1, *self.epoch_shape), matrix[:, self.eeg_width:])


def _validate_log_every(log_every: int) -> None:
    """Validate the requested concise per-fold training log interval."""
    if not isinstance(log_every, int) or isinstance(log_every, bool) or log_every <= 0:
        raise ValueError("log_every must be a positive integer")


def _cnn_history_callback(held_out_participant: str, log_every: int):
    """Return a compact loss-health printer for one CNN outer fold."""
    _validate_log_every(log_every)
    previous_train_loss: float | None = None
    best_validation_loss = float("inf")

    def report(record: dict[str, float | int | bool]) -> None:
        nonlocal previous_train_loss, best_validation_loss
        epoch, train_loss, validation_loss = int(record["epoch"]), float(record["train_mse"]), float(record["validation_mse"])
        if bool(record["non_finite_loss"]):
            print(f"[{held_out_participant}] TRAINING WARNING: NaN/Inf loss at epoch {epoch:03d}", flush=True)
        elif previous_train_loss is not None and train_loss > previous_train_loss * 10.0:
            print(f"[{held_out_participant}] TRAINING WARNING: train MSE rose sharply from {previous_train_loss:.6f} to {train_loss:.6f}", flush=True)
        if bool(record["is_best_validation"]):
            best_validation_loss = validation_loss
            print(f"[{held_out_participant}] new best validation MSE={validation_loss:.6f} at epoch {epoch:03d}", flush=True)
        if epoch == 1 or epoch % log_every == 0:
            best = validation_loss if bool(record["is_best_validation"]) else best_validation_loss
            print(f"[{held_out_participant}] epoch {epoch:03d} | train MSE={train_loss:.6f} | val MSE={validation_loss:.6f} | best={best:.6f}", flush=True)
        previous_train_loss = train_loss
    return report


def training_health_status(history: list[dict[str, float | int | bool]], best_validation_loss: float) -> str:
    """Classify numerical training health without using predictive-performance cutoffs."""
    finite_history = bool(history) and all(
        not bool(row["non_finite_loss"]) and np.isfinite(float(row["train_mse"])) and np.isfinite(float(row["validation_mse"]))
        for row in history
    )
    return "TRAINING_OK" if finite_history and np.isfinite(best_validation_loss) else "TRAINING_WARNING"


def _load_condition_data(derived_root: Path, participants: list[str] | tuple[str, ...], target: str, modality: str):
    """Load EEG-only or trigger-aligned EEG-plus-face data."""
    if modality == "multimodal":
        return load_general_multimodal_epoch_data(derived_root, participants, target)
    loaded = load_general_eeg_epoch_data(derived_root, participants, target, include_triggers=True, include_times=True)
    if len(loaded) == 7:
        epochs, ratings, groups, channels, sampling_hz, triggers, times = loaded
    elif len(loaded) == 6:
        epochs, ratings, groups, channels, sampling_hz, triggers = loaded
        times = np.arange(epochs.shape[2], dtype=float) / float(sampling_hz)
    else:
        epochs, ratings, groups, channels, sampling_hz = loaded
        triggers = np.full(len(ratings), np.nan)
        times = np.arange(epochs.shape[2], dtype=float) / float(sampling_hz)
    return epochs, ratings, groups, tuple(channels), float(sampling_hz), np.asarray(times), np.asarray(triggers), np.empty((len(ratings), 0)), [], []


def run_general_eeg_cnn_fold(derived_root: Path, participants: list[str] | tuple[str, ...], target: str, held_out_participant: str, seed: int = 42, progress: bool = False, architecture: str = "baseline", modality: str = "eeg", epoch_window: str = FULL_EPOCH_WINDOW, log_every: int = 10) -> dict[str, object]:
    """Train one LOSO fold for full/poststim EEG or poststim late fusion."""
    if modality not in {"eeg", "multimodal"}:
        raise ValueError("CNN modality must be eeg or multimodal")
    if modality == "multimodal" and epoch_window != POSTSTIM_EPOCH_WINDOW:
        raise ValueError("multimodal CNN requires --epoch-window poststim")
    if modality == "multimodal" and architecture != "baseline":
        raise ValueError("multimodal CNN uses the fixed baseline EEG branch")
    _validate_log_every(log_every)
    epochs, ratings, groups, channels, sampling_hz, times, triggers, faces, face_names, alignment = _load_condition_data(derived_root, participants, target, modality)
    cropped_epochs, cropped_times = crop_epoch_window(epochs, times, epoch_window)
    binary_labels = (ratings >= 4.0).astype(int)
    row_ids = np.arange(len(ratings), dtype=float).reshape(-1, 1)
    outer_rows, outer_labels, outer_groups, test_rows, _, test_groups = make_loso_split(row_ids, binary_labels, groups, held_out_participant)
    train_rows, _, train_groups, validation_rows, _, validation_groups = make_grouped_validation_split(outer_rows, outer_labels, outer_groups, held_out_participant, seed)
    train_indices, validation_indices, test_indices = (train_rows[:, 0].astype(int), validation_rows[:, 0].astype(int), test_rows[:, 0].astype(int))
    raw_epochs = {"train": cropped_epochs[train_indices], "validation": cropped_epochs[validation_indices], "test": cropped_epochs[test_indices]}
    row_groups = {"train": train_groups, "validation": validation_groups, "test": test_groups}
    eeg_tensors = {name: torch.tensor(participant_channel_zscore_epochs(raw_epochs[name], row_groups[name]), dtype=torch.float32) for name in raw_epochs}
    targets = {"train": torch.tensor(ratings[train_indices], dtype=torch.float32), "validation": torch.tensor(ratings[validation_indices], dtype=torch.float32), "test": torch.tensor(ratings[test_indices], dtype=torch.float32)}
    if modality == "eeg":
        model: torch.nn.Module = build_named_eeg_cnn(len(channels), architecture)
        tensors = eeg_tensors
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        face_scaling = None
    else:
        if face_names != list(FACE_FEATURE_NAMES):
            raise ValueError("Unexpected face feature schema")
        face_scaler = StandardScaler().fit(faces[train_indices])
        tensors = {name: torch.cat((eeg_tensors[name].flatten(start_dim=1), torch.tensor(face_scaler.transform(faces[indices]), dtype=torch.float32)), dim=1) for name, indices in {"train": train_indices, "validation": validation_indices, "test": test_indices}.items()}
        late_fusion = build_late_fusion_eeg_cnn(len(channels))
        model = _LateFusionTrainingAdapter(late_fusion, (len(channels), cropped_epochs.shape[2]))
        parameter_count = sum(parameter.numel() for parameter in late_fusion.parameters())
        face_scaling = "StandardScaler fit on inner-training face trials only"
    fit = fit_with_early_stopping(
        model, build_optimizer(model, learning_rate=.001), build_regression_loss(),
        tensors["train"], targets["train"], tensors["validation"], targets["validation"],
        max_epochs=200, batch_size=32, patience=20, seed=seed,
        history_callback=_cnn_history_callback(str(held_out_participant).upper(), log_every),
    )
    regression = {name: evaluate_regression(model, tensors[name], targets[name]) for name in tensors}
    classification = {name: evaluate_rating_classification(model, tensors[name], targets[name]) for name in ("validation", "test")}
    predicted_test_ratings = predict_regression(model, tensors["test"]).numpy()
    dummy_validation = _regression_dummy_metrics(ratings[train_indices], ratings[validation_indices])
    dummy_test = _regression_dummy_metrics(ratings[train_indices], ratings[test_indices])
    history = list(fit.get("history", []))
    health = training_health_status(history, float(fit["best_validation_loss"]))
    return {"held_out_participant": str(held_out_participant).upper(), "architecture": architecture, "modality": modality, "channels": channels, "sampling_hz": sampling_hz, "epoch_samples": cropped_epochs.shape[2], "epoch_window": {"name": epoch_window, "start_seconds": float(cropped_times[0]), "end_seconds": float(cropped_times[-1])}, "face_features": list(FACE_FEATURE_NAMES) if modality == "multimodal" else [], "face_scaling": face_scaling, "alignment": alignment, "normalization": "participant-wise transductive channel z-score over participant trials and selected time samples; no labels", "n_train": len(train_indices), "n_validation": len(validation_indices), "n_test": len(test_indices), "groups": {"train": sorted(set(train_groups)), "validation": sorted(set(validation_groups)), "test": sorted(set(test_groups))}, "parameter_count": parameter_count, "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_validation_mse": fit["best_validation_loss"], "training_history": history, "training_health": health, "stopped_early": bool(fit.get("stopped_early", False)), "non_finite_loss": bool(fit.get("non_finite_loss", False)), "regression": regression, "classification": classification, "dummy": {"training_mean_rating": dummy_validation["mean_training_rating"], "validation": dummy_validation, "test": dummy_test}, "validation_rmse_delta": regression["validation"]["rmse"] - dummy_validation["rmse"], "test_rmse_delta": regression["test"]["rmse"] - dummy_test["rmse"], "test_predictions": [{"held_out_participant": str(held_out_participant).upper(), "trigger": None if not np.isfinite(triggers[index]) else int(triggers[index]), "true_rating": float(ratings[index]), "predicted_rating": float(predicted), "true_binary_LOW_HIGH": int(ratings[index] >= 4.0), "predicted_binary_LOW_HIGH": int(predicted >= 4.0)} for index, predicted in zip(test_indices, predicted_test_ratings)]}


def print_cnn_result(result: dict[str, object]) -> None:
    """Print one concise CNN fold report."""
    print(f"held-out={result['held_out_participant']}; modality={result['modality']}; rows train/validation/test={result['n_train']}/{result['n_validation']}/{result['n_test']}")
    print(f"channels={result['channels']}; samples={result['epoch_samples']}; epoch window={result['epoch_window']}; sampling_hz={result['sampling_hz']}; parameters={result['parameter_count']}")
    for report in result["alignment"]:
        print(f"{report['participant']}: EEG epochs={report['eeg_epochs']}; face trials={report['face_trials']}; aligned multimodal trials={report['aligned_multimodal_trials']}; dropped EEG-only={report['dropped_eeg_only']}; dropped face-only={report['dropped_face_only']}")
    print(f"normalization={result['normalization']}")
    history = result["training_history"]
    if history:
        initial, final = history[0], history[-1]
        best_train = next(row["train_mse"] for row in history if bool(row["is_best_validation"]) and int(row["epoch"]) == result["best_epoch"])
        print(f"fold health: held-out={result['held_out_participant']}; best epoch={result['best_epoch']}; initial/final/train-at-best MSE={float(initial['train_mse']):.6f}/{float(final['train_mse']):.6f}/{float(best_train):.6f}; initial/best validation MSE={float(initial['validation_mse']):.6f}/{result['best_validation_mse']:.6f}")
    else:
        print(f"fold health: held-out={result['held_out_participant']}; no per-epoch history available")
    test, dummy_test = result["regression"]["test"], result["dummy"]["test"]
    print(f"test RMSE={test['rmse']:.6f}; test MAE={test['mae']:.6f}; dummy test RMSE={dummy_test['rmse']:.6f}; delta vs dummy={result['test_rmse_delta']:.6f}")
    if result["stopped_early"]:
        reason = "non-finite loss" if result["non_finite_loss"] else "patience exhausted"
        print(f"[{result['held_out_participant']}] early stopping: {reason}", flush=True)
    print(result["training_health"], flush=True)


FOLD_COLUMNS = ("held_out_participant", "architecture", "train_rows", "validation_rows", "test_rows", "best_epoch", "best_validation_mse", "validation_rmse", "validation_mae", "test_rmse", "test_mae", "dummy_validation_rmse", "dummy_validation_mae", "dummy_test_rmse", "dummy_test_mae", "validation_rmse_delta", "test_rmse_delta", "validation_ba", "test_ba", "test_accuracy")


def _fold_row(result: dict[str, object]) -> dict[str, object]:
    """Convert a completed fold to the persistent compact output schema."""
    return {"held_out_participant": result["held_out_participant"], "architecture": result["architecture"], "train_rows": result["n_train"], "validation_rows": result["n_validation"], "test_rows": result["n_test"], "best_epoch": result["best_epoch"], "best_validation_mse": result["best_validation_mse"], "validation_rmse": result["regression"]["validation"]["rmse"], "validation_mae": result["regression"]["validation"]["mae"], "test_rmse": result["regression"]["test"]["rmse"], "test_mae": result["regression"]["test"]["mae"], "dummy_validation_rmse": result["dummy"]["validation"]["rmse"], "dummy_validation_mae": result["dummy"]["validation"]["mae"], "dummy_test_rmse": result["dummy"]["test"]["rmse"], "dummy_test_mae": result["dummy"]["test"]["mae"], "validation_rmse_delta": result["validation_rmse_delta"], "test_rmse_delta": result["test_rmse_delta"], "validation_ba": result["classification"]["validation"]["balanced_accuracy"], "test_ba": result["classification"]["test"]["balanced_accuracy"], "test_accuracy": result["classification"]["test"]["accuracy"]}


def write_cnn_outputs(output_dir: Path, results: list[dict[str, object]]) -> None:
    """Write only the compact persistent tables requested for a new CNN run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_fold_row(result) for result in results]
    pd.DataFrame(rows, columns=FOLD_COLUMNS).to_csv(output_dir / "fold_results.csv", index=False)
    predictions = [prediction for result in results for prediction in result["test_predictions"]]
    pd.DataFrame(predictions, columns=("held_out_participant", "trigger", "true_rating", "predicted_rating", "true_binary_LOW_HIGH", "predicted_binary_LOW_HIGH")).to_csv(output_dir / "predictions.csv", index=False)
    if rows:
        frame = pd.DataFrame(rows)
        aggregate = pd.DataFrame([{"number_of_folds": len(frame), "mean_validation_rmse": frame.validation_rmse.mean(), "median_validation_rmse": frame.validation_rmse.median(), "mean_validation_rmse_delta": frame.validation_rmse_delta.mean(), "median_validation_rmse_delta": frame.validation_rmse_delta.median(), "number_validation_folds_beating_dummy": int((frame.validation_rmse_delta < 0).sum()), "fraction_validation_folds_beating_dummy": (frame.validation_rmse_delta < 0).mean(), "mean_test_rmse": frame.test_rmse.mean(), "median_test_rmse": frame.test_rmse.median(), "mean_test_rmse_delta": frame.test_rmse_delta.mean(), "median_test_rmse_delta": frame.test_rmse_delta.median(), "number_test_folds_beating_dummy": int((frame.test_rmse_delta < 0).sum()), "fraction_test_folds_beating_dummy": (frame.test_rmse_delta < 0).mean(), "mean_test_mae": frame.test_mae.mean(), "median_test_mae": frame.test_mae.median(), "mean_secondary_test_ba": frame.test_ba.mean(), "median_secondary_test_ba": frame.test_ba.median()}])
    else:
        aggregate = pd.DataFrame()
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)


def write_training_history(output_dir: Path, results: list[dict[str, object]]) -> None:
    """Refresh the compact, experiment-wide per-epoch loss history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"held_out_participant": result["held_out_participant"], **{column: record[column] for column in TRAINING_HISTORY_COLUMNS[1:]}}
        for result in results for record in result.get("training_history", [])
    ]
    pd.DataFrame(rows, columns=TRAINING_HISTORY_COLUMNS).to_csv(output_dir / "training_history.csv", index=False)


def write_cnn_run_config(output_dir: Path, participants: tuple[str, ...], target: str, modality: str, epoch_window: str, seed: int, result: dict[str, object], log_every: int = 10) -> None:
    """Save resolved settings and normalization details for a new run."""
    _validate_log_every(log_every)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"model": "late_fusion_eeg_cnn" if modality == "multimodal" else "cnn", "task": "general LOSO regression", "target": target, "modality": modality, "architecture": "late_fusion_baseline" if modality == "multimodal" else "baseline", "epoch_window": result["epoch_window"], "sampling_frequency": result["sampling_hz"], "channels": list(result["channels"]), "face_features": result["face_features"], "parameter_count": result["parameter_count"], "normalization": {"eeg": result["normalization"], "face": result["face_scaling"]}, "loss": "MSELoss", "optimizer": "Adam", "learning_rate": .001, "batch_size": 32, "max_epochs": 200, "patience": 20, "seed": seed, "log_every": log_every, "training_history_file": "training_history.csv", "validation_method": "LOSO outer test; deterministic 5-fold StratifiedGroupKFold inner validation; held-out participant excluded from both training and validation", "participants": list(participants), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def print_comparison(target: str, current_output: Path) -> None:
    """Print available full/poststim aggregates without retraining any run."""
    candidates = (("FULL EEG -0.5..2.0", Path("outputs/cnn_all_participants_regression_general") / target), ("POSTSTIM EEG 0..2.0", Path("outputs/cnn_poststim_regression_general") / target), ("POSTSTIM EEG + FACE", Path("outputs/cnn_poststim_multimodal_regression_general") / target))
    rows = []
    for label, directory in candidates:
        path = current_output / "aggregate_summary.csv" if directory.resolve() == current_output.resolve() else directory / "aggregate_summary.csv"
        if path.is_file(): rows.append((label, pd.read_csv(path).iloc[0]))
    if not rows: return
    print("\nAvailable CNN condition comparison")
    for label, row in rows:
        print(f"{label}: mean RMSE={row['mean_test_rmse']:.6f}; median RMSE={row['median_test_rmse']:.6f}; mean MAE={row['mean_test_mae']:.6f}; mean delta vs dummy={row['mean_test_rmse_delta']:.6f}; folds beating dummy={int(row['number_test_folds_beating_dummy'])}")
    baseline = next((row for label, row in rows if label.startswith("FULL EEG")), None)
    if baseline is not None:
        for label, row in rows:
            if not label.startswith("FULL EEG"): print(f"{label}: RMSE delta vs full-epoch CNN={row['mean_test_rmse'] - baseline['mean_test_rmse']:.6f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the original one-fold command and controlled persistent runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", help="Run one held-out participant only (legacy terminal-only mode).")
    parser.add_argument("--participants", default=",".join(ALL_PARTICIPANTS), help="LOSO cohort for persistent runs.")
    parser.add_argument("--target", choices=("valence", "arousal"), required=True)
    parser.add_argument("--modality", choices=("eeg", "multimodal"), default="eeg")
    parser.add_argument("--epoch-window", choices=(FULL_EPOCH_WINDOW, POSTSTIM_EPOCH_WINDOW), default=FULL_EPOCH_WINDOW)
    parser.add_argument("--output-dir", type=Path, help="New empty directory for persistent LOSO outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10, help="Print CNN loss health every N epochs (default: 10).")
    parser.add_argument("--architecture", choices=tuple(CNN_ARCHITECTURES), default="baseline")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    args.participants = tuple(item.strip().upper() for item in args.participants.split(",") if item.strip())
    if not args.participants or len(set(args.participants)) != len(args.participants): parser.error("--participants must be non-empty and contain no duplicates")
    if unknown := sorted(set(args.participants).difference(ALL_PARTICIPANTS)): parser.error(f"Unknown participant IDs: {unknown}")
    if args.modality == "multimodal" and args.epoch_window != POSTSTIM_EPOCH_WINDOW: parser.error("--modality multimodal requires --epoch-window poststim")
    if args.modality == "multimodal" and args.architecture != "baseline": parser.error("--modality multimodal uses the fixed baseline EEG branch")
    if args.participant and args.output_dir: parser.error("--participant is terminal-only and cannot be combined with --output-dir")
    if not args.participant and not args.output_dir: parser.error("Specify --participant for one legacy fold or --output-dir for a persistent LOSO run")
    if args.log_every <= 0: parser.error("--log-every must be a positive integer")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run a controlled CNN condition; persistent runs never overwrite outputs."""
    args = parse_args(argv)
    if args.participant:
        print_cnn_result(run_general_eeg_cnn_fold(Path("derived"), args.participants, args.target, args.participant, args.seed, args.progress, args.architecture, args.modality, args.epoch_window, args.log_every))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for index, participant in enumerate(args.participants, start=1):
        print(f"[{index}/{len(args.participants)}] held-out={participant} started", flush=True)
        result = run_general_eeg_cnn_fold(Path("derived"), args.participants, args.target, participant, args.seed, args.progress, args.architecture, args.modality, args.epoch_window, args.log_every)
        print_cnn_result(result); results.append(result)
        if index == 1: write_cnn_run_config(args.output_dir, args.participants, args.target, args.modality, args.epoch_window, args.seed, result, args.log_every)
        write_cnn_outputs(args.output_dir, results)
        write_training_history(args.output_dir, results)
        print(f"[{index}/{len(args.participants)}] held-out={participant} complete", flush=True)
    print_comparison(args.target, args.output_dir)


if __name__ == "__main__":
    main()
