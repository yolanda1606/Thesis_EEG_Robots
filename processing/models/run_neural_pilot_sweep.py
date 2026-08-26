"""Terminal-only pilot LOSO sweeps for existing MLP and CNN neural experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from processing.models.train_cnn import run_general_eeg_cnn_fold
from processing.models.cnn_model import CNN_ARCHITECTURES, build_named_eeg_cnn
from processing.models.train_mlp import (
    run_general_branched_multimodal_regression_experiment,
    run_general_eeg_normalization_experiment,
    run_general_modality_regression_experiment,
)


ALL_PARTICIPANTS = tuple(f"P{index:02d}" for index in range(1, 47))
DEFAULT_PILOT_PARTICIPANTS = ("P05", "P10", "P15", "P20", "P25", "P30", "P35", "P45")
ALLOWED_MODELS = ("mlp", "cnn")
ALLOWED_MODALITIES = ("eeg", "face", "multimodal", "branched-multimodal")
ALLOWED_NORMALIZATIONS = ("global_train_scaler", "participant_zscore")


def parse_csv_option(value: str, participant_ids: bool = False) -> tuple[str, ...]:
    """Parse a non-empty comma-separated option, normalizing participant IDs."""
    items = tuple(item.strip().upper() if participant_ids else item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("option must contain at least one value")
    return items


def _result_row(result: dict[str, object], model: str, configuration: str) -> dict[str, object]:
    """Collect common regression/dummy diagnostics from an existing fold result."""
    validation, test, dummy = result["regression"]["validation"], result["regression"]["test"], result["dummy"]
    return {
        "held_out_participant": result["groups"]["test"][0], "model": model, "configuration": configuration,
        "architecture": result.get("architecture", configuration.split(":")[1] if model == "cnn" else None),
        "modality": result.get("modality", "eeg"), "input_dim": result.get("input_dim"),
        "normalization_mode": result.get("normalization_mode", result.get("normalization", "global_train_scaler")),
        "train_rows": result["n_train"], "validation_rows": result["n_validation"], "test_rows": result["n_test"],
        "best_epoch": result["best_epoch"], "best_validation_mse": result["best_validation_mse"],
        "validation_rmse": validation["rmse"], "validation_mae": validation["mae"],
        "test_rmse": test["rmse"], "test_mae": test["mae"],
        "dummy_validation_rmse": dummy["validation"]["rmse"], "dummy_validation_mae": dummy["validation"]["mae"],
        "dummy_test_rmse": dummy["test"]["rmse"], "dummy_test_mae": dummy["test"]["mae"],
        "validation_rmse_delta": validation["rmse"] - dummy["validation"]["rmse"],
        "test_rmse_delta": test["rmse"] - dummy["test"]["rmse"],
        "validation_ba": result["classification"]["validation"]["balanced_accuracy"],
        "test_ba": result["classification"]["test"]["balanced_accuracy"],
        "test_accuracy": result["classification"]["test"]["accuracy"],
        "predicted_high_fraction": result["classification"]["test"].get("predicted_high_fraction", float("nan")),
        "test_predictions": result.get("test_predictions", []),
    }


def collect_neural_pilot_rows(
    held_out_participants: tuple[str, ...], models: tuple[str, ...], modalities: tuple[str, ...],
    target: str, seed: int, derived_root: Path = Path("derived"), progress: bool = False,
    normalizations: tuple[str, ...] | None = None,
    on_fold_complete: Callable[[list[dict[str, object]]], None] | None = None,
    cnn_architectures: tuple[str, ...] = ("baseline",),
) -> list[dict[str, object]]:
    """Dispatch to existing MLP/CNN one-fold functions; never train locally."""
    if normalizations and modalities != ("eeg",):
        raise ValueError("participant normalization pilot supports EEG modality only")
    rows: list[dict[str, object]] = []
    total = len(held_out_participants)
    for index, participant in enumerate(held_out_participants, start=1):
        if "mlp" in models:
            cells = (("eeg", normalization) for normalization in normalizations) if normalizations else ((modality, None) for modality in modalities)
            for modality, normalization in cells:
                configuration = f"mlp:{modality}:{normalization or 'default'}"
                if progress:
                    print(f"[{index}/{total}] MLP held-out={participant} ({configuration}) started", flush=True)
                if normalization:
                    result = run_general_eeg_normalization_experiment(
                        derived_root, ALL_PARTICIPANTS, target, participant, normalization, seed, progress
                    )
                elif modality == "branched-multimodal":
                    result = run_general_branched_multimodal_regression_experiment(
                        derived_root, ALL_PARTICIPANTS, target, participant, seed, progress
                    )
                else:
                    result = run_general_modality_regression_experiment(
                        derived_root, ALL_PARTICIPANTS, target, participant, modality, seed, progress
                    )
                rows.append(_result_row(result, "mlp", configuration))
                if on_fold_complete is not None:
                    on_fold_complete(rows)
                if progress:
                    print(f"[{index}/{total}] MLP held-out={participant} complete", flush=True)
        if "cnn" in models:
            for architecture in cnn_architectures:
                if progress:
                    print(f"[{index}/{total}] CNN {architecture} held-out={participant} started", flush=True)
                result = run_general_eeg_cnn_fold(
                    derived_root, ALL_PARTICIPANTS, target, participant, seed, progress, architecture
                )
                rows.append(_result_row(result, "cnn", f"cnn:{architecture}"))
                if on_fold_complete is not None:
                    on_fold_complete(rows)
                if progress:
                    print(f"[{index}/{total}] CNN {architecture} held-out={participant} complete", flush=True)
    return rows


def summarize_rows(rows: list[dict[str, object]], configurations: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Aggregate fixed fold results by model configuration without selection."""
    summary: dict[str, dict[str, object]] = {}
    for configuration in configurations:
        subset = [row for row in rows if row["configuration"] == configuration]
        if not subset:
            continue
        values = lambda key: np.asarray([float(row[key]) for row in subset], dtype=float)
        validation_delta, test_delta = values("validation_rmse_delta"), values("test_rmse_delta")
        summary[configuration] = {
            "folds_completed": len(subset),
            "mean_validation_rmse": float(values("validation_rmse").mean()),
            "median_validation_rmse": float(np.median(values("validation_rmse"))),
            "mean_validation_rmse_delta": float(validation_delta.mean()),
            "median_validation_rmse_delta": float(np.median(validation_delta)),
            "validation_dummy_beats": int((validation_delta < 0).sum()),
            "mean_test_rmse": float(values("test_rmse").mean()),
            "median_test_rmse": float(np.median(values("test_rmse"))),
            "mean_test_rmse_delta": float(test_delta.mean()),
            "median_test_rmse_delta": float(np.median(test_delta)),
            "test_dummy_beats": int((test_delta < 0).sum()),
            "mean_test_mae": float(values("test_mae").mean()),
            "mean_test_ba": float(values("test_ba").mean()),
            "individual_test_rmse_deltas": {str(row["held_out_participant"]): float(row["test_rmse_delta"]) for row in subset},
        }
    return summary


def print_neural_pilot_results(rows: list[dict[str, object]], summary: dict[str, dict[str, object]]) -> None:
    """Print per-fold and aggregate terminal-only results."""
    print("held_out model configuration                         val_RMSE dummy_val delta     test_RMSE dummy_test delta     test_MAE test_BA epoch")
    for row in rows:
        print(
            f"{row['held_out_participant']:8} {row['model']:5} {row['configuration']:37} "
            f"{row['validation_rmse']:8.4f} {row['dummy_validation_rmse']:9.4f} {row['validation_rmse_delta']:8.4f} "
            f"{row['test_rmse']:9.4f} {row['dummy_test_rmse']:10.4f} {row['test_rmse_delta']:8.4f} "
            f"{row['test_mae']:8.4f} {row['test_ba']:7.4f} {row['best_epoch']:5}"
        )
    for configuration, values in summary.items():
        print(f"\n{configuration} aggregate ({values['folds_completed']} folds)")
        print(f"  validation RMSE mean/median={values['mean_validation_rmse']:.4f}/{values['median_validation_rmse']:.4f}; delta mean/median={values['mean_validation_rmse_delta']:.4f}/{values['median_validation_rmse_delta']:.4f}; beats dummy={values['validation_dummy_beats']}")
        print(f"  test RMSE mean/median={values['mean_test_rmse']:.4f}/{values['median_test_rmse']:.4f}; delta mean/median={values['mean_test_rmse_delta']:.4f}/{values['median_test_rmse_delta']:.4f}; beats dummy={values['test_dummy_beats']}; mean MAE={values['mean_test_mae']:.4f}; mean secondary BA={values['mean_test_ba']:.4f}")
        print(f"  individual test RMSE deltas: {values['individual_test_rmse_deltas']}")


def print_cnn_mlp_pairs(rows: list[dict[str, object]]) -> None:
    """Print descriptive paired CNN-minus-MLP test RMSE differences only."""
    cnn_rows = {row["held_out_participant"]: row for row in rows if row["model"] == "cnn"}
    mlp_configurations = sorted({row["configuration"] for row in rows if row["model"] == "mlp"})
    for configuration in mlp_configurations:
        mlp_rows = {row["held_out_participant"]: row for row in rows if row["configuration"] == configuration}
        paired = {participant: cnn_rows[participant]["test_rmse"] - mlp_row["test_rmse"]
                  for participant, mlp_row in mlp_rows.items() if participant in cnn_rows}
        print(f"\npaired CNN test RMSE - {configuration} test RMSE: {paired}")


def print_cnn_architecture_comparison(summary: dict[str, dict[str, object]], architectures: tuple[str, ...]) -> None:
    """Print descriptive aggregate CNN architecture comparison without selection."""
    print("\nCNN architecture comparison (descriptive; do not select on outer-test results)")
    print("architecture                    mean_test_RMSE median_test_RMSE mean_delta median_delta folds_beating_dummy")
    for architecture in architectures:
        values = summary.get(f"cnn:{architecture}")
        if values is None:
            continue
        print(
            f"{architecture:31} {values['mean_test_rmse']:14.4f} {values['median_test_rmse']:16.4f} "
            f"{values['mean_test_rmse_delta']:10.4f} {values['median_test_rmse_delta']:12.4f} "
            f"{values['test_dummy_beats']:20}"
        )


FOLD_COLUMNS = (
    "held_out_participant", "architecture", "train_rows", "validation_rows", "test_rows", "best_epoch", "best_validation_mse",
    "validation_rmse", "validation_mae", "test_rmse", "test_mae", "dummy_validation_rmse",
    "dummy_validation_mae", "dummy_test_rmse", "dummy_test_mae", "validation_rmse_delta",
    "test_rmse_delta", "validation_ba", "test_ba", "test_accuracy",
)


def _cnn_aggregate_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build the requested one-row CNN aggregate from completed folds only."""
    if not rows:
        return pd.DataFrame()
    values = lambda key: np.asarray([float(row[key]) for row in rows], dtype=float)
    validation_delta, test_delta = values("validation_rmse_delta"), values("test_rmse_delta")
    return pd.DataFrame([{
        "number_of_folds": len(rows),
        "mean_validation_rmse": float(values("validation_rmse").mean()), "median_validation_rmse": float(np.median(values("validation_rmse"))),
        "mean_validation_rmse_delta": float(validation_delta.mean()), "median_validation_rmse_delta": float(np.median(validation_delta)),
        "number_validation_folds_beating_dummy": int((validation_delta < 0).sum()), "fraction_validation_folds_beating_dummy": float((validation_delta < 0).mean()),
        "mean_test_rmse": float(values("test_rmse").mean()), "median_test_rmse": float(np.median(values("test_rmse"))),
        "mean_test_rmse_delta": float(test_delta.mean()), "median_test_rmse_delta": float(np.median(test_delta)),
        "number_test_folds_beating_dummy": int((test_delta < 0).sum()), "fraction_test_folds_beating_dummy": float((test_delta < 0).mean()),
        "mean_test_mae": float(values("test_mae").mean()), "median_test_mae": float(np.median(values("test_mae"))),
        "mean_secondary_test_ba": float(values("test_ba").mean()), "median_secondary_test_ba": float(np.median(values("test_ba"))),
    }])


def write_cnn_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Atomically refresh the three incremental CNN result tables from completed rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cnn_rows = [row for row in rows if row["model"] == "cnn"]
    pd.DataFrame([
        {column: row.get(column, "baseline" if column == "architecture" else None) for column in FOLD_COLUMNS}
        for row in cnn_rows
    ], columns=FOLD_COLUMNS).to_csv(output_dir / "fold_results.csv", index=False)
    predictions = [prediction for row in cnn_rows for prediction in row["test_predictions"]]
    pd.DataFrame(predictions, columns=("held_out_participant", "trigger", "true_rating", "predicted_rating", "true_binary_LOW_HIGH", "predicted_binary_LOW_HIGH")).to_csv(output_dir / "predictions.csv", index=False)
    _cnn_aggregate_frame(cnn_rows).to_csv(output_dir / "aggregate_summary.csv", index=False)


def write_cnn_run_config(output_dir: Path, participants: tuple[str, ...], target: str, seed: int) -> None:
    """Write the static reproducibility record for the intended CNN full run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_named_eeg_cnn(8)
    config = {
        "model": "cnn", "task": "general regression", "target": target, "seed": seed,
        "architecture": "baseline",
        "architecture_definition": "Conv1d(8,16,k=7,p=3)->ReLU->MaxPool1d(2)->Conv1d(16,32,k=5,p=2)->ReLU->AdaptiveAvgPool1d(1)->Linear(32,16)->ReLU->Linear(16,1)",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "channel_order": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"], "epoch_samples": 626,
        "sampling_rate_hz": 250.0,
        "normalization_method": "participant-wise transductive channel z-score over participant trials and time; no labels",
        "loss": "MSELoss", "optimizer": "Adam", "learning_rate": .001, "batch_size": 32,
        "max_epochs": 200, "patience": 20, "participants": list(participants),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the terminal-only neural pilot sweep command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(DEFAULT_PILOT_PARTICIPANTS))
    parser.add_argument("--target", choices=("valence", "arousal"), default="valence")
    parser.add_argument("--models", default="mlp")
    parser.add_argument("--modalities", default="eeg,face,multimodal")
    parser.add_argument("--normalizations", help="Comma-separated: global_train_scaler,participant_zscore; MLP EEG only")
    parser.add_argument("--cnn-architectures", default="baseline", help="Comma-separated CNN architectures")
    parser.add_argument("--output-dir", type=Path, help="CNN-only persistent output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true", help="Print fold and epoch-level progress")
    args = parser.parse_args(argv)
    args.participants = parse_csv_option(args.participants, participant_ids=True)
    args.models = parse_csv_option(args.models)
    args.modalities = parse_csv_option(args.modalities)
    args.normalizations = parse_csv_option(args.normalizations) if args.normalizations else None
    args.cnn_architectures = parse_csv_option(args.cnn_architectures)
    if len(set(args.participants)) != len(args.participants): parser.error("--participants must not contain duplicates")
    if unknown := sorted(set(args.participants).difference(ALL_PARTICIPANTS)): parser.error(f"Unknown participant IDs: {unknown}")
    if invalid := sorted(set(args.models).difference(ALLOWED_MODELS)): parser.error(f"Invalid --models: {invalid}")
    if invalid := sorted(set(args.modalities).difference(ALLOWED_MODALITIES)): parser.error(f"Invalid --modalities: {invalid}")
    if invalid := sorted(set(args.normalizations or ()).difference(ALLOWED_NORMALIZATIONS)): parser.error(f"Invalid --normalizations: {invalid}")
    if invalid := sorted(set(args.cnn_architectures).difference(CNN_ARCHITECTURES)): parser.error(f"Invalid --cnn-architectures: {invalid}")
    if args.normalizations and "mlp" not in args.models: parser.error("--normalizations requires --models mlp")
    if args.normalizations and args.modalities != ("eeg",): parser.error("--normalizations requires --modalities eeg")
    if args.output_dir and args.models != ("cnn",): parser.error("--output-dir currently supports --models cnn only")
    if args.output_dir and args.cnn_architectures != ("baseline",): parser.error("--output-dir currently supports only the baseline CNN")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run a user-requested neural pilot sweep without output files."""
    args = parse_args(argv)
    if args.output_dir:
        write_cnn_run_config(args.output_dir, args.participants, args.target, args.seed)
        write_cnn_outputs(args.output_dir, [])
    callback = (lambda completed_rows: write_cnn_outputs(args.output_dir, completed_rows)) if args.output_dir else None
    rows = collect_neural_pilot_rows(args.participants, args.models, args.modalities, args.target, args.seed, progress=args.progress, normalizations=args.normalizations, on_fold_complete=callback, cnn_architectures=args.cnn_architectures)
    configurations = tuple(dict.fromkeys(row["configuration"] for row in rows))
    summary = summarize_rows(rows, configurations)
    print_neural_pilot_results(rows, summary)
    if set(args.models) == {"mlp", "cnn"}:
        print_cnn_mlp_pairs(rows)
    if "cnn" in args.models and len(args.cnn_architectures) > 1:
        print_cnn_architecture_comparison(summary, args.cnn_architectures)


if __name__ == "__main__":
    main()
