"""Focused tests for controlled post-stimulus and late-fusion CNN conditions."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from processing.models.cnn_model import build_eeg_cnn, build_late_fusion_eeg_cnn
from processing.models.mlp_data import FACE_FEATURE_NAMES, load_general_multimodal_epoch_data
from processing.models.mlp_model import build_optimizer, build_regression_loss, fit_with_early_stopping
from processing.models.train_cnn import (
    POSTSTIM_EPOCH_WINDOW, TRAINING_HISTORY_COLUMNS, crop_epoch_window, parse_args,
    run_general_eeg_cnn_fold, training_health_status, write_cnn_outputs,
    write_cnn_run_config, write_training_history,
)


class TestCNNWindowsAndModel(unittest.TestCase):
    def test_full_epoch_is_unchanged_and_poststim_uses_timestamps(self):
        epochs = np.zeros((2, 8, 626))
        times = np.arange(626, dtype=float) / 250.0 - .5
        full, full_times = crop_epoch_window(epochs, times, "full")
        poststim, poststim_times = crop_epoch_window(epochs, times, POSTSTIM_EPOCH_WINDOW)
        self.assertIs(full, epochs)
        np.testing.assert_array_equal(full_times, times)
        self.assertEqual(poststim.shape, (2, 8, 501))
        self.assertGreaterEqual(poststim_times[0], 0.0)
        self.assertEqual((float(poststim_times[0]), float(poststim_times[-1])), (0.0, 2.0))

    def test_late_fusion_has_separate_branches_and_expected_output(self):
        model = build_late_fusion_eeg_cnn(8)
        output = model(torch.randn(3, 8, 501), torch.randn(3, 10))
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertEqual(tuple(model.encode_eeg(torch.randn(3, 8, 501)).shape), (3, 16))
        self.assertEqual(tuple(model.encode_face(torch.randn(3, 10)).shape), (3, 4))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 4333)

    def test_original_eeg_cnn_interface_remains_compatible(self):
        model = build_eeg_cnn(8)
        self.assertEqual(tuple(model(torch.randn(4, 8, 626)).shape), (4, 1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 4049)


class TestMultimodalAlignment(unittest.TestCase):
    def test_trigger_alignment_drops_only_missing_face_trials(self):
        epochs = np.ones((3, 8, 4))
        source = (epochs, np.array([1., 5., 7.]), np.array(["P01"] * 3), ("Fz",) * 8, 250., np.array([10, 11, 12]), np.array([-.5, 0., .5, 1.]))
        table = pd.DataFrame({"trigger": [10, 12, 99], "valence_rating": [1., 7., 5.], **{name: [1., 2., 3.] for name in FACE_FEATURE_NAMES}})
        with patch("processing.models.mlp_data.load_general_eeg_epoch_data", return_value=source), patch("processing.models.mlp_data.train_classification.load_participant_table", return_value=(table, Path("synthetic.csv"))):
            aligned = load_general_multimodal_epoch_data(Path("derived"), ["P01"], "valence")
        aligned_epochs, ratings, groups, _, _, _, triggers, faces, names, report = aligned
        self.assertEqual(aligned_epochs.shape[0], 2)
        np.testing.assert_array_equal(ratings, [1., 7.])
        self.assertEqual(names, list(FACE_FEATURE_NAMES))
        self.assertEqual(report, [{"participant": "P01", "eeg_epochs": 3, "face_trials": 3, "aligned_multimodal_trials": 2, "dropped_eeg_only": 1, "dropped_face_only": 1}])


class TestLateFusionFoldAndOutputs(unittest.TestCase):
    def test_face_scaling_is_fit_only_on_inner_training_trials(self):
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        groups = np.repeat(participant_ids, 6)
        ratings = np.tile(np.array([1., 2., 3., 5., 6., 7.]), 6)
        epochs = np.random.default_rng(1).normal(size=(36, 8, 8))
        faces = np.arange(360, dtype=float).reshape(36, 10)
        source = (epochs, ratings, groups, ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"), 250., np.linspace(-.5, 2., 8), np.arange(36), faces, list(FACE_FEATURE_NAMES), [])
        no_op = {"epochs_run": 1, "best_epoch": 1, "best_validation_loss": 1., "final_training_loss": 1.}
        with patch("processing.models.train_cnn.load_general_multimodal_epoch_data", return_value=source), patch("processing.models.train_cnn.fit_with_early_stopping", return_value=no_op) as fit:
            result = run_general_eeg_cnn_fold(Path("derived"), participant_ids, "valence", "P15", modality="multimodal", epoch_window="poststim")
        training_features = fit.call_args.args[3]
        self.assertEqual(result["parameter_count"], 4333)
        self.assertTrue(torch.allclose(training_features[:, -10:].mean(dim=0), torch.zeros(10), atol=1e-6))
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])

    def test_metadata_records_window_modality_and_face_features(self):
        result = {"epoch_window": {"name": "poststim", "start_seconds": 0., "end_seconds": 2.}, "sampling_hz": 250., "channels": ("Fz",) * 8, "face_features": list(FACE_FEATURE_NAMES), "parameter_count": 4333, "normalization": "participant/channel", "face_scaling": "training only"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_cnn_run_config(output, ("P01",), "valence", "multimodal", "poststim", 42, result)
            config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["modality"], "multimodal")
        self.assertEqual(config["epoch_window"]["start_seconds"], 0.)
        self.assertEqual(config["face_features"], list(FACE_FEATURE_NAMES))

    def test_config_writer_creates_nonexistent_nested_directory(self):
        result = {"epoch_window": {"name": "poststim", "start_seconds": 0., "end_seconds": 2.}, "sampling_hz": 250., "channels": ("Fz",) * 8, "face_features": [], "parameter_count": 4049, "normalization": "participant/channel", "face_scaling": None}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "nested" / "cnn"
            write_cnn_run_config(output, ("P01",), "valence", "eeg", "poststim", 42, result)
            self.assertTrue((output / "run_config.json").is_file())

    def test_output_tables_remain_compact(self):
        result = {"held_out_participant": "P01", "architecture": "baseline", "modality": "eeg", "epoch_window": {"name": "poststim"}, "n_train": 10, "n_validation": 2, "n_test": 2, "best_epoch": 1, "best_validation_mse": 1., "regression": {"validation": {"rmse": 1., "mae": .8}, "test": {"rmse": 1.2, "mae": .9}}, "dummy": {"validation": {"rmse": 1.1, "mae": .9}, "test": {"rmse": 1.3, "mae": 1.}}, "validation_rmse_delta": -.1, "test_rmse_delta": -.1, "classification": {"validation": {"balanced_accuracy": .5}, "test": {"balanced_accuracy": .6, "accuracy": .7}}, "test_predictions": []}
        with tempfile.TemporaryDirectory() as directory:
            write_cnn_outputs(Path(directory), [result])
            self.assertEqual({path.name for path in Path(directory).iterdir()}, {"fold_results.csv", "aggregate_summary.csv", "predictions.csv"})

    def test_training_history_csv_has_requested_schema(self):
        result = {"held_out_participant": "P01", "training_history": [{"epoch": 1, "train_mse": 4., "validation_mse": 3., "is_best_validation": True, "epochs_without_improvement": 0, "learning_rate": .001, "non_finite_loss": False}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "run"
            write_training_history(output, [result])
            history = pd.read_csv(output / "training_history.csv")
        self.assertEqual(tuple(history.columns), TRAINING_HISTORY_COLUMNS)
        self.assertEqual(history.loc[0, "held_out_participant"], "P01")
        self.assertTrue(bool(history.loc[0, "is_best_validation"]))


class TestTrainingHistory(unittest.TestCase):
    def test_history_records_losses_and_best_epoch(self):
        model = build_eeg_cnn(2)
        x_train, y_train = torch.zeros((6, 2, 16)), torch.ones(6)
        x_validation, y_validation = torch.zeros((3, 2, 16)), torch.ones(3)
        result = fit_with_early_stopping(model, build_optimizer(model), build_regression_loss(), x_train, y_train, x_validation, y_validation, max_epochs=4, batch_size=3, patience=2)
        history = result["history"]
        self.assertEqual(len(history), result["epochs_run"])
        self.assertEqual(set(history[0]), {"epoch", "train_mse", "validation_mse", "is_best_validation", "epochs_without_improvement", "learning_rate", "non_finite_loss"})
        self.assertTrue(bool(history[int(result["best_epoch"]) - 1]["is_best_validation"]))
        self.assertTrue(all(np.isfinite(float(row["train_mse"])) and np.isfinite(float(row["validation_mse"])) for row in history))

    def test_early_stopping_history_records_stale_epochs(self):
        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.))

            def forward(self, features):
                return self.value.expand(len(features), 1)

        model = ConstantModel()
        x, y = torch.zeros((4, 2)), torch.ones(4)
        with patch("processing.models.mlp_model.train_one_epoch", return_value=1.):
            result = fit_with_early_stopping(model, build_optimizer(model), build_regression_loss(), x, y, x, y, max_epochs=10, batch_size=2, patience=2)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["history"][-1]["epochs_without_improvement"], 2)

    def test_non_finite_history_is_a_training_warning(self):
        history = [{"train_mse": float("nan"), "validation_mse": 1., "non_finite_loss": True}]
        self.assertEqual(training_health_status(history, 1.), "TRAINING_WARNING")
        self.assertEqual(training_health_status([{"train_mse": 1., "validation_mse": .5, "non_finite_loss": False}], .5), "TRAINING_OK")

    def test_log_every_requires_a_positive_integer(self):
        with self.assertRaises(SystemExit):
            parse_args(["--target", "valence", "--participant", "P01", "--log-every", "0"])
        self.assertEqual(parse_args(["--target", "valence", "--participant", "P01", "--log-every", "1"]).log_every, 1)


if __name__ == "__main__":
    unittest.main()
