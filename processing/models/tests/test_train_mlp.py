"""Focused checks for the first EEG-only MLP experiment implementation."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from processing.models.train_mlp import (
    build_binary_loss,
    build_regression_loss,
    build_balanced_binary_loss,
    build_optimizer,
    build_small_mlp,
    evaluate_binary,
    fit_with_early_stopping,
    fit_eeg_scaler,
    load_general_eeg_data,
    make_loso_split,
    make_grouped_validation_split,
    predict_binary,
    prediction_diagnostics,
    run_single_participant_experiment,
    run_general_loso_fold_experiment,
    run_overfitting_sanity_check,
    run_general_loso_regression_fold,
    run_general_regression_input_selection_comparison,
    run_general_modality_regression_comparison,
    run_general_modality_regression_experiment,
    run_general_branched_multimodal_regression_experiment,
    run_general_eeg_normalization_experiment,
    prepare_eeg_data,
    train_one_step,
    train_one_epoch,
    transform_eeg_features,
    EEG_INPUT_CONFIGURATIONS,
    select_eeg_input_features,
)
from processing.multimodal_image.modeling import train_classification
from processing.models.mlp_data import (
    FACE_FEATURE_NAMES,
    modality_rating_columns,
    prepare_modality_rating_data,
    participant_zscore_features,
    participant_channel_zscore_epochs,
)
from processing.models.cnn_model import CNN_ARCHITECTURES, build_eeg_cnn, build_named_eeg_cnn
from processing.models.train_cnn import main as cnn_main, run_general_eeg_cnn_fold
from processing.models.run_neural_pilot_sweep import (
    collect_neural_pilot_rows, main as neural_pilot_sweep_main, summarize_rows,
    parse_args as parse_neural_pilot_args, write_cnn_outputs, write_cnn_run_config,
)
from processing.models.multimodal_mlp_model import (
    EEG_INPUT_DIM, FACE_INPUT_DIM, FUSED_EMBEDDING_DIM, build_branched_multimodal_mlp,
)


def eeg_table() -> pd.DataFrame:
    """Return a minimal complete EEG table with LOW and HIGH ratings."""
    table = pd.DataFrame(
        {
            column: [float(index), float(index + 1), float(index + 2)]
            for index, column in enumerate(train_classification.EEG_COLUMNS)
        }
    )
    table["valence_rating"] = [3.9, 4.0, 7.0]
    table["arousal_rating"] = [1.0, 4.0, 6.0]
    return table


class TestEEGDataPreparation(unittest.TestCase):
    def test_returns_the_expected_number_of_eeg_predictors(self):
        x, _, feature_names = prepare_eeg_data(eeg_table(), "P15", "valence")
        self.assertEqual(x.shape[1], len(train_classification.EEG_COLUMNS))
        self.assertEqual(len(feature_names), len(train_classification.EEG_COLUMNS))
        self.assertEqual(feature_names, train_classification.EEG_COLUMNS)

    def test_returns_low_and_high_as_binary_labels(self):
        _, y, _ = prepare_eeg_data(eeg_table(), "P15", "valence")
        np.testing.assert_array_equal(y, np.array([0, 1, 1]))
        self.assertTrue(set(y).issubset({0, 1}))

    def test_missing_or_invalid_rows_match_classical_preparation(self):
        table = eeg_table()
        table.loc[1, train_classification.EEG_COLUMNS[0]] = np.nan
        table["valence_rating"] = table["valence_rating"].astype(object)
        table.loc[2, "valence_rating"] = "not-a-rating"

        expected_x, expected_y, expected_names = train_classification.prepare_modality_data(
            table, "P15", "valence", "eeg"
        )
        x, y, names = prepare_eeg_data(table, "P15", "valence")

        np.testing.assert_array_equal(x, expected_x.to_numpy(dtype=float))
        np.testing.assert_array_equal(y, expected_y.to_numpy(dtype=int))
        self.assertEqual(names, expected_names)
        self.assertEqual(x.shape[0], 1)


class TestEEGInputSelection(unittest.TestCase):
    def setUp(self):
        self.names = list(train_classification.EEG_COLUMNS)
        self.x = np.arange(3 * len(self.names), dtype=float).reshape(3, -1)

    def test_each_named_configuration_has_expected_feature_count(self):
        expected = {
            "all_features_all_channels": 120,
            "bandpower_all_channels": 40,
            "bandpower_entropy_all_channels": 80,
            "posterior_bandpower": 20,
            "midline_bandpower": 20,
            "frontal_central_bandpower": 20,
        }
        self.assertEqual(set(EEG_INPUT_CONFIGURATIONS), set(expected))
        for name, count in expected.items():
            selected, feature_names = select_eeg_input_features(self.x, self.names, name)
            self.assertEqual(selected.shape[1], count)
            self.assertEqual(len(feature_names), count)

    def test_channel_and_feature_family_filters_use_existing_names(self):
        _, names = select_eeg_input_features(self.x, self.names, "posterior_bandpower")
        self.assertTrue(all(name.split("__")[0] in {"eeg_bp_delta", "eeg_bp_theta", "eeg_bp_alpha", "eeg_bp_beta", "eeg_bp_gamma"} for name in names))
        self.assertTrue(all(name.split("__")[1] in {"Pz", "PO7", "Oz", "PO8"} for name in names))

    def test_selection_order_is_stable_and_default_preserves_all_features(self):
        first_x, first_names = select_eeg_input_features(self.x, self.names)
        second_x, second_names = select_eeg_input_features(self.x, self.names)
        np.testing.assert_array_equal(first_x, self.x)
        self.assertEqual(first_names, self.names)
        np.testing.assert_array_equal(first_x, second_x)
        self.assertEqual(first_names, second_names)

    def test_missing_requested_column_raises_clear_error(self):
        missing_names = self.names.copy()
        missing_names.remove("eeg_bp_delta__Pz")
        with self.assertRaisesRegex(ValueError, "Missing EEG columns.*eeg_bp_delta__Pz"):
            select_eeg_input_features(self.x[:, :-1], missing_names, "posterior_bandpower")


class TestModalityRatingData(unittest.TestCase):
    def setUp(self):
        self.table = eeg_table()
        for index, column in enumerate(FACE_FEATURE_NAMES):
            self.table[column] = [float(index), float(index + 1), float(index + 2)]

    def test_face_uses_exact_classical_video_columns(self):
        self.assertEqual(modality_rating_columns("face"), train_classification.FACE_COLUMNS)
        self.assertEqual(len(modality_rating_columns("face")), 10)

    def test_midline_eeg_selection_has_twenty_inputs(self):
        columns = modality_rating_columns("eeg")
        self.assertEqual(len(columns), 20)
        self.assertTrue(all(column.split("__")[1] in {"Fz", "Cz", "Pz", "Oz"} for column in columns))

    def test_multimodal_concatenates_eeg_then_face_and_drops_missing_rows(self):
        self.table.loc[1, FACE_FEATURE_NAMES[0]] = np.nan
        x, ratings, names = prepare_modality_rating_data(self.table, "P15", "valence", "multimodal")
        self.assertEqual(names, modality_rating_columns("eeg") + modality_rating_columns("face"))
        self.assertEqual(x.shape, (2, 30))
        np.testing.assert_array_equal(ratings, np.array([3.9, 7.0]))


class TestGeneralModalityRegression(unittest.TestCase):
    @staticmethod
    def cohort(feature_count: int = 2):
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        groups = np.repeat(participant_ids, 6)
        ratings = np.tile(np.array([1., 2., 3., 5., 6., 7.]), len(participant_ids))
        participant_code = np.repeat(np.array([1., 2., 3., 4., 5., 15.]), 6)
        x = np.column_stack([participant_code + index for index in range(feature_count)])
        return x, ratings, groups, [f"feature_{index}" for index in range(feature_count)]

    @staticmethod
    def no_op_fit_result():
        return {"epochs_run": 1, "best_epoch": 1, "best_validation_loss": 1., "final_training_loss": 1.}

    def test_fixed_groups_scaler_and_modality_dummy_are_training_only(self):
        reference = self.cohort(3)
        modality = self.cohort(2)
        with patch("processing.models.train_mlp.load_general_eeg_rating_data", return_value=reference):
            with patch("processing.models.train_mlp.load_general_modality_rating_data", return_value=modality):
                with patch("processing.models.train_mlp.fit_eeg_scaler", wraps=fit_eeg_scaler) as scaler:
                    with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
                        result = run_general_modality_regression_experiment(
                            Path("derived"), ["P01", "P02", "P03", "P04", "P05", "P15"],
                            "valence", "P15", "face",
                        )
        self.assertEqual(result["input_dim"], 2)
        self.assertEqual((result["n_train"], result["n_validation"], result["n_test"]), (24, 6, 6))
        self.assertTrue(np.all(scaler.call_args.args[0][:, 0] != 15.0))
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])
        self.assertAlmostEqual(result["dummy"]["training_mean_rating"], 4.0)

    def test_comparison_ranks_only_by_validation_rmse(self):
        synthetic = [
            {"modality": "eeg", "regression": {"validation": {"rmse": 2.}, "test": {"rmse": 1.}}},
            {"modality": "face", "regression": {"validation": {"rmse": 1.}, "test": {"rmse": 3.}}},
        ]
        with patch("processing.models.train_mlp.run_general_modality_regression_experiment", side_effect=synthetic):
            ranked = run_general_modality_regression_comparison(Path("derived"), ["P01"], modalities=("eeg", "face"))
        self.assertEqual([result["modality"] for result in ranked], ["face", "eeg"])


class TestMLPPilotSweep(unittest.TestCase):
    @staticmethod
    def experiment_result(held_out: str, modality: str, validation_rmse: float, test_rmse: float) -> dict[str, object]:
        return {
            "groups": {"test": [held_out]}, "modality": modality,
            "input_dim": {"eeg": 20, "face": 10, "multimodal": 30, "branched-multimodal": 30}[modality],
            "n_train": 100, "n_validation": 20, "n_test": 10,
            "best_epoch": 3, "best_validation_mse": 1.5,
            "regression": {
                "validation": {"rmse": validation_rmse, "mae": 1.0},
                "test": {"rmse": test_rmse, "mae": 1.2},
            },
            "dummy": {
                "validation": {"rmse": 2.0, "mae": 1.4},
                "test": {"rmse": 2.0, "mae": 1.5},
            },
            "classification": {
                "validation": {"balanced_accuracy": .5},
                "test": {"balanced_accuracy": .6, "accuracy": .7, "predicted_high_fraction": .4},
            },
        }

    def test_collects_one_row_per_fixed_fold_and_modality_without_reimplementing_experiment(self):
        def fake_experiment(_root, _cohort, _target, held_out, modality, _seed, _progress=False):
            return self.experiment_result(held_out, modality, 1.5, 1.8)

        with patch("processing.models.run_neural_pilot_sweep.run_general_modality_regression_experiment", side_effect=fake_experiment) as runner:
            rows = collect_neural_pilot_rows(("P05", "P15"), ("mlp",), ("eeg", "face"), "valence", 42)
        self.assertEqual(runner.call_count, 4)
        self.assertEqual([(row["held_out_participant"], row["modality"]) for row in rows], [("P05", "eeg"), ("P05", "face"), ("P15", "eeg"), ("P15", "face")])
        self.assertAlmostEqual(rows[0]["validation_rmse_delta"], -0.5)
        self.assertAlmostEqual(rows[0]["test_rmse_delta"], -0.2)

    def test_aggregate_reports_dummy_deltas_and_individual_test_results(self):
        rows = [
            {"held_out_participant": "P05", "configuration": "mlp:eeg:default", "validation_rmse": 1.5, "validation_rmse_delta": -.5, "test_rmse": 1.8, "test_rmse_delta": -.2, "test_mae": 1.1, "test_ba": .6},
            {"held_out_participant": "P15", "configuration": "mlp:eeg:default", "validation_rmse": 2.5, "validation_rmse_delta": .5, "test_rmse": 2.2, "test_rmse_delta": .2, "test_mae": 1.3, "test_ba": .4},
        ]
        summary = summarize_rows(rows, ("mlp:eeg:default",))
        self.assertEqual(summary["mlp:eeg:default"]["folds_completed"], 2)
        self.assertAlmostEqual(summary["mlp:eeg:default"]["mean_validation_rmse"], 2.0)
        self.assertEqual(summary["mlp:eeg:default"]["validation_dummy_beats"], 1)
        self.assertEqual(summary["mlp:eeg:default"]["test_dummy_beats"], 1)
        self.assertEqual(summary["mlp:eeg:default"]["individual_test_rmse_deltas"], {"P05": -.2, "P15": .2})

    def test_branched_multimodal_routes_to_the_existing_branched_experiment(self):
        result = self.experiment_result("P15", "branched-multimodal", 1.5, 1.8)
        result["input_dim"] = 30
        with patch("processing.models.run_neural_pilot_sweep.run_general_branched_multimodal_regression_experiment", return_value=result) as branched:
            with patch("processing.models.run_neural_pilot_sweep.run_general_modality_regression_experiment") as flat:
                rows = collect_neural_pilot_rows(("P15",), ("mlp",), ("branched-multimodal",), "valence", 42)
        self.assertEqual(branched.call_count, 1)
        flat.assert_not_called()
        self.assertEqual(rows[0]["modality"], "branched-multimodal")

    def test_cnn_routes_to_the_existing_cnn_fold_experiment(self):
        result = self.experiment_result("P15", "eeg", 1.5, 1.8)
        with patch("processing.models.run_neural_pilot_sweep.run_general_eeg_cnn_fold", return_value=result) as cnn:
            with patch("processing.models.run_neural_pilot_sweep.run_general_modality_regression_experiment") as mlp:
                rows = collect_neural_pilot_rows(("P15",), ("cnn",), ("eeg",), "valence", 42)
        cnn.assert_called_once()
        mlp.assert_not_called()
        self.assertEqual(rows[0]["model"], "cnn")

    def test_cnn_sweep_aggregates_each_architecture_separately(self):
        def fake_cnn(_root, _cohort, _target, held_out, _seed, _progress, architecture):
            result = self.experiment_result(held_out, "eeg", 1.5 if architecture == "baseline" else 1.4, 1.8)
            result["architecture"] = architecture
            return result

        completed_row_counts = []
        with patch("processing.models.run_neural_pilot_sweep.run_general_eeg_cnn_fold", side_effect=fake_cnn):
            rows = collect_neural_pilot_rows(
                ("P15",), ("cnn",), ("eeg",), "valence", 42,
                on_fold_complete=lambda completed: completed_row_counts.append(len(completed)),
                cnn_architectures=("baseline", "larger_kernels"),
            )
        summary = summarize_rows(rows, ("cnn:baseline", "cnn:larger_kernels"))
        self.assertEqual(set(summary), {"cnn:baseline", "cnn:larger_kernels"})
        self.assertEqual(summary["cnn:baseline"]["folds_completed"], 1)
        self.assertEqual(completed_row_counts, [1, 2])

    def test_cnn_architecture_comparison_cli_uses_the_computed_summary(self):
        rows = [
            {
                "held_out_participant": "P15", "configuration": "cnn:baseline",
                "validation_rmse": 1.5, "validation_rmse_delta": -0.1,
                "test_rmse": 1.8, "test_rmse_delta": -0.2,
                "test_mae": 1.2, "test_ba": 0.6,
            },
            {
                "held_out_participant": "P15", "configuration": "cnn:larger_kernels",
                "validation_rmse": 1.4, "validation_rmse_delta": -0.2,
                "test_rmse": 1.7, "test_rmse_delta": -0.3,
                "test_mae": 1.1, "test_ba": 0.7,
            },
        ]
        with patch("processing.models.run_neural_pilot_sweep.collect_neural_pilot_rows", return_value=rows):
            with patch("processing.models.run_neural_pilot_sweep.print_neural_pilot_results"):
                with patch("processing.models.run_neural_pilot_sweep.print_cnn_architecture_comparison") as comparison:
                    neural_pilot_sweep_main([
                        "--participants", "P15", "--models", "cnn",
                        "--cnn-architectures", "baseline,larger_kernels",
                    ])
        comparison.assert_called_once()
        self.assertEqual(
            set(comparison.call_args.args[0]), {"cnn:baseline", "cnn:larger_kernels"}
        )

    def test_neural_pilot_cli_accepts_arousal_for_the_existing_cnn_experiment(self):
        args = parse_neural_pilot_args([
            "--participants", "P15", "--models", "cnn", "--target", "arousal",
        ])
        self.assertEqual(args.target, "arousal")

    def test_one_fold_cnn_cli_forwards_arousal_without_changing_the_experiment(self):
        with patch("processing.models.train_cnn.run_general_eeg_cnn_fold", return_value={}) as runner:
            with patch("processing.models.train_cnn.print_cnn_result"):
                cnn_main(["--participant", "P15", "--target", "arousal"])
        self.assertEqual(runner.call_args.args[2:4], ("arousal", "P15"))


class TestBranchedMultimodalMLP(unittest.TestCase):
    def test_fixed_branch_dimensions_embedding_and_scalar_output(self):
        model = build_branched_multimodal_mlp()
        features = torch.zeros((3, EEG_INPUT_DIM + FACE_INPUT_DIM), dtype=torch.float32)
        eeg, face = model.split_inputs(features)
        self.assertEqual(eeg.shape[1], 20)
        self.assertEqual(face.shape[1], 10)
        embedding = torch.cat((model.encode_eeg(eeg), model.encode_face(face)), dim=1)
        self.assertEqual(embedding.shape, (3, FUSED_EMBEDDING_DIM))
        self.assertEqual(model(features).shape, (3, 1))

    def test_initialization_is_deterministic(self):
        first, second = build_branched_multimodal_mlp(), build_branched_multimodal_mlp()
        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter)

    def test_branched_experiment_uses_aligned_rows_separate_training_scalers_and_no_p15_leakage(self):
        reference = TestGeneralModalityRegression.cohort(3)
        multimodal = TestGeneralModalityRegression.cohort(30)
        with patch("processing.models.train_mlp.load_general_eeg_rating_data", return_value=reference):
            with patch("processing.models.train_mlp.load_general_modality_rating_data", return_value=multimodal):
                with patch("processing.models.train_mlp.fit_eeg_scaler", wraps=fit_eeg_scaler) as scaler:
                    with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=TestGeneralModalityRegression.no_op_fit_result()):
                        result = run_general_branched_multimodal_regression_experiment(
                            Path("derived"), ["P01", "P02", "P03", "P04", "P05", "P15"], "valence", "P15"
                        )
        self.assertEqual((result["n_train"], result["n_validation"], result["n_test"]), (24, 6, 6))
        self.assertEqual(len(scaler.call_args_list), 2)
        self.assertEqual(scaler.call_args_list[0].args[0].shape[1], EEG_INPUT_DIM)
        self.assertEqual(scaler.call_args_list[1].args[0].shape[1], FACE_INPUT_DIM)
        self.assertTrue(np.all(scaler.call_args_list[0].args[0][:, 0] != 15.0))
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])


class TestParticipantWiseNormalization(unittest.TestCase):
    def test_participant_zscore_has_per_participant_zero_mean_unit_std_and_safe_constants(self):
        features = np.array([[1., 5., 2.], [3., 5., 2.], [10., 9., 2.], [30., 9., 2.]])
        groups = np.array(["P01", "P01", "P02", "P02"])
        normalized = participant_zscore_features(features, groups)
        for participant in ("P01", "P02"):
            subset = normalized[groups == participant]
            np.testing.assert_allclose(subset.mean(axis=0), np.zeros(3), atol=1e-7)
            np.testing.assert_allclose(subset[:, 0].std(), 1.0, atol=1e-7)
            np.testing.assert_array_equal(subset[:, 1:], np.zeros((2, 2)))

    def test_participant_zscore_does_not_mix_groups_or_accept_labels(self):
        features = np.array([[1.], [3.], [100.], [104.]])
        groups = np.array(["P01", "P01", "P02", "P02"])
        normalized = participant_zscore_features(features, groups)
        np.testing.assert_allclose(normalized[:, 0], np.array([-1., 1., -1., 1.]))

    def test_global_mode_delegates_to_unchanged_pooled_scaler_experiment(self):
        expected = {"modality": "eeg", "regression": {}, "classification": {}, "dummy": {}}
        with patch("processing.models.train_mlp.run_general_modality_regression_experiment", return_value=expected) as baseline:
            result = run_general_eeg_normalization_experiment(Path("derived"), ["P01"], "valence", "P15", "global_train_scaler")
        baseline.assert_called_once_with(Path("derived"), ["P01"], "valence", "P15", "eeg", 42, False)
        self.assertEqual(result["normalization_mode"], "global_train_scaler")

    def test_participant_zscore_keeps_p15_out_of_training_and_uses_no_global_scaler(self):
        reference = TestGeneralModalityRegression.cohort(20)
        with patch("processing.models.train_mlp.load_general_eeg_rating_data", return_value=reference):
            with patch("processing.models.train_mlp.load_general_modality_rating_data", return_value=reference):
                with patch("processing.models.train_mlp.fit_eeg_scaler") as global_scaler:
                    with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=TestGeneralModalityRegression.no_op_fit_result()) as fit:
                        result = run_general_eeg_normalization_experiment(
                            Path("derived"), ["P01", "P02", "P03", "P04", "P05", "P15"],
                            "valence", "P15", "participant_zscore"
                        )
        global_scaler.assert_not_called()
        self.assertTrue(torch.all(fit.call_args.args[3][:, 0] != 15.0).item())
        self.assertTrue(torch.all(fit.call_args.args[5][:, 0] != 15.0).item())
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])


class TestEEGCNN(unittest.TestCase):
    def test_expected_input_and_scalar_output_shape(self):
        model = build_eeg_cnn(8)
        self.assertEqual(model(torch.zeros((4, 8, 626), dtype=torch.float32)).shape, (4, 1))
        with self.assertRaisesRegex(ValueError, "8 channels"):
            model(torch.zeros((4, 7, 626), dtype=torch.float32))

    def test_default_builder_is_the_named_baseline(self):
        default, baseline = build_eeg_cnn(8), build_named_eeg_cnn(8, "baseline")
        self.assertEqual((default.kernel_size_1, default.kernel_size_2, default.dropout_probability), (7, 5, 0.0))
        for first_parameter, second_parameter in zip(default.parameters(), baseline.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter)

    def test_larger_kernels_preserve_output_shape_and_dropout_stays_in_dense_head(self):
        larger = build_named_eeg_cnn(8, "larger_kernels")
        dropout = build_named_eeg_cnn(8, "larger_kernels_dropout")
        self.assertEqual(larger(torch.zeros((2, 8, 626))).shape, (2, 1))
        self.assertEqual(dropout(torch.zeros((2, 8, 626))).shape, (2, 1))
        self.assertIsInstance(larger.dropout, torch.nn.Identity)
        self.assertIsInstance(dropout.dropout, torch.nn.Dropout)
        self.assertEqual(dropout.dropout.p, .3)
        self.assertEqual(CNN_ARCHITECTURES["larger_kernels"].kernel_size_1, 15)
        self.assertEqual(CNN_ARCHITECTURES["larger_kernels"].kernel_size_2, 9)

    def test_deterministic_initialization_and_one_step_parameter_update(self):
        first, second = build_eeg_cnn(8), build_eeg_cnn(8)
        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter)
        optimizer = build_optimizer(first, .001)
        before = [parameter.detach().clone() for parameter in first.parameters()]
        train_one_step(first, optimizer, build_regression_loss(), torch.randn((4, 8, 32)), torch.tensor([1., 2., 3., 4.]))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, first.parameters())))

    def test_tiny_synthetic_overfit_reduces_mse(self):
        model = build_eeg_cnn(2)
        optimizer, loss = build_optimizer(model, .001), build_regression_loss()
        x = torch.cat((torch.zeros((4, 2, 32)), torch.ones((4, 2, 32))), dim=0)
        y = torch.tensor([0.] * 4 + [1.] * 4)
        losses = [train_one_step(model, optimizer, loss, x, y) for _ in range(80)]
        self.assertLess(losses[-1], losses[0])

    def test_participant_channel_normalization_preserves_group_boundaries_and_handles_constants(self):
        epochs = np.array([[[1., 3.], [5., 5.]], [[3., 5.], [5., 5.]], [[10., 30.], [9., 9.]], [[30., 50.], [9., 9.]]])
        groups = np.array(["P01", "P01", "P02", "P02"])
        normalized = participant_channel_zscore_epochs(epochs, groups)
        for participant in ("P01", "P02"):
            subset = normalized[groups == participant]
            np.testing.assert_allclose(subset.mean(axis=(0, 2)), np.zeros(2), atol=1e-7)
            self.assertAlmostEqual(float(subset[:, 0, :].std()), 1.0)
            np.testing.assert_array_equal(subset[:, 1, :], np.zeros((2, 2)))

    def test_cnn_fold_uses_aligned_partitions_without_p15_training_leakage_and_dummy_plumbing(self):
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        groups = np.repeat(participant_ids, 6)
        ratings = np.tile(np.array([1., 2., 3., 5., 6., 7.]), len(participant_ids))
        epochs = np.random.default_rng(1).normal(size=(36, 8, 32))
        source = (epochs, ratings, groups, ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"), 250.)
        no_op = {"epochs_run": 1, "best_epoch": 1, "best_validation_loss": 1., "final_training_loss": 1.}
        with patch("processing.models.train_cnn.load_general_eeg_epoch_data", return_value=source):
            with patch("processing.models.train_cnn.fit_with_early_stopping", return_value=no_op) as fit:
                result = run_general_eeg_cnn_fold(Path("derived"), participant_ids, "valence", "P15")
        self.assertEqual((result["n_train"], result["n_validation"], result["n_test"]), (24, 6, 6))
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])
        self.assertEqual(result["groups"]["test"], ["P15"])
        self.assertEqual(fit.call_args.args[3].shape, (24, 8, 32))
        self.assertAlmostEqual(result["dummy"]["training_mean_rating"], 4.0)

    def test_cnn_fold_selects_named_architecture(self):
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        source = (np.random.default_rng(2).normal(size=(36, 8, 32)), np.tile(np.array([1., 2., 3., 5., 6., 7.]), 6), np.repeat(participant_ids, 6), ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"), 250.)
        no_op = {"epochs_run": 1, "best_epoch": 1, "best_validation_loss": 1., "final_training_loss": 1.}
        with patch("processing.models.train_cnn.load_general_eeg_epoch_data", return_value=source):
            with patch("processing.models.train_cnn.fit_with_early_stopping", return_value=no_op):
                result = run_general_eeg_cnn_fold(Path("derived"), participant_ids, "valence", "P15", architecture="larger_kernels_dropout")
        self.assertEqual(result["architecture"], "larger_kernels_dropout")


class TestPersistentCNNOutputs(unittest.TestCase):
    def test_writes_only_requested_cnn_output_files(self):
        row = {
            "model": "cnn", "held_out_participant": "P15", "train_rows": 100, "validation_rows": 20, "test_rows": 10,
            "best_epoch": 4, "best_validation_mse": 1.5, "validation_rmse": 1.1, "validation_mae": .9,
            "test_rmse": 1.2, "test_mae": 1.0, "dummy_validation_rmse": 1.3, "dummy_validation_mae": 1.1,
            "dummy_test_rmse": 1.4, "dummy_test_mae": 1.2, "validation_rmse_delta": -.2, "test_rmse_delta": -.2,
            "validation_ba": .6, "test_ba": .7, "test_accuracy": .8,
            "test_predictions": [{"held_out_participant": "P15", "trigger": 101, "true_rating": 5., "predicted_rating": 4.5, "true_binary_LOW_HIGH": 1, "predicted_binary_LOW_HIGH": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_cnn_run_config(output_dir, ("P15",), "valence", 42)
            write_cnn_outputs(output_dir, [row])
            self.assertEqual({path.name for path in output_dir.iterdir()}, {"fold_results.csv", "aggregate_summary.csv", "predictions.csv", "run_config.json"})
            fold_results = pd.read_csv(output_dir / "fold_results.csv")
            predictions = pd.read_csv(output_dir / "predictions.csv")
            aggregate = pd.read_csv(output_dir / "aggregate_summary.csv")
            config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(list(fold_results.columns), [
                "held_out_participant", "architecture", "train_rows", "validation_rows", "test_rows",
                "best_epoch", "best_validation_mse", "validation_rmse", "validation_mae", "test_rmse",
                "test_mae", "dummy_validation_rmse", "dummy_validation_mae", "dummy_test_rmse",
                "dummy_test_mae", "validation_rmse_delta", "test_rmse_delta", "validation_ba", "test_ba",
                "test_accuracy",
            ])
            self.assertEqual(fold_results.loc[0, "held_out_participant"], "P15")
            self.assertEqual(predictions.loc[0, "trigger"], 101)
            self.assertEqual(int(aggregate.loc[0, "number_of_folds"]), 1)
            self.assertEqual(config["model"], "cnn")
            self.assertEqual(config["architecture"], "baseline")
            self.assertEqual(config["target"], "valence")


class TestEEGScaling(unittest.TestCase):
    def test_scaled_training_features_have_zero_mean(self):
        x_train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaled_train = transform_eeg_features(x_train, fit_eeg_scaler(x_train))
        np.testing.assert_allclose(scaled_train.mean(axis=0), np.zeros(2), atol=1e-12)

    def test_non_constant_scaled_training_features_have_unit_std(self):
        x_train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaled_train = transform_eeg_features(x_train, fit_eeg_scaler(x_train))
        np.testing.assert_allclose(scaled_train.std(axis=0), np.ones(2), atol=1e-12)

    def test_validation_features_use_training_statistics(self):
        x_train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        x_validation = np.array([[7.0, 40.0], [9.0, 50.0]])
        scaled_validation = transform_eeg_features(x_validation, fit_eeg_scaler(x_train))
        expected = (x_validation - np.array([3.0, 20.0])) / np.sqrt(np.array([8.0 / 3.0, 200.0 / 3.0]))
        np.testing.assert_allclose(scaled_validation, expected)

    def test_zero_variance_columns_remain_finite(self):
        x_train = np.array([[2.0, 10.0], [2.0, 20.0], [2.0, 30.0]])
        x_validation = np.array([[2.0, 40.0]])
        scaler = fit_eeg_scaler(x_train)
        scaled_train = transform_eeg_features(x_train, scaler)
        scaled_validation = transform_eeg_features(x_validation, scaler)
        self.assertTrue(np.isfinite(scaled_train).all())
        self.assertTrue(np.isfinite(scaled_validation).all())
        np.testing.assert_array_equal(scaled_train[:, 0], np.zeros(3))


class TestSmallMLP(unittest.TestCase):
    def test_returns_one_logit_per_input_row(self):
        model = build_small_mlp(input_dim=3)
        output = model(torch.ones((4, 3)))
        self.assertEqual(output.shape, (4, 1))
        self.assertIsInstance(output, torch.Tensor)

    def test_rejects_wrong_feature_width(self):
        model = build_small_mlp(input_dim=3)
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            model(torch.ones((2, 2)))

    def test_rejects_non_positive_input_dimension(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_small_mlp(input_dim=0)

    def test_initialization_is_deterministic(self):
        first_model = build_small_mlp(input_dim=3)
        second_model = build_small_mlp(input_dim=3)
        for first_parameter, second_parameter in zip(first_model.parameters(), second_model.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter)

    def test_returns_raw_logits_without_sigmoid(self):
        model = build_small_mlp(input_dim=3)
        with torch.no_grad():
            model.output_layer.weight.zero_()
            model.output_layer.bias.fill_(2.0)
        output = model(torch.zeros((1, 3)))
        torch.testing.assert_close(output, torch.tensor([[2.0]]))

    def test_supports_requested_alternative_hidden_widths(self):
        model = build_small_mlp(input_dim=3, hidden_dims=(32, 16))
        self.assertEqual(model.first_layer.weight.shape, (32, 3))
        self.assertEqual(model.second_layer.weight.shape, (16, 32))
        self.assertEqual(model.output_layer.weight.shape, (1, 16))


class TestBinaryLoss(unittest.TestCase):
    def test_returns_bce_with_logits_loss(self):
        self.assertIsInstance(build_binary_loss(), torch.nn.BCEWithLogitsLoss)

    def test_accepts_raw_logits_and_binary_targets(self):
        loss = build_binary_loss()
        logits = torch.tensor([[-2.0], [2.0]])
        targets = torch.tensor([[0.0], [1.0]])
        self.assertEqual(loss(logits, targets).ndim, 0)

    def test_loss_is_finite(self):
        loss = build_binary_loss()
        value = loss(torch.tensor([[-2.0], [2.0]]), torch.tensor([[0.0], [1.0]]))
        self.assertTrue(torch.isfinite(value).item())

    def test_correct_logit_has_lower_loss_than_incorrect_logit(self):
        loss = build_binary_loss()
        target = torch.tensor([[1.0]])
        correct_loss = loss(torch.tensor([[10.0]]), target)
        incorrect_loss = loss(torch.tensor([[-10.0]]), target)
        self.assertLess(correct_loss.item(), incorrect_loss.item())

    def test_balanced_loss_uses_training_low_to_high_ratio(self):
        loss = build_balanced_binary_loss(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(loss.pos_weight, torch.tensor([3.0]))


class TestOptimizer(unittest.TestCase):
    def test_returns_adam_optimizer(self):
        self.assertIsInstance(build_optimizer(build_small_mlp(3)), torch.optim.Adam)

    def test_default_learning_rate_is_exactly_point_zero_zero_one(self):
        optimizer = build_optimizer(build_small_mlp(3))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_accepts_custom_positive_learning_rate(self):
        optimizer = build_optimizer(build_small_mlp(3), learning_rate=0.01)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_rejects_zero_and_negative_learning_rates(self):
        model = build_small_mlp(3)
        for learning_rate in (0.0, -0.001):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaisesRegex(ValueError, "greater than zero"):
                    build_optimizer(model, learning_rate=learning_rate)

    def test_optimizer_contains_the_model_trainable_parameters(self):
        model = build_small_mlp(3)
        optimizer = build_optimizer(model)
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertSetEqual(actual, expected)


class TestOneTrainingStep(unittest.TestCase):
    @staticmethod
    def components() -> tuple[torch.nn.Module, torch.optim.Adam, torch.nn.Module, torch.Tensor, torch.Tensor]:
        model = build_small_mlp(2)
        optimizer = build_optimizer(model, learning_rate=0.05)
        loss_fn = build_binary_loss()
        x_batch = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y_batch = torch.tensor([0.0, 0.0, 1.0, 1.0])
        return model, optimizer, loss_fn, x_batch, y_batch

    def test_returns_a_finite_loss(self):
        model, optimizer, loss_fn, x_batch, y_batch = self.components()
        loss = train_one_step(model, optimizer, loss_fn, x_batch, y_batch)
        self.assertTrue(np.isfinite(loss))

    def test_model_parameters_change_after_one_step(self):
        model, optimizer, loss_fn, x_batch, y_batch = self.components()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        train_one_step(model, optimizer, loss_fn, x_batch, y_batch)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_gradients_are_produced(self):
        model, optimizer, loss_fn, x_batch, y_batch = self.components()
        train_one_step(model, optimizer, loss_fn, x_batch, y_batch)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_adam_records_an_update_for_model_parameters(self):
        model, optimizer, loss_fn, x_batch, y_batch = self.components()
        train_one_step(model, optimizer, loss_fn, x_batch, y_batch)
        self.assertTrue(all(parameter in optimizer.state for parameter in model.parameters()))
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer.state.values()))

    def test_repeated_steps_reduce_loss_on_tiny_easy_data(self):
        model, optimizer, loss_fn, x_batch, y_batch = self.components()
        losses = [train_one_step(model, optimizer, loss_fn, x_batch, y_batch) for _ in range(20)]
        self.assertLess(losses[-1], losses[0])


class TestOneTrainingEpoch(unittest.TestCase):
    @staticmethod
    def sample_tensors(sample_count: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        x_train = torch.stack((torch.arange(sample_count, dtype=torch.float32), torch.ones(sample_count)), dim=1)
        y_train = (torch.arange(sample_count) % 2).to(dtype=torch.float32)
        return x_train, y_train

    def test_every_sample_is_used_once(self):
        x_train, y_train = self.sample_tensors()
        with patch("processing.models.mlp_model.train_one_step", return_value=0.5) as one_step:
            train_one_epoch(None, None, None, x_train, y_train, batch_size=2, shuffle=False)
        used = torch.cat([call.args[3][:, 0] for call in one_step.call_args_list])
        torch.testing.assert_close(used.sort().values, torch.arange(5, dtype=torch.float32))

    def test_final_smaller_batch_is_included(self):
        x_train, y_train = self.sample_tensors()
        with patch("processing.models.mlp_model.train_one_step", return_value=0.5) as one_step:
            train_one_epoch(None, None, None, x_train, y_train, batch_size=2, shuffle=False)
        self.assertEqual([call.args[3].shape[0] for call in one_step.call_args_list], [2, 2, 1])

    def test_returns_a_finite_mean_loss(self):
        model, optimizer, loss_fn, x_train, y_train = TestOneTrainingStep.components()
        loss = train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size=3, shuffle=False)
        self.assertTrue(np.isfinite(loss))

    def test_shuffling_is_deterministic_with_fixed_seed(self):
        x_train, y_train = self.sample_tensors()

        def sample_order() -> torch.Tensor:
            with patch("processing.models.mlp_model.train_one_step", return_value=0.5) as one_step:
                train_one_epoch(None, None, None, x_train, y_train, batch_size=2, shuffle=True, seed=7)
            return torch.cat([call.args[3][:, 0] for call in one_step.call_args_list])

        torch.testing.assert_close(sample_order(), sample_order())

    def test_model_parameters_change_over_the_epoch(self):
        model, optimizer, loss_fn, x_train, y_train = TestOneTrainingStep.components()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size=2, shuffle=False)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_repeated_epochs_reduce_loss_on_tiny_easy_data(self):
        model, optimizer, loss_fn, x_train, y_train = TestOneTrainingStep.components()
        losses = [
            train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size=4, shuffle=False)
            for _ in range(10)
        ]
        self.assertLess(losses[-1], losses[0])


class FixedLogitModel(torch.nn.Module):
    """Small inference-only model returning supplied raw logits for tests."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32).reshape(-1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits


class TestBinaryPredictionAndEvaluation(unittest.TestCase):
    def test_probabilities_are_between_zero_and_one(self):
        probabilities, _ = predict_binary(FixedLogitModel([-10.0, 0.0, 10.0]), torch.zeros((3, 1)))
        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(torch.all((probabilities >= 0.0) & (probabilities <= 1.0)).item())

    def test_probability_threshold_at_point_five_behaves_correctly(self):
        _, labels = predict_binary(FixedLogitModel([-1.0, 0.0, 1.0]), torch.zeros((3, 1)))
        torch.testing.assert_close(labels, torch.tensor([0, 1, 1]))

    def test_predicted_labels_are_binary(self):
        _, labels = predict_binary(FixedLogitModel([-2.0, 2.0]), torch.zeros((2, 1)))
        self.assertTrue(set(labels.tolist()).issubset({0, 1}))

    def test_evaluation_returns_finite_metrics(self):
        metrics = evaluate_binary(
            FixedLogitModel([-2.0, 2.0]),
            torch.zeros((2, 1)),
            torch.tensor([0.0, 1.0]),
        )
        self.assertTrue(np.isfinite(metrics["balanced_accuracy"]))
        self.assertTrue(np.isfinite(metrics["accuracy"]))

    def test_perfect_predictions_have_unit_scores(self):
        metrics = evaluate_binary(
            FixedLogitModel([-10.0, 10.0]),
            torch.zeros((2, 1)),
            torch.tensor([[0.0], [1.0]]),
        )
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_always_one_imbalanced_predictions_have_expected_scores(self):
        metrics = evaluate_binary(
            FixedLogitModel([10.0, 10.0, 10.0, 10.0]),
            torch.zeros((4, 1)),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.25)

    def test_prediction_diagnostics_report_confusion_and_probabilities(self):
        diagnostics = prediction_diagnostics(
            FixedLogitModel([-10.0, 10.0]),
            torch.zeros((2, 1)),
            torch.tensor([0.0, 1.0]),
        )
        self.assertEqual((diagnostics["tn"], diagnostics["fp"], diagnostics["fn"], diagnostics["tp"]), (1, 0, 0, 1))
        self.assertLess(diagnostics["mean_p_high_true_low"], diagnostics["mean_p_high_true_high"])
        self.assertEqual(diagnostics["predicted_high_fraction"], 0.5)


class ScalarLogitModel(torch.nn.Module):
    """One-parameter model for controlled early-stopping tests."""

    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(features.shape[0], 1)


class TestEarlyStoppingFit(unittest.TestCase):
    @staticmethod
    def components() -> tuple[ScalarLogitModel, torch.optim.Adam, torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = ScalarLogitModel()
        optimizer = build_optimizer(model, learning_rate=0.1)
        loss_fn = build_binary_loss()
        x_train = torch.zeros((4, 1))
        y_train = torch.ones(4)
        x_val = torch.zeros((4, 1))
        y_val = torch.zeros(4)
        return model, optimizer, loss_fn, x_train, y_train, x_val, y_val

    def fit_contradictory_validation_case(self, max_epochs: int = 10, patience: int = 2) -> tuple[ScalarLogitModel, dict[str, float | int]]:
        model, optimizer, loss_fn, x_train, y_train, x_val, y_val = self.components()
        result = fit_with_early_stopping(
            model, optimizer, loss_fn, x_train, y_train, x_val, y_val,
            max_epochs=max_epochs, batch_size=4, patience=patience,
        )
        return model, result

    def test_stops_before_max_epochs_on_easy_overfit_prone_data(self):
        _, result = self.fit_contradictory_validation_case(max_epochs=10, patience=2)
        self.assertLess(result["epochs_run"], 10)

    def test_best_model_weights_are_restored(self):
        expected_model, expected_optimizer, loss_fn, x_train, y_train, _, _ = self.components()
        train_one_epoch(expected_model, expected_optimizer, loss_fn, x_train, y_train, batch_size=4, seed=42)
        fitted_model, result = self.fit_contradictory_validation_case()
        self.assertEqual(result["best_epoch"], 1)
        torch.testing.assert_close(fitted_model.logit, expected_model.logit)

    def test_best_validation_loss_is_finite(self):
        _, result = self.fit_contradictory_validation_case()
        self.assertTrue(np.isfinite(result["best_validation_loss"]))

    def test_epochs_run_never_exceeds_max_epochs(self):
        _, result = self.fit_contradictory_validation_case(max_epochs=2, patience=10)
        self.assertLessEqual(result["epochs_run"], 2)

    def test_patience_is_respected(self):
        _, result = self.fit_contradictory_validation_case(max_epochs=10, patience=2)
        self.assertEqual(result["epochs_run"], 3)

    def test_validation_does_not_update_model_parameters(self):
        model = build_small_mlp(2)
        optimizer = build_optimizer(model)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        x = torch.zeros((2, 2))
        y = torch.tensor([0.0, 1.0])
        with patch("processing.models.train_mlp.train_one_epoch", return_value=0.0):
            fit_with_early_stopping(
                model, optimizer, build_binary_loss(), x, y, x, y,
                max_epochs=1, batch_size=2, patience=1,
            )
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, model.parameters())))


class TestSingleParticipantExperiment(unittest.TestCase):
    @staticmethod
    def synthetic_table() -> pd.DataFrame:
        row_count = 30
        labels = np.array([0] * 15 + [1] * 15)
        table = pd.DataFrame(
            {
                column: np.arange(row_count, dtype=float) + index
                for index, column in enumerate(train_classification.EEG_COLUMNS)
            }
        )
        table[train_classification.EEG_COLUMNS[0]] = labels * 2.0 - 1.0
        table["valence_rating"] = np.where(labels == 0, 1.0, 6.0)
        table["arousal_rating"] = table["valence_rating"]
        return table

    def prepared_synthetic_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return prepare_eeg_data(self.synthetic_table(), "P15", "valence")

    @staticmethod
    def no_op_fit_result() -> dict[str, float | int]:
        return {
            "epochs_run": 1,
            "best_epoch": 1,
            "best_validation_loss": 0.5,
            "final_training_loss": 0.5,
        }

    def run_with_prepared_data(self, **kwargs: object) -> dict[str, object]:
        with patch(
            "processing.models.train_mlp.load_participant_eeg_data",
            return_value=self.prepared_synthetic_data(),
        ):
            return run_single_participant_experiment(Path("derived"), "P15", "valence", **kwargs)

    def test_split_sizes_approximately_follow_sixty_twenty_twenty(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
            result = self.run_with_prepared_data()
        self.assertEqual((result["n_train"], result["n_val"], result["n_test"]), (18, 6, 6))

    def test_splits_are_deterministic(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
            first = self.run_with_prepared_data(seed=7)
            second = self.run_with_prepared_data(seed=7)
        self.assertEqual(first, second)

    def test_scaler_is_fitted_only_on_training_rows(self):
        x, y, _ = self.prepared_synthetic_data()
        train_validation_indices, _ = train_test_split(
            np.arange(len(y)), test_size=0.20, stratify=y, random_state=42
        )
        expected_train_indices, _ = train_test_split(
            train_validation_indices, test_size=0.25, stratify=y[train_validation_indices], random_state=42
        )
        with patch("processing.models.train_mlp.fit_eeg_scaler", wraps=fit_eeg_scaler) as fit_scaler:
            with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
                self.run_with_prepared_data()
        np.testing.assert_array_equal(fit_scaler.call_args.args[0], x[expected_train_indices])

    def test_test_labels_are_not_passed_to_fitting(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()) as fit:
            result = self.run_with_prepared_data()
        fitting_arguments = fit.call_args.args
        self.assertEqual(len(fitting_arguments), 7)
        self.assertEqual(fitting_arguments[4].shape[0], result["n_train"])
        self.assertEqual(fitting_arguments[6].shape[0], result["n_val"])

    def test_balanced_loss_uses_only_training_labels(self):
        with patch("processing.models.train_mlp.build_balanced_binary_loss", wraps=build_balanced_binary_loss) as loss_builder:
            with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
                result = self.run_with_prepared_data(balanced_loss=True)
        self.assertEqual(loss_builder.call_args.args[0].shape[0], result["n_train"])

    def test_experiment_accepts_requested_architecture_and_learning_rate(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
            result = self.run_with_prepared_data(hidden_dims=(32, 16), learning_rate=0.0005)
        self.assertEqual(result["architecture"], (120, 32, 16, 1))
        self.assertEqual(result["learning_rate"], 0.0005)

    def test_metrics_are_finite_and_bounded(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
            result = self.run_with_prepared_data()
        for key in (
            "train_balanced_accuracy", "train_accuracy", "val_balanced_accuracy", "val_accuracy",
            "test_balanced_accuracy", "test_accuracy",
        ):
            self.assertTrue(np.isfinite(result[key]))
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    def test_runs_end_to_end_on_tiny_synthetic_table(self):
        result = self.run_with_prepared_data(seed=42)
        self.assertEqual(result["n_total"], 30)
        self.assertLessEqual(result["epochs_run"], 200)


class TestGeneralEEGDataLoading(unittest.TestCase):
    @staticmethod
    def participant_data() -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
        return [
            (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]), ["feature_a", "feature_b"]),
            (np.array([[5.0, 6.0]]), np.array([1]), ["feature_a", "feature_b"]),
        ]

    def test_preserves_participant_groups_and_row_order(self):
        with patch("processing.models.mlp_data.load_participant_eeg_data", side_effect=self.participant_data()):
            x, y, groups, names = load_general_eeg_data(Path("derived"), ["p01", "P02"], "valence")
        np.testing.assert_array_equal(x, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        np.testing.assert_array_equal(y, np.array([0, 1, 1]))
        np.testing.assert_array_equal(groups, np.array(["P01", "P01", "P02"]))
        self.assertEqual(names, ["feature_a", "feature_b"])

    def test_features_labels_and_groups_have_matching_row_counts(self):
        with patch("processing.models.mlp_data.load_participant_eeg_data", side_effect=self.participant_data()):
            x, y, groups, _ = load_general_eeg_data(Path("derived"), ["P01", "P02"], "arousal")
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertEqual(y.shape[0], groups.shape[0])
        self.assertTrue(set(y).issubset({0, 1}))

    def test_rejects_different_feature_schemas(self):
        inconsistent = self.participant_data()
        inconsistent[1] = (inconsistent[1][0], inconsistent[1][1], ["feature_a", "other_feature"])
        with patch("processing.models.mlp_data.load_participant_eeg_data", side_effect=inconsistent):
            with self.assertRaisesRegex(ValueError, "schema mismatch for P02"):
                load_general_eeg_data(Path("derived"), ["P01", "P02"], "valence")

    def test_missing_participant_source_error_names_participant(self):
        with patch("processing.models.mlp_data.load_participant_eeg_data", side_effect=FileNotFoundError("merged table missing")):
            with self.assertRaisesRegex(FileNotFoundError, "P46 no-ICA EEG source"):
                load_general_eeg_data(Path("derived"), ["P46"], "valence")


class TestLOSOSplit(unittest.TestCase):
    @staticmethod
    def cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
            np.array([0, 1, 0, 1, 1]),
            np.array(["P01", "P02", "P01", "P03", "P02"]),
        )

    def test_held_out_participant_is_absent_from_training(self):
        x, y, groups = self.cohort()
        _, _, groups_train, _, _, _ = make_loso_split(x, y, groups, "P02")
        self.assertNotIn("P02", groups_train)

    def test_test_contains_only_held_out_participant(self):
        x, y, groups = self.cohort()
        _, _, _, _, _, groups_test = make_loso_split(x, y, groups, "P02")
        np.testing.assert_array_equal(groups_test, np.array(["P02", "P02"]))

    def test_train_and_test_row_counts_match_original(self):
        x, y, groups = self.cohort()
        x_train, y_train, groups_train, x_test, y_test, groups_test = make_loso_split(x, y, groups, "P02")
        self.assertEqual(x_train.shape[0] + x_test.shape[0], x.shape[0])
        self.assertEqual(y_train.shape[0] + y_test.shape[0], y.shape[0])
        self.assertEqual(groups_train.shape[0] + groups_test.shape[0], groups.shape[0])

    def test_labels_and_groups_remain_aligned(self):
        x, y, groups = self.cohort()
        x_train, y_train, groups_train, x_test, y_test, groups_test = make_loso_split(x, y, groups, "P02")
        np.testing.assert_array_equal(x_train[:, 0], np.array([1.0, 3.0, 4.0]))
        np.testing.assert_array_equal(y_train, np.array([0, 0, 1]))
        np.testing.assert_array_equal(groups_train, np.array(["P01", "P01", "P03"]))
        np.testing.assert_array_equal(x_test[:, 0], np.array([2.0, 5.0]))
        np.testing.assert_array_equal(y_test, np.array([1, 1]))
        np.testing.assert_array_equal(groups_test, np.array(["P02", "P02"]))

    def test_unknown_held_out_participant_is_clear(self):
        x, y, groups = self.cohort()
        with self.assertRaisesRegex(ValueError, "Unknown held-out participant: P99"):
            make_loso_split(x, y, groups, "P99")


class TestGroupedValidationSplit(unittest.TestCase):
    @staticmethod
    def outer_training_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups = np.repeat(["P01", "P02", "P03", "P04", "P05"], 2)
        labels = np.tile(np.array([0, 1]), 5)
        features = np.column_stack((np.arange(len(labels), dtype=float), labels.astype(float)))
        return features, labels, groups

    def test_train_and_validation_participants_do_not_overlap(self):
        x, y, groups = self.outer_training_pool()
        _, _, groups_subtrain, _, _, groups_validation = make_grouped_validation_split(x, y, groups, "P15")
        self.assertFalse(set(groups_subtrain).intersection(groups_validation))

    def test_outer_test_participant_is_absent_from_both_inner_partitions(self):
        x, y, groups = self.outer_training_pool()
        _, _, groups_subtrain, _, _, groups_validation = make_grouped_validation_split(x, y, groups, "P15")
        self.assertNotIn("P15", groups_subtrain)
        self.assertNotIn("P15", groups_validation)

    def test_inner_partitions_reconstruct_outer_training_pool(self):
        x, y, groups = self.outer_training_pool()
        x_subtrain, y_subtrain, groups_subtrain, x_validation, y_validation, groups_validation = make_grouped_validation_split(
            x, y, groups, "P15"
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate((x_subtrain[:, 0], x_validation[:, 0]))), x[:, 0]
        )
        self.assertEqual(len(y_subtrain) + len(y_validation), len(y))
        self.assertEqual(len(groups_subtrain) + len(groups_validation), len(groups))

    def test_labels_and_groups_remain_aligned(self):
        x, y, groups = self.outer_training_pool()
        x_subtrain, y_subtrain, groups_subtrain, x_validation, y_validation, groups_validation = make_grouped_validation_split(
            x, y, groups, "P15"
        )
        np.testing.assert_array_equal(x_subtrain[:, 1], y_subtrain)
        np.testing.assert_array_equal(x_validation[:, 1], y_validation)
        self.assertTrue(all(group.startswith("P") for group in groups_subtrain))
        self.assertTrue(all(group.startswith("P") for group in groups_validation))

    def test_split_is_deterministic_for_seed_42(self):
        x, y, groups = self.outer_training_pool()
        first = make_grouped_validation_split(x, y, groups, "P15", seed=42)
        second = make_grouped_validation_split(x, y, groups, "P15", seed=42)
        for first_part, second_part in zip(first, second):
            np.testing.assert_array_equal(first_part, second_part)


class TestGeneralLOSOFoldExperiment(unittest.TestCase):
    @staticmethod
    def cohort_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        labels = np.tile(np.array([0, 0, 0, 1, 1, 1]), len(participant_ids))
        groups = np.repeat(participant_ids, 6)
        participant_codes = np.repeat(np.array([1, 2, 3, 4, 5, 15], dtype=float), 6)
        features = np.column_stack((participant_codes, labels.astype(float)))
        return features, labels, groups, ["participant_code", "label_feature"]

    @staticmethod
    def no_op_fit_result() -> dict[str, float | int]:
        return {
            "epochs_run": 1,
            "best_epoch": 1,
            "best_validation_loss": 0.5,
            "final_training_loss": 0.5,
        }

    def run_with_synthetic_cohort(self) -> dict[str, object]:
        with patch("processing.models.train_mlp.load_general_eeg_data", return_value=self.cohort_data()):
            return run_general_loso_fold_experiment(
                Path("derived"), ["P01", "P02", "P03", "P04", "P05", "P15"], "valence", "P15"
            )

    def test_scaler_is_fitted_only_on_inner_training_rows(self):
        x, y, groups, _ = self.cohort_data()
        outer = make_loso_split(x, y, groups, "P15")
        expected_subtrain, _, _, _, _, _ = make_grouped_validation_split(*outer[:3], "P15", seed=42)
        with patch("processing.models.train_mlp.fit_eeg_scaler", wraps=fit_eeg_scaler) as fit_scaler:
            with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
                self.run_with_synthetic_cohort()
        np.testing.assert_array_equal(fit_scaler.call_args.args[0], expected_subtrain)

    def test_no_p15_leakage_into_inner_training_or_validation(self):
        with patch("processing.models.train_mlp.make_grouped_validation_split", wraps=make_grouped_validation_split) as split:
            with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()):
                result = self.run_with_synthetic_cohort()
        np.testing.assert_array_equal(np.sort(split.call_args.args[2]), np.array(["P01", "P01", "P01", "P01", "P01", "P01", "P02", "P02", "P02", "P02", "P02", "P02", "P03", "P03", "P03", "P03", "P03", "P03", "P04", "P04", "P04", "P04", "P04", "P04", "P05", "P05", "P05", "P05", "P05", "P05"]))
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])
        self.assertEqual(result["groups"]["test"], ["P15"])

    def test_validation_data_is_not_passed_as_optimizer_training_data(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()) as fit:
            self.run_with_synthetic_cohort()
        fitting_arguments = fit.call_args.args
        self.assertEqual(fitting_arguments[3].shape[0], 24)
        self.assertEqual(fitting_arguments[5].shape[0], 6)
        self.assertFalse(torch.equal(fitting_arguments[3], fitting_arguments[5]))

    def test_p15_test_data_is_not_passed_to_early_stopping(self):
        with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()) as fit:
            self.run_with_synthetic_cohort()
        fitting_arguments = fit.call_args.args
        self.assertTrue(torch.all(fitting_arguments[3][:, 0] != 15.0).item())
        self.assertTrue(torch.all(fitting_arguments[5][:, 0] != 15.0).item())

    def test_final_evaluation_uses_model_after_best_weight_restoration(self):
        def restored_fit(model: torch.nn.Module, *args: object, **kwargs: object) -> dict[str, float | int]:
            with torch.no_grad():
                model.output_layer.bias.fill_(2.0)
            return self.no_op_fit_result()

        observed_biases: list[float] = []

        def evaluate_restored_model(model: torch.nn.Module, *args: object, **kwargs: object) -> dict[str, float]:
            observed_biases.append(float(model.output_layer.bias.item()))
            return {"balanced_accuracy": 0.5, "accuracy": 0.5}

        with patch("processing.models.train_mlp.fit_with_early_stopping", side_effect=restored_fit):
            with patch("processing.models.train_mlp.evaluate_binary", side_effect=evaluate_restored_model):
                self.run_with_synthetic_cohort()
        self.assertEqual(observed_biases, [2.0, 2.0, 2.0])


class TestOverfittingSanityCheck(unittest.TestCase):
    @staticmethod
    def cohort_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        labels = np.tile(np.array([0] * 20 + [1] * 20), len(participant_ids))
        groups = np.repeat(participant_ids, 40)
        participant_codes = np.repeat(np.array([1, 2, 3, 4, 5, 15], dtype=float), 40)
        features = np.column_stack((participant_codes, labels.astype(float), np.arange(len(labels), dtype=float)))
        return features, labels, groups, ["participant_code", "label_feature", "row_id"]

    def test_subset_is_balanced_and_scaler_uses_only_selected_inner_training_rows(self):
        with patch("processing.models.train_mlp.load_general_eeg_data", return_value=self.cohort_data()):
            with patch("processing.models.train_mlp.fit_eeg_scaler", wraps=fit_eeg_scaler) as fit_scaler:
                result = run_overfitting_sanity_check(
                    Path("derived"), ["P01", "P02", "P03", "P04", "P05", "P15"], "valence", "P15", epochs=1
                )
        self.assertEqual(result["subset_size"], 64)
        self.assertEqual(result["class_counts"], {"low": 32, "high": 32})
        self.assertEqual(fit_scaler.call_args.args[0].shape, (64, 3))
        self.assertTrue(np.all(fit_scaler.call_args.args[0][:, 0] != 15.0))


class TestGeneralLOSORegressionFold(unittest.TestCase):
    @staticmethod
    def cohort_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        participant_ids = ["P01", "P02", "P03", "P04", "P05", "P15"]
        ratings = np.tile(np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0]), len(participant_ids))
        groups = np.repeat(participant_ids, 6)
        participant_codes = np.repeat(np.array([1, 2, 3, 4, 5, 15], dtype=float), 6)
        features = np.column_stack((participant_codes, ratings, np.arange(len(ratings), dtype=float)))
        return features, ratings, groups, ["participant_code", "rating_feature", "row_id"]

    @staticmethod
    def no_op_fit_result() -> dict[str, float | int]:
        return {
            "epochs_run": 1,
            "best_epoch": 1,
            "best_validation_loss": 1.0,
            "final_training_loss": 1.0,
        }

    def test_regression_fold_uses_mse_and_inner_training_scaler_only(self):
        with patch("processing.models.train_mlp.load_general_eeg_rating_data", return_value=self.cohort_data()):
            with patch("processing.models.train_mlp.fit_eeg_scaler", wraps=fit_eeg_scaler) as fit_scaler:
                with patch("processing.models.train_mlp.fit_with_early_stopping", return_value=self.no_op_fit_result()) as fit:
                    result = run_general_loso_regression_fold(
                        Path("derived"), ["P01", "P02", "P03", "P04", "P05", "P15"], "valence", "P15"
                    )
        self.assertIsInstance(fit.call_args.args[2], torch.nn.MSELoss)
        self.assertTrue(np.all(fit_scaler.call_args.args[0][:, 0] != 15.0))
        self.assertNotIn("P15", result["groups"]["train"])
        self.assertNotIn("P15", result["groups"]["validation"])
        self.assertEqual(result["groups"]["test"], ["P15"])

    def test_build_regression_loss_returns_mse(self):
        self.assertIsInstance(build_regression_loss(), torch.nn.MSELoss)

    def test_named_regression_comparison_ranks_by_validation_rmse_only(self):
        def result(name: str, validation_rmse: float, test_rmse: float) -> dict[str, object]:
            return {
                "input_configuration": name,
                "regression": {"validation": {"rmse": validation_rmse}, "test": {"rmse": test_rmse}},
            }

        synthetic = [
            result("all_features_all_channels", 2.0, 1.0),
            result("bandpower_all_channels", 1.0, 3.0),
        ]
        with patch("processing.models.train_mlp.EEG_INPUT_CONFIGURATIONS", {"a": object(), "b": object()}):
            with patch("processing.models.train_mlp.run_general_loso_regression_fold", side_effect=synthetic):
                ranked = run_general_regression_input_selection_comparison(Path("derived"), ["P01"])
        self.assertEqual([row["input_configuration"] for row in ranked], ["bandpower_all_channels", "all_features_all_channels"])


if __name__ == "__main__":
    unittest.main()
