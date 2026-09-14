"""Focused unit tests for the isolated latency replay layer."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from processing.multimodal_robot.transfer import run_final_frozen_eeg_transfer as eeg_transfer

from processing.multimodal_robot.analysis.latency.replay_window import assemble_and_predict, eeg_wide_row, summarize_timings
from processing.multimodal_robot.analysis.latency.validate_replay_compatibility import compare_frames, recorded_model_input_features, validation_passes


class LatencyReplayTests(unittest.TestCase):
    def test_wide_eeg_names_match_transfer_convention(self) -> None:
        row = eeg_wide_row({"window_id": "x"}, ["Fz", "Cz"], [{"eeg_sd": 1.0}, {"eeg_sd": 2.0}])
        self.assertEqual(row["eeg_sd__Fz"], 1.0)
        self.assertEqual(row["eeg_sd__Cz"], 2.0)

    def test_instrumented_probability_equals_pipeline_probability(self) -> None:
        x = pd.DataFrame({"a": [0., 1., 2., 3.], "b": [1., 0., 1., 0.]})
        y = np.array([0, 0, 1, 1])
        pipeline = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(random_state=1))]).fit(x, y)
        record = {"classifier": "logreg", "candidate_features": ["a", "b"], "selected_features": ["a", "b"],
                  "pipeline": pipeline, "probability_model": pipeline, "high_probability_column": 1}
        probability, hard, assembly, transform, hard_time, inference = assemble_and_predict(record, {"a": 2.5, "b": 1.0})
        self.assertAlmostEqual(probability, pipeline.predict_proba(pd.DataFrame([{"a": 2.5, "b": 1.0}]))[0, 1])
        self.assertEqual(hard, int(pipeline.predict(pd.DataFrame([{"a": 2.5, "b": 1.0}]))[0]))
        self.assertGreaterEqual(assembly, 0.0); self.assertGreaterEqual(transform, 0.0); self.assertGreaterEqual(hard_time, 0.0); self.assertGreaterEqual(inference, 0.0)

    def test_actual_p27_calibrated_svm_wrapper_uses_canonical_selected_input(self) -> None:
        root = Path(__file__).resolve().parents[3]
        args = SimpleNamespace(
            participant="P27",
            top3_csv=root / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv",
            model_root=root / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_models",
            calibration_root=root / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_probability_calibration_v1",
        )
        record = next(item for item in eeg_transfer.load_frozen_models(args) if item["classifier"] == "svm")
        self.assertEqual(type(record["probability_model"]).__name__, "CalibratedClassifierCV")
        feature_row = {name: 0.0 for name in record["candidate_features"]}
        probability, hard, _, transform, _, _ = assemble_and_predict(record, feature_row)
        candidate = pd.DataFrame([feature_row], columns=record["candidate_features"])
        selected = candidate.loc[:, record["selected_features"]]
        expected = record["probability_model"].predict_proba(selected)[0, record["calibrated_high_probability_column"]]
        self.assertAlmostEqual(probability, expected)
        self.assertEqual(hard, int(record["pipeline"].predict(candidate)[0]))
        self.assertTrue(np.isnan(transform))

    def test_compatibility_reports_differences_and_tolerance(self) -> None:
        keys = {"participant": "P01", "task": "stack", "segment_id": 1, "window_id": "w", "window_start_s": 0.0, "window_end_s": 2.0}
        replay = pd.DataFrame([{**keys, "eeg_sd__Fz": 1.0}])
        same = pd.DataFrame([{**keys, "eeg_sd__Fz": 1.0 + 1e-10}])
        results = compare_frames(replay, same, ["eeg_sd__Fz"], tolerance=1e-9, comparison="eeg_features")
        self.assertTrue(results.loc[0, "pass"])
        self.assertTrue(validation_passes(results))
        changed = same.assign(**{"eeg_sd__Fz": 1.1})
        self.assertFalse(validation_passes(compare_frames(replay, changed, ["eeg_sd__Fz"], tolerance=1e-9, comparison="eeg_features")))

    def test_recorded_model_input_schema_excludes_structural_nan_columns(self) -> None:
        """Different model schemas must not add NaN-only union columns to a comparison."""
        replay_input = pd.DataFrame([
            {"target": "valence", "model_rank": 1, "model_modality": "eeg",
             "probability_input_features": '["eeg_sd__Fz"]', "eeg_sd__Fz": 1.0,
             "eeg_bp_alpha__C4": np.nan},
            {"target": "valence", "model_rank": 1, "model_modality": "eeg",
             "probability_input_features": '["eeg_sd__Fz"]', "eeg_sd__Fz": 2.0,
             "eeg_bp_alpha__C4": np.nan},
        ])
        features = recorded_model_input_features(replay_input)
        self.assertEqual(features, ["eeg_sd__Fz"])
        self.assertFalse(replay_input.loc[:, features].isna().any().any())

    def test_summary_includes_required_descriptives(self) -> None:
        summary = summarize_timings(pd.DataFrame({"mode": ["eeg_only", "eeg_only"], "eeg_feature_ms": [1.0, 3.0]}), ["eeg_feature_ms"], ["mode"])
        self.assertEqual(summary.loc[0, "count"], 2)
        self.assertEqual(summary.loc[0, "median"], 2.0)
        self.assertIn("p95", summary.columns)


if __name__ == "__main__":
    unittest.main()
