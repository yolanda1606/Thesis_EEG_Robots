"""Focused checks for participant-agnostic personalized inference helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from processing.multimodal_robot.inference import run_personalized_inference as inference


class TestPersonalizedInference(unittest.TestCase):
    def test_model_spec_is_generic_and_validated(self):
        spec = inference.parse_spec("arousal:multimodal:knn:all")
        self.assertEqual(spec.identifier, "arousal__multimodal__knn__all")
        with self.assertRaises(Exception): inference.parse_spec("P15:arousal:knn:all")

    def test_multimodal_uses_exact_one_to_one_shared_window_keys(self):
        keys = {"participant": ["P00"], "task": ["stack"], "segment_id": [1], "window_id": ["stack_s1_0.000"], "window_start_s": [0.0], "window_end_s": [2.0]}
        eeg = pd.DataFrame({**keys, "eeg_sd__Fz": [1.0]})
        video = pd.DataFrame({**keys, "video_irisdo_norm_mean": [2.0]})
        spec = inference.parse_spec("valence:multimodal:gnb:5")
        self.assertEqual(len(inference.robot_input(spec, eeg, video)), 1)
        extra = eeg.copy(); extra.loc[0, "window_id"] = "eeg_only"
        self.assertEqual(len(inference.robot_input(spec, pd.concat([eeg, extra], ignore_index=True), video)), 1)

    def test_final_model_has_high_probability_column(self):
        x = pd.DataFrame({"a": np.arange(12), "b": np.tile([0.0, 1.0], 6)})
        y = pd.Series([0, 1] * 6)
        fitted, metadata = inference.fit_final_model(x, y, inference.parse_spec("valence:face:gnb:all"), 42)
        self.assertEqual(metadata["classifier_classes"], [0, 1])
        self.assertEqual(metadata["high_probability_column"], 1)
        self.assertTrue(hasattr(fitted.named_steps["classifier"], "predict_proba"))

    def test_trajectory_segments_break_at_missing_window_gap(self):
        frame = pd.DataFrame({"window_start_s": [0.0, 1.0, 4.0], "window_center_s": [1.0, 2.0, 5.0], "high_probability": [.1, .2, .3]})
        self.assertEqual([len(part) for part in inference._segments(frame)], [2, 1])


if __name__ == "__main__":
    unittest.main()
