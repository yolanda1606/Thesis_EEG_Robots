"""Focused tests for post-hoc causal-versus-canonical Robot-rating evaluation."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from processing.multimodal_robot.analysis.causal_streaming.evaluate_causal_vs_canonical_robot_ratings import (
    complete_window_task_predictions, performance,
)


class RobotRatingEvaluationTests(unittest.TestCase):
    def test_definition_b_medians_are_computed_from_paired_complete_windows(self) -> None:
        windows = pd.DataFrame({
            "participant": ["P01"] * 3, "task": ["pick_place"] * 3, "target": ["valence"] * 3,
            "segment_id": [1, 1, 1], "window_id": ["w1", "w2", "w3"],
            "top3_consensus_probability_canonical": [.1, .7, .4], "top3_consensus_probability_causal": [.2, .8, .6],
        })
        task = complete_window_task_predictions(windows).iloc[0]
        self.assertEqual(task.number_complete_top3_windows, 3)
        self.assertAlmostEqual(task.canonical_P_task, .4)
        self.assertAlmostEqual(task.causal_P_task, .6)
        self.assertEqual(task.canonical_verdict, "LOW")
        self.assertEqual(task.causal_verdict, "HIGH")

    def test_performance_uses_high_as_positive_and_one_class_ba_is_na(self) -> None:
        balanced = pd.DataFrame({"robot_label": ["HIGH", "HIGH", "LOW", "LOW"], "canonical_verdict": ["HIGH", "LOW", "LOW", "LOW"]})
        result = performance(balanced, "canonical", "overall")
        self.assertAlmostEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["balanced_accuracy"], .75)
        self.assertAlmostEqual(result["precision"], 1.0)
        one_class = performance(balanced.loc[:1], "canonical", "overall")
        self.assertTrue(np.isnan(one_class["balanced_accuracy"]))


if __name__ == "__main__":
    unittest.main()
