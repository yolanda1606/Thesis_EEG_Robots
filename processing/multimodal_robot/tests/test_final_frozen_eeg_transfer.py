"""Focused tests for final frozen Image-to-Robot consensus reporting."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "processing" / "multimodal_robot" / "inference"))
import run_final_frozen_eeg_transfer as transfer


class TestFinalFrozenEegTransfer(unittest.TestCase):
    def predictions_with_incomplete_window(self) -> pd.DataFrame:
        rows = []
        probabilities = {
            "complete": {1: 0.2, 2: 0.6, 3: 0.9},
            "incomplete": {1: 0.8, 2: 0.7},
        }
        for window_index, (window_id, ranks) in enumerate(probabilities.items()):
            for rank, probability in ranks.items():
                rows.append({
                    "participant": "P00", "target": "valence", "task": "stack", "segment_id": 1,
                    "window_id": window_id, "window_start_s": float(window_index), "window_end_s": float(window_index + 2),
                    "model_rank": rank, "high_probability": probability,
                    "original_hard_prediction": int(probability >= 0.5),
                })
        return pd.DataFrame(rows)

    def test_complete_window_mask_excludes_incomplete_top_three_windows(self):
        complete, completeness = transfer.complete_top3_windows(self.predictions_with_incomplete_window())
        self.assertEqual(len(complete), 1)
        self.assertEqual(completeness.loc[0, "n_candidate_windows"], 2)
        self.assertEqual(completeness.loc[0, "n_complete_top3_windows"], 1)
        self.assertEqual(completeness.loc[0, "n_incomplete_top3_windows"], 1)
        self.assertEqual(completeness.loc[0, "fraction_complete_top3_windows"], 0.5)
        self.assertAlmostEqual(complete.loc[0, "top3_consensus_probability"], 0.6)
        self.assertEqual(complete.loc[0, "top3_consensus_class"], "HIGH")
        self.assertFalse(complete.loc[0, "all_three_hard_agree"])

    def test_task_verdict_uses_median_windows_of_median_models(self):
        complete, completeness = transfer.complete_top3_windows(self.predictions_with_incomplete_window())
        agreement = transfer.agreement(complete, completeness)
        ratings = pd.DataFrame([{"target": "valence", "task": "stack", "robot_rating": 5.0, "robot_rating_class": "HIGH"}])
        shift = pd.DataFrame([
            {"target": "valence", "task": "stack", "model_rank": rank, "median_nearest_image_distance": value}
            for rank, value in [(1, 2.0), (2, 3.0), (3, 4.0)]
        ])
        statistics = transfer.task_statistics(complete, agreement, ratings, {"stack": "FAULTY"}, shift)
        row = statistics.iloc[0]
        self.assertAlmostEqual(row.median_top3_consensus_probability, 0.6)
        self.assertEqual(row.descriptive_task_verdict, "HIGH")
        self.assertTrue(row.verdict_matches_robot_rating)
        self.assertEqual(row.n_complete_top3_windows, 1)
        self.assertAlmostEqual(row.unanimous_agreement_rate, 0.0)

    def test_zero_complete_windows_are_reported_unavailable(self):
        predictions = self.predictions_with_incomplete_window().query("window_id == 'incomplete'").copy()
        complete, completeness = transfer.complete_top3_windows(predictions)
        agreement = transfer.agreement(complete, completeness)
        ratings = pd.DataFrame([{"target": "valence", "task": "stack", "robot_rating": 2.0, "robot_rating_class": "LOW"}])
        shift = pd.DataFrame([
            {"target": "valence", "task": "stack", "model_rank": rank, "median_nearest_image_distance": 1.0}
            for rank in [1, 2, 3]
        ])
        statistics = transfer.task_statistics(complete, agreement, ratings, {"stack": "N/A"}, shift)
        row = statistics.iloc[0]
        self.assertEqual(row.n_complete_top3_windows, 0)
        self.assertTrue(np.isnan(row.median_top3_consensus_probability))
        self.assertTrue(pd.isna(row.descriptive_task_verdict))
        self.assertTrue(pd.isna(row.verdict_matches_robot_rating))

    def test_consensus_segments_sort_time_and_do_not_merge_segments(self):
        frame = pd.DataFrame({
            "segment_id": [2, 1, 1, 2],
            "window_center_s": [2.0, 3.0, 1.0, 1.0],
            "top3_consensus_probability": [0.4, 0.5, 0.6, 0.7],
        })
        pieces = transfer.consensus_segments(frame)
        self.assertEqual([piece.segment_id.iloc[0] for piece in pieces], [1, 2])
        self.assertEqual([piece.window_center_s.tolist() for piece in pieces], [[1.0, 3.0], [1.0, 2.0]])

    def test_event_source_path_selection_matches_canonical_unavailable_mapping(self):
        self.assertEqual(transfer.selected_configured_paths("a.csv"), [Path("a.csv")])
        self.assertEqual(transfer.selected_configured_paths({"file": "a.csv"}), [Path("a.csv")])
        self.assertEqual(
            transfer.selected_configured_paths({"segments": [{"file": "a.csv"}, {"file": "b.csv"}]}),
            [Path("a.csv"), Path("b.csv")],
        )
        self.assertEqual(
            transfer.selected_configured_paths({"unavailable": True, "reason": "not required"}),
            [],
        )


if __name__ == "__main__":
    unittest.main()
