"""Unit checks for leakage-safe threshold selection helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "processing" / "multimodal_image" / "modeling"
sys.path.insert(0, str(MODEL_DIR))
import run_focused_personalized_eeg_threshold_nested as threshold


class TestFocusedPersonalizedThresholdNested(unittest.TestCase):
    def test_threshold_grid_is_frozen(self):
        self.assertEqual(threshold.THRESHOLDS, (.30, .35, .40, .45, .50, .55, .60, .65, .70))

    def test_threshold_tie_prefers_closest_to_half(self):
        # Several thresholds have identical perfect BA; the prespecified first
        # tie-break must keep the conventional 0.50 threshold.
        y = np.array([0, 0, 1, 1])
        probability = np.array([.10, .44, .56, .90])
        selected, score = threshold.choose_threshold(y, probability)
        self.assertEqual(selected, .50)
        self.assertEqual(score, 1.0)

    def test_dry_run_count_matches_completed_frozen_outer_folds(self):
        preview = threshold.dry_run_summary()
        self.assertEqual(preview["candidate_configurations"], 2070)
        self.assertEqual(preview["outer_folds"], 459)
        self.assertEqual(preview["selection_candidate_fold_evaluations"], 30996)

    def test_scoped_argument_parsing(self):
        args = threshold.parse_args(["--participant", "p01", "--target", "valence", "--progress", "--verbose", "--dry-run"])
        self.assertEqual(args.participant, "P01")
        self.assertEqual(args.target, "valence")
        self.assertTrue(args.progress)
        self.assertTrue(args.verbose)

    def test_candidate_progress_format(self):
        line = threshold.format_candidate_progress(
            "P01", 1, 46, "valence", 2, 5, 14, 27,
            {"classifier": "svm", "feature_family": "all_eeg", "feature_count_request": "10"}, 511.9,
        )
        self.assertEqual(line, "P01 [1/46] | valence | fold 2/5 | candidate 14/27 | svm | all_eeg | k=10 | elapsed 00:08:31")


if __name__ == "__main__":
    unittest.main()
