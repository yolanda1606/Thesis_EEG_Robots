"""Focused checks for the midpoint-label ambiguity diagnostic."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
MODELING_DIR = ROOT / "processing" / "multimodal_image" / "modeling"
sys.path.insert(0, str(MODELING_DIR))
import run_label_ambiguity_diagnostic as diagnostic
import run_classifier_search as classifier_search


def synthetic_table(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    columns = classifier_search.baseline.EEG_COLUMNS + classifier_search.baseline.FACE_COLUMNS
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in columns})
    frame["valence_rating"] = [1, 3, 4, 5] * (rows // 4)
    frame["arousal_rating"] = [1, 3, 4, 5] * (rows // 4)
    return frame


class TestLabelAmbiguityDiagnostic(unittest.TestCase):
    def test_standard_and_exclude_midpoint_label_rules(self):
        ratings = pd.Series([1, 3, 4, 5])
        standard_mask, standard_labels = diagnostic.label_condition(ratings, "standard")
        filtered_mask, filtered_labels = diagnostic.label_condition(ratings, "exclude_midpoint")
        self.assertEqual(standard_mask.tolist(), [True, True, True, True])
        self.assertEqual(standard_labels.tolist(), [0, 0, 1, 1])
        self.assertEqual(filtered_mask.tolist(), [True, True, False, True])
        self.assertEqual(filtered_labels.tolist(), [0, 0, 1])

    def test_midpoint_trial_is_removed_before_features_and_cv(self):
        table = synthetic_table()
        x, y = diagnostic.prepare_condition_data(table, "P36", "valence", "face", "exclude_midpoint")
        self.assertEqual(len(x), 15)
        self.assertEqual(y.tolist(), [0, 0, 1] * 5)
        splits = list(diagnostic.StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(x, y))
        self.assertTrue(all(len(train) + len(test) == 15 for train, test in splits))

    def test_trial_counts_and_retained_fraction(self):
        row = diagnostic.trial_count_row("P36", "valence", pd.Series([1, 3, 4, 5, 5]), "exclude_midpoint")
        self.assertEqual((row["total_original_trials"], row["n_trials"], row["n_low"], row["n_high"], row["n_dropped_midpoint"]), (5, 4, 2, 2, 1))
        self.assertAlmostEqual(row["retained_fraction"], 0.8)

    def test_rating_distribution_audit_counts_all_requested_rules(self):
        row = diagnostic.rating_distribution_row("P36", "valence", pd.Series([1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual([row[f"n_rating_{rating}"] for rating in range(1, 8)], [1] * 7)
        self.assertEqual((row["standard_n_low"], row["standard_n_high"]), (3, 4))
        self.assertEqual((row["exclude_midpoint_n_low"], row["exclude_midpoint_n_high"], row["n_dropped_midpoint"]), (3, 3, 1))
        self.assertAlmostEqual(row["retained_fraction"], 6 / 7)
        self.assertEqual((row["moderate_extremes_n_low"], row["moderate_extremes_n_high"], row["moderate_extremes_n_dropped"]), (2, 3, 2))
        self.assertAlmostEqual(row["moderate_extremes_retained_fraction"], 5 / 7)
        self.assertEqual((row["strong_extremes_n_low"], row["strong_extremes_n_high"], row["strong_extremes_n_dropped"]), (2, 2, 3))
        self.assertAlmostEqual(row["strong_extremes_retained_fraction"], 4 / 7)

    def test_audit_only_writes_counts_without_model_or_cv_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory) / "out"
            args = diagnostic.parse_args(["--participants", "P36", "--targets", "valence", "--audit-only", "--output-dir", str(output)])
            with patch.object(classifier_search.baseline, "load_participant_table", return_value=(synthetic_table(), root / "P36.csv")), patch.object(diagnostic, "evaluate_configuration") as evaluate, patch.object(classifier_search, "make_pipeline") as make_pipeline, patch.object(diagnostic, "StratifiedKFold") as splitter:
                result = diagnostic.run(args)
            self.assertTrue(result["audit_only"])
            self.assertTrue((output / "rating_distribution_audit.csv").is_file())
            evaluate.assert_not_called()
            make_pipeline.assert_not_called()
            splitter.assert_not_called()

    def test_insufficient_class_is_marked_without_cv(self):
        x = pd.DataFrame({"feature": [1., 2., 3.]})
        status, reason = diagnostic.condition_status(x, pd.Series([0, 0, 0]), "P36 valence")
        self.assertEqual(status, "insufficient_data")
        self.assertIn("one retained class", reason)

    def test_standard_vs_filtered_delta(self):
        best = pd.DataFrame([
            {"participant": "P36", "target": "valence", "label_condition": "standard", "mean_balanced_accuracy": .60, "mean_accuracy": .61, "classifier": "gnb", "modality": "face", "feature_count": "all"},
            {"participant": "P36", "target": "valence", "label_condition": "exclude_midpoint", "mean_balanced_accuracy": .70, "mean_accuracy": .71, "classifier": "svm", "modality": "eeg", "feature_count": 8},
        ])
        counts = pd.DataFrame([{"participant": "P36", "target": "valence", "label_condition": "exclude_midpoint", "n_trials": 16, "n_low": 8, "n_high": 8, "n_dropped_midpoint": 4, "retained_fraction": .8}])
        comparison = diagnostic.comparison_frame(best, counts)
        self.assertAlmostEqual(comparison.iloc[0].BA_delta, .10)
        self.assertAlmostEqual(comparison.iloc[0].accuracy_delta, .10)

    def test_feature_count_filtering_and_participant_subset_validation(self):
        self.assertEqual(classifier_search.resolved_feature_requests("face"), ["3", "5", "8", "10", "all"])
        self.assertEqual(diagnostic.parse_args(["--participants", "P36,P22"]).participants, ["P36", "P22"])

    def test_resume_skips_completed_configurations_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory) / "out"
            arguments = ["--participants", "P36", "--targets", "valence", "--modalities", "face", "--output-dir", str(output), "--resume"]
            def fake_load(_root, _participant, _source):
                return synthetic_table(), root / "P36.csv"
            calls: list[str] = []
            def interrupted(participant, _table, target, condition, modality, model, request, _seed, _jobs):
                calls.append(request)
                if len(calls) == 2:
                    raise RuntimeError("interrupted")
                return {"participant": participant, "target": target, "label_condition": condition, "modality": modality, "classifier": model, "feature_count": request, "feature_count_request": request, "total_original_trials": 20, "n_trials": 15 if condition == "exclude_midpoint" else 20, "n_low": 10, "n_high": 10, "n_dropped_midpoint": 5 if condition == "exclude_midpoint" else 0, "retained_fraction": .75 if condition == "exclude_midpoint" else 1., "class_balance_high_fraction": .5, "status": "complete", "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4}
            with patch.object(diagnostic, "MODELS", ("gnb",)), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load), patch.object(diagnostic, "evaluate_configuration", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    diagnostic.run(diagnostic.parse_args(arguments))
            resumed: list[str] = []
            def complete(*values):
                resumed.append(values[6])
                return interrupted(*values) if False else {"participant": values[0], "target": values[2], "label_condition": values[3], "modality": values[4], "classifier": values[5], "feature_count": values[6], "feature_count_request": values[6], "total_original_trials": 20, "n_trials": 15 if values[3] == "exclude_midpoint" else 20, "n_low": 10, "n_high": 10, "n_dropped_midpoint": 5 if values[3] == "exclude_midpoint" else 0, "retained_fraction": .75 if values[3] == "exclude_midpoint" else 1., "class_balance_high_fraction": .5, "status": "complete", "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4}
            with patch.object(diagnostic, "MODELS", ("gnb",)), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load), patch.object(diagnostic, "evaluate_configuration", side_effect=complete):
                diagnostic.run(diagnostic.parse_args(arguments))
            results = pd.read_csv(output / "results_summary.csv")
            self.assertEqual(len(resumed), 9)
            self.assertEqual(len(results), 10)
            self.assertFalse(results.duplicated(["participant", "target", "label_condition", "modality", "classifier", "feature_count_request"]).any())


if __name__ == "__main__":
    unittest.main()
