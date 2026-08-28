"""Focused unit tests for the fixed individual classifier search."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
MODELING_DIR = ROOT / "processing" / "multimodal_image" / "modeling"
sys.path.insert(0, str(MODELING_DIR))
import run_classifier_search as search
import train_classification as baseline


def synthetic_table(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    table = pd.DataFrame({column: rng.normal(size=rows) for column in baseline.EEG_COLUMNS + baseline.FACE_COLUMNS})
    table["valence_rating"] = [1, 5] * (rows // 2)
    table["arousal_rating"] = [2, 6] * (rows // 2)
    return table


class TestClassifierSearch(unittest.TestCase):
    def test_participant_parsing_allows_valid_ordered_cohort_subsets(self):
        args = search.parse_args([])
        self.assertEqual(args.participants, list(search.DEVELOPMENT_PARTICIPANTS))
        self.assertEqual(args.n_jobs, 4)
        self.assertEqual(
            search.parse_participants("P36,P22,P06,P17,P40"),
            ["P36", "P22", "P06", "P17", "P40"],
        )
        self.assertEqual(search.parse_participants("P36"), ["P36"])

    def test_participant_parsing_rejects_outside_cohort_and_duplicates(self):
        with self.assertRaises(ValueError):
            search.parse_participants("P36,P46")
        with self.assertRaises(ValueError):
            search.parse_participants("P36,P36")

    def test_low_high_labeling_is_reused_unchanged(self):
        self.assertEqual(baseline.label_ratings(pd.Series([3.9, 4.0])).tolist(), [0, 1])

    def test_face_feature_requests_filter_invalid_duplicates(self):
        self.assertEqual(search.resolved_feature_requests("face"), ["3", "5", "8", "10", "all"])
        self.assertEqual(search.resolved_feature_requests("eeg"), list(search.FEATURE_COUNTS))

    def test_knn_neighbors_are_filtered_to_inner_training_size(self):
        _, grid = search.make_pipeline("knn", "all", 42)
        splits = [(np.arange(10), np.arange(10, 12)), (np.arange(8), np.arange(8, 12))]
        filtered = search.filter_knn_grid(grid, splits)
        self.assertEqual(filtered[0]["classifier__n_neighbors"], [3, 5, 7])

    def test_svm_gamma_is_only_in_rbf_grid_and_balanced_options_exist(self):
        _, grids = search.make_pipeline("svm", "all", 42)
        linear, rbf = grids
        self.assertNotIn("classifier__gamma", linear)
        self.assertEqual(linear["classifier__class_weight"], [None, "balanced"])
        self.assertEqual(rbf["classifier__gamma"], ["scale", "auto", 0.001, 0.01, 0.1, 1])

    def test_logistic_and_extra_trees_pipeline_configuration(self):
        logistic, _ = search.make_pipeline("logreg", 3, 42)
        trees, tree_grid = search.make_pipeline("extra_trees", 3, 42)
        self.assertIsInstance(logistic.named_steps["classifier"], search.LogisticRegression)
        self.assertIn("scale", logistic.named_steps)
        self.assertIsInstance(trees.named_steps["classifier"], search.ExtraTreesClassifier)
        self.assertNotIn("scale", trees.named_steps)
        self.assertEqual(trees.named_steps["classifier"].n_jobs, 1)
        self.assertEqual(tree_grid[0]["classifier__class_weight"], [None, "balanced"])

    def test_inner_grid_search_receives_n_jobs_and_outer_rows_do_not_leak(self):
        captured: dict[str, int] = {}
        original = search.GridSearchCV
        def recording_search(*args, **kwargs):
            captured["n_jobs"] = kwargs["n_jobs"]
            return original(*args, **kwargs)
        table = synthetic_table()
        x, y, _ = baseline.prepare_modality_data(table, "P01", "valence", "face")
        with patch.object(search, "GridSearchCV", side_effect=recording_search):
            row = search.evaluate_configuration("P01", table, "valence", "face", "gnb", "all", 42, 2)
        self.assertEqual(captured["n_jobs"], 2)
        self.assertEqual(row["outer_folds"], 5)
        self.assertIn("mean_balanced_accuracy", row)

    def test_baseline_matching_and_delta_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.csv"
            pd.DataFrame([
                {"mode": "individual", "participant": "P36", "target": "valence", "modality": "eeg", "classifier": "gnb", "feature_count_resolved": "all", "balanced_accuracy_mean": .60, "accuracy_mean": .61},
                {"mode": "individual", "participant": "P36", "target": "valence", "modality": "face", "classifier": "svm", "feature_count_resolved": "5", "balanced_accuracy_mean": .70, "accuracy_mean": .65},
            ]).to_csv(path, index=False)
            previous = search.load_baseline_best(path, ["P36"], ["valence"])
        optimized = pd.DataFrame([{"participant": "P36", "target": "valence", "modality": "face", "classifier": "logreg", "feature_count": 5, "balanced_accuracy": .75, "accuracy": .70, "best_hyperparameters": "[]"}])
        comparison = search.baseline_comparison(previous, optimized)
        self.assertEqual(comparison.iloc[0].previous_classifier, "svm")
        self.assertAlmostEqual(comparison.iloc[0].BA_delta, .05)
        self.assertAlmostEqual(comparison.iloc[0].accuracy_delta, .05)

    def test_resume_uses_checkpoint_and_skips_completed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, baseline_csv = root / "out", root / "baseline.csv"
            pairs = [(participant, target) for participant in search.DEVELOPMENT_PARTICIPANTS for target in search.TARGETS]
            pd.DataFrame([{"mode": "individual", "participant": p, "target": t, "modality": "face", "classifier": "gnb", "feature_count_resolved": "all", "balanced_accuracy_mean": .5, "accuracy_mean": .5} for p, t in pairs]).to_csv(baseline_csv, index=False)
            arguments = ["--output-dir", str(output), "--baseline-csv", str(baseline_csv), "--targets", "valence", "--modalities", "face"]
            calls: list[tuple[str, str]] = []
            def fake_load(_root, participant, _source):
                return synthetic_table(), root / f"{participant}.csv"
            def interrupted(participant, _table, target, modality, model, request, _seed, _jobs):
                calls.append((participant, model))
                if len(calls) == 2:
                    raise RuntimeError("interrupted")
                return {"participant": participant, "target": target, "modality": modality, "classifier": model, "feature_count": "all", "feature_count_request": request, "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4, "estimator_n_jobs": None}
            with patch.object(baseline, "load_participant_table", side_effect=fake_load), patch.object(search, "evaluate_configuration", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    search.run(search.parse_args(arguments + ["--resume"]))
            state = json.loads((output / search.CHECKPOINT_FILE).read_text())
            self.assertEqual(len(state["completed_configurations"]), 1)
            resumed: list[tuple[str, str]] = []
            def complete(participant, table, target, modality, model, request, seed, jobs):
                resumed.append((participant, model))
                return interrupted(participant, table, target, modality, model, request, seed, jobs) if False else {"participant": participant, "target": target, "modality": modality, "classifier": model, "feature_count": "all", "feature_count_request": request, "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4, "estimator_n_jobs": None}
            with patch.object(baseline, "load_participant_table", side_effect=fake_load), patch.object(search, "evaluate_configuration", side_effect=complete):
                search.run(search.parse_args(arguments + ["--resume"]))
            self.assertEqual(len(resumed), 374)  # 15 participants x 5 models x 5 face requests, less one checkpointed row


if __name__ == "__main__":
    unittest.main()
