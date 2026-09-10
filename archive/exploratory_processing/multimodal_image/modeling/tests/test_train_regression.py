"""Synthetic checks for the Image Experiment continuous-rating trainer."""
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
from processing.multimodal_image.modeling import train_regression as tr


def synthetic_table(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in tr.EEG_COLUMNS + tr.FACE_COLUMNS})
    frame["valence_rating"] = np.linspace(1.0, 7.0, rows)
    frame["arousal_rating"] = np.linspace(6.8, 1.2, rows)
    return frame


class TestTrainRegression(unittest.TestCase):
    @staticmethod
    def run_args(root: Path, run_name: str, *extra: str):
        readiness = root / "readiness.csv"
        pd.DataFrame({"participant": ["P19"], "qc_group": ["Very clean"]}).to_csv(readiness, index=False)
        return tr.parse_args(["--mode", "individual", "--target", "valence", "--modality", "face", "--models", "ridge", "--feature-counts", "all,5", "--participants", "P19", "--run-name", run_name, "--readiness-csv", str(readiness), "--output-root", str(root / "out"), *extra])

    def test_progress_and_verbose_flags_are_accepted(self):
        base = ["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "flags"]
        self.assertTrue(tr.parse_args(base + ["--progress"]).progress)
        args = tr.parse_args(base + ["--verbose"])
        self.assertTrue(args.verbose)
        reporter = tr.ProgressReporter(args.progress, args.verbose)
        self.assertTrue(reporter.progress_enabled)
        self.assertTrue(reporter.verbose_enabled)

    def test_n_jobs_is_accepted_and_reaches_grid_search(self):
        base = ["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "workers"]
        self.assertEqual(tr.parse_args(base + ["--n-jobs", "2"]).n_jobs, 2)
        x = pd.DataFrame({"x": np.arange(12), "y": np.arange(12)[::-1]})
        with patch.object(tr, "GridSearchCV", wraps=tr.GridSearchCV) as search:
            tr.fit_outer_fold(x.iloc[:8], pd.Series(np.arange(8, dtype=float)), x.iloc[8:], "ridge", "all", list(tr.KFold(2).split(x.iloc[:8])), 42, n_jobs=2)
        self.assertEqual(search.call_args.kwargs["n_jobs"], 2)

    def test_worker_counts_do_not_change_scientific_results(self):
        x, y, _ = tr.prepare_modality_data(synthetic_table(), "P19", "valence", "face")
        serial = pd.DataFrame(tr.evaluate_individual("P19", x, y, "valence", "face", "ridge", "5", 42, n_jobs=1)[0]).sort_index(axis=1)
        parallel = pd.DataFrame(tr.evaluate_individual("P19", x, y, "valence", "face", "ridge", "5", 42, n_jobs=2)[0]).sort_index(axis=1)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_checkpoint_resume_skips_completed_work_and_matches_clean_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); interrupted = self.run_args(root, "interrupted")
            original = tr.evaluate_individual
            def fail_on_second(*args, **kwargs):
                if args[6] == "5": raise RuntimeError("simulated interruption")
                return original(*args, **kwargs)
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")), patch.object(tr, "evaluate_individual", side_effect=fail_on_second):
                with self.assertRaises(RuntimeError): tr.run(interrupted)
            state = json.loads((root / "out" / "interrupted" / tr.CHECKPOINT_STATE_FILE).read_text())
            self.assertEqual(len(state["completed_configurations"]), 1)
            resumed = self.run_args(root, "interrupted", "--resume")
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")), patch.object(tr, "evaluate_individual", wraps=original) as resumed_evaluation:
                tr.run(resumed)
            self.assertEqual(resumed_evaluation.call_count, 1)
            clean = self.run_args(root, "clean")
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")):
                tr.run(clean)
            for filename in ("fold_results.csv", "predictions.csv", "best_hyperparameters.csv", "selected_features_by_fold.csv", "results_summary.csv"):
                left = pd.read_csv(root / "out" / "interrupted" / filename).sort_index(axis=1)
                right = pd.read_csv(root / "out" / "clean" / filename).sort_index(axis=1)
                pd.testing.assert_frame_equal(left, right)

    def test_resume_rejects_incompatible_settings_and_normal_run_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.run_args(root, "saved")
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")): tr.run(args)
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")):
                with self.assertRaises(FileExistsError): tr.run(args)
            incompatible = self.run_args(root, "saved", "--resume", "--seed", "99")
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")):
                with self.assertRaises(ValueError): tr.run(incompatible)

    def test_continuous_targets_are_preserved_without_labels(self):
        _, ratings, _ = tr.prepare_modality_data(synthetic_table(), "P19", "valence", "face")
        self.assertEqual(ratings.iloc[[0, -1]].tolist(), [1.0, 7.0])
        self.assertGreater(ratings.nunique(), 2)

    def test_missing_modality_rows_match_classification_policy(self):
        frame = synthetic_table(); frame.loc[0, tr.EEG_COLUMNS] = np.nan
        self.assertEqual(len(tr.prepare_modality_data(frame, "P19", "valence", "eeg")[0]), len(frame) - 1)
        self.assertEqual(len(tr.prepare_modality_data(frame, "P19", "valence", "face")[0]), len(frame))

    def test_scaling_and_regression_selection_are_in_pipeline(self):
        pipeline, _ = tr.make_pipeline("ridge", 5, 42)
        self.assertIsInstance(pipeline.named_steps["scale"], tr.StandardScaler)
        self.assertIsInstance(pipeline.named_steps["selector"], tr.SelectKBest)
        self.assertIs(pipeline.named_steps["selector"].score_func, tr.f_regression)
        self.assertNotIn("selector", tr.make_pipeline("ridge", "all", 42)[0].named_steps)

    def test_metrics_include_negative_r2(self):
        metrics = tr.metric_row(pd.Series([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
        self.assertEqual(set(metrics), {"rmse", "mae", "r2", "explained_variance"})
        self.assertLess(metrics["r2"], 0)

    def test_selected_feature_names_are_recovered(self):
        x = pd.DataFrame({"first": np.arange(12), "second": np.arange(12)[::-1]})
        fitted, _, _, _ = tr.fit_outer_fold(x.iloc[:8], pd.Series(np.arange(8, dtype=float)), x.iloc[8:], "ridge", 1, list(tr.KFold(2).split(x.iloc[:8])), 42)
        rows = tr.selected_feature_rows(fitted, list(x.columns), {"fold": 1})
        self.assertEqual({row["feature"] for row in rows}, set(x.columns))
        self.assertEqual(sum(row["selected"] for row in rows), 1)

    def test_individual_rows_never_name_other_participants(self):
        x, y, _ = tr.prepare_modality_data(synthetic_table(), "P19", "arousal", "face")
        rows, _, _, _ = tr.evaluate_individual("P19", x, y, "arousal", "face", "ridge", "all", 42)
        self.assertTrue(all(row["participant"] == row["training_participants"] == "P19" for row in rows))

    def test_general_loso_and_inner_splits_are_group_safe(self):
        groups = pd.Series(["P10"] * 8 + ["P11"] * 8 + ["P19"] * 8)
        for train, test in tr.LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups):
            self.assertEqual(groups.iloc[test].nunique(), 1)
            self.assertFalse(set(groups.iloc[train]).intersection(groups.iloc[test]))
        for train, test in tr.grouped_inner_splits(pd.DataFrame({"x": range(len(groups))}), groups):
            self.assertFalse(set(groups.iloc[train]).intersection(groups.iloc[test]))

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); readiness = root / "readiness.csv"; output = root / "out"
            pd.DataFrame({"participant": ["P19"], "qc_group": ["Very clean"]}).to_csv(readiness, index=False)
            (output / "existing").mkdir(parents=True)
            args = tr.parse_args(["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "existing", "--readiness-csv", str(readiness), "--output-root", str(output)])
            with patch.object(tr, "load_participant_table", return_value=(synthetic_table(), root / "table.csv")):
                with self.assertRaises(FileExistsError): tr.run(args)


if __name__ == "__main__":
    unittest.main()
