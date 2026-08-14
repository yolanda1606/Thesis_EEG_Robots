"""Synthetic checks for the Image Experiment continuous-rating trainer."""
from __future__ import annotations

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
