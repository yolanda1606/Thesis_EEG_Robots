"""Focused synthetic checks for the unified Image Experiment classifier."""
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
from processing.multimodal_image.modeling import train_classification as tc


def synthetic_table(rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in tc.EEG_COLUMNS + tc.FACE_COLUMNS})
    frame["valence_rating"] = [1, 5] * (rows // 2)
    frame["arousal_rating"] = [2, 6] * (rows // 2)
    frame["trigger"] = range(rows)
    return frame


class TestTrainClassification(unittest.TestCase):
    def test_label_definition(self):
        self.assertEqual(tc.label_ratings(pd.Series([1, 3.9, 4, 7])).tolist(), [0, 0, 1, 1])

    def test_missing_eeg_does_not_remove_face_row(self):
        frame = synthetic_table()
        frame.loc[0, tc.EEG_COLUMNS] = np.nan
        eeg, _, _ = tc.prepare_modality_data(frame, "P00", "valence", "eeg")
        face, _, _ = tc.prepare_modality_data(frame, "P00", "valence", "face")
        multi, _, _ = tc.prepare_modality_data(frame, "P00", "valence", "multimodal")
        self.assertEqual(len(eeg), len(frame)-1)
        self.assertEqual(len(face), len(frame))
        self.assertEqual(len(multi), len(frame)-1)

    def test_predictor_columns_exclude_metadata(self):
        self.assertNotIn("participant", tc.modality_columns("eeg"))
        self.assertNotIn("trigger", tc.modality_columns("face"))
        self.assertNotIn("valence_rating", tc.modality_columns("multimodal"))

    def test_selector_is_inside_pipeline_and_all_has_none(self):
        selected, _ = tc.make_pipeline("svm", 5, 42)
        all_features, _ = tc.make_pipeline("svm", "all", 42)
        self.assertIn("selector", selected.named_steps)
        self.assertIsInstance(selected.named_steps["selector"], tc.SelectKBest)
        self.assertNotIn("selector", all_features.named_steps)
        self.assertEqual(tc.resolved_feature_count("20", 10), 10)

    def test_general_outer_and_inner_splits_keep_groups_separate(self):
        groups = pd.Series(["P01"] * 6 + ["P02"] * 6 + ["P03"] * 6)
        labels = pd.Series([0, 1, 0, 1, 0, 1] * 3)
        for train, test in tc.general_outer_splits(groups):
            self.assertFalse(set(groups.iloc[train]).intersection(groups.iloc[test]))
        x = pd.DataFrame({"x": np.arange(len(groups))})
        for train, test in tc.grouped_inner_splits(x, labels, groups, 42):
            self.assertFalse(set(groups.iloc[train]).intersection(groups.iloc[test]))

    def test_individual_evaluation_does_not_accept_other_participant_rows(self):
        frame = synthetic_table(20)
        x, y, _ = tc.prepare_modality_data(frame, "P10", "valence", "face")
        rows, _, _ = tc.evaluate_individual("P10", x, y, "valence", "face", "gnb", "all", 42)
        self.assertTrue(rows)
        self.assertTrue(all(row["participant"] == "P10" and row["training_participants"] == "P10" for row in rows))

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); readiness = root / "readiness.csv"
            pd.DataFrame({"participant": ["P10"], "qc_group": ["Very clean"]}).to_csv(readiness, index=False)
            existing = root / "outputs" / "already_here"; existing.mkdir(parents=True)
            args = tc.parse_args(["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "already_here", "--readiness-csv", str(readiness), "--output-root", str(root / "outputs")])
            with patch.object(tc, "load_participant_table", return_value=(synthetic_table(20), root / "table.csv")):
                with self.assertRaises(FileExistsError): tc.run(args)

    def test_dry_run_does_not_train_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); readiness = root / "readiness.csv"
            pd.DataFrame({"participant": ["P10"], "qc_group": ["Very clean"]}).to_csv(readiness, index=False)
            output = root / "outputs"
            args = tc.parse_args(["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "dry", "--readiness-csv", str(readiness), "--output-root", str(output), "--dry-run"])
            with patch.object(tc, "load_participant_table", return_value=(synthetic_table(20), root / "table.csv")), patch.object(tc, "evaluate_individual") as evaluate:
                result = tc.run(args)
            self.assertTrue(result["dry_run"])
            evaluate.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
