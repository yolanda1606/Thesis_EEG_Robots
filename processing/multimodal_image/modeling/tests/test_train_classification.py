"""Focused synthetic checks for the unified Image Experiment classifier."""
from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from processing.multimodal_image.modeling.core import train_classification as tc


def synthetic_table(rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in tc.EEG_COLUMNS + tc.FACE_COLUMNS})
    frame["valence_rating"] = [1, 5] * (rows // 2)
    frame["arousal_rating"] = [2, 6] * (rows // 2)
    frame["trigger"] = range(rows)
    return frame


class TestTrainClassification(unittest.TestCase):
    def test_progress_and_verbose_flags_are_accepted(self):
        base = ["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "flags"]
        self.assertTrue(tc.parse_args(base + ["--progress"]).progress)
        args = tc.parse_args(base + ["--verbose"])
        self.assertTrue(args.verbose)
        reporter = tc.ProgressReporter(args.progress, args.verbose)
        self.assertTrue(reporter.progress_enabled)
        self.assertTrue(reporter.verbose_enabled)

    def test_n_jobs_flag_is_accepted(self):
        base = ["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "flags"]
        self.assertEqual(tc.parse_args(base + ["--n-jobs", "-1"]).n_jobs, -1)
        with self.assertRaises(SystemExit): tc.parse_args(base + ["--n-jobs", "0"])

    def test_source_run_defaults_to_final_and_resolves_no_ica_template(self):
        base = ["--mode", "individual", "--target", "valence", "--modality", "face", "--run-name", "source"]
        self.assertEqual(tc.parse_args(base).source_run, tc.DEFAULT_SOURCE_RUN)
        args = tc.parse_args(base + ["--source-run", "{participant_lower}_no_ica"])
        self.assertEqual(tc.source_run_name("P15", args.source_run), "p15_no_ica")
        self.assertEqual(
            tc.final_run_path(Path("derived"), "P15", args.source_run),
            Path("derived/P15/Image_Experiment/runs/p15_no_ica"),
        )

    def test_grid_search_receives_requested_n_jobs(self):
        captured = {}
        original = tc.GridSearchCV
        def recording_search(*args, **kwargs):
            captured["n_jobs"] = kwargs["n_jobs"]
            return original(*args, **kwargs)
        x = pd.DataFrame({"x": range(12), "y": [0, 1] * 6})
        y = pd.Series([0, 1] * 6)
        splits = list(tc.StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(x, y))
        with patch.object(tc, "GridSearchCV", side_effect=recording_search):
            tc.fit_outer_fold(x, y, x.iloc[:2], model="gnb", feature_count="all", inner_splits=splits, seed=42, n_jobs=2)
        self.assertEqual(captured["n_jobs"], 2)

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

    def test_serial_and_parallel_inner_search_have_identical_scientific_results(self):
        frame = synthetic_table(20)
        x, y, _ = tc.prepare_modality_data(frame, "P19", "valence", "face")
        serial, _, _ = tc.evaluate_individual("P19", x, y, "valence", "face", "gnb", "all", 42, n_jobs=1)
        parallel, _, _ = tc.evaluate_individual("P19", x, y, "valence", "face", "gnb", "all", 42, n_jobs=2)
        pd.testing.assert_frame_equal(pd.DataFrame(serial), pd.DataFrame(parallel), check_exact=True)

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

    def test_checkpoint_resume_skips_completed_and_rejects_mismatch(self):
        def rows(participant, _x, _y, target, modality, model, request, _seed, _reporter, n_jobs=1):
            common = {"mode": "individual", "participant": participant, "held_out_participant": participant, "target": target, "modality": modality, "classifier": model, "feature_count_request": request, "feature_count_resolved": "all", "fold": 1}
            result = {**common, "training_participants": participant, "sample_size": 20, "train_size": 16, "test_size": 4, "low_count": 10, "high_count": 10, "balanced_accuracy": .5, "accuracy": .5, "precision": .5, "recall": .5, "f1": .5, "tn": 1, "fp": 1, "fn": 1, "tp": 1}
            selected = {**common, "feature": "feature", "selected": True, "feature_score": 1.0}
            params = {**common, "best_parameters": "{}", "inner_best_balanced_accuracy": .5}
            return [result], [selected], [params]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); readiness = root / "readiness.csv"; output = root / "outputs"
            pd.DataFrame({"participant": ["P10", "P11"], "qc_group": ["Very clean", "Very clean"]}).to_csv(readiness, index=False)
            base = ["--mode", "individual", "--target", "valence", "--modality", "face", "--models", "gnb", "--feature-counts", "all", "--run-name", "checkpoint", "--readiness-csv", str(readiness), "--output-root", str(output)]
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(args[0])
                if len(calls) == 2: raise RuntimeError("simulated interruption")
                return rows(*args, **kwargs)
            with patch.object(tc, "load_participant_table", side_effect=lambda _root, p, _source: (synthetic_table(20), root / f"{p}.csv")), patch.object(tc, "evaluate_individual", side_effect=interrupted):
                with self.assertRaises(RuntimeError): tc.run(tc.parse_args(base))
            state = json.loads((output / "checkpoint" / tc.CHECKPOINT_STATE_FILE).read_text())
            self.assertEqual(len(state["completed_configurations"]), 1)
            resumed = []
            def record(*args, **kwargs): resumed.append(args[0]); return rows(*args, **kwargs)
            with patch.object(tc, "load_participant_table", side_effect=lambda _root, p, _source: (synthetic_table(20), root / f"{p}.csv")), patch.object(tc, "evaluate_individual", side_effect=record):
                tc.run(tc.parse_args(base + ["--resume"]))
            self.assertEqual(resumed, ["P11"])
            self.assertEqual(len(pd.read_csv(output / "checkpoint" / "fold_results.csv")), 2)
            with patch.object(tc, "load_participant_table", side_effect=lambda _root, p, _source: (synthetic_table(20), root / f"{p}.csv")):
                with self.assertRaises(ValueError): tc.run(tc.parse_args(base + ["--resume", "--seed", "7"]))
            clean_base = base.copy(); clean_base[clean_base.index("checkpoint")] = "clean"
            with patch.object(tc, "load_participant_table", side_effect=lambda _root, p, _source: (synthetic_table(20), root / f"{p}.csv")), patch.object(tc, "evaluate_individual", side_effect=rows):
                tc.run(tc.parse_args(clean_base))
            pd.testing.assert_frame_equal(
                pd.read_csv(output / "checkpoint" / "fold_results.csv"),
                pd.read_csv(output / "clean" / "fold_results.csv"),
            )


if __name__ == "__main__":
    unittest.main()
