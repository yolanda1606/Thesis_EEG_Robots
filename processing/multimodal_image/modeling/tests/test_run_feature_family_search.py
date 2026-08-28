"""Focused tests for the no-ICA EEG feature-family experiment."""
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
MODELING_DIR = ROOT / "processing" / "multimodal_image" / "modeling"
sys.path.insert(0, str(MODELING_DIR))
import run_feature_family_search as family_search
import run_classifier_search as classifier_search


def realistic_eeg_columns() -> list[str]:
    return classifier_search.baseline.EEG_COLUMNS.copy()


def synthetic_table(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in realistic_eeg_columns()})
    frame["valence_rating"] = [1, 5] * (rows // 2)
    frame["arousal_rating"] = [2, 6] * (rows // 2)
    return frame


class TestFeatureFamilySearch(unittest.TestCase):
    def test_real_style_feature_family_detection_and_exact_counts(self):
        families = family_search.build_feature_families(realistic_eeg_columns())
        self.assertEqual(len(families["all_eeg"]), 120)
        self.assertEqual(len(families["hjorth"]), 16)
        self.assertEqual(len(families["entropy_all_bands"]), 40)
        self.assertEqual(len(families["entropy_beta_gamma"]), 16)
        self.assertEqual(len(families["bandpower_all"]), 40)
        self.assertEqual(len(families["bandpower_beta_gamma"]), 16)
        self.assertEqual(len(families["paper_compact"]), 64)
        self.assertEqual(len(families["time_statistical"]), 8)

    def test_beta_gamma_matching_never_includes_other_bands(self):
        families = family_search.build_feature_families(realistic_eeg_columns())
        for name in ("entropy_beta_gamma", "bandpower_beta_gamma"):
            types = {family_search.parse_eeg_column(column)[0] for column in families[name]}
            self.assertTrue(all(feature_type.endswith(("_beta", "_gamma")) for feature_type in types))
            self.assertFalse(any(feature_type.endswith("_alpha") for feature_type in types))

    def test_channel_parser_and_declared_channel_subsets(self):
        self.assertEqual(family_search.parse_eeg_column("eeg_bp_beta__PO8"), ("eeg_bp_beta", "PO8", "beta"))
        self.assertEqual(family_search.parse_eeg_column("video_mwo_norm_mean"), None)
        families = family_search.build_feature_families(realistic_eeg_columns())
        paper = families["paper_compact"]
        self.assertEqual(len(family_search.subset_family_columns(paper, "all_channels")), 64)
        self.assertEqual(len(family_search.subset_family_columns(paper, "midline")), 32)
        self.assertEqual(len(family_search.subset_family_columns(paper, "posterior")), 32)
        self.assertEqual(len(family_search.subset_family_columns(paper, "frontal_central")), 32)

    def test_paper_compact_contains_only_intended_types(self):
        families = family_search.build_feature_families(realistic_eeg_columns())
        types = {family_search.parse_eeg_column(column)[0] for column in families["paper_compact"]}
        self.assertEqual(types, {"eeg_sd", "eeg_hm", "eeg_hc", "eeg_mf_hz", "eeg_se_beta", "eeg_se_gamma", "eeg_bp_beta", "eeg_bp_gamma"})

    def test_feature_count_filtering_removes_invalid_and_all_duplicate(self):
        self.assertEqual(family_search.feature_count_requests(8), ["3", "5", "all"])
        self.assertEqual(family_search.feature_count_requests(16), ["3", "5", "8", "10", "15", "all"])
        pipeline, _ = classifier_search.make_pipeline("svm", 3, 42)
        self.assertIn("selector", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["selector"], classifier_search.SelectKBest)

    def test_original_and_optimized_baseline_matching_and_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_path, optimized_path = root / "original.csv", root / "optimized.csv"
            pd.DataFrame([{"mode": "individual", "participant": "P36", "target": "valence", "modality": "eeg", "classifier": "gnb", "feature_count_resolved": "all", "balanced_accuracy_mean": .60, "accuracy_mean": .61}]).to_csv(original_path, index=False)
            pd.DataFrame([{"participant": "P36", "target": "valence", "modality": "eeg", "classifier": "svm", "feature_count": 8, "balanced_accuracy": .70, "accuracy": .71}]).to_csv(optimized_path, index=False)
            original = classifier_search.load_baseline_best(original_path, ["P36"], ["valence"])
            optimized = family_search.load_optimized_baseline(optimized_path, ["P36"], ["valence"])
        winners = pd.DataFrame([{"participant": "P36", "target": "valence", "winning_feature_family": "paper_compact", "winning_channel_subset": "midline", "winning_classifier": "svm", "winning_feature_count": 5, "feature_family_balanced_accuracy": .75, "feature_family_accuracy": .76, "winning_hyperparameters": "[]"}])
        comparison = family_search.comparison_frame(original, optimized, winners)
        self.assertAlmostEqual(comparison.iloc[0].BA_delta_vs_original, .15)
        self.assertAlmostEqual(comparison.iloc[0].BA_delta_vs_optimized_classifier, .05)

    def test_participant_subset_validation_is_reused(self):
        self.assertEqual(family_search.parse_args(["--participants", "P36,P22"]).participants, ["P36", "P22"])
        with self.assertRaises(SystemExit):
            family_search.parse_args(["--participants", "P36,P46"])

    def test_resume_skips_completed_feature_family_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, original_path, optimized_path = root / "out", root / "original.csv", root / "optimized.csv"
            participants = list(family_search.DEFAULT_PARTICIPANTS)
            pd.DataFrame([{"mode": "individual", "participant": participant, "target": "valence", "modality": "eeg", "classifier": "gnb", "feature_count_resolved": "all", "balanced_accuracy_mean": .5, "accuracy_mean": .5} for participant in participants]).to_csv(original_path, index=False)
            pd.DataFrame([{"participant": participant, "target": "valence", "modality": "eeg", "classifier": "gnb", "feature_count": "all", "balanced_accuracy": .5, "accuracy": .5} for participant in participants]).to_csv(optimized_path, index=False)
            initial_participants = ["P36", "P17"]
            base_arguments = ["--targets", "valence", "--output-dir", str(output), "--original-baseline-csv", str(original_path), "--optimized-baseline-csv", str(optimized_path), "--resume"]
            initial_arguments = ["--participants", ",".join(initial_participants), *base_arguments]
            expanded_arguments = ["--participants", ",".join(participants), *base_arguments]
            subset_arguments = ["--participants", ",".join(initial_participants), *base_arguments]
            def fake_load(_root, participant, _source):
                return synthetic_table(), root / f"{participant}.csv"
            calls: list[str] = []
            def interrupted(participant, _table, target, family, subset, _columns, classifier, request, _seed, _jobs):
                calls.append(participant)
                if len(calls) == 2:
                    raise RuntimeError("interrupted")
                return {"participant": participant, "target": target, "feature_family": family, "channel_subset": subset, "classifier": classifier, "feature_count": "all", "feature_count_request": request, "available_feature_count": 8, "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4}
            constants = {"FAMILY_CHANNEL_PLANS": (("hjorth", ("midline",)),), "MODELS": ("gnb",)}
            with patch.multiple(family_search, **constants), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load), patch.object(family_search, "evaluate_configuration", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    family_search.run(family_search.parse_args(initial_arguments))
            state = json.loads((output / family_search.CHECKPOINT_FILE).read_text())
            self.assertEqual(len(state["completed_configurations"]), 1)
            resumed_same: list[str] = []
            def complete(participant, _table, target, family, subset, _columns, classifier, request, _seed, _jobs):
                resumed_same.append(participant)
                return {"participant": participant, "target": target, "feature_family": family, "channel_subset": subset, "classifier": classifier, "feature_count": "all", "feature_count_request": request, "available_feature_count": 8, "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4}
            with patch.multiple(family_search, **constants), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load), patch.object(family_search, "evaluate_configuration", side_effect=complete):
                family_search.run(family_search.parse_args(initial_arguments))
            self.assertEqual(len(resumed_same), 5)  # 2 participants x 3 valid k values, less one checkpointed configuration

            resumed_expanded: list[str] = []
            def complete_expanded(participant, _table, target, family, subset, _columns, classifier, request, _seed, _jobs):
                resumed_expanded.append(participant)
                return {"participant": participant, "target": target, "feature_family": family, "channel_subset": subset, "classifier": classifier, "feature_count": "all", "feature_count_request": request, "available_feature_count": 8, "outer_folds": 2, "mean_balanced_accuracy": .5, "std_balanced_accuracy": 0., "mean_accuracy": .5, "std_accuracy": 0., "best_hyperparameters": "[]", "seed": 42, "gridsearch_n_jobs": 4}
            with patch.multiple(family_search, **constants), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load), patch.object(family_search, "evaluate_configuration", side_effect=complete_expanded):
                family_search.run(family_search.parse_args(expanded_arguments))
            self.assertEqual(set(resumed_expanded), {"P22", "P06", "P40"})
            self.assertEqual(len(resumed_expanded), 9)
            results = pd.read_csv(output / "results_summary.csv")
            self.assertEqual(len(results), 15)
            self.assertFalse(results.duplicated(["participant", "target", "feature_family", "channel_subset", "classifier", "feature_count_request"]).any())
            self.assertEqual(len(pd.read_csv(output / "best_per_participant_target.csv")), 5)

            with patch.multiple(family_search, **constants), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load), patch.object(family_search, "evaluate_configuration") as evaluate:
                family_search.run(family_search.parse_args(subset_arguments))
            evaluate.assert_not_called()
            self.assertEqual(len(pd.read_csv(output / "best_per_participant_target.csv")), 2)

            with self.assertRaises(SystemExit):
                family_search.parse_args(initial_arguments + ["--seed", "7"])
            changed_target_arguments = ["--participants", ",".join(initial_participants), "--targets", "arousal", "--output-dir", str(output), "--original-baseline-csv", str(original_path), "--optimized-baseline-csv", str(optimized_path), "--resume"]
            with patch.multiple(family_search, **constants), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load):
                with self.assertRaises(ValueError):
                    family_search.run(family_search.parse_args(changed_target_arguments))
            with patch.multiple(family_search, FAMILY_CHANNEL_PLANS=(("hjorth", ("midline",)),), MODELS=("svm",)), patch.object(classifier_search.baseline, "load_participant_table", side_effect=fake_load):
                with self.assertRaises(ValueError):
                    family_search.run(family_search.parse_args(subset_arguments))


if __name__ == "__main__":
    unittest.main()
