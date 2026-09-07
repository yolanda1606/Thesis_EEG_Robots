"""Focused synthetic checks for the frozen personalized binary-search design."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "processing" / "multimodal_image" / "modeling"
sys.path.insert(0, str(MODEL_DIR))
import run_focused_personalized_binary_search as focused


class TestFocusedPersonalizedBinarySearch(unittest.TestCase):
    def test_frozen_grid_and_target_specific_plan(self):
        self.assertEqual(focused.model_grid("gnb"), [{"classifier__var_smoothing": [1e-11]}])
        self.assertEqual(len(focused.model_grid("logreg")[0]["classifier__C"]), 3)
        self.assertEqual(len(focused.representation_plan("valence")), 9)
        self.assertEqual(len(focused.representation_plan("arousal")), 6)
        self.assertEqual(len(focused.feature_columns("paper_compact", "all_channels")), 64)
        self.assertEqual(len(focused.feature_columns("paper_compact", "frontal_central")), 32)

    def test_full_cohort_plan_has_2070_configurations(self):
        plan = focused.planned_configurations(list(focused.PARTICIPANTS), list(focused.TARGETS))
        self.assertEqual(len(plan), 2070)
        self.assertEqual(sum(row["target"] == "valence" for row in plan), 1242)
        self.assertEqual(sum(row["target"] == "arousal" for row in plan), 828)

    def test_evidence_driven_face_and_multimodal_extension_plan(self):
        self.assertEqual(focused.feature_columns("face_geometry", "all_face", "face"), focused.baseline.FACE_COLUMNS)
        self.assertEqual(len(focused.feature_columns("all_eeg_plus_face", "all_channels", "multimodal")), 130)
        self.assertEqual([row["feature_count_request"] for row in focused.modality_representation_plan("arousal", "face")], ["5", "all"])
        multimodal = focused.modality_representation_plan("valence", "multimodal")
        self.assertEqual([row["feature_count_request"] for row in multimodal], ["5", "10", "20"])
        self.assertEqual({row["feature_family"] for row in multimodal}, {"all_eeg_plus_face"})
        plan = focused.planned_configurations(list(focused.PARTICIPANTS), list(focused.TARGETS), ["face", "multimodal"])
        self.assertEqual(len(plan), 1380)
        self.assertEqual(sum(row["modality"] == "face" for row in plan), 552)
        self.assertEqual(sum(row["modality"] == "multimodal" for row in plan), 828)

    def test_extension_svm_grid_restores_evidence_supported_regions_only(self):
        self.assertEqual(focused.model_grid("svm", "eeg"), [
            {"classifier__kernel": ["linear"], "classifier__C": [1, 10], "classifier__class_weight": [None, "balanced"]},
            {"classifier__kernel": ["rbf"], "classifier__C": [1, 10], "classifier__gamma": [0.001, 0.01], "classifier__class_weight": [None, "balanced"]},
        ])
        self.assertEqual(focused.model_grid("svm", "face"), [
            {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10], "classifier__class_weight": [None, "balanced"]},
            {"classifier__kernel": ["rbf"], "classifier__C": [1, 10], "classifier__gamma": ["scale", 0.01], "classifier__class_weight": [None, "balanced"]},
        ])
        self.assertEqual(focused.model_grid("svm", "multimodal"), focused.model_grid("svm", "face"))

    def test_face_multimodal_cli_uses_extension_default(self):
        args = focused.parse_args(["--modalities", "face,multimodal", "--dry-run"])
        self.assertEqual(args.output_dir, focused.DEFAULT_EXTENSION_OUTPUT)
        self.assertEqual(args.modalities, ["face", "multimodal"])

    def test_combined_builder_adds_modalities_without_rewriting_eeg_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); eeg = root / "eeg"; extension = root / "extension"; eeg.mkdir(); extension.mkdir()
            config = pd.DataFrame([{"participant":"P01", "target":"valence", "classifier":"gnb", "feature_family":"all_eeg", "channel_subset":"all_channels", "feature_count_request":"8", "analysis_tier":"primary", "available_feature_count":120, "feature_count_resolved":8, "outer_folds":5, "mean_outer_cv_balanced_accuracy":.6, "fold_ba_sd":.1, "outer_fold_ba_values":"[]", "mean_accuracy":.6, "low_count":50, "high_count":50, "sample_size":100}])
            extension_config = config.assign(modality="face", feature_family="face_geometry", channel_subset="all_face", available_feature_count=10)
            files = ["complete_search_results.csv", "fold_results.csv", "predictions_by_fold.csv", "selected_features_by_fold.csv", "selected_hyperparameters_by_fold.csv", "top3_models_per_participant_target.csv"]
            for name in files:
                config.to_csv(eeg / name, index=False)
            extension_files = ["complete_search_results.csv", "fold_results.csv", "predictions_by_fold.csv", "selected_features_by_fold.csv", "selected_hyperparameters_by_fold.csv", "top3_models_per_participant_target_modality.csv"]
            for name in extension_files:
                extension_config.to_csv(extension / name, index=False)
            focused.write_combined_all_modality_outputs(eeg, extension, root)
            combined = pd.read_csv(root / "configuration_summary_all_modalities.csv")
            self.assertEqual(combined.modality.tolist(), ["eeg", "face"])
            self.assertTrue((root / "participant_summary_all_modalities.csv").exists())

    def test_diversity_uses_competitive_classifier_then_feature(self):
        rows = []
        for classifier, family, subset, ba, sd in [("logreg", "all_eeg", "all_channels", .70, .08), ("gnb", "hjorth", "all_channels", .69, .07), ("svm", "paper_compact", "frontal_central", .685, .06), ("logreg", "all_eeg", "all_channels", .68, .05)]:
            rows.append({"participant":"P01", "target":"valence", "classifier":classifier, "feature_family":family, "channel_subset":subset, "feature_count_request":"8", "feature_count_resolved":8, "analysis_tier":"primary", "available_feature_count":120, "mean_outer_cv_balanced_accuracy":ba, "fold_ba_sd":sd})
        absolute, transfer = focused.ranked_top3(pd.DataFrame(rows))
        self.assertEqual(len(absolute), 3)
        self.assertEqual(transfer.selection_reason.tolist(), ["best", "diverse_classifier", "diverse_classifier"])
        self.assertEqual(transfer.classifier.tolist(), ["logreg", "gnb", "svm"])

    def test_image_domain_top3_selects_across_modalities_without_refitting(self):
        rows = []
        for modality, classifier, family, ba, sd in [
            ("eeg", "logreg", "all_eeg", .70, .08),
            ("face", "gnb", "face_geometry", .695, .07),
            ("multimodal", "svm", "all_eeg_plus_face", .69, .06),
            ("eeg", "svm", "paper_compact", .66, .05),
        ]:
            rows.append({"participant": "P01", "target": "valence", "modality": modality,
                         "classifier": classifier, "feature_family": family, "channel_subset": "all_channels",
                         "mean_outer_cv_balanced_accuracy": ba, "fold_ba_sd": sd,
                         "final_model_path": f"/models/{modality}_{classifier}.joblib"})
        selected = focused.ranked_image_domain_top3_across_modalities(pd.DataFrame(rows))
        self.assertEqual(selected.transfer_rank.tolist(), [1, 2, 3])
        self.assertEqual(selected.modality.tolist(), ["eeg", "face", "multimodal"])
        self.assertEqual(selected.selection_reason.tolist(), ["best", "diverse_classifier", "diverse_classifier"])
        self.assertEqual(selected.delta_ba_from_best.round(3).tolist(), [0.0, -.005, -.01])

    def test_dry_run_does_not_create_output(self):
        args = focused.parse_args(["--participants", "P01,P02", "--dry-run"])
        result = focused.run(args)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["configurations"], 90)


if __name__ == "__main__":
    unittest.main()
