"""Focused tests for Image-only modality-agnostic SVM calibration helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "modeling"
    / "calib"
    / "calibrate_modality_agnostic_top3_svms.py"
)
SPEC = importlib.util.spec_from_file_location("modality_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


class ModalityAgnosticCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.image_output_root = self.root / "outputs" / "image_classification"
        self.image_output_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def build_record(self, modality: str) -> dict:
        candidate_names = {
            "eeg": ["eeg_alpha", "eeg_beta"],
            "face": ["video_irisdo_norm_mean", "video_mwo_norm_std"],
            "multimodal": ["eeg_alpha", "video_irisdo_norm_mean"],
        }[modality]
        values = np.array(
            [[0.0, 1.0], [0.2, 0.9], [0.4, 0.8], [0.6, 0.2], [0.8, 0.1], [1.0, 0.0], [1.2, 0.1], [1.4, 0.2]]
        )
        ratings = [1, 2, 3, 3, 4, 5, 5, 4]
        table = pd.DataFrame(values, columns=candidate_names)
        table["arousal_rating"] = ratings
        source = (
            self.root
            / "derived"
            / "P99"
            / "Image_Experiment"
            / "runs"
            / "p99_no_ica"
            / "merged"
            / "p99_image_trial_dataset.csv"
        )
        source.parent.mkdir(parents=True)
        table.to_csv(source, index=False)

        pipeline = Pipeline(
            [("scale", StandardScaler()), ("classifier", SVC(C=1.0, kernel="linear"))]
        )
        pipeline.fit(table[candidate_names], (table["arousal_rating"] >= 4).astype(int))
        model_path = self.image_output_root / f"P99_arousal_{modality}_rank1.joblib"
        joblib.dump(pipeline, model_path)
        row = pd.Series(
            {
                "participant": "P99",
                "target": "arousal",
                "transfer_rank": 1,
                "modality": modality,
                "preprocessing_condition": "no_ica source run {participant_lower}_no_ica",
                "actual_selected_feature_names": json.dumps(candidate_names),
                "final_model_path": str(model_path),
            }
        )
        return calibration.frozen_record(row, self.image_output_root, self.root)

    def test_eeg_svm_calibration_is_image_only(self) -> None:
        self.assert_calibrates("eeg")

    def test_face_svm_calibration_is_image_only(self) -> None:
        self.assert_calibrates("face")

    def test_multimodal_svm_calibration_is_image_only(self) -> None:
        self.assert_calibrates("multimodal")

    def assert_calibrates(self, modality: str) -> None:
        record = self.build_record(modality)
        selected, labels = calibration.validate_reproduction(record)
        calibrated, folds = calibration.calibrate(record, selected, labels)
        metadata = calibration.deployment_metadata(record, len(labels), labels, folds)
        self.assertEqual(record["modality"], modality)
        self.assertEqual(folds, 4)
        self.assertTrue(np.isfinite(calibrated.predict_proba(selected)).all())
        self.assertFalse(metadata["robot_data_used"])
        self.assertEqual(metadata["candidate_features"], record["candidate_features"])
        self.assertEqual(metadata["selected_features"], record["selected_features"])

    def test_hash_mismatch_is_incompatible(self) -> None:
        record = self.build_record("face")
        existing = self.root / "old_calibration"
        path = calibration.artifact_path(existing, record)
        path.parent.mkdir(parents=True)
        joblib.dump(
            {"deployment_metadata": {"original_frozen_model_sha256": "0" * 64}}, path
        )
        self.assertEqual(calibration.existing_artifact_state(existing, record), "hash_mismatch")

    def test_script_has_no_robot_data_dependency(self) -> None:
        source = inspect.getsource(calibration)
        self.assertNotIn("multimodal_robot", source)
        self.assertNotIn("outputs/robot_", source)
        self.assertIn('"robot_data_used": False', source)


if __name__ == "__main__":
    unittest.main()
