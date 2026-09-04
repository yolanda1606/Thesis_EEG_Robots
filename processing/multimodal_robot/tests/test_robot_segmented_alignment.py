"""Focused regression checks for independent Robot EEG recording segments."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ROBOT_ROOT = ROOT / "processing" / "multimodal_robot"
sys.path.insert(0, str(ROBOT_ROOT))
import continuous
import run_robot_pipeline as pipeline


def qc(*, method: str, overlap: float, segment_id: int | None = None) -> dict:
    value = {
        "task": "stack",
        "method": method,
        "eeg_duration_s": 10.0,
        "video_duration_s": 30.0,
        "status_event_count": 3,
        "original_anchor_count": 3,
        "retained_anchor_count": 3,
        "rejected_anchor_count": 0,
        "matched_anchor_count": 3,
        "aligned_overlap_s": overlap,
        "offset_s": 0.0,
        "drift_s_per_s": 0.0,
        "rmse_s": 0.01,
        "max_residual_s": 0.02,
        "median_absolute_residual_s": 0.01,
    }
    if segment_id is not None:
        value["segment_id"] = segment_id
    return value


class TestRobotSegmentedAlignment(unittest.TestCase):
    def test_segmented_alignment_keeps_models_and_rejects_only_bad_segment(self):
        task = {
            "eeg": {"segments": [{"file": "one.bdf", "order": 1}, {"file": "two.bdf", "order": 2}]},
            "video": "video.avi",
            "vision_log": "vision.csv",
        }
        failed = qc(method="invalid_residual", overlap=8.0)
        accepted = qc(method="status_constant_offset", overlap=10.0)
        with patch.object(continuous, "alignment_for_task", side_effect=[(failed, [{"task": "stack"}], [(Path("one.bdf"), object())]), (accepted, [{"task": "stack"}], [(Path("two.bdf"), object())])]) as aligned:
            task_qc, pairs, segments = continuous.alignment_for_segmented_task(task, "stack", "P99_2026-01-01")
        self.assertEqual([call.args[0]["eeg"] for call in aligned.call_args_list], ["one.bdf", "two.bdf"])
        self.assertEqual([item["qc"]["segment_id"] for item in segments], [1, 2])
        self.assertEqual([item["qc"]["accepted"] for item in segments], [False, True])
        self.assertEqual(segments[0]["qc"]["rejection_reason"], "synchronization_qc_failed")
        self.assertTrue(task_qc["accepted"])
        self.assertEqual(task_qc["accepted_segment_count"], 1)
        self.assertEqual(task_qc["rejected_segment_count"], 1)
        self.assertEqual([pair["segment_id"] for pair in pairs], [1, 2])

    def test_segmented_extraction_uses_only_accepted_models_and_preserves_ids(self):
        task = {"video": "video.avi", "vision_log": "vision.csv"}
        accepted_two = {**qc(method="status_constant_offset", overlap=10.0), "video_duration_s": 10.0, "segment_id": 2, "accepted": True}
        rejected_one = {**qc(method="invalid_residual", overlap=10.0), "video_duration_s": 10.0, "segment_id": 1, "accepted": False}
        accepted_three = {**qc(method="status_constant_offset", overlap=10.0), "video_duration_s": 10.0, "segment_id": 3, "accepted": True}

        def extracted(_participant, _task_id, _task, model, _raws, _resources, _progress):
            windows = pd.DataFrame({"segment_id": [1, 1, 1], "window_id": ["stack_s1_0.000", "stack_s1_1.000", "stack_s1_9.000"], "window_start_s": [0.0, 1.0, 9.0], "window_end_s": [2.0, 3.0, 11.0]})
            return windows.copy(), windows.copy(), windows.copy()

        records = [
            {"qc": rejected_one, "raws": [(Path("one.bdf"), object())]},
            {"qc": accepted_two, "raws": [(Path("two.bdf"), object())]},
            {"qc": accepted_three, "raws": [(Path("three.bdf"), object())]},
        ]
        with patch.object(continuous, "extract_task", side_effect=extracted) as extract:
            eeg, video, merged = continuous.extract_segmented_task("P99", "stack", task, records, Path("resources"))
        self.assertEqual(extract.call_count, 2)
        self.assertEqual([call.args[3]["segment_id"] for call in extract.call_args_list], [2, 3])
        self.assertEqual(set(eeg["segment_id"]), {2, 3})
        self.assertEqual(set(video["segment_id"]), {2, 3})
        self.assertEqual(set(merged["segment_id"]), {2, 3})
        # The final 9--11 s windows exceed the 10 s model coverage and are
        # excluded only from video/multimodal output, not EEG output.
        self.assertEqual(len(eeg), 6)
        self.assertEqual(len(video), 4)
        self.assertEqual(len(merged), 4)

    def test_normal_single_bdf_uses_existing_alignment_and_feature_path(self):
        config = {"tasks": {"pick_place": {"category": "observation", "eeg": "one.bdf", "video": "video.avi", "vision_log": "vision.csv"}}, "video_settings": {"crop": {}}}
        single_qc = qc(method="status_constant_offset", overlap=10.0)
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "alignment_for_task", return_value=(single_qc, [], [(Path("one.bdf"), object())])) as aligned, patch.object(pipeline, "alignment_for_segmented_task") as segmented, patch.object(pipeline, "extract_task", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())) as extracted:
            pipeline.continuous_run(config, "P99", Path("resources"), Path(directory) / "run", True, ("pick_place",))
        aligned.assert_called_once()
        segmented.assert_not_called()
        extracted.assert_called_once()

    @unittest.skipUnless((ROOT / "data" / "P44_2026-07-24" / "Stack" / "EEG" / "UnicornRecorder_24_07_2026_14_35_13.bdf").is_file(), "P44 raw fixture is unavailable")
    def test_p44_stack_aligns_each_configured_bdf_independently(self):
        config = pipeline.localize_task_paths(pipeline.load(ROBOT_ROOT / "configs" / "participants" / "P44.yaml"), None, "linux")
        task_qc, _pairs, segments = continuous.alignment_for_segmented_task(config["tasks"]["stack"], "stack", config["raw_participant_dir"])
        self.assertEqual([item["qc"]["segment_id"] for item in segments], [1, 2, 3])
        self.assertEqual([item["qc"]["accepted"] for item in segments], [False, True, True])
        self.assertEqual([item["qc"]["method"] for item in segments], ["invalid_residual", "status_constant_offset", "status_constant_offset"])
        self.assertEqual(task_qc["accepted_segment_count"], 2)
        self.assertEqual(task_qc["rejected_segments"], [{"segment_id": 1, "reason": "synchronization_qc_failed"}])


if __name__ == "__main__":
    unittest.main()
