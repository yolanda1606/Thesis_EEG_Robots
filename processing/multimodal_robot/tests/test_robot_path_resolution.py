"""Focused path-resolution checks for portable Robot participant YAML files."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROBOT_ROOT = ROOT / "processing" / "multimodal_robot"
sys.path.insert(0, str(ROBOT_ROOT))
import run_robot_pipeline as pipeline
import continuous


class TestRobotPathResolution(unittest.TestCase):
    def setUp(self):
        self.roots = {
            "linux": "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data",
            "windows": r"D:\Thesis\data",
        }

    def test_p01_linux_path_with_spaces(self):
        config = pipeline.load(ROBOT_ROOT / "configs" / "participants" / "P01.yaml")
        result = pipeline.localize_task_paths(config, None, "linux", self.roots)
        self.assertEqual(
            result["tasks"]["pick_place"]["eeg"],
            "/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/"
            "P01_2026-05-26/Pick and Place/EEG/UnicornRecorder_26_05_2026_15_32_27.bdf",
        )

    def test_p01_windows_path(self):
        config = pipeline.load(ROBOT_ROOT / "configs" / "participants" / "P01.yaml")
        result = pipeline.localize_task_paths(config, None, "windows", self.roots)
        self.assertEqual(
            result["tasks"]["pick_place"]["eeg"],
            r"D:\Thesis\data\P01_2026-05-26\Pick and Place\EEG\UnicornRecorder_26_05_2026_15_32_27.bdf",
        )

    def test_raw_root_participant_directory_overrides_configured_root(self):
        config = pipeline.load(ROBOT_ROOT / "configs" / "participants" / "P01.yaml")
        with tempfile.TemporaryDirectory() as directory:
            participant_root = Path(directory) / "P01_2026-05-26"
            participant_root.mkdir()
            result = pipeline.localize_task_paths(config, participant_root, "windows", self.roots)
        self.assertEqual(
            result["tasks"]["pick_place"]["video"],
            str(participant_root / "Pick and Place" / "vision_video_15-32-35_PnP_CONTROL_SLOW.avi"),
        )

    def test_p44_segment_paths_resolve_on_linux(self):
        config = pipeline.load(ROBOT_ROOT / "configs" / "participants" / "P44.yaml")
        result = pipeline.localize_task_paths(config, None, "linux", self.roots)
        segments = result["tasks"]["stack"]["eeg"]["segments"]
        self.assertEqual(len(segments), 3)
        self.assertTrue(all(item["file"].startswith(self.roots["linux"] + "/P44_2026-07-24/Stack/EEG/") for item in segments))

    def test_robot_uses_global_125_ms_sync_residual_limit(self):
        config = continuous.robot_eeg_config()
        self.assertEqual(config["health"]["sync"]["linear_max_residual_warning_s"], 0.125)


if __name__ == "__main__":
    unittest.main()
