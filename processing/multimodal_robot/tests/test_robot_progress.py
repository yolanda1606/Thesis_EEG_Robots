"""Focused checks for opt-in Robot terminal progress reporting."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "processing" / "multimodal_robot"))
from prep import continuous
import run_robot_pipeline as pipeline


class TestRobotProgress(unittest.TestCase):
    def test_progress_messages_are_opt_in_and_throttled(self):
        lines: list[str] = []
        reporter = continuous.ProgressReporter(show_progress=True, video_progress=True, eeg_progress=True, timing=True, writer=lines.append)
        reporter.participant_start("P01", 1)
        reporter.task_start(1, 1, "pick_place")
        reporter.eeg("Loading...")
        reporter.alignment({"matched_anchor_count": 11, "method": "status_constant_offset", "rmse_s": 0.011})
        reporter.video_start(100)
        reporter.video_frame(1, 100, 5)
        reporter.video_frame(5, 100, 5)
        reporter.features(32)
        reporter.task_done("pick_place")
        reporter.participant_done("P01", Path("derived/P01/Robot_Experiment/runs/example"))
        self.assertIn("[P01] Starting Robot processing: 1 tasks", lines)
        self.assertIn("[EEG] Loading...", lines)
        self.assertIn("[ALIGN] 11 anchors, constant offset, RMSE=0.011 s", lines)
        self.assertIn("[VIDEO] 5/100 frames (5.0%)", lines)
        self.assertFalse(any("1/100 frames" in line for line in lines))
        self.assertIn("[FEATURES] 32 windows", lines)
        self.assertTrue(any(line.startswith("[DONE] pick_place") for line in lines))
        self.assertIn("[P01] Failed tasks: none", lines)

    def test_cli_accepts_progress_flags(self):
        with patch.object(sys, "argv", [
            "run_robot_pipeline.py", "--participant-config", "P01.yaml", "--participant", "P01", "--run-name", "example",
            "--continuous-features", "--show-progress", "--video-progress", "--eeg-progress", "--timing",
        ]):
            parsed = pipeline.args()
        self.assertTrue(parsed.show_progress)
        self.assertTrue(parsed.video_progress)
        self.assertTrue(parsed.eeg_progress)
        self.assertTrue(parsed.timing)


if __name__ == "__main__":
    unittest.main()
