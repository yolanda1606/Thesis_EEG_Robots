"""Focused orchestration tests; scientific replay behavior is tested elsewhere."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

from processing.multimodal_robot.analysis.causal_streaming import run_causal_robot_cohort as cohort


class CausalRobotCohortTests(unittest.TestCase):
    def test_participant_selection_and_output_path(self) -> None:
        self.assertEqual(cohort.selected_participants(["P01", "P46"], False), ("P01", "P46"))
        self.assertEqual(len(cohort.selected_participants(None, True)), 46)
        with self.assertRaisesRegex(ValueError, "Unknown participant"):
            cohort.selected_participants(["P47"], False)
        paths = cohort.participant_paths("P01", Path("/tmp/cohort"), Path("/raw"))
        self.assertEqual(paths["output_dir"], Path("/tmp/cohort/P01"))
        self.assertIn("p01_robot_final_features", str(paths["canonical_eeg_csv"]))

    def test_completeness_requires_all_final_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "P01"; output.mkdir()
            self.assertFalse(cohort.output_complete(output))
            for name in cohort.REQUIRED_OUTPUT_FILES:
                (output / name).touch()
            self.assertTrue(cohort.output_complete(output))

    def test_failure_continues_to_later_participant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"
            calls: list[str] = []
            def validate(participant, paths):
                if participant == "P01":
                    raise FileNotFoundError("expected missing input")
                return [Path("/raw/P02.bdf")]
            def execute(command, **kwargs):
                calls.append(command[command.index("--participant") + 1])
            with patch.object(cohort, "validate_inputs", side_effect=validate):
                rows = cohort.run_participants(("P01", "P02"), output, Path("/raw"), dry_run=False, force=False, execute=execute, writer=lambda _: None)
            self.assertEqual([row["status"] for row in rows], ["failed", "success"])
            self.assertEqual(calls, ["P02"])
            self.assertTrue((output / "cohort_run_status.csv").is_file())

    def test_dry_run_never_starts_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"
            executed = []
            with patch.object(cohort, "validate_inputs", return_value=[Path("/raw/P01.bdf")]):
                rows = cohort.run_participants(("P01",), output, Path("/raw"), dry_run=True, force=False,
                                               execute=lambda *args, **kwargs: executed.append(args), writer=lambda _: None)
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(executed, [])

    def test_subprocess_failure_writes_log_and_preserves_existing_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"; output.mkdir()
            (output / "cohort_run_status.csv").write_text("participant,status\nP02,success\n", encoding="utf-8")
            error = subprocess.CalledProcessError(1, ["runner"], output="runner stdout", stderr="Traceback\nValueError: exact failure")
            with patch.object(cohort, "validate_inputs", return_value=[Path("/raw/P01.bdf")]), \
                 patch.object(cohort, "output_complete", return_value=False):
                rows = cohort.run_participants(("P01",), output, Path("/raw"), dry_run=False, force=False,
                                               execute=lambda *args, **kwargs: (_ for _ in ()).throw(error), writer=lambda _: None)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("ValueError: exact failure", rows[0]["error_message"])
            log = output / "logs/P01.log"
            self.assertIn("runner stdout", log.read_text(encoding="utf-8"))
            self.assertIn("ValueError: exact failure", log.read_text(encoding="utf-8"))
            self.assertIn("P02,success", (output / "cohort_run_status.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
