"""Focused tests for modality-agnostic causal cohort orchestration only."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from processing.multimodal_robot.analysis.causal_streaming import run_causal_modality_agnostic_robot_cohort as cohort
from processing.multimodal_robot.analysis.causal_streaming import run_causal_modality_agnostic_robot_replay as replay


def eligible_audit(participant: str) -> dict[str, object]:
    return {"participant": participant, "modalities": ["eeg", "face", "multimodal"], "model_count": 6,
            "reference_status": "available", "reference_causal_output": "/reference/P01",
            "cases": [{"participant": participant, "task": "pick_place", "segment_id": 1,
                       "eligible": True, "reason": ""}]}


def frozen_models() -> list[dict[str, object]]:
    return [{"target": target, "model_rank": rank, "modality": "eeg",
             "candidate_features": ["eeg_a"], "selected_features": ["eeg_a"]}
            for target in ("valence", "arousal") for rank in (1, 2, 3)]


def make_valid_output(output: Path, participant: str) -> None:
    output.mkdir(parents=True)
    for name, fields in cohort.REQUIRED_OUTPUT_SCHEMAS.items():
        path = output / name
        if name == "run_manifest.json":
            path.write_text(json.dumps({"participant": participant, "tasks_requested": ["pick_place"],
                                        "realtime_scheduler": {}, "frozen_prediction_equivalence": {}}), encoding="utf-8")
        else:
            pd.DataFrame(columns=sorted(fields)).to_csv(path, index=False)


class CausalModalityAgnosticCohortTests(unittest.TestCase):
    def test_participant_paths_and_selection(self) -> None:
        paths = cohort.participant_paths("P01", Path("/tmp/cohort"), Path("/raw"), Path("/tmp/reference"))
        self.assertEqual(paths["output_dir"], Path("/tmp/cohort/P01"))
        self.assertEqual(paths["reference_causal_output"], Path("/tmp/reference/P01"))
        self.assertIn("p01_robot_final_features", str(paths["alignment_metadata"]))
        self.assertEqual(cohort.selected_participants(["P01", "P46"], False), ("P01", "P46"))
        self.assertEqual(len(cohort.selected_participants(None, True)), 46)

    def test_output_root_must_be_new_v5_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(cohort, "V5_OUTPUT_PARENT", root / "v5"):
                self.assertEqual(cohort.validate_v5_output_root(root / "v5" / "cohort", resume=False), root / "v5" / "cohort")
                with self.assertRaisesRegex(ValueError, "v5 run directory"):
                    cohort.validate_v5_output_root(root / "v5", resume=False)
                old = root / "v5" / "existing"; old.mkdir(parents=True); (old / "old.csv").touch()
                with self.assertRaises(FileExistsError):
                    cohort.validate_v5_output_root(old, resume=False)

    def test_completeness_requires_every_final_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "P01"; output.mkdir()
            self.assertFalse(cohort.output_complete(output))
            output.rmdir(); make_valid_output(output, "P01")
            self.assertTrue(cohort.output_complete(output))
            (output / "deadline_summary.csv").write_text("participant\nP01\n", encoding="utf-8")
            valid, reason = cohort.validate_output(output, "P01")
            self.assertFalse(valid); self.assertIn("missing columns", reason)
            make_valid_output(Path(temporary) / "P02", "P02")
            valid, reason = cohort.validate_output(Path(temporary) / "P02", "P02", ("pick_place", "stack"))
            self.assertFalse(valid); self.assertIn("do not match", reason)

    def test_no_anchor_and_fragmented_segments_are_excluded_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "raw"; raw.mkdir()
            paths = {"participant_config": root / "P44.yaml", "canonical_root": root / "canonical",
                     "image_reference_csv": root / "image.csv", "alignment_metadata": root / "alignment.json",
                     "synchronization_anchors": root / "anchors.csv", "raw_root": raw, "output_dir": root / "out",
                     "reference_causal_output": root / "reference"}
            paths["canonical_root"].mkdir(); paths["participant_config"].touch(); paths["image_reference_csv"].touch()
            tasks = {}
            prediction_rows = []
            alignment_rows = []
            anchor_rows = []
            for task in cohort.TASKS:
                video = raw / f"{task}.avi"; log = raw / f"{task}.csv"; video.touch(); log.touch()
                if task == "stack":
                    eeg_files = [raw / f"stack_{index}.bdf" for index in (1, 2, 3)]
                    for path in eeg_files: path.touch()
                    tasks[task] = {"eeg": {"segments": [{"file": str(path)} for path in eeg_files]}, "video": str(video), "vision_log": str(log)}
                    alignment_rows.append({"task": task, "method": "segmented", "segments": [
                        {"segment_id": 1, "method": "invalid_residual", "accepted": False},
                        {"segment_id": 2, "method": "status_constant_offset", "accepted": True},
                        {"segment_id": 3, "method": "status_constant_offset", "accepted": True}]})
                    prediction_rows.extend([{"task": task, "segment_id": 2}, {"task": task, "segment_id": 3}])
                    anchor_rows.extend([{"task": task, "segment_id": 1, "vision_time_s": 3.0},
                                        {"task": task, "segment_id": 2, "vision_time_s": 17.0},
                                        {"task": task, "segment_id": 3, "vision_time_s": 92.0}])
                else:
                    eeg = raw / f"{task}.bdf"; eeg.touch()
                    tasks[task] = {"eeg": str(eeg), "video": str(video), "vision_log": str(log)}
                    method = "filename_fallback" if task == "shape_sorter_alone" else "status_constant_offset"
                    alignment_rows.append({"task": task, "method": method, "offset_s": -5.0})
                    prediction_rows.append({"task": task, "segment_id": 1})
                    if task != "shape_sorter_alone": anchor_rows.append({"task": task, "segment_id": None, "vision_time_s": 2.0})
            (paths["alignment_metadata"]).write_text(json.dumps({"tasks": alignment_rows}), encoding="utf-8")
            pd.DataFrame(anchor_rows).to_csv(paths["synchronization_anchors"], index=False)
            pd.DataFrame(prediction_rows).to_csv(paths["canonical_root"] / "P44_window_predictions.csv", index=False)
            pd.DataFrame().to_csv(paths["canonical_root"] / "P44_window_consensus.csv", index=False)
            pd.DataFrame().to_csv(paths["canonical_root"] / "P44_task_consensus_analysis.csv", index=False)
            config = {"participant": "P44", "tasks": tasks}
            models = frozen_models()
            with patch.object(cohort, "load", return_value=config), patch.object(cohort, "localize_task_paths", side_effect=lambda value, _: value), \
                 patch.object(replay.transfer, "load_models", return_value=models):
                result = cohort.preflight_participant("P44", paths)
            cases = pd.DataFrame(result["cases"])
            stack = cases.loc[cases.task.eq("stack")].set_index("segment_id")
            self.assertIn("canonical_unavailable", stack.loc[1, "reason"])
            self.assertIn("ambiguous_multi_segment_video_log_mapping", stack.loc[1, "reason"])
            self.assertIn("ambiguous_multi_segment_video_log_mapping", stack.loc[2, "reason"])
            self.assertIn("ambiguous_multi_segment_video_log_mapping", stack.loc[3, "reason"])
            alone = cases.loc[cases.task.eq("shape_sorter_alone")].iloc[0]
            self.assertIn("no_status_alignment", alone.reason)
            self.assertIn("no_early_anchor", alone.reason)

    def test_task_summary_does_not_double_count_fragment_reasons(self) -> None:
        cases = pd.DataFrame([
            {"participant": "P44", "task": "stack", "segment_id": 1, "eligible": False, "reason": "canonical_unavailable;no_status_alignment"},
            {"participant": "P44", "task": "stack", "segment_id": 2, "eligible": False, "reason": "no_early_anchor"},
            {"participant": "P44", "task": "stack", "segment_id": 3, "eligible": False, "reason": "no_early_anchor"},
            {"participant": "P44", "task": "pick_place", "segment_id": 1, "eligible": True, "reason": ""},
        ])
        eligible, reasons = cohort.task_eligibility_summary(cases)
        self.assertEqual(eligible, 1)
        self.assertEqual(reasons["no_early_anchor"], 1)
        self.assertEqual(reasons["canonical_unavailable"], 0)

    def test_resume_skips_only_schema_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"; participant_output = output / "P01"; make_valid_output(participant_output, "P01")
            executed = []
            with patch.object(cohort, "preflight_participant", return_value=eligible_audit("P01")):
                rows, _ = cohort.run_participants(("P01",), output, Path("/raw"), Path("/reference"), ("pick_place",), dry_run=False, resume=True,
                                                  execute=lambda *args, **kwargs: executed.append(args), writer=lambda _: None)
            self.assertEqual(rows[0]["status"], "skipped_existing")
            self.assertEqual(executed, [])

    def test_incomplete_existing_output_is_blocked_without_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"; participant_output = output / "P01"; participant_output.mkdir(parents=True)
            (participant_output / "partial.csv").touch(); executed = []
            with patch.object(cohort, "preflight_participant", return_value=eligible_audit("P01")):
                rows, _ = cohort.run_participants(("P01",), output, Path("/raw"), Path("/reference"), ("pick_place",), dry_run=False, resume=True,
                                                  execute=lambda *args, **kwargs: executed.append(args),
                                                  writer=lambda _: None)
            self.assertEqual(rows[0]["status"], "blocked_existing")
            self.assertEqual(executed, [])
            self.assertTrue(participant_output.is_dir())
            self.assertEqual(list(output.glob("P01.archived_*")), [])

    def test_failure_continues_and_dry_run_never_executes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"; calls = []
            def execute(command, **kwargs):
                participant = command[command.index("--participant") + 1]; calls.append(participant)
                if participant == "P01": raise subprocess.CalledProcessError(1, command, stderr="exact failure")
                return SimpleNamespace(stdout="ok", stderr="")
            with patch.object(cohort, "preflight_participant", side_effect=lambda participant, paths, tasks: eligible_audit(participant)):
                rows, _ = cohort.run_participants(("P01", "P02"), output, Path("/raw"), Path("/reference"), ("pick_place",), dry_run=False, resume=False,
                                                  execute=execute, writer=lambda _: None)
            self.assertEqual([row["status"] for row in rows], ["failed", "success"])
            self.assertEqual(calls, ["P01", "P02"])
            dry_calls = []
            with patch.object(cohort, "preflight_participant", return_value=eligible_audit("P03")):
                dry_rows, _ = cohort.run_participants(("P03",), output, Path("/raw"), Path("/reference"), ("pick_place",), dry_run=True, resume=False,
                                                      execute=lambda *args, **kwargs: dry_calls.append(args), writer=lambda _: None)
            self.assertEqual(dry_rows[0]["status"], "dry_run_ok")
            self.assertEqual(dry_calls, [])

    def test_missing_reference_is_reported_without_command_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = cohort.participant_paths("P01", root / "out", root / "raw", root / "missing_reference")
            self.assertEqual(cohort.reference_status(paths["reference_causal_output"]), "missing")
            command = cohort.runner_command("P01", paths, ("pick_place",), "missing")
            self.assertNotIn("--reference-causal-output", command)

    def test_frozen_model_completeness_is_required(self) -> None:
        cohort.validate_frozen_models(frozen_models(), "P01")
        with self.assertRaisesRegex(ValueError, "exactly six"):
            cohort.validate_frozen_models(frozen_models()[:-1], "P01")

    def test_frozen_calibration_rule_remains_15_seconds(self) -> None:
        self.assertEqual(replay.CALIBRATION_SECONDS, 15.0)
        offset, anchors = replay.fixed_offset_calibration([("1", 2.0), ("2", 30.0)], [("1", 3.0), ("2", 300.0)])
        self.assertEqual(offset, 1.0); self.assertEqual(len(anchors), 1)


if __name__ == "__main__":
    unittest.main()
