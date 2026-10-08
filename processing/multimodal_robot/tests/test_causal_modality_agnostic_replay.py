"""Focused boundaries for the constrained modality-agnostic causal pilot."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from processing.multimodal_robot.analysis.causal_streaming.run_causal_modality_agnostic_robot_replay import (
    CALIBRATION_SECONDS, ambiguous_segment_mapping_reason, build_realtime_schedule, delivery_accounting, face_window_features, fixed_offset_calibration,
    compare_reference_predictions, prediction_window_starts, route_source, single_worker_schedule,
    source_video_timestamp, timed_window_consensus, timestamp_assignment_audit, validate_frame_telemetry,
    validate_reference_invariants, validate_route_schema,
)
from processing.multimodal_robot.analysis.causal_streaming.causal_filter import car_filter_stream, causal_sos, filter_stream
from processing.multimodal_robot.transfer.run_final_modality_agnostic_transfer import consensus


class CausalModalityAgnosticReplayTests(unittest.TestCase):
    def _reference_comparison_rows(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        keys = {"participant": "P99", "task": "stack", "segment_id": 1, "window_id": "stack_s1_1.000",
                "window_start_s": 1.0, "window_end_s": 3.0, "target": "valence"}
        predictions = pd.DataFrame([{**keys, "model_rank": 1, "modality": "eeg", "classifier": "logreg",
                                     "high_probability": 0.2}])
        consensus_rows = pd.DataFrame([{**keys, "top3_consensus_probability": 0.2,
                                        "top3_consensus_class": "LOW"}])
        return predictions, consensus_rows

    def _compare_with_reference(self, predictions: pd.DataFrame, consensus_rows: pd.DataFrame,
                                reference_predictions: pd.DataFrame, reference_consensus: pd.DataFrame,
                                validation_mode: str = "strict_historical",
                                affected_windows: pd.DataFrame | None = None) -> dict[str, object]:
        with TemporaryDirectory() as directory:
            reference_dir = Path(directory)
            reference_predictions.to_csv(reference_dir / "causal_window_predictions.csv", index=False)
            reference_consensus.to_csv(reference_dir / "causal_window_consensus.csv", index=False)
            return compare_reference_predictions(predictions, consensus_rows, reference_dir, validation_mode, affected_windows)

    def test_reference_equivalence_reports_fully_equivalent_predictions(self) -> None:
        predictions, consensus_rows = self._reference_comparison_rows()
        result = self._compare_with_reference(predictions, consensus_rows, predictions, consensus_rows)
        self.assertTrue(result["pass"])
        self.assertEqual(result["model_differing_probability_count"], 0)
        self.assertEqual(result["consensus_differing_probability_count"], 0)
        self.assertEqual(result["changed_consensus_class_count"], 0)

    def test_reference_equivalence_reports_model_probability_difference(self) -> None:
        predictions, consensus_rows = self._reference_comparison_rows()
        reference_predictions = predictions.copy(); reference_predictions.loc[0, "high_probability"] = 0.7
        result = self._compare_with_reference(predictions, consensus_rows, reference_predictions, consensus_rows)
        self.assertFalse(result["pass"])
        self.assertEqual(result["model_differing_probability_count"], 1)
        example = result["model_probability_mismatch_examples"][0]
        self.assertEqual(example["task"], "stack")
        self.assertAlmostEqual(example["high_probability_replay"], 0.2)
        self.assertAlmostEqual(example["high_probability_reference"], 0.7)

    def test_reference_equivalence_reports_consensus_probability_difference(self) -> None:
        predictions, consensus_rows = self._reference_comparison_rows()
        reference_consensus = consensus_rows.copy(); reference_consensus.loc[0, "top3_consensus_probability"] = 0.7
        result = self._compare_with_reference(predictions, consensus_rows, predictions, reference_consensus)
        self.assertFalse(result["pass"])
        self.assertEqual(result["consensus_differing_probability_count"], 1)
        self.assertEqual(result["changed_consensus_class_count"], 0)

    def test_reference_equivalence_reports_changed_consensus_class(self) -> None:
        predictions, consensus_rows = self._reference_comparison_rows()
        reference_consensus = consensus_rows.copy(); reference_consensus.loc[0, "top3_consensus_class"] = "HIGH"
        result = self._compare_with_reference(predictions, consensus_rows, predictions, reference_consensus)
        self.assertFalse(result["pass"])
        self.assertEqual(result["changed_consensus_class_count"], 1)
        self.assertEqual(result["consensus_class_mismatch_examples"][0]["top3_consensus_class_reference"], "HIGH")

    def test_reference_equivalence_reports_replay_and_reference_only_rows(self) -> None:
        predictions, consensus_rows = self._reference_comparison_rows()
        replay_predictions = predictions.copy(); replay_predictions.loc[0, "window_id"] = "stack_s1_2.000"
        replay_consensus = consensus_rows.copy(); replay_consensus.loc[0, "window_id"] = "stack_s1_2.000"
        result = self._compare_with_reference(replay_predictions, replay_consensus, predictions, consensus_rows)
        self.assertFalse(result["pass"])
        for level in ("model", "consensus"):
            self.assertEqual(result["coverage"][level]["replay_only_count"], 1)
            self.assertEqual(result["coverage"][level]["reference_only_count"], 1)

    def test_corrected_timestamp_mode_permits_only_affected_face_or_multimodal_rows(self) -> None:
        predictions, consensus_rows = self._reference_comparison_rows()
        predictions.loc[0, "modality"] = "face"
        reference_predictions = predictions.copy(); reference_predictions.loc[0, "high_probability"] = 0.7
        affected = predictions.loc[:, ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]]
        accepted = self._compare_with_reference(predictions, consensus_rows, reference_predictions, consensus_rows,
                                                validation_mode="corrected_timestamp", affected_windows=affected)
        self.assertTrue(accepted["pass"])
        self.assertEqual(accepted["timestamp_affected_model_difference_count"], 1)
        predictions.loc[0, "modality"] = "eeg"
        reference_predictions.loc[0, "modality"] = "eeg"
        rejected = self._compare_with_reference(predictions, consensus_rows, reference_predictions, consensus_rows,
                                                validation_mode="corrected_timestamp", affected_windows=affected)
        self.assertFalse(rejected["pass"])
        self.assertEqual(rejected["unexpected_model_difference_count"], 1)

    def test_timestamp_assignment_audit_uses_half_open_legacy_and_corrected_membership(self) -> None:
        base = {"participant": "P02", "task": "shape_sorter_interaction", "segment_id": 1}
        frames = pd.DataFrame([{**base, "frame_index": 2233, "timestamp_provenance": "source_fps_continuation",
                                "legacy_video_time_s": 74.43333333333334, "video_time_s": 75.57933333333334,
                                "legacy_eeg_time_s": 98.17933333333333, "eeg_time_s": 99.32533333333333}])
        windows = pd.DataFrame([{**base, "window_id": f"w{start}", "window_start_s": float(start), "window_end_s": float(start + 2)}
                                for start in (97, 98, 99)])
        audit = timestamp_assignment_audit(frames, windows)
        self.assertEqual(audit.window_id.tolist(), ["w97", "w99"])
        self.assertEqual(audit.legacy_member.tolist(), [True, False])
        self.assertEqual(audit.corrected_member.tolist(), [False, True])

    def test_reference_invariants_require_identical_routes_features_and_calibration(self) -> None:
        route = {"target": "valence", "model_rank": 1, "modality": "face", "classifier": "logreg",
                 "candidate_features": ["video_a"], "selected_features": ["video_a"],
                 "model_path": Path("/models/frozen.joblib"), "probability_source": "original_frozen_pipeline"}
        calibration = pd.DataFrame([{"participant": "P99", "task": "stack", "segment_id": 1,
                                     "frozen_offset_s": -3.0, "early_calibration_anchor_count": 2,
                                     "calibration_vision_end_s": 15.0, "calibration_eeg_end_s": 18.0}])
        with TemporaryDirectory() as directory:
            reference = Path(directory)
            (reference / "run_manifest.json").write_text(json.dumps({"routes": [{"target": "valence", "rank": 1,
                "modality": "face", "classifier": "logreg", "candidate_features": ["video_a"],
                "selected_features": ["video_a"], "model_path": "/models/frozen.joblib",
                "probability_source": "original_frozen_pipeline"}]}))
            calibration.to_csv(reference / "calibration_offsets.csv", index=False)
            self.assertTrue(validate_reference_invariants([route], calibration, reference)["pass"])
            changed = calibration.copy(); changed.loc[0, "frozen_offset_s"] = -2.0
            self.assertFalse(validate_reference_invariants([route], changed, reference)["pass"])

    def test_calibration_uses_only_first_15_seconds_and_never_fits_drift(self) -> None:
        eeg = [("1", 4.0), ("2", 9.0), ("3", 19.0)]
        vision = [("1", 1.0), ("2", 6.0), ("3", 30.0)]
        offset, used = fixed_offset_calibration(eeg, vision)
        self.assertEqual(offset, -3.0)
        self.assertEqual(used.event_identity.tolist(), ["1", "2"])
        changed_late, _ = fixed_offset_calibration(eeg, [("1", 1.0), ("2", 6.0), ("3", 300.0)])
        self.assertEqual(changed_late, offset)
        self.assertNotIn("drift", used.columns)

    def test_predictions_start_only_after_calibration_on_eeg_clock(self) -> None:
        offset = -5.5
        calibration_eeg_end = CALIBRATION_SECONDS - offset
        starts = prediction_window_starts(30.0, calibration_eeg_end)
        self.assertEqual(starts[0], 21.0)
        self.assertTrue(all(start >= calibration_eeg_end for start in starts))

    def test_face_window_never_uses_future_frames(self) -> None:
        frames = pd.DataFrame({"eeg_time_s":[1.0, 1.9, 2.0, 4.0], "face_detected":[True]*4, "irisdo_norm":[1.,3.,99.,100.]})
        values = face_window_features(frames, 0.0, 2.0)
        self.assertEqual(values["video_frame_count"], 2)
        self.assertEqual(values["video_irisdo_norm_mean"], 2.0)

    def test_trailing_video_frames_continue_the_existing_vision_log_clock(self) -> None:
        # P02 and P04 each have one trailing AVI frame after their final
        # Frame_Count -> Experiment_Time mapping.  The continuation must not
        # jump back to the independent frame_index / fps clock.
        for final_time, expected in ((75.546, 75.546 + 1.0 / 30.0),
                                     (70.379, 70.379 + 1.0 / 30.0)):
            mapped, mapped_source = source_video_timestamp(10, 30.0, {11: final_time}, None)
            continued, continued_source = source_video_timestamp(11, 30.0, {11: final_time}, mapped)
            second_continued, second_continued_source = source_video_timestamp(12, 30.0, {11: final_time}, continued)
            self.assertEqual(mapped_source, "vision_log")
            self.assertEqual(continued_source, "source_fps_continuation")
            self.assertEqual(second_continued_source, "source_fps_continuation")
            self.assertAlmostEqual(continued, expected)
            self.assertAlmostEqual(second_continued, expected + 1.0 / 30.0)
            self.assertGreater(continued, mapped)

    def test_timestamp_continuation_is_causal_and_preserves_eeg_clock_order(self) -> None:
        log_times = {1: 4.0, 2: 4.04, 4: 99.0}
        first, first_source = source_video_timestamp(0, 25.0, log_times, None)
        second, second_source = source_video_timestamp(1, 25.0, log_times, first)
        # Frame 3 has no log row.  The later frame-4 mapping must not be read.
        third, third_source = source_video_timestamp(2, 25.0, log_times, second)
        fourth, fourth_source = source_video_timestamp(3, 25.0, log_times, third)
        self.assertEqual((first_source, second_source, third_source, fourth_source),
                         ("vision_log", "vision_log", "source_fps_continuation", "vision_log"))
        self.assertAlmostEqual(third, 4.08)
        self.assertNotEqual(third, 99.0)
        eeg_times = [value - 1.5 for value in (first, second, third)]
        self.assertTrue(pd.Series(eeg_times).is_monotonic_increasing)
        self.assertEqual(fourth, 99.0)

    def test_timestamp_without_vision_mapping_uses_only_source_fps(self) -> None:
        timestamp, provenance = source_video_timestamp(3, 20.0, {}, None)
        self.assertEqual(provenance, "source_fps_no_vision_log")
        self.assertEqual(timestamp, 0.15)

    def test_eeg_face_and_multimodal_routing_preserves_schema_order(self) -> None:
        eeg={"eeg_a":1.}; face={"video_b":2.}
        self.assertEqual(route_source("eeg",eeg,face),eeg); self.assertEqual(route_source("face",eeg,face),face)
        multi=route_source("multimodal",eeg,face); self.assertEqual(list(multi),["eeg_a","video_b"])
        for modality,features in (("eeg",["eeg_a"]),("face",["video_b"]),("multimodal",["video_b","eeg_a"])):
            validate_route_schema({"target":"x","model_rank":1,"modality":modality,"candidate_features":features},route_source(modality,eeg,face))

    def test_complete_top3_requires_all_ranks(self) -> None:
        base={"participant":"P27","task":"stack","segment_id":1,"window_id":"w","window_start_s":15.,"window_end_s":17.,"target":"valence","classifier":"x","feature_family":"x","selected_feature_count":1,"frozen_image_balanced_accuracy":.5,"original_hard_prediction":1}
        predictions=pd.DataFrame([{**base,"model_rank":rank,"modality":modality,"high_probability":p} for rank,modality,p in ((1,"multimodal",.2),(2,"eeg",.6),(3,"multimodal",.8))])
        models=[{"target":"valence","model_rank":rank,"modality":modality} for rank,modality in ((1,"multimodal"),(2,"eeg"),(3,"multimodal"))]
        complete,counts=consensus(predictions,models); self.assertEqual(len(complete),1); self.assertAlmostEqual(complete.iloc[0].top3_consensus_probability,.6)
        incomplete,_=consensus(predictions.iloc[:2],models); self.assertTrue(incomplete.empty)

    def test_timing_total_is_sum_of_declared_categories(self) -> None:
        frame_work = {"video_frame_access_decode_ms": .1, "video_crop_ms": .2,
                      "face_image_prepare_ms": .3, "face_landmark_ms": .4,
                      "face_geometric_measure_ms": .5}
        self.assertAlmostEqual(float(sum(frame_work.values())), 1.5)

    def test_chunkwise_car_preserves_the_existing_causal_filter_values(self) -> None:
        rng=np.random.default_rng(7); data=rng.normal(size=(8, 125))
        expected,_=filter_stream(data-data.mean(axis=0,keepdims=True),causal_sos(),25)
        observed,timing=car_filter_stream(data,causal_sos(),25)
        np.testing.assert_allclose(observed,expected,rtol=0,atol=1e-14)
        self.assertEqual(len(timing),5); self.assertTrue(all("causal_car_ms" in row for row in timing))

    def test_single_worker_backlog_and_deadline_accounting(self) -> None:
        jobs=[
            {"job_type":"video_frame","sequence":0,"arrival_s":0.0,"service_ms":800.0,"deadline_s":np.nan},
            {"job_type":"prediction_update","sequence":1,"arrival_s":0.5,"service_ms":800.0,"deadline_s":1.5},
            {"job_type":"prediction_update","sequence":2,"arrival_s":1.5,"service_ms":100.0,"deadline_s":2.5},
        ]
        result=single_worker_schedule(jobs)
        self.assertAlmostEqual(result.iloc[1].start_s,.8)
        self.assertAlmostEqual(result.iloc[1].completion_s,1.6)
        self.assertFalse(result.iloc[1].deadline_met)
        self.assertAlmostEqual(result.iloc[2].start_s,1.6)
        self.assertTrue(result.iloc[2].deadline_met)

    def test_timed_consensus_matches_definition_b_and_missing_face_is_not_imputed(self) -> None:
        meta={"participant":"P27","task":"pick_place","segment_id":1,"window_id":"w","window_start_s":1.0,"window_end_s":3.0}
        rows=[meta|{"target":"valence","model_rank":rank,"high_probability":value,"original_hard_prediction":int(value>=.5)}
              for rank,value in ((1,.2),(2,.6),(3,.8))]
        output=timed_window_consensus(meta,rows,"valence")
        self.assertTrue(output["complete_top3"]); self.assertAlmostEqual(output["top3_consensus_probability"],.6)
        incomplete=timed_window_consensus(meta,rows[:2],"valence")
        self.assertFalse(incomplete["complete_top3"])
        no_face=face_window_features(pd.DataFrame({"eeg_time_s":[1.0],"face_detected":[False]}),0.,2.)
        self.assertTrue(np.isnan(no_face["video_irisdo_norm_mean"]))
        empty_face=face_window_features(pd.DataFrame(columns=["eeg_time_s","face_detected"]),0.,2.)
        self.assertEqual(empty_face["video_frame_count"],0.0)
        self.assertTrue(np.isnan(empty_face["face_detection_rate"]))

    def test_frame_telemetry_rejects_duplicate_or_out_of_order_frames_and_reuses_shared_multimodal_row(self) -> None:
        frames=pd.DataFrame({"participant":["P27","P27"],"task":["pick_place"]*2,"segment_id":[1,1],"frame_index":[0,1],"eeg_time_s":[0.,.1]})
        validate_frame_telemetry(frames)
        with self.assertRaisesRegex(RuntimeError,"more than once"):
            validate_frame_telemetry(pd.concat([frames,frames.iloc[[1]]],ignore_index=True))
        with self.assertRaisesRegex(RuntimeError,"chronological"):
            validate_frame_telemetry(frames.iloc[::-1].reset_index(drop=True))
        combined={"eeg_a":1.,"video_b":2.}
        self.assertIs(route_source("multimodal",{"eeg_a":1.},{"video_b":2.},combined),combined)

    def test_delivery_accounting_preserves_invalid_windows_in_the_denominator(self) -> None:
        keys={"participant":"P27","task":"pick_place","segment_id":1,"window_id":"w1","window_start_s":1.,"window_end_s":3.}
        second={**keys,"window_id":"w2","window_start_s":2.,"window_end_s":4.}
        timing=pd.DataFrame([{**keys,"valid_model_count":6,"prediction_compute_ms":100.},
                             {**second,"valid_model_count":3,"prediction_compute_ms":1200.}])
        consensus_rows=pd.DataFrame([{**keys,"target":"valence"},{**keys,"target":"arousal"},
                                     {**second,"target":"valence"}])
        schedule=build_realtime_schedule(pd.DataFrame(),pd.DataFrame(),timing)
        self.assertNotIn("window_start_s",schedule.columns)
        self.assertNotIn("window_end_s",schedule.columns)
        delivery, summary=delivery_accounting(timing,consensus_rows,schedule)
        self.assertEqual(delivery.complete_paired_consensus.tolist(),[True,False])
        self.assertEqual(delivery.complete_paired_delivered_on_time.tolist(),[True,False])
        self.assertEqual(int(summary.iloc[0].scheduled_updates),2)
        self.assertEqual(int(summary.iloc[0].complete_paired_delivered_on_time),1)

    def test_multi_segment_task_is_rejected_without_media_mapping(self) -> None:
        task={"eeg":{"segments":[{"file":"one.bdf"},{"file":"two.bdf"}]},"video":"task.avi","vision_log":"task.csv"}
        self.assertEqual(ambiguous_segment_mapping_reason(task),"ambiguous_multi_segment_video_log_mapping")


if __name__ == "__main__": unittest.main()
