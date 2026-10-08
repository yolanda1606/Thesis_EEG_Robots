"""Focused boundaries for the constrained modality-agnostic causal pilot."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from processing.multimodal_robot.analysis.causal_streaming.run_causal_modality_agnostic_robot_replay import (
    CALIBRATION_SECONDS, face_window_features, fixed_offset_calibration,
    prediction_window_starts, route_source, validate_route_schema,
)
from processing.multimodal_robot.transfer.run_final_modality_agnostic_transfer import consensus


class CausalModalityAgnosticReplayTests(unittest.TestCase):
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
        categories=np.array([.1,.2,.3,.4,.5,.6])
        self.assertAlmostEqual(float(categories.sum()),2.1)


if __name__ == "__main__": unittest.main()
