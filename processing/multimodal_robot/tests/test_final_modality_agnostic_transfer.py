"""Focused non-inference tests for modality-agnostic transfer aggregation."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "inference" / "run_final_modality_agnostic_transfer.py"
SPEC = importlib.util.spec_from_file_location("modality_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


class ModalityAgnosticConsensusTests(unittest.TestCase):
    def test_consensus_requires_all_three_modalities_and_uses_definition_b_window_values(self) -> None:
        rows = []
        for window_id, start, probabilities, ranks in [
            ("task_s1_0.000", 0.0, {1: 0.2, 2: 0.8, 3: 0.6}, (1, 2, 3)),
            ("task_s1_1.000", 1.0, {1: 0.9, 2: 0.1}, (1, 2)),
        ]:
            for rank in ranks:
                rows.append({
                    "participant": "P99", "task": "pick_place", "segment_id": 1,
                    "window_id": window_id, "window_start_s": start, "window_end_s": start + 2.0,
                    "target": "arousal", "model_rank": rank,
                    "high_probability": probabilities[rank], "original_hard_prediction": int(probabilities[rank] >= .5),
                })
        models = [
            {"target": "arousal", "model_rank": 1, "modality": "eeg"},
            {"target": "arousal", "model_rank": 2, "modality": "face"},
            {"target": "arousal", "model_rank": 3, "modality": "multimodal"},
        ]
        complete, counts = transfer.consensus(pd.DataFrame(rows), models)
        self.assertEqual(int(counts.loc[0, "n_candidate_windows"]), 2)
        self.assertEqual(int(counts.loc[0, "n_complete_top3_windows"]), 1)
        self.assertEqual(complete.loc[0, "rank1_modality"], "eeg")
        self.assertEqual(complete.loc[0, "rank2_modality"], "face")
        self.assertEqual(complete.loc[0, "rank3_modality"], "multimodal")
        self.assertAlmostEqual(float(complete.loc[0, "top3_consensus_probability"]), 0.6)


if __name__ == "__main__":
    unittest.main()
