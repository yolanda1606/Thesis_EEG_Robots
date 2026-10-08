"""Focused tests for read-only causal cohort aggregation formulas."""
from __future__ import annotations

import unittest

import pandas as pd

from processing.multimodal_robot.analysis.causal_streaming.summarize_causal_robot_cohort import aggregation_participants, task_verdicts


class CausalCohortSummaryTests(unittest.TestCase):
    def test_primary_aggregation_keeps_successful_partial_participants(self) -> None:
        completion = pd.DataFrame({"participant": ["P01", "P02", "P03"], "status": ["success", "success", "failed"], "tasks_completed": [6, 4, 6]})
        usable, complete_six = aggregation_participants(completion)
        self.assertEqual(usable, ["P01", "P02"])
        self.assertEqual(complete_six, ["P01"])

    def test_definition_b_uses_window_median_then_task_median(self) -> None:
        frame = pd.DataFrame({
            "participant": ["P01"] * 3, "task": ["stack"] * 3, "target": ["valence"] * 3,
            "window_id": ["w1", "w2", "w3"],
            "top3_consensus_probability_canonical": [.2, .6, .8],
            "top3_consensus_probability_causal": [.1, .4, .7],
        })
        verdicts, summary, _ = task_verdicts(frame)
        row = verdicts.iloc[0]
        self.assertEqual(row.canonical_P_task, .6)
        self.assertEqual(row.causal_P_task, .4)
        self.assertEqual(row.canonical_verdict, "HIGH")
        self.assertEqual(row.causal_verdict, "LOW")
        self.assertEqual(summary.loc[summary.summary_scope.eq("overall"), "verdict_changes"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
