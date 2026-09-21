"""Focused tests for the isolated causal-streaming experiment."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import signal

from processing.multimodal_robot.analysis.causal_streaming.causal_filter import (
    SAMPLE_RATE_HZ, causal_sos, causal_warmup_excluded, filter_stream, initial_state,
)
from processing.multimodal_robot.analysis.causal_streaming.run_causal_robot_replay import (
    CANONICAL_VALIDATION_ATOL, CANONICAL_VALIDATION_RTOL,
)


class CausalStreamingTests(unittest.TestCase):
    def test_chunked_filter_matches_one_continuous_stateful_filter(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(8, 1000))
        sos = causal_sos()
        expected, _ = signal.sosfilt(sos, data, axis=-1, zi=initial_state(sos, data[:, 0]))
        observed, timings = filter_stream(data, sos, chunk_samples=25)
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(len(timings), 40)
        self.assertEqual(sum(row["sample_count"] for row in timings), data.shape[1])

    def test_chunking_does_not_reset_state(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(1, 100))
        sos = causal_sos()
        persistent, _ = filter_stream(data, sos, chunk_samples=25)
        reset = np.concatenate([
            signal.sosfilt(sos, data[:, start:start + 25], axis=-1,
                           zi=initial_state(sos, data[:, start]))[0]
            for start in range(0, 100, 25)
        ], axis=-1)
        self.assertFalse(np.allclose(persistent, reset))

    def test_preregistered_warmup_keeps_grid_and_flags_start_windows(self) -> None:
        self.assertTrue(causal_warmup_excluded(0.0))
        self.assertTrue(causal_warmup_excluded(1.0))
        self.assertFalse(causal_warmup_excluded(2.0))
        self.assertEqual(SAMPLE_RATE_HZ, 250.0)

    def test_canonical_validation_accepts_roundoff_but_rejects_meaningful_difference(self) -> None:
        reference = 2.1038082477099076e8
        self.assertTrue(np.isclose(reference + 2.9802322387695312e-8, reference,
                                   rtol=CANONICAL_VALIDATION_RTOL, atol=CANONICAL_VALIDATION_ATOL))
        self.assertFalse(np.isclose(reference + 1e-4, reference,
                                    rtol=CANONICAL_VALIDATION_RTOL, atol=CANONICAL_VALIDATION_ATOL))


if __name__ == "__main__":
    unittest.main()
