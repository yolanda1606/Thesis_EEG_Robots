"""Stateful causal SOS filtering for the experimental Robot branch only."""
from __future__ import annotations

from time import perf_counter_ns
from typing import Any

import numpy as np
from scipy import signal


SAMPLE_RATE_HZ = 250.0
LOW_CUTOFF_HZ = 1.0
HIGH_CUTOFF_HZ = 40.0
BUTTERWORTH_ORDER = 4
WARMUP_SECONDS = 2.0
SENSITIVITY_WARMUP_SECONDS = (1.58, 3.0)


def causal_sos() -> np.ndarray:
    """Create the canonical forward Butterworth SOS prototype.

    MNE constructs this same order-4, 1--40 Hz, 250-Hz Butterworth band-pass
    prototype before applying it twice for its zero-phase reference branch.
    """
    return signal.iirfilter(
        N=BUTTERWORTH_ORDER,
        Wn=[LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ],
        btype="bandpass",
        ftype="butter",
        fs=SAMPLE_RATE_HZ,
        output="sos",
    )


def filter_design_metadata(sos: np.ndarray) -> dict[str, Any]:
    """Return auditable filter-design metadata without touching raw data."""
    return {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "cutoffs_hz": [LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ],
        "nominal_butterworth_order": BUTTERWORTH_ORDER,
        "forward_bandpass_transfer_order": BUTTERWORTH_ORDER * 2,
        "sos_section_count": int(sos.shape[0]),
        "implementation": "scipy.signal.sosfilt",
        "state_policy": "scipy.signal.sosfilt_zi scaled by first post-CAR sample; zf carried across chunks",
        "state_reset_policy": "reset only at independent BDF/recording boundaries",
        "warmup_seconds_preregistered": WARMUP_SECONDS,
        "warmup_sensitivity_seconds": list(SENSITIVITY_WARMUP_SECONDS),
        "canonical_reference": "MNE phase='zero' SOS forward-backward filtering; effective band-pass order 16",
    }


def initial_state(sos: np.ndarray, first_samples: np.ndarray) -> np.ndarray:
    """Return per-channel steady-state SOS initial conditions.

    ``first_samples`` is one post-CAR value per channel. The shape matches
    scipy's ``sosfilt(..., axis=-1, zi=...)`` for channel x sample arrays.
    """
    first_samples = np.asarray(first_samples, dtype=float)
    if first_samples.ndim != 1:
        raise ValueError("first_samples must be one-dimensional")
    return signal.sosfilt_zi(sos)[:, None, :] * first_samples[None, :, None]


def filter_stream(data: np.ndarray, sos: np.ndarray, chunk_samples: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Filter channels x samples continuously, preserving state across chunks."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("data must have shape (channels, non-empty samples)")
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive")
    state = initial_state(sos, data[:, 0])
    filtered = np.empty_like(data)
    timings: list[dict[str, float]] = []
    for start in range(0, data.shape[1], chunk_samples):
        stop = min(start + chunk_samples, data.shape[1])
        started = perf_counter_ns()
        filtered[:, start:stop], state = signal.sosfilt(sos, data[:, start:stop], axis=-1, zi=state)
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
        timings.append({"chunk_start_sample": start, "chunk_end_sample": stop,
                        "sample_count": stop - start, "chunk_duration_s": (stop - start) / SAMPLE_RATE_HZ,
                        "causal_filter_ms": elapsed_ms})
    return filtered, timings


def causal_warmup_excluded(window_start_s: float, warmup_s: float = WARMUP_SECONDS) -> bool:
    """Keep canonical IDs/grid while flagging windows starting during warm-up."""
    return bool(window_start_s < warmup_s)
