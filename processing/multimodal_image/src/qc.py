"""Non-interactive participant-level EEG quality-control outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def autoreject_rows(reject_log, events: np.ndarray) -> list[dict[str, Any]]:
    """Make a compact, event-aligned record of an AutoReject decision log."""
    if len(events) != len(reject_log.bad_epochs):
        raise ValueError("AutoReject log and pre-rejection event list have different lengths")
    labels = np.asarray(reject_log.labels)
    channel_names = list(reject_log.ch_names)
    rows = []
    for index, (event, is_rejected, row_labels) in enumerate(zip(events, reject_log.bad_epochs, labels)):
        # AutoReject 0.4.3: 0 = good, 1 = bad, 2 = bad and interpolated.
        interpolated = [name for name, state in zip(channel_names, row_labels) if state == 2]
        bad = [name for name, state in zip(channel_names, row_labels) if state == 1]
        rows.append({
            "epoch_index": index,
            "trigger_id": int(event[2]),
            "epoch_rejected": bool(is_rejected),
            "interpolation_count": len(interpolated),
            "interpolated_channel_names": ";".join(interpolated),
            "bad_channel_names": ";".join(bad),
        })
    return rows


def validate_autoreject_event_alignment(events: np.ndarray, cleaned_events: np.ndarray, reject_log) -> None:
    """Ensure AutoReject epoch removal did not detach trigger IDs from epochs."""
    retained = np.asarray(events)[~np.asarray(reject_log.bad_epochs, dtype=bool), 2]
    if not np.array_equal(retained, np.asarray(cleaned_events)[:, 2]):
        raise RuntimeError("Trigger IDs no longer align with epochs retained by AutoReject")


def save_autoreject_plot(reject_log, destination: Path, participant: str, run_name: str) -> None:
    """Save AutoReject's horizontal epoch-by-channel decision display."""
    figure, axis = plt.subplots(figsize=(14, 4))
    # AutoReject 0.4.3 natively retains its good/interpolated/bad color map and
    # legend when rendered horizontally: epochs on x, channels on y.
    reject_log.plot(orientation="horizontal", aspect="auto", show=False, ax=axis)
    figure.suptitle(f"AutoReject epoch/channel QC — {participant} — {run_name}", y=1.03)
    axis.set_xlabel("Epochs")
    axis.set_ylabel("Channels")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _category_epochs(epochs: mne.Epochs, ranges: dict[str, Any]) -> dict[str, mne.Epochs]:
    result = {}
    trigger_ids = epochs.events[:, 2]
    for category, (low, high) in ranges.items():
        mask = (trigger_ids >= int(low)) & (trigger_ids <= int(high))
        result[category] = epochs[np.flatnonzero(mask)]
    return result


def save_fz_time_frequency(epochs: mne.Epochs, config: dict[str, Any], destination: Path,
                           participant: str, run_name: str, logger) -> None:
    """Save the requested four-category Fz Morlet event-related TFR figure."""
    if "Fz" not in epochs.ch_names:
        logger.warning("Fz is absent; skipping Fz event-related time-frequency QC")
        return
    baseline = tuple(float(value) for value in config["eeg"]["epoch"]["baseline_s"])
    if epochs.tmin > baseline[0] or epochs.tmax < 2.0:
        raise ValueError("Fz TFR QC needs complete cleaned -0.5..2.0 s epochs")
    freqs = np.logspace(*np.log10([4.0, 30.0]), num=20)
    n_cycles = freqs / 2.0
    category_epochs = _category_epochs(epochs, config["events"]["image_ranges"])
    powers = {}
    for category, selected in category_epochs.items():
        if not len(selected):
            continue
        power = selected.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles,
                                     return_itc=False, average=True, output="power", n_jobs=1,
                                     verbose="ERROR")
        # This is the same MNE log-ratio baseline transformation previously
        # supplied directly to plot(), applied here only to establish one
        # shared color range before drawing the panels.
        power.apply_baseline(baseline=baseline, mode="logratio", verbose="ERROR")
        powers[category] = power
    values = [power.copy().pick("Fz").data for power in powers.values()]
    finite = np.concatenate([value[np.isfinite(value)] for value in values]) if values else np.array([])
    vlim = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (-1.0, 1.0)
    if vlim[0] == vlim[1]:
        vlim = (vlim[0] - 1.0, vlim[1] + 1.0)
    figure, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 7))
    image = None
    for axis, category in zip(np.ravel(axes), ("HAHV", "LALV", "HALV", "LAHV")):
        count = len(category_epochs[category])
        if category not in powers:
            axis.text(0.5, 0.5, f"{category} (n=0, no retained epochs)", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(f"{category} (n=0, no retained epochs)")
        else:
            powers[category].plot(["Fz"], baseline=None, mode="mean", vlim=vlim, axes=axis,
                                  show=False, colorbar=False, title=None)
            image = axis.collections[-1]
            axis.set_title(f"{category} (n={count})")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    if image is not None:
        # Reserve a dedicated right-side axis so the shared magnitude scale
        # cannot be laid over either right-hand category panel.
        colorbar_axis = figure.add_axes((0.90, 0.14, 0.022, 0.64))
        figure.colorbar(image, cax=colorbar_axis, label="Power (log ratio)")
    figure.suptitle(f"Event-related time-frequency QC — Fz — {participant} — {run_name}\n"
                     "Morlet 4–30 Hz (20 log-spaced frequencies), n_cycles = frequency / 2", fontsize=11)
    figure.subplots_adjust(top=0.84, right=0.86, wspace=0.25, hspace=0.35)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_event_related_band_power(
    epochs: mne.Epochs,
    config: dict[str, Any],
    destination_png: Path,
    destination_csv: Path,
    participant: str,
    run_name: str,
) -> list[str]:
    """Write descriptive event-related, baseline-normalized band-power QC.

    The passed epochs are never cropped or otherwise mutated.  Power is
    calculated per trial/channel/frequency, transformed to dB relative to the
    configured prestimulus baseline, then averaged for the displayed summary.
    """
    epoch_cfg = config["eeg"]["epoch"]
    baseline = tuple(float(value) for value in epoch_cfg["baseline_s"])
    if epochs.tmin > baseline[0] or epochs.tmax < 2.0 or baseline[1] > 0:
        raise ValueError("QC needs complete cleaned -0.5..2.0 s epochs with a prestimulus baseline")
    bands = config["features"]["bands_hz"]
    category_epochs = _category_epochs(epochs, config["events"]["image_ranges"])
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {band: {} for band in bands}
    counts = {category: len(value) for category, value in category_epochs.items()}
    times = epochs.times.copy()
    for category, selected in category_epochs.items():
        if not len(selected):
            continue
        for band, (low, high) in bands.items():
            # Integer-spaced frequencies retain a compact QC calculation while
            # resolving every configured inclusive band endpoint.
            frequencies = np.arange(float(low), float(high) + 0.001, 1.0)
            n_cycles = np.maximum(frequencies / 2.0, 1.0)
            power = selected.compute_tfr(
                method="morlet", freqs=frequencies, n_cycles=n_cycles,
                return_itc=False, average=False, output="power", n_jobs=1,
                verbose="ERROR",
            ).data
            baseline_mask = (times >= baseline[0] - 1e-9) & (times <= baseline[1] + 1e-9)
            baseline_power = np.nanmean(power[..., baseline_mask], axis=-1, keepdims=True)
            finite_baseline = np.isfinite(baseline_power) & (baseline_power > np.finfo(float).tiny)
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_db = np.where(finite_baseline, 10.0 * np.log10(power / baseline_power), np.nan)
            curve = np.nanmean(normalized_db, axis=(0, 1, 2))
            if not np.isfinite(curve).any():
                continue
            curves[band][category] = curve
            rows.extend({
                "participant": participant, "run_name": run_name, "category": category,
                "band": band, "time_s": float(time), "mean_power_db": float(value),
                "n_epochs": int(len(selected)), "n_channels": int(len(selected.ch_names)),
            } for time, value in zip(times, curve))
    if not rows:
        raise RuntimeError("No finite event-related spectral-power QC values were produced")
    pd.DataFrame(rows).to_csv(destination_csv, index=False)
    figure, axes = plt.subplots(len(bands), 1, sharex=True, figsize=(10, 2.5 * len(bands)))
    axes = np.atleast_1d(axes)
    for axis, band in zip(axes, bands):
        for category, curve in curves[band].items():
            axis.plot(times, curve, label=f"{category} (n={counts[category]})")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1, label="Stimulus onset")
        axis.set_ylabel(f"{band}\npower (dB)")
        axis.legend(loc="best", fontsize="small")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time from image onset (s)")
    axes[-1].set_xlim(float(epochs.tmin), float(epochs.tmax))
    figure.suptitle(
        f"Descriptive event-related spectral-power QC — {participant} — {run_name}\n"
        "Baseline-normalized band power; retained trials: " + ", ".join(f"{key}={value}" for key, value in counts.items()),
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination_png, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [category for category, count in counts.items() if not count]
