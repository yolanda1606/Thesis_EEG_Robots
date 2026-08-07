from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from scipy.stats import pearsonr

from .eeg_sources import load_eeg_session
from .validation import available_image_codes


def _prepare_raw(paths, config: dict[str, Any]):
    raw, all_events, source_metadata = load_eeg_session(paths, config, preload=True)
    mapping = config["channels"]["eeg_mapping"]
    expected = set(mapping).union(config["channels"]["motion_channels"]).union({config["channels"]["stim_channel"]})
    missing = expected.difference(raw.ch_names)
    if missing:
        raise ValueError(f"EEG file is missing expected channels: {sorted(missing)}")
    channel_types = {name: "eeg" for name in mapping}
    channel_types.update({name: "misc" for name in config["channels"]["motion_channels"]})
    # Recorder bookkeeping fields are not electrodes and must not be assigned
    # scalp locations by the montage.
    channel_types.update({name: "misc" for name in ("CNT", "VALID", "DT") if name in raw.ch_names})
    channel_types[config["channels"]["stim_channel"]] = "stim"
    raw.set_channel_types(channel_types, verbose=False)
    raw.rename_channels(mapping)
    raw.set_montage(mne.channels.make_standard_montage(config["channels"]["montage"]),
                    match_case=False, on_missing="raise")
    return raw, all_events, source_metadata


def preprocess_eeg(paths, config: dict[str, Any], logger) -> tuple[mne.Epochs, dict[str, Any], pd.DataFrame, Any | None, np.ndarray]:
    """Apply the approved CAR, IIR, epoch, ICA, AutoReject, and interpolation stages."""
    raw, all_events, source_metadata = _prepare_raw(paths, config)
    valid_codes = available_image_codes(config)
    events = all_events[np.isin(all_events[:, 2], list(valid_codes))]
    if set(map(int, events[:, 2])) != valid_codes or len(events) != len(valid_codes):
        raise ValueError("Available image events are not complete and unique; preprocessing stopped")

    eeg_names = list(config["channels"]["eeg_mapping"].values())
    logger.info("Applying common average reference across %d EEG channels", len(eeg_names))
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    filter_config = config["eeg"]["filter"]
    logger.info("Applying fourth-order Butterworth IIR band-pass: %.1f-%.1f Hz", filter_config["l_freq_hz"], filter_config["h_freq_hz"])
    raw.filter(l_freq=filter_config["l_freq_hz"], h_freq=filter_config["h_freq_hz"], method="iir",
               iir_params={"order": filter_config["iir_order"], "ftype": "butter", "output": "sos"}, verbose=False)
    epoch_config = config["eeg"]["epoch"]
    event_id = {str(code): code for code in sorted(valid_codes)}
    epochs_all = mne.Epochs(raw, events, event_id=event_id, tmin=epoch_config["tmin_s"], tmax=epoch_config["tmax_s"],
                            baseline=tuple(epoch_config["baseline_s"]), preload=True, on_missing="raise", verbose=False)
    boundary_omitted = sorted(valid_codes.difference(map(int, epochs_all.events[:, 2])))
    eeg_epochs = epochs_all.copy().pick(eeg_names)
    motion_names = [name for name in config["channels"]["motion_channels"] if name in epochs_all.ch_names]
    motion_epochs = epochs_all.copy().pick(motion_names)
    ica_rows: list[dict[str, Any]] = []
    selected_components: list[int] = []
    if config["eeg"]["ica"]["enabled"]:
        ica_cfg = config["eeg"]["ica"]
        logger.info("Fitting ICA with %d components and random seed %d", len(eeg_names), ica_cfg["random_seed"])
        ica = mne.preprocessing.ICA(n_components=len(eeg_names), method=ica_cfg["method"], random_state=ica_cfg["random_seed"], verbose=False)
        ica.fit(eeg_epochs, verbose=False)
        source = ica.get_sources(eeg_epochs).get_data()
        motion = motion_epochs.get_data()
        correlations = np.full((source.shape[1], motion.shape[1]), np.nan)
        for component in range(source.shape[1]):
            for motion_index in range(motion.shape[1]):
                values = []
                for epoch_index in range(source.shape[0]):
                    r, _ = pearsonr(source[epoch_index, component], motion[epoch_index, motion_index])
                    if np.isfinite(r):
                        values.append(abs(float(r)))
                correlations[component, motion_index] = float(np.mean(values)) if values else np.nan
        threshold = float(ica_cfg["motion_correlation_threshold"])
        for component in range(source.shape[1]):
            maximum = float(np.nanmax(correlations[component]))
            rejected = maximum > threshold
            selected_components.extend([component] if rejected else [])
            ica_rows.append({"component": component, "maximum_mean_abs_motion_correlation": maximum,
                             "threshold": threshold, "rejected": rejected,
                             "motion_channel": motion_names[int(np.nanargmax(correlations[component]))]})
        ica.exclude = selected_components
        if selected_components:
            logger.warning("ICA components selected for removal: %s", selected_components)
            eeg_epochs = ica.apply(eeg_epochs, verbose=False)
        else:
            logger.info("No ICA components exceeded the approved correlation threshold")

    interpolated = list(config["eeg"]["bad_channels"]["approved_for_interpolation"])
    unknown = set(interpolated).difference(eeg_epochs.ch_names)
    if unknown:
        raise ValueError(f"Configured bad channels do not exist: {sorted(unknown)}")
    if interpolated:
        logger.warning("Interpolating researcher-approved bad channels: %s", interpolated)
        eeg_epochs.info["bads"] = interpolated
        eeg_epochs.interpolate_bads(reset_bads=False, verbose=False)
    else:
        logger.info("No channels are approved for interpolation in this run")

    rejection_summary: dict[str, Any] = {"autoreject_enabled": bool(config["eeg"]["autoreject"]["enabled"]), "dropped_epochs": []}
    autoreject_events = eeg_epochs.events.copy()
    reject_log = None
    if config["eeg"]["autoreject"]["enabled"]:
        seed = int(config["eeg"]["autoreject"]["random_seed"])
        allow_interpolation = bool(config["eeg"]["autoreject"].get("allow_epoch_channel_interpolation", False))
        logger.info("Running AutoReject with random seed %d", seed)
        logger.info("AutoReject epoch-wise channel interpolation: %s", "enabled" if allow_interpolation else "disabled")
        # AutoReject 0.4.3 supports [0], which leaves channel observations
        # marked bad in the log but makes affected epochs drop rather than be
        # repaired by its epoch-wise interpolation step.
        ar = AutoReject(n_interpolate=None if allow_interpolation else [0], n_jobs=1, random_state=seed, verbose=False)
        eeg_epochs, reject_log = ar.fit_transform(eeg_epochs, return_log=True)
        retained_triggers = autoreject_events[~np.asarray(reject_log.bad_epochs, dtype=bool), 2]
        if not np.array_equal(retained_triggers, eeg_epochs.events[:, 2]):
            raise RuntimeError("Trigger IDs no longer align with epochs retained by AutoReject")
        rejected_indices = np.flatnonzero(reject_log.bad_epochs).astype(int).tolist()
        rejection_summary["dropped_epochs"] = rejected_indices
        rejection_summary["autoreject_labels"] = reject_log.labels.tolist()
        rejection_summary["allow_epoch_channel_interpolation"] = allow_interpolation
        logger.info("AutoReject retained %d epochs; marked %d epochs for rejection", len(eeg_epochs), len(rejected_indices))

    qc = {"all_event_count": int(len(all_events)), "image_event_count": int(len(events)), "epochs_before_cleaning": int(len(epochs_all)),
          "epochs_after_cleaning": int(len(eeg_epochs)), "ica_rejected_components": selected_components,
          "interpolated_channels": interpolated, "rejection": rejection_summary, "sampling_hz": float(eeg_epochs.info["sfreq"]),
          "epoch_tmin_s": float(eeg_epochs.tmin), "epoch_tmax_s": float(eeg_epochs.tmax),
          "eeg_sources": source_metadata, "boundary_omitted_epoch_triggers": boundary_omitted}
    return eeg_epochs, qc, pd.DataFrame(ica_rows), reject_log, autoreject_events
