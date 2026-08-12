from __future__ import annotations

from typing import Any
import warnings

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr
from scipy.signal import butter, sosfiltfilt

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
    # ICA is fitted to the filtered, un-baselined epochs.  Baseline correction
    # is applied after source correction so it cannot reduce the ICA rank or
    # trigger MNE's warning about fitting ICA to baseline-corrected data.
    epochs_all = mne.Epochs(raw, events, event_id=event_id, tmin=epoch_config["tmin_s"], tmax=epoch_config["tmax_s"],
                            baseline=None, preload=True, on_missing="raise", verbose=False)
    boundary_omitted = sorted(valid_codes.difference(map(int, epochs_all.events[:, 2])))
    eeg_epochs = epochs_all.copy().pick(eeg_names)
    # Only the three accelerometer axes are motion references for ICA.
    motion_names = [name for name in ("ACC X", "ACC Y", "ACC Z") if name in epochs_all.ch_names]
    if len(motion_names) != 3:
        raise ValueError(f"ICA requires ACC X/Y/Z motion references; found {motion_names}")
    motion_epochs = epochs_all.copy().pick(motion_names)
    ica_rows: list[dict[str, Any]] = []
    selected_components: list[dict[str, Any]] = []
    ica_qc: dict[str, Any] = {"enabled": bool(config["eeg"]["ica"]["enabled"]), "trials": []}
    if config["eeg"]["ica"]["enabled"]:
        ica_cfg = config["eeg"]["ica"]
        # Fit one decomposition to the valid image-epoch collection.  This
        # supplies enough samples for a stable ICA fit while retaining the
        # requested trial-specific ACC source assessment and correction below.
        concatenated = eeg_epochs.get_data().transpose(1, 0, 2).reshape(len(eeg_names), -1)
        rank = int(np.linalg.matrix_rank(concatenated))
        fit_duration_s = float(concatenated.shape[1] / eeg_epochs.info["sfreq"])
        if rank < 1:
            raise RuntimeError("ICA cannot be fitted: concatenated post-CAR EEG rank is zero")
        if rank < len(eeg_names) - 1:
            logger.warning("ICA QC review: concatenated post-CAR EEG rank is unexpectedly low: %d", rank)
        logger.info("Fitting rank-aware ICA on %.3f s of concatenated valid image-epoch data: rank=%d, requested components=%d, seed=%d",
                    fit_duration_s, rank, rank, ica_cfg["random_seed"])
        ica = mne.preprocessing.ICA(n_components=rank, method=ica_cfg["method"],
                                     random_state=ica_cfg["random_seed"], verbose=False)
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            ica.fit(eeg_epochs, verbose=False)
        convergence_warnings = [str(item.message) for item in fit_warnings if issubclass(item.category, ConvergenceWarning)]
        if convergence_warnings:
            logger.warning("ICA convergence warnings: %s", convergence_warnings)
            raise RuntimeError("Rank-aware concatenated ICA did not converge; preprocessing stopped")
        mixing_warnings = [str(item.message) for item in fit_warnings if "unstable mixing matrix" in str(item.message)]
        if mixing_warnings:
            logger.warning("ICA mixing-matrix warnings: %s", mixing_warnings)
            raise RuntimeError("Rank-aware concatenated ICA reported an unstable mixing matrix; preprocessing stopped")
        if int(ica.n_components_) > rank:
            raise RuntimeError("ICA fitted more components than the detected EEG rank")
        logger.info("Rank-aware ICA fit converged: detected rank=%d, fitted components=%d", rank, ica.n_components_)
        highpass_sos = butter(4, 3.0, btype="highpass", fs=float(eeg_epochs.info["sfreq"]), output="sos")
        corrected = eeg_epochs.get_data(copy=True)
        all_sources = ica.get_sources(eeg_epochs).get_data()
        for epoch_index in range(len(eeg_epochs)):
            trigger_id = int(eeg_epochs.events[epoch_index, 2])
            trial_qc = {"epoch_index": epoch_index, "trigger_id": trigger_id, "detected_eeg_rank": rank,
                        "requested_ica_components": rank, "fitted_ica_components": None,
                        "fit_failed": False, "motion_related_components": [], "corrected_source_count": 0,
                        "corrected_axes": [], "rank_below_normal_post_car": rank < len(eeg_names) - 1}
            try:
                sources = all_sources[epoch_index].copy()
                trial_qc["fitted_ica_components"] = int(ica.n_components_)
                motion = motion_epochs.get_data()[epoch_index]
                correlations = np.full((ica.n_components_, len(motion_names)), np.nan)
                axis_mean = np.full(len(motion_names), np.nan)
                axis_std = np.full(len(motion_names), np.nan)
                for axis_index, axis_name in enumerate(motion_names):
                    for component in range(ica.n_components_):
                        r, _ = pearsonr(sources[component], motion[axis_index])
                        correlations[component, axis_index] = float(r) if np.isfinite(r) else np.nan
                    finite = correlations[:, axis_index][np.isfinite(correlations[:, axis_index])]
                    if len(finite):
                        axis_mean[axis_index] = float(np.mean(finite))
                        axis_std[axis_index] = float(np.std(finite, ddof=0))
                flagged = []
                for component in range(ica.n_components_):
                    axes = [motion_names[axis] for axis in range(len(motion_names))
                            if np.isfinite(correlations[component, axis]) and np.isfinite(axis_mean[axis])
                            and correlations[component, axis] > axis_mean[axis] + 2.0 * axis_std[axis]]
                    if axes:
                        sources[component] = sosfiltfilt(highpass_sos, sources[component])
                        flagged.append({"component": component, "axes": axes})
                        selected_components.append({"trigger_id": trigger_id, "component": component, "axes": axes})
                    for axis_index, axis_name in enumerate(motion_names):
                        ica_rows.append({"epoch_index": epoch_index, "trigger_id": trigger_id, "detected_eeg_rank": rank,
                                         "requested_ica_components": rank, "fitted_ica_components": int(ica.n_components_),
                                         "component": component, "acc_axis": axis_name,
                                         "correlation": correlations[component, axis_index],
                                         "axis_mean_correlation": axis_mean[axis_index], "axis_std_correlation": axis_std[axis_index],
                                         "motion_related": axis_name in axes, "source_highpass_3hz": bool(axes)})
                # Reconstruct manually from the edited ICA sources. This is the
                # same back-projection used by MNE, except selected sources are
                # high-pass filtered rather than zeroed/excluded.
                prewhitened = ica.pca_components_[:ica.n_components_].T @ (ica.mixing_matrix_ @ sources)
                if ica.pca_mean_ is not None:
                    prewhitened += ica.pca_mean_[:, None]
                if ica.noise_cov is None:
                    reconstructed = prewhitened * ica.pre_whitener_
                else:
                    reconstructed = np.linalg.pinv(ica.pre_whitener_, rcond=1e-14) @ prewhitened
                corrected[epoch_index] = reconstructed
                trial_qc["motion_related_components"] = flagged
                trial_qc["corrected_source_count"] = len(flagged)
                trial_qc["corrected_axes"] = sorted({axis for item in flagged for axis in item["axes"]})
            except Exception as exc:
                trial_qc["fit_failed"] = True
                trial_qc["failure_reason"] = f"{type(exc).__name__}: {exc}"
                logger.warning("ICA failed for trigger %d: %s", trigger_id, trial_qc["failure_reason"])
            ica_qc["trials"].append(trial_qc)
        eeg_epochs._data = corrected
        ica_qc.update({"fit_data_duration_s": fit_duration_s, "detected_eeg_rank": rank,
                       "requested_ica_components": rank, "fitted_ica_components": int(ica.n_components_),
                       "convergence_warnings": convergence_warnings, "mixing_matrix_warnings": mixing_warnings})
        logger.info("Concatenated ICA trial correction complete: %d fit failures; %d corrected sources",
                    sum(item["fit_failed"] for item in ica_qc["trials"]),
                    sum(item["corrected_source_count"] for item in ica_qc["trials"]))
    # Keep the approved baseline interval, but apply it only after ICA
    # reconstruction and before AutoReject/feature extraction.
    eeg_epochs.apply_baseline(tuple(epoch_config["baseline_s"]), verbose=False)

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
          "ica_trialwise": ica_qc,
          "interpolated_channels": interpolated, "rejection": rejection_summary, "sampling_hz": float(eeg_epochs.info["sfreq"]),
          "epoch_tmin_s": float(eeg_epochs.tmin), "epoch_tmax_s": float(eeg_epochs.tmax),
          "eeg_sources": source_metadata, "boundary_omitted_epoch_triggers": boundary_omitted}
    return eeg_epochs, qc, pd.DataFrame(ica_rows), reject_log, autoreject_events
