"""Strict loading and merging of shared, experiment, and participant YAML."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import warnings

import yaml


SCHEMA: dict[str, Any] = {
    "schema_version": None, "resources": {"face_landmarker_filename": None},
    "participant": None, "experiment": None,
    "inputs": {"participant_dir": None, "experiment_dir": None, "eeg_file": None, "eeg_files": None,
               "ratings_file": None, "video_file": None, "vision_log_file": None,
               "authoritative_ratings": None, "face_landmarker_model": None},
    "channels": {"eeg_mapping": None, "montage": None, "stim_channel": None,
                 "motion_channels": None, "recorder_channels": None},
    "events": {"image_ranges": None},
    "ratings": {"required_columns": None, "rating_columns": {"valence": None, "arousal": None}, "missing_tokens": None},
    "trials": {"expected_image_trials": None, "event_locked": None,
               "allow_incomplete_image_trials": None, "expected_available_triggers": None,
               "known_missing_triggers": None},
    "eeg": {"reference": None, "filter": {"l_freq_hz": None, "h_freq_hz": None, "iir_order": None},
            "epoch": {"tmin_s": None, "tmax_s": None, "baseline_s": None},
            "ica": {"enabled": None, "method": None, "random_seed": None, "motion_correlation_threshold": None},
            "autoreject": {"enabled": None, "random_seed": None, "allow_epoch_channel_interpolation": None},
            "bad_channels": {"approved_for_interpolation": None}},
    "features": {"eeg_feature_window_s": None, "bands_hz": None, "deferred": None},
    "video": {"analysis_window_s": None,
              "face_landmarker": {"min_detection_confidence": None, "min_presence_confidence": None, "min_tracking_confidence": None},
              "landmark_indices": None,
              "crop": {"left": None, "top": None, "right": None, "bottom": None, "scale": None, "approved": None, "approval_note": None}},
    "health": {"near_flat_variance_v2": None, "large_amplitude_v": None, "clipping_fraction_warning": None, "line_noise_ratio_warning": None,
               "video_frame_samples": None, "video_timestamp_gap_factor": None,
               "sync": {"linear_rmse_warning_s": None, "linear_max_residual_warning_s": None, "outlier_residual_s": None}},
    "output": {"experiment_output_dir": None, "log_filename": None, "video_progress_interval_frames": None},
    "synchronization": {"offset_override_s": None, "drift_override_s_per_s": None},
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def _validate_known_keys(value: dict[str, Any], schema: dict[str, Any], source: Path, prefix: str = "") -> None:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            raise ValueError(f"Unknown configuration key '{path}' in {source}")
        if isinstance(child, dict) and isinstance(schema[key], dict):
            _validate_known_keys(child, schema[key], source, path)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _require(config: dict[str, Any], *paths: str) -> None:
    for dotted in paths:
        value: Any = config
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Resolved configuration is missing required key: {dotted}")
            value = value[key]
        if value is None or value == "":
            raise ValueError(f"Resolved configuration has no value for: {dotted}")


def load_resolved_config(shared_path: Path, experiment_path: Path, participant_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    """Load layered configuration, or temporarily accept a legacy single YAML file."""
    sources = [str(shared_path.resolve())]
    shared = _read_yaml(shared_path)
    _validate_known_keys(shared, SCHEMA, shared_path)
    if participant_path is None and "participant" in shared and "inputs" in shared:
        warnings.warn("Single-YAML configuration is deprecated; use --config plus --participant-config.", UserWarning, stacklevel=2)
        _require(shared, "participant", "experiment", "inputs.participant_dir", "inputs.ratings_file", "inputs.video_file", "inputs.vision_log_file")
        _validate_eeg_input_choice(shared)
        return shared, sources
    experiment = _read_yaml(experiment_path)
    _validate_known_keys(experiment, SCHEMA, experiment_path)
    if participant_path is None:
        raise ValueError("--participant-config is required with layered configuration")
    participant = _read_yaml(participant_path)
    _validate_known_keys(participant, SCHEMA, participant_path)
    sources.extend([str(experiment_path.resolve()), str(participant_path.resolve())])
    config = _merge(_merge(shared, experiment), participant)
    _require(config, "participant", "experiment", "inputs.participant_dir", "inputs.experiment_dir",
             "inputs.ratings_file", "inputs.video_file", "inputs.vision_log_file", "resources.face_landmarker_filename")
    _validate_eeg_input_choice(config)
    return config, sources


def _validate_eeg_input_choice(config: dict[str, Any]) -> None:
    inputs = config["inputs"]
    has_single = bool(inputs.get("eeg_file"))
    has_multiple = "eeg_files" in inputs and inputs.get("eeg_files") is not None
    if has_single == has_multiple:
        raise ValueError("Configuration must define exactly one of inputs.eeg_file or inputs.eeg_files")
    if has_multiple:
        files = inputs["eeg_files"]
        if not isinstance(files, list) or not files or not all(isinstance(value, str) and value for value in files):
            raise ValueError("inputs.eeg_files must be a non-empty ordered list of filenames")
