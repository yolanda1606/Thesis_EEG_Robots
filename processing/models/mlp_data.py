"""EEG data loading, cohort splitting, scaling, and input selection for MLPs."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
import mne

from processing.multimodal_image.modeling import train_classification


DEFAULT_NO_ICA_SOURCE_RUN = "{participant_lower}_no_ica"
EEG_FEATURE_NAMES = tuple(train_classification.EEG_COLUMNS)
FACE_FEATURE_NAMES = tuple(train_classification.FACE_COLUMNS)
EEG_CHANNELS = tuple(train_classification.EEG_CHANNELS)
EEG_FAMILIES = tuple(train_classification.EEG_FAMILIES)


def prepare_eeg_data(table: pd.DataFrame, participant: str, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return unscaled EEG features, LOW=0/HIGH=1 labels, and names."""
    if target not in train_classification.TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    features, labels, feature_names = train_classification.prepare_modality_data(
        table=table, participant=participant, target=target, modality="eeg"
    )
    return features.to_numpy(dtype=float), labels.to_numpy(dtype=int), feature_names


def load_participant_eeg_data(derived_root: Path, participant: str, target: str,
                              source_run: str = DEFAULT_NO_ICA_SOURCE_RUN) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one participant's existing no-ICA EEG Image table."""
    table, _ = train_classification.load_participant_table(derived_root, participant, source_run)
    return prepare_eeg_data(table, participant, target)


def prepare_eeg_rating_data(table: pd.DataFrame, participant: str, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return usable EEG rows and their original continuous ratings."""
    if target not in train_classification.TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    feature_names = train_classification.modality_columns("eeg")
    target_column = train_classification.TARGET_COLUMNS[target]
    missing = sorted(set(feature_names + [target_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{participant} missing required EEG columns: {missing}")
    numeric_features = table[feature_names].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[target_column], errors="coerce")
    usable = numeric_features.notna().all(axis=1) & ratings.notna()
    features = numeric_features.loc[usable].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError(f"{participant} EEG predictors contain infinity")
    return features, ratings.loc[usable].to_numpy(dtype=float), feature_names


def load_participant_eeg_rating_data(derived_root: Path, participant: str, target: str,
                                     source_run: str = DEFAULT_NO_ICA_SOURCE_RUN) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one no-ICA EEG table with continuous ratings."""
    table, _ = train_classification.load_participant_table(derived_root, participant, source_run)
    return prepare_eeg_rating_data(table, participant, target)


def _load_general(derived_root: Path, participants: list[str] | tuple[str, ...], target: str, loader):
    participant_ids = [str(participant).upper() for participant in participants]
    if not participant_ids:
        raise ValueError("participants must contain at least one participant ID")
    feature_matrices, targets, group_vectors = [], [], []
    expected_feature_names: list[str] | None = None
    for participant in participant_ids:
        try:
            features, labels, feature_names = loader(derived_root, participant, target)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Cannot load {participant} no-ICA EEG source: {error}") from error
        except ValueError as error:
            raise ValueError(f"Cannot prepare {participant} EEG data: {error}") from error
        if expected_feature_names is None:
            expected_feature_names = feature_names
        elif feature_names != expected_feature_names:
            raise ValueError(f"EEG feature schema mismatch for {participant}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"{participant} EEG features and labels have different row counts")
        feature_matrices.append(features)
        targets.append(labels)
        group_vectors.append(np.full(labels.shape[0], participant, dtype=object))
    return np.vstack(feature_matrices), np.concatenate(targets), np.concatenate(group_vectors), expected_feature_names or []


def load_general_eeg_data(derived_root: Path, participants: list[str] | tuple[str, ...], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load binary EEG trials for a participant cohort, retaining row groups."""
    x, y, groups, names = _load_general(derived_root, participants, target, load_participant_eeg_data)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("EEG labels must be binary LOW=0/HIGH=1")
    return x, y, groups, names


def load_general_eeg_rating_data(derived_root: Path, participants: list[str] | tuple[str, ...], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load continuous-rating EEG trials for a participant cohort."""
    return _load_general(derived_root, participants, target, load_participant_eeg_rating_data)


def modality_rating_columns(modality: str, eeg_configuration: str = "midline_bandpower") -> list[str]:
    """Return the established continuous-rating predictors for one MLP modality."""
    if modality == "eeg":
        configuration = get_eeg_input_configuration(eeg_configuration)
        requested = {f"{family}__{channel}" for family in configuration.feature_families for channel in configuration.channels}
        return [name for name in EEG_FEATURE_NAMES if name in requested]
    if modality == "face":
        return list(FACE_FEATURE_NAMES)
    if modality == "multimodal":
        return modality_rating_columns("eeg", eeg_configuration) + modality_rating_columns("face")
    raise ValueError(f"Unknown MLP modality: {modality}")


def prepare_modality_rating_data(table: pd.DataFrame, participant: str, target: str,
                                 modality: str, eeg_configuration: str = "midline_bandpower") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return finite modality predictors and original observed target ratings.

    The usable-row rule matches the classical classifier: every required
    predictor and the target rating must be numeric and non-missing.
    """
    if target not in train_classification.TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    feature_names = modality_rating_columns(modality, eeg_configuration)
    target_column = train_classification.TARGET_COLUMNS[target]
    missing = sorted(set(feature_names + [target_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{participant} missing required {modality} columns: {missing}")
    numeric_features = table[feature_names].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[target_column], errors="coerce")
    usable = numeric_features.notna().all(axis=1) & ratings.notna()
    features = numeric_features.loc[usable].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError(f"{participant} {modality} predictors contain infinity")
    return features, ratings.loc[usable].to_numpy(dtype=float), feature_names


def load_general_modality_rating_data(derived_root: Path, participants: list[str] | tuple[str, ...],
                                      target: str, modality: str,
                                      eeg_configuration: str = "midline_bandpower") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load one MLP modality across participants while retaining row groups."""
    participant_ids = [str(participant).upper() for participant in participants]
    if not participant_ids:
        raise ValueError("participants must contain at least one participant ID")
    matrices: list[np.ndarray] = []
    rating_vectors: list[np.ndarray] = []
    group_vectors: list[np.ndarray] = []
    expected_names: list[str] | None = None
    for participant in participant_ids:
        try:
            table, _ = train_classification.load_participant_table(
                derived_root, participant, DEFAULT_NO_ICA_SOURCE_RUN
            )
            features, ratings, feature_names = prepare_modality_rating_data(
                table, participant, target, modality, eeg_configuration
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Cannot load {participant} no-ICA {modality} source: {error}") from error
        except ValueError as error:
            raise ValueError(f"Cannot prepare {participant} {modality} data: {error}") from error
        if expected_names is None:
            expected_names = feature_names
        elif feature_names != expected_names:
            raise ValueError(f"{modality} feature schema mismatch for {participant}")
        matrices.append(features)
        rating_vectors.append(ratings)
        group_vectors.append(np.full(len(ratings), participant, dtype=object))
    return np.vstack(matrices), np.concatenate(rating_vectors), np.concatenate(group_vectors), expected_names or []


def make_loso_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, held_out_participant: str):
    """Partition all rows into all-other-participant training and one test participant."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    if x.ndim != 2 or y.ndim != 1 or groups.ndim != 1 or not (len(x) == len(y) == len(groups)):
        raise ValueError("X must be two-dimensional and X, y, and groups must have matching row counts")
    held_out_id = str(held_out_participant).upper()
    test_mask = np.char.upper(groups.astype(str)) == held_out_id
    if not test_mask.any():
        raise ValueError(f"Unknown held-out participant: {held_out_id}")
    return x[~test_mask], y[~test_mask], groups[~test_mask], x[test_mask], y[test_mask], groups[test_mask]


def make_grouped_validation_split(x_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                                  outer_held_out_participant: str, seed: int = 42):
    """Return the first deterministic five-fold stratified group partition."""
    x_train, y_train, groups_train = np.asarray(x_train), np.asarray(y_train), np.asarray(groups_train)
    if x_train.ndim != 2 or y_train.ndim != 1 or groups_train.ndim != 1 or not (len(x_train) == len(y_train) == len(groups_train)):
        raise ValueError("X_train must be two-dimensional and inputs must have matching row counts")
    held_out_id = str(outer_held_out_participant).upper()
    if (np.char.upper(groups_train.astype(str)) == held_out_id).any():
        raise ValueError(f"Outer held-out participant {held_out_id} must be absent from the training pool")
    if len(np.unique(groups_train)) < 5:
        raise ValueError("Grouped validation requires at least five training participants")
    try:
        train_indices, val_indices = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x_train, y_train, groups_train))
    except ValueError as error:
        raise ValueError(f"Cannot form 5-fold stratified grouped validation split: {error}") from error
    if set(groups_train[train_indices]).intersection(groups_train[val_indices]):
        raise RuntimeError("Grouped validation split mixed participant groups")
    return x_train[train_indices], y_train[train_indices], groups_train[train_indices], x_train[val_indices], y_train[val_indices], groups_train[val_indices]


@dataclass(frozen=True)
class EEGScaler:
    """Z-score parameters fitted from one training feature matrix."""
    mean: np.ndarray
    std: np.ndarray


def fit_eeg_scaler(x_train: np.ndarray) -> EEGScaler:
    """Fit z-score parameters from training rows only."""
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim != 2:
        raise ValueError("x_train must be a two-dimensional array")
    if not len(x_train):
        raise ValueError("x_train must contain at least one row")
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    return EEGScaler(mean=mean, std=np.where(std == 0.0, 1.0, std))


def transform_eeg_features(features: np.ndarray, scaler: EEGScaler) -> np.ndarray:
    """Transform a feature matrix using pre-fitted training parameters."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if features.shape[1] != scaler.mean.shape[0]:
        raise ValueError(f"Expected {scaler.mean.shape[0]} input features, received {features.shape[1]}.")
    return (features - scaler.mean) / scaler.std


def participant_zscore_features(features: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Normalize each participant independently from feature values only.

    This is intentionally participant-wise transductive normalization when
    applied to validation or held-out rows: it uses that participant's
    unlabeled feature distribution and must not be treated as deployable
    real-time normalization.
    """
    feature_matrix = np.asarray(features, dtype=float)
    participant_groups = np.asarray(groups)
    if feature_matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if participant_groups.ndim != 1 or len(feature_matrix) != len(participant_groups):
        raise ValueError("features and groups must have matching row counts")
    normalized = np.empty_like(feature_matrix, dtype=float)
    for participant in np.unique(participant_groups):
        mask = participant_groups == participant
        participant_features = feature_matrix[mask]
        mean = participant_features.mean(axis=0)
        std = participant_features.std(axis=0)
        normalized[mask] = (participant_features - mean) / np.where(std == 0.0, 1.0, std)
    return normalized


EEG_EPOCH_CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")


def cleaned_epoch_path(derived_root: Path, participant: str) -> Path:
    """Return the existing saved no-ICA MNE epoch artifact for one participant."""
    participant_id = str(participant).upper()
    participant_lower = participant_id.lower()
    return (
        Path(derived_root) / participant_id / "Image_Experiment" / "runs" / f"{participant_lower}_no_ica"
        / "eeg" / "cleaned_epochs" / f"{participant_lower}_image_cleaned-epo.fif"
    )


def load_general_eeg_epoch_data(derived_root: Path, participants: list[str] | tuple[str, ...],
                                target: str, include_triggers: bool = False,
                                include_times: bool = False):
    """Load retained no-ICA trial epochs aligned to observed original ratings.

    Each row follows the saved MNE epoch event order and has shape
    ``(channels, time)``. Ratings are joined by trigger ID, never by implicit
    table row position.
    """
    if target not in train_classification.TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    participant_ids = [str(participant).upper() for participant in participants]
    if not participant_ids:
        raise ValueError("participants must contain at least one participant ID")
    epoch_matrices, rating_vectors, group_vectors, trigger_vectors = [], [], [], []
    expected_shape: tuple[int, int] | None = None
    expected_channels: tuple[str, ...] | None = None
    expected_sampling_hz: float | None = None
    expected_times: np.ndarray | None = None
    target_column = train_classification.TARGET_COLUMNS[target]
    for participant in participant_ids:
        table, _ = train_classification.load_participant_table(
            derived_root, participant, DEFAULT_NO_ICA_SOURCE_RUN
        )
        if "trigger" not in table or target_column not in table:
            raise ValueError(f"{participant} merged table requires trigger and {target_column}")
        trigger_ratings = pd.DataFrame({
            "trigger": pd.to_numeric(table["trigger"], errors="coerce"),
            "rating": pd.to_numeric(table[target_column], errors="coerce"),
        }).dropna()
        if trigger_ratings.trigger.duplicated().any():
            raise ValueError(f"{participant} merged table has duplicate rating triggers")
        rating_by_trigger = dict(zip(trigger_ratings.trigger.astype(int), trigger_ratings.rating.astype(float)))
        path = cleaned_epoch_path(derived_root, participant)
        if not path.is_file():
            raise FileNotFoundError(f"{participant} cleaned no-ICA epochs missing: {path}")
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        channels = tuple(epochs.ch_names)
        data = epochs.get_data(copy=True)
        triggers = epochs.events[:, 2].astype(int)
        times = epochs.times.copy()
        usable = np.asarray([trigger in rating_by_trigger for trigger in triggers], dtype=bool)
        if not usable.any():
            raise ValueError(f"{participant} has no retained epochs with usable {target} ratings")
        data, triggers = data[usable], triggers[usable]
        ratings = np.asarray([rating_by_trigger[trigger] for trigger in triggers], dtype=float)
        shape, sampling_hz = data.shape[1:], float(epochs.info["sfreq"])
        if expected_shape is None:
            expected_shape, expected_channels, expected_sampling_hz, expected_times = shape, channels, sampling_hz, times
        elif (shape != expected_shape or channels != expected_channels or sampling_hz != expected_sampling_hz
              or not np.array_equal(times, expected_times)):
            raise ValueError(f"EEG epoch schema mismatch for {participant}")
        epoch_matrices.append(data)
        rating_vectors.append(ratings)
        group_vectors.append(np.full(len(ratings), participant, dtype=object))
        trigger_vectors.append(triggers)
    result = (
        np.concatenate(epoch_matrices), np.concatenate(rating_vectors), np.concatenate(group_vectors),
        expected_channels or (), float(expected_sampling_hz or 0.0),
    )
    if include_triggers:
        result += (np.concatenate(trigger_vectors),)
    if include_times:
        result += (np.asarray(expected_times, dtype=float),)
    return result


def load_general_multimodal_epoch_data(
    derived_root: Path, participants: list[str] | tuple[str, ...], target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], float, np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, int | str]]]:
    """Load trigger-aligned raw epochs, ratings, and existing face predictors.

    Face rows are joined to EEG rows by their saved ``trigger`` identifiers,
    never by CSV or epoch row position.  Missing face features remove only the
    affected multimodal trial and are reported per participant.
    """
    epochs, ratings, groups, channels, sampling_hz, triggers, times = load_general_eeg_epoch_data(
        derived_root, participants, target, include_triggers=True, include_times=True
    )
    face_names = list(FACE_FEATURE_NAMES)
    target_column = train_classification.TARGET_COLUMNS[target]
    aligned_indices: list[int] = []
    aligned_face_rows: list[np.ndarray] = []
    alignment_report: list[dict[str, int | str]] = []
    for participant in dict.fromkeys(groups.astype(str)):
        eeg_indices = np.flatnonzero(groups.astype(str) == participant)
        participant_triggers = triggers[eeg_indices].astype(int)
        table, _ = train_classification.load_participant_table(
            derived_root, participant, DEFAULT_NO_ICA_SOURCE_RUN
        )
        required = set(face_names + ["trigger", target_column])
        if missing := sorted(required.difference(table.columns)):
            raise ValueError(f"{participant} merged table missing required multimodal columns: {missing}")
        face_table = table.loc[:, ["trigger", target_column] + face_names].copy()
        face_table["trigger"] = pd.to_numeric(face_table["trigger"], errors="coerce")
        face_table[target_column] = pd.to_numeric(face_table[target_column], errors="coerce")
        face_values = face_table[face_names].apply(pd.to_numeric, errors="coerce")
        usable_face = face_table["trigger"].notna() & face_table[target_column].notna() & face_values.notna().all(axis=1)
        face_table = face_table.loc[usable_face]
        if face_table["trigger"].duplicated().any():
            raise ValueError(f"{participant} has duplicate usable face-feature triggers")
        face_by_trigger = {
            int(trigger): values.to_numpy(dtype=float)
            for trigger, (_, values) in zip(face_table["trigger"], face_table[face_names].iterrows())
        }
        face_triggers = set(face_by_trigger)
        eeg_trigger_set = set(participant_triggers)
        selected = [index for index in eeg_indices if int(triggers[index]) in face_by_trigger]
        aligned_indices.extend(selected)
        aligned_face_rows.extend(face_by_trigger[int(triggers[index])] for index in selected)
        alignment_report.append({
            "participant": participant,
            "eeg_epochs": int(len(eeg_indices)),
            "face_trials": int(len(face_table)),
            "aligned_multimodal_trials": int(len(selected)),
            "dropped_eeg_only": int(len(eeg_trigger_set.difference(face_triggers))),
            "dropped_face_only": int(len(face_triggers.difference(eeg_trigger_set))),
        })
    if not aligned_indices:
        raise ValueError("No EEG and face trials aligned by trigger")
    indices = np.asarray(aligned_indices, dtype=int)
    return (
        epochs[indices], ratings[indices], groups[indices], channels, sampling_hz, times, triggers[indices],
        np.vstack(aligned_face_rows), face_names, alignment_report,
    )


def participant_channel_zscore_epochs(epochs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score each participant/channel over that participant's trial and time axes.

    Means and standard deviations use axes ``(trials, time)`` independently
    for each channel. This feature-only transform preserves trial-to-trial
    amplitude differences and is participant-wise transductive for validation
    and held-out test partitions.
    """
    epoch_array, participant_groups = np.asarray(epochs, dtype=float), np.asarray(groups)
    if epoch_array.ndim != 3:
        raise ValueError("epochs must have shape (trials, channels, time)")
    if participant_groups.ndim != 1 or len(epoch_array) != len(participant_groups):
        raise ValueError("epochs and groups must have matching trial counts")
    normalized = np.empty_like(epoch_array, dtype=float)
    for participant in np.unique(participant_groups):
        mask = participant_groups == participant
        participant_epochs = epoch_array[mask]
        mean = participant_epochs.mean(axis=(0, 2), keepdims=True)
        std = participant_epochs.std(axis=(0, 2), keepdims=True)
        normalized[mask] = (participant_epochs - mean) / np.where(std == 0.0, 1.0, std)
    return normalized


@dataclass(frozen=True)
class EEGInputConfiguration:
    """Named subset of the established project EEG schema."""
    name: str
    channels: tuple[str, ...]
    feature_families: tuple[str, ...]


_ALL_CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
_BANDPOWER_FAMILIES = ("eeg_bp_delta", "eeg_bp_theta", "eeg_bp_alpha", "eeg_bp_beta", "eeg_bp_gamma")
_BANDPOWER_ENTROPY_FAMILIES = ("eeg_bp_delta", "eeg_se_delta", "eeg_bp_theta", "eeg_se_theta", "eeg_bp_alpha", "eeg_se_alpha", "eeg_bp_beta", "eeg_se_beta", "eeg_bp_gamma", "eeg_se_gamma")
EEG_INPUT_CONFIGURATIONS = {
    "all_features_all_channels": EEGInputConfiguration("all_features_all_channels", _ALL_CHANNELS, EEG_FAMILIES),
    "bandpower_all_channels": EEGInputConfiguration("bandpower_all_channels", _ALL_CHANNELS, _BANDPOWER_FAMILIES),
    "bandpower_entropy_all_channels": EEGInputConfiguration("bandpower_entropy_all_channels", _ALL_CHANNELS, _BANDPOWER_ENTROPY_FAMILIES),
    "posterior_bandpower": EEGInputConfiguration("posterior_bandpower", ("Pz", "PO7", "Oz", "PO8"), _BANDPOWER_FAMILIES),
    "midline_bandpower": EEGInputConfiguration("midline_bandpower", ("Fz", "Cz", "Pz", "Oz"), _BANDPOWER_FAMILIES),
    "frontal_central_bandpower": EEGInputConfiguration("frontal_central_bandpower", ("Fz", "C3", "Cz", "C4"), _BANDPOWER_FAMILIES),
}


def get_eeg_input_configuration(name: str) -> EEGInputConfiguration:
    """Return one supported named EEG selection configuration."""
    try:
        return EEG_INPUT_CONFIGURATIONS[name]
    except KeyError as error:
        raise ValueError(f"Unknown EEG input configuration: {name}") from error


def select_eeg_input_features(x: np.ndarray, feature_names: list[str], configuration_name: str = "all_features_all_channels") -> tuple[np.ndarray, list[str]]:
    """Select an ordered EEG subset from the actual established feature schema."""
    features = np.asarray(x)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("X columns must match feature_names")
    config = get_eeg_input_configuration(configuration_name)
    requested = {f"{family}__{channel}" for family in config.feature_families for channel in config.channels}
    missing = sorted(requested.difference(feature_names))
    if missing:
        raise ValueError(f"Missing EEG columns for {configuration_name}: {missing}")
    # Schema order is the established classifier order; the full default remains byte-for-byte compatible.
    selected_names = [name for name in EEG_FEATURE_NAMES if name in requested]
    indices = [feature_names.index(name) for name in selected_names]
    return features[:, indices], selected_names
