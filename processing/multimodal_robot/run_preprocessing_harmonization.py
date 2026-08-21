#!/usr/bin/env python3
"""Experiment C: reproducible no-ICA P19 Robot preprocessing harmonization.

This separate diagnostic path reads raw P19 Robot BDF files and existing frozen
Image models. It does not modify raw data, the canonical robot run, Image code,
or any Experiment A/B output. C0 is explicitly a reproducible harmonized
preprocessing baseline without ICA, not a perfectly Image-matched pipeline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path, PureWindowsPath

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "processing/multimodal_image"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGE_ROOT))

from src.features import extract_eeg_features  # noqa: E402  # Direct, frozen Image feature definitions.
from processing.multimodal_image.modeling.train_classification import EEG_COLUMNS, load_participant_table, prepare_modality_data  # noqa: E402
from processing.multimodal_robot.audit_personalized_transfer import distance_audit, percentile  # noqa: E402
from processing.multimodal_robot.run_personalized_inference import (  # noqa: E402
    PARTICIPANT, TASK_LABELS, high_probabilities, prepare_robot_windows,
)

RAW_ROOT = Path("data/P19_2026-06-11")
ROBOT_RESOLVED_CONFIG = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/resolved_participant.yaml")
ALIGNMENT_METADATA = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/alignment/alignment_metadata.json")
EXISTING_FEATURES = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/features/eeg_window_features.csv")
IMAGE_RESOLVED_CONFIG = Path("derived/P19/Image_Experiment/runs/p19_final/config/resolved_configuration.yaml")
INFERENCE_DIR = Path("outputs/robot_personalized_inference/P19")
OUTPUT_DIR = INFERENCE_DIR / "preprocessing_harmonization"
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction")
PLOT_TASKS = ("shape_sorter_observation", "sisyphus", "shape_sorter_interaction")
TARGETS = ("valence", "arousal")
PROBLEM_FEATURES = ("eeg_bp_delta__Fz", "eeg_hc__Fz", "eeg_se__Fz")
OUTPUTS = (
    "P19_C0_harmonized_window_features.csv", "P19_preprocessing_harmonization_window_diagnostics.csv",
    "P19_preprocessing_harmonization_task_summary.csv", "P19_preprocessing_harmonization_feature_summary.csv",
    "P19_preprocessing_harmonization_summary.md", "P19_preprocessing_harmonization_manifest.json",
    "P19_domain_distance_existing_vs_C0.png", "P19_problematic_feature_existing_vs_C0.png",
    "P19_shape_sorter_observation_C0_trajectory_comparison.png", "P19_sisyphus_C0_trajectory_comparison.png",
    "P19_shape_sorter_interaction_C0_trajectory_comparison.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default=PARTICIPANT, choices=(PARTICIPANT,))
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--robot-resolved-config", type=Path, default=ROBOT_RESOLVED_CONFIG)
    parser.add_argument("--alignment-metadata", type=Path, default=ALIGNMENT_METADATA)
    parser.add_argument("--image-resolved-config", type=Path, default=IMAGE_RESOLVED_CONFIG)
    parser.add_argument("--existing-robot-features", type=Path, default=EXISTING_FEATURES)
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--inference-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace only Experiment C outputs in --output-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input contracts without preprocessing, model application, or writes.")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_raw_paths(raw_root: Path, robot_config: dict) -> dict[str, Path]:
    """Resolve only BDF names explicitly recorded in the existing P19 config."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"P19 raw root missing: {raw_root}")
    paths = {}
    for task in TASKS:
        configured = str(robot_config["tasks"][task]["eeg"])
        name = PureWindowsPath(configured).name
        matches = list(raw_root.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"{task}: expected exactly one raw BDF named {name}; found {len(matches)}")
        paths[task] = matches[0]
    return paths


def validate_contract(image_config: dict, alignment: dict, raw_paths: dict[str, Path]) -> None:
    expected_mapping = {"EEG 1": "Fz", "EEG 2": "C3", "EEG 3": "Cz", "EEG 4": "C4", "EEG 5": "Pz", "EEG 6": "PO7", "EEG 7": "Oz", "EEG 8": "PO8"}
    if image_config["channels"]["eeg_mapping"] != expected_mapping:
        raise ValueError("P19 Image EEG mapping differs from approved Experiment C mapping")
    filt = image_config["eeg"]["filter"]
    if (float(filt["l_freq_hz"]), float(filt["h_freq_hz"]), int(filt["iir_order"])) != (1.0, 40.0, 4):
        raise ValueError("P19 Image filter is not the approved 1.0–40.0 Hz fourth-order Butterworth IIR")
    task_records = {row["task"]: row for row in alignment.get("tasks", [])}
    if set(TASKS).difference(task_records):
        raise ValueError("Existing alignment metadata lacks one or more approved Status-aligned tasks")
    if any(not raw_paths[task].is_file() for task in TASKS):
        raise FileNotFoundError("One or more configured raw BDF files are unavailable")


def preprocess_task_c0(raw_path: Path, task: str, image_config: dict, expected_duration_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C0: direct raw BDF -> map -> CAR -> 1–40 Hz IIR -> 500-sample windows.

    The Image feature extractor is invoked directly via an EpochsArray wrapper.
    Its local [0, 1.996] time extent represents exactly 500 samples at 250 Hz;
    this avoids adding a 501st endpoint sample to the specified 2.0-s robot
    windows while retaining the exact feature mathematics.
    """
    mapping = image_config["channels"]["eeg_mapping"]
    eeg_names = list(mapping.values())
    raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], 250.0):
        raise ValueError(f"{task}: expected 250 Hz raw data, found {raw.info['sfreq']}")
    missing = set(mapping).difference(raw.ch_names)
    if missing:
        raise ValueError(f"{task}: raw BDF lacks mapped EEG channels: {sorted(missing)}")
    raw.set_channel_types({name: "eeg" for name in mapping}, verbose="ERROR")
    raw.rename_channels(mapping)
    raw.pick(eeg_names)
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    filt = image_config["eeg"]["filter"]
    raw.filter(float(filt["l_freq_hz"]), float(filt["h_freq_hz"]), method="iir",
               iir_params={"order": int(filt["iir_order"]), "ftype": "butter", "output": "sos"}, verbose="ERROR")
    sfreq, samples = float(raw.info["sfreq"]), raw.n_times
    if abs(samples / sfreq - expected_duration_s) > 1.0 / sfreq:
        raise ValueError(f"{task}: raw duration conflicts with existing validated alignment metadata")
    starts = np.arange(0, samples - 500 + 1, 250, dtype=int)
    data = np.stack([raw.get_data(start=int(start), stop=int(start + 500)) for start in starts])
    if data.ndim != 3 or data.shape[1:] != (8, 500):
        raise RuntimeError(f"{task}: C0 window construction did not produce 8 × 500 samples")
    info = mne.create_info(eeg_names, sfreq=sfreq, ch_types="eeg")
    events = np.column_stack([np.arange(len(data)), np.zeros(len(data), dtype=int), np.ones(len(data), dtype=int)])
    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0, baseline=None, verbose="ERROR")
    feature_config = copy.deepcopy(image_config)
    feature_config["features"]["eeg_feature_window_s"] = [0.0, (500 - 1) / sfreq]
    long = extract_eeg_features(epochs, feature_config)
    feature_names = [name for name in long.columns if name.startswith("eeg_")]
    wide = long.pivot(index="epoch_index", columns="channel", values=feature_names)
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    wide = wide.reindex(columns=EEG_COLUMNS).reset_index(drop=True)
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy(float)).all():
        raise RuntimeError(f"{task}: Image feature extraction produced missing/non-finite C0 values")
    metadata = pd.DataFrame({"participant": PARTICIPANT, "representation": "C0_reproducible_harmonized_no_ICA", "task": task,
                             "task_label": TASK_LABELS[task], "window_id": [f"{task}_s1_{start / sfreq:.3f}" for start in starts],
                             "window_index": np.arange(1, len(starts) + 1), "window_start_s": starts / sfreq,
                             "window_end_s": (starts + 500) / sfreq, "sample_count": 500, "sampling_hz": sfreq})
    return metadata, wide


def selected_features(model) -> list[str]:
    support = model.named_steps["selector"].get_support(indices=True)
    return [EEG_COLUMNS[index] for index in support]


def representation_diagnostics(representation: str, windows: pd.DataFrame, image_data: dict, models: dict) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        model, image, labels = models[target], image_data[target]["features"], image_data[target]["labels"]
        selected = selected_features(model)
        scaled_image = model.named_steps["selector"].transform(model.named_steps["scale"].transform(image))
        scaled_robot_full = model.named_steps["scale"].transform(windows[EEG_COLUMNS])
        scaled_robot = model.named_steps["selector"].transform(scaled_robot_full)
        reference, distance = distance_audit(scaled_image, scaled_robot, k=11)
        p95, maximum = percentile(reference["nearest"], 95), float(np.max(reference["nearest"]))
        predicted, scores = high_probabilities(model, windows[EEG_COLUMNS])
        z = np.abs(pd.DataFrame(scaled_robot_full, columns=EEG_COLUMNS)[selected].to_numpy(float))
        for index, row in enumerate(windows.itertuples(index=False)):
            nearest = float(distance["nearest"][index])
            status = "beyond_image_reference" if nearest > maximum else ("above_image_95pct" if nearest > p95 else "within_reference")
            rows.append({"participant": PARTICIPANT, "representation": representation, "target": target, "task": row.task,
                         "window_id": row.window_id, "window_index": int(row.window_index), "window_start_s": float(row.window_start_s),
                         "predicted_class": predicted[index], "high_class_score": float(scores[index]), "nearest_image_distance": nearest,
                         "image_reference_distance_percentile": float(np.mean(reference["nearest"] <= nearest) * 100),
                         "image_reference_nearest_distance_p95": p95, "image_reference_nearest_distance_max": maximum,
                         "nearest_distance_exceeds_image_p95": bool(nearest > p95), "nearest_distance_exceeds_image_max": bool(nearest > maximum),
                         "feature_domain_reliability": status, "selected_feature_abs_z_median": float(np.median(z[index])),
                         "selected_feature_abs_z_p95": percentile(z[index], 95)})
    return pd.DataFrame(rows)


def feature_summary(representations: dict[str, pd.DataFrame], image_data: dict, models: dict) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        image, labels, model = image_data[target]["features"], image_data[target]["labels"], models[target]
        scaler = model.named_steps["scale"]
        for feature in selected_features(model):
            low, high = image.loc[labels == 0, feature].to_numpy(float), image.loc[labels == 1, feature].to_numpy(float)
            for representation, windows in representations.items():
                scaled = pd.DataFrame(scaler.transform(windows[EEG_COLUMNS]), columns=EEG_COLUMNS)
                for task in TASKS:
                    mask = windows.task.eq(task).to_numpy()
                    values, z = windows.loc[mask, feature].to_numpy(float), np.abs(scaled.loc[mask, feature].to_numpy(float))
                    rows.append({"participant": PARTICIPANT, "representation": representation, "target": target, "task": task,
                                 "task_label": TASK_LABELS[task], "feature": feature, "known_problem_feature": feature in PROBLEM_FEATURES,
                                 "image_low_median": float(np.median(low)), "image_low_iqr": float(np.percentile(low, 75) - np.percentile(low, 25)),
                                 "image_high_median": float(np.median(high)), "image_high_iqr": float(np.percentile(high, 75) - np.percentile(high, 25)),
                                 "image_min": float(np.min(image[feature])), "image_max": float(np.max(image[feature])),
                                 "robot_median": float(np.median(values)), "robot_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                                 "robot_z_median_abs": float(np.median(z)), "robot_z_p95_abs": percentile(z, 95),
                                 "robot_pct_outside_image_min_max": float(np.mean((values < np.min(image[feature])) | (values > np.max(image[feature])))*100)})
    return pd.DataFrame(rows)


def task_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in diagnostics.groupby(["representation", "target", "task"], sort=False):
        representation, target, task = keys
        scores = frame.high_class_score.to_numpy(float)
        rows.append({"participant": PARTICIPANT, "representation": representation, "target": target, "task": task, "task_label": TASK_LABELS[task],
                     "window_count": len(frame), "nearest_image_distance_median": float(np.median(frame.nearest_image_distance)),
                     "pct_above_image_reference_p95": float(np.mean(frame.nearest_distance_exceeds_image_p95)*100),
                     "pct_beyond_image_reference_max": float(np.mean(frame.nearest_distance_exceeds_image_max)*100),
                     "selected_feature_abs_z_median": float(np.median(frame.selected_feature_abs_z_median)),
                     "selected_feature_abs_z_p95": percentile(frame.selected_feature_abs_z_p95, 95),
                     "high_score_mean": float(np.mean(scores)), "high_score_median": float(np.median(scores)),
                     "pct_predicted_high": float(np.mean(frame.predicted_class.eq("HIGH"))*100)})
    return pd.DataFrame(rows)


def plot_distances(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, target in zip(axes, TARGETS):
        part = summary[summary.target == target]
        for offset, representation in enumerate(("A_existing_robot_features", "C0_reproducible_harmonized_no_ICA")):
            values = part[part.representation == representation].set_index("task").loc[list(TASKS), "nearest_image_distance_median"]
            axis.plot(range(len(TASKS)), values, marker="o", linewidth=1.4, label=representation.replace("A_existing_robot_features", "Existing A").replace("C0_reproducible_harmonized_no_ICA", "C0 no-ICA"))
        axis.set(xticks=range(len(TASKS)), xticklabels=[TASK_LABELS[t] for t in TASKS], ylabel="Median nearest Image distance", title=target.capitalize())
        axis.tick_params(axis="x", rotation=35, labelsize=7); axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Experiment C: frozen Image-scaled selected-feature geometry")
    figure.savefig(output_dir / "P19_domain_distance_existing_vs_C0.png", dpi=160); plt.close(figure)


def plot_problem_features(features: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(PROBLEM_FEATURES), figsize=(14, 4.6), constrained_layout=True)
    for axis, feature in zip(axes, PROBLEM_FEATURES):
        part = features[features.feature == feature]
        if part.empty: axis.set_visible(False); continue
        first = part.iloc[0]
        series = [np.repeat(first.image_low_median, 2), np.repeat(first.image_high_median, 2)]
        labels = ["Image LOW", "Image HIGH"]
        for representation in ("A_existing_robot_features", "C0_reproducible_harmonized_no_ICA"):
            series.append(part[part.representation == representation].robot_median.to_numpy(float)); labels.append("Existing A" if representation.startswith("A_") else "C0 no-ICA")
        axis.boxplot(series, labels=labels, showfliers=False)
        axis.set_title(feature); axis.tick_params(axis="x", rotation=35, labelsize=7); axis.set_ylabel("Per-task median feature value")
    figure.suptitle("Problematic selected features: Image class medians vs Robot task medians")
    figure.savefig(output_dir / "P19_problematic_feature_existing_vs_C0.png", dpi=160); plt.close(figure)


def plot_trajectories(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    for task in PLOT_TASKS:
        figure, axes = plt.subplots(2, 1, figsize=(9.2, 5.7), sharex=True, constrained_layout=True)
        for axis, target in zip(axes, TARGETS):
            part = diagnostics[(diagnostics.task == task) & (diagnostics.target == target)]
            for representation, color, label in (("A_existing_robot_features", "#777777", "Existing A"), ("C0_reproducible_harmonized_no_ICA", "#1f77b4", "C0 no-ICA")):
                data = part[part.representation == representation]
                axis.plot(data.window_start_s, data.high_class_score, color=color, marker="o", markersize=2.4, linewidth=1.1, label=label)
                far = data.feature_domain_reliability.eq("beyond_image_reference")
                if far.any(): axis.scatter(data.loc[far, "window_start_s"], data.loc[far, "high_class_score"], marker="x", s=21, color=color, zorder=4)
            axis.axhline(.5, color="black", linestyle="--", linewidth=1, label="0.5 classification threshold")
            axis.set(ylim=(-.03, 1.03), ylabel=f"High-{target} probability"); axis.legend(frameon=False, fontsize=8)
        axes[-1].set_xlabel("Task-relative EEG window start (s)")
        figure.suptitle(f"P19 {TASK_LABELS[task]}: Existing A vs C0 reproducible no-ICA baseline\nCrosses: candidate feature space beyond Image reference maximum")
        figure.savefig(output_dir / f"P19_{task}_C0_trajectory_comparison.png", dpi=160); plt.close(figure)


def write_report(summary: pd.DataFrame, features: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# P19 Experiment C: Preprocessing Harmonization", "", "## Branch decision", "", "`IMAGE_EQUIVALENT_ICA_NOT_REPRODUCIBLE_FOR_CONTINUOUS_ROBOT_DATA`", "",
             "C0 is a reproducible harmonized preprocessing baseline without ICA: raw BDF -> Image channel mapping -> CAR -> Image-configured 1–40 Hz fourth-order Butterworth IIR -> task-local 500-sample/2.0-s windows at 1.0-s step -> direct Image feature extraction. It is not claimed to be perfectly Image-matched.", "",
             "## Why C1 was not implemented", "", "The Image code uses MNE FastICA, seed 42, rank-aware component count, fit once on concatenated unbaselined valid Image stimulus epochs after CAR/filtering. For each Image epoch it then identifies positive ACC X/Y/Z-correlated components relative to that epoch's component-correlation mean plus 2 SD and high-pass filters those source signals at 3 Hz before reconstruction. This fit corpus and trial-specific decision rule depend on Image stimulus epochs; applying it to continuous Robot tasks would require inventing a non-equivalent segmentation and component-selection rule.", "",
             "## Representation comparison", "", "| Representation | Target | Task | Median nearest Image distance | % beyond Image reference max | Median selected |z| | 95th selected |z| | Mean HIGH score | % predicted HIGH |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.representation} | {row.target} | {row.task_label} | {row.nearest_image_distance_median:.3f} | {row.pct_beyond_image_reference_max:.1f} | {row.selected_feature_abs_z_median:.3f} | {row.selected_feature_abs_z_p95:.3f} | {row.high_score_mean:.3f} | {row.pct_predicted_high:.1f} |")
    lines.extend(["", "## Reproducibility artifact", "", "`P19_C0_harmonized_window_features.csv` is retained inside this Experiment C output only because the numeric comparison, frozen inference, and per-feature summary must be reproducible from the exact regenerated C0 representation. It is not a canonical derived Robot run.", "", "## Interpretation boundary", "", "Use domain geometry and feature distributions—not probability aesthetics—to assess whether C0 changes cross-context proximity. C1 is absent by scientific design; no ICA comparison is implied."])
    (output_dir / "P19_preprocessing_harmonization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    robot_config, image_config = read_yaml(args.robot_resolved_config), read_yaml(args.image_resolved_config)
    if not args.alignment_metadata.is_file(): raise FileNotFoundError(f"Alignment metadata missing: {args.alignment_metadata}")
    alignment = json.loads(args.alignment_metadata.read_text(encoding="utf-8")); raw_paths = resolve_raw_paths(args.raw_root, robot_config)
    validate_contract(image_config, alignment, raw_paths)
    for target in TARGETS:
        if not (args.inference_dir / f"P19_{target}_image_calibration_model.joblib").is_file(): raise FileNotFoundError("Frozen strict-transfer model missing")
    if args.dry_run:
        print("Dry-run validation passed: five Status-aligned raw BDFs, Image 1–40 Hz/CAR contract, existing alignment metadata, and frozen strict models are available.")
        print("C1 intentionally not implemented: IMAGE_EQUIVALENT_ICA_NOT_REPRODUCIBLE_FOR_CONTINUOUS_ROBOT_DATA")
        return 0
    existing = [args.output_dir / name for name in OUTPUTS if (args.output_dir / name).exists()]
    if existing and not args.overwrite: raise FileExistsError("Refusing to overwrite Experiment C outputs without --overwrite")
    primary_prediction = args.inference_dir / "P19_robot_window_predictions.csv"; prediction_hash = sha256(primary_prediction)
    alignment_by_task = {row["task"]: row for row in alignment["tasks"]}
    c0_parts = []
    for task in TASKS:
        metadata, features = preprocess_task_c0(raw_paths[task], task, image_config, float(alignment_by_task[task]["eeg_duration_s"]))
        c0_parts.append(pd.concat([metadata, features], axis=1))
    c0 = pd.concat(c0_parts, ignore_index=True)
    if list(c0[EEG_COLUMNS].columns) != EEG_COLUMNS or len(c0) != 337: raise RuntimeError("C0 schema/window count is incompatible with the approved five-task Robot analysis")
    _, existing_windows = prepare_robot_windows(args.existing_robot_features)
    if len(existing_windows) != len(c0) or set(existing_windows.window_id) != set(c0.window_id): raise RuntimeError("C0 task-local window grid does not exactly match existing approved Robot windows")
    table, _ = load_participant_table(args.derived_root, PARTICIPANT)
    image_data, models = {}, {}
    for target in TARGETS:
        features, labels, columns = prepare_modality_data(table, PARTICIPANT, target, "eeg")
        if columns != EEG_COLUMNS: raise RuntimeError("P19 Image EEG schema changed")
        image_data[target] = {"features": features, "labels": labels.to_numpy(int)}
        models[target] = joblib.load(args.inference_dir / f"P19_{target}_image_calibration_model.joblib")
    representations = {"A_existing_robot_features": existing_windows, "C0_reproducible_harmonized_no_ICA": c0}
    diagnostics = pd.concat([representation_diagnostics(name, frame, image_data, models) for name, frame in representations.items()], ignore_index=True)
    features = feature_summary(representations, image_data, models); summary = task_summary(diagnostics)
    for frame in (c0, diagnostics, features, summary):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(float)
        if not np.isfinite(numeric).all(): raise RuntimeError("Experiment C produced NaN or infinity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    c0.to_csv(args.output_dir / "P19_C0_harmonized_window_features.csv", index=False)
    diagnostics.to_csv(args.output_dir / "P19_preprocessing_harmonization_window_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "P19_preprocessing_harmonization_task_summary.csv", index=False)
    features.to_csv(args.output_dir / "P19_preprocessing_harmonization_feature_summary.csv", index=False)
    plot_distances(summary, args.output_dir); plot_problem_features(features, args.output_dir); plot_trajectories(diagnostics, args.output_dir)
    write_report(summary, features, args.output_dir)
    manifest = {"experiment": "C_preprocessing_harmonization", "participant": PARTICIPANT, "branch": "C0_reproducible_harmonized_no_ICA", "C1_status": "IMAGE_EQUIVALENT_ICA_NOT_REPRODUCIBLE_FOR_CONTINUOUS_ROBOT_DATA", "raw_bdf_inputs": {task: {"path": str(path), "sha256": sha256(path)} for task, path in raw_paths.items()}, "image_config_source": str(args.image_resolved_config), "robot_config_source": str(args.robot_resolved_config), "alignment_source": str(args.alignment_metadata), "existing_robot_features_read_only": str(args.existing_robot_features), "frozen_models_read_only": [str(args.inference_dir / f"P19_{target}_image_calibration_model.joblib") for target in TARGETS], "settings": {"sampling_hz": 250, "channels": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"], "reference": "CAR", "filter": "Butterworth IIR order 4, 1.0–40.0 Hz", "ica": "not applied", "window_samples": 500, "window_length_s": 2.0, "step_samples": 250, "step_s": 1.0}}
    (args.output_dir / "P19_preprocessing_harmonization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if sha256(primary_prediction) != prediction_hash: raise RuntimeError("Existing strict-transfer prediction CSV changed during Experiment C")
    print(f"P19 Experiment C complete: {args.output_dir}"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
