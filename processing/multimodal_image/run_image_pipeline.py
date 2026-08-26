#!/usr/bin/env python3
"""Safe, read-only-input multimodal Image Experiment pipeline."""
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/multimodal_image_matplotlib")

import cv2
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.alignment import build_alignment
from src.configuration import load_resolved_config
from src.eeg import preprocess_eeg
from src.eeg_sources import load_eeg_session
from src.export import write_json, write_rows
from src.features import extract_eeg_features
from src.health import eeg_health, markdown_report, ratings_health, synchronization_health, video_health
from src.validation import file_record, inspect_inputs, resolve_inputs
from src.video import create_crop_preview, process_video
from src.qc import (autoreject_rows, save_autoreject_plot,
                    save_event_related_band_power, save_fz_time_frequency)


def ensure_parent(path: Path) -> None:
    """Create an output branch only immediately before its first write."""
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML, or a deprecated single YAML.")
    parser.add_argument("--participant-config", type=Path, help="Participant YAML for layered configuration.")
    parser.add_argument("--pipeline-config", type=Path, default=HERE / "configs" / "pipeline" / "eeg_video_defaults.yaml")
    parser.add_argument("--participant", help="Optional consistency check against participant YAML.")
    parser.add_argument("--experiment", help="Optional consistency check against experiment YAML.")
    parser.add_argument("--resource-root", type=Path, default=HERE / "models", help="Directory containing the configured landmark-model filename.")
    parser.add_argument("--output-root", type=Path, default=Path("derived"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overwrite-run", action="store_true", help="Replace only the exact requested run directory; ignored by --dry-run.")
    parser.add_argument("--health-check-only", action="store_true", help="Run read-only EEG, ratings, video, log, and synchronization checks only.")
    parser.add_argument("--crop-preview-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preprocess-eeg", action="store_true")
    parser.add_argument("--preprocess-video", action="store_true")
    parser.add_argument("--extract-eeg-features", action="store_true")
    parser.add_argument("--extract-video-features", action="store_true")
    parser.add_argument("--reuse-video-features-from", type=Path,
                        help="Existing video_trial_features.csv to validate and reuse only when merging a new run.")
    parser.add_argument("--merge-modalities", action="store_true")
    parser.add_argument("--save-clean-epochs", action="store_true")
    parser.add_argument("--save-landmarks", action="store_true")
    parser.add_argument("--save-qc", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def make_run_dir(root: Path, config: dict, args: argparse.Namespace) -> Path:
    section = "health_checks" if args.health_check_only else "runs"
    return (root / config["participant"] / config["output"]["experiment_output_dir"] / section / args.run_name).resolve()


def validate_output_root(output_root: Path, root: Path, raw_root: Path) -> None:
    """Reject roots that could make a run replacement broader than intended."""
    if output_root == root:
        raise ValueError("--output-root must not be the repository root")
    if output_root == Path(output_root.anchor):
        raise ValueError("--output-root must not be a drive root")
    if output_root == raw_root or raw_root in output_root.parents or output_root in raw_root.parents:
        raise ValueError("--output-root must be outside data/ and must not contain it")


def validate_run_directory(output_root: Path, run_dir: Path, config: dict, args: argparse.Namespace) -> None:
    """Require the exact participant/experiment/mode/run-name location before replacement."""
    run_name = args.run_name
    if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise ValueError("--run-name must be a single non-empty directory name")
    section = "health_checks" if args.health_check_only else "runs"
    expected = (str(config["participant"]), str(config["output"]["experiment_output_dir"]), section, run_name)
    try:
        relative = run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Resolved run directory must be inside --output-root") from exc
    if relative.parts != expected:
        raise ValueError(
            "Resolved run directory does not match the requested participant, experiment, mode, and run name: "
            f"expected {expected}, got {relative.parts}"
        )


def replace_run_directory_if_requested(run_dir: Path, output_root: Path, config: dict, args: argparse.Namespace, logger) -> None:
    """Replace only an already validated exact run directory when explicitly requested."""
    validate_run_directory(output_root, run_dir, config, args)
    if not args.overwrite_run or not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"Requested run path exists but is not a directory: {run_dir}")
    logger.info("Replacing existing output directory: %s", run_dir)
    shutil.rmtree(run_dir)


def configure_logging(run_dir: Path | None, config: dict, level: str) -> logging.Logger:
    logger = logging.getLogger("multimodal_image")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(); console.setLevel(getattr(logging, level)); console.setFormatter(formatter); logger.addHandler(console)
    if run_dir is not None:
        ensure_parent(run_dir / "logs" / config["output"]["log_filename"])
        handler = logging.FileHandler(run_dir / "logs" / config["output"]["log_filename"], encoding="utf-8")
        handler.setLevel(logging.DEBUG); handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def package_versions() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name)
        for name in ("mne", "autoreject", "mediapipe", "numpy", "pandas", "scipy", "PyYAML")
    }
    versions["opencv_runtime"] = cv2.__version__
    for distribution in ("opencv-contrib-python", "opencv-python", "opencv-contrib-python-headless", "opencv-python-headless"):
        try:
            versions["opencv_distribution"] = f"{distribution}=={importlib.metadata.version(distribution)}"
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        versions["opencv_distribution"] = "metadata not found"
    return versions


def setup_run(run_dir: Path, config: dict, sources: list[str], args: argparse.Namespace, health_only: bool,
              paths: dict | None = None, reused_video_features: Path | None = None) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}. Choose a new --run-name.")
    run_dir.mkdir(parents=True, exist_ok=False)
    ensure_parent(run_dir / "config" / "resolved_configuration.yaml")
    with (run_dir / "config" / "resolved_configuration.yaml").open("x", encoding="utf-8") as handle: yaml.safe_dump(config, handle, sort_keys=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1], text=True, capture_output=True, check=False).stdout.strip() or None
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(map(str, sys.argv)), "configuration_sources": sources, "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "git_commit": git, "package_versions": package_versions(), "resolved_configuration": config, "flags": vars(args), "incomplete_session": bool(config.get("trials", {}).get("allow_incomplete_image_trials", False)), "input_sources": input_manifest(paths, config) if paths is not None else {}}
    if reused_video_features is not None:
        manifest["reused_video_feature_source"] = file_record(reused_video_features)
    write_json(run_dir / "manifest" / "run_manifest.json", manifest)


def input_manifest(paths: dict[str, Path], config: dict) -> dict:
    result = {}
    for role, path in paths.items():
        if role == "experiment_dir": continue
        if role == "eeg_files":
            _, _, metadata = load_eeg_session(paths, config, preload=False)
            result[role] = [{"role": "eeg_fragment", **file_record(item), **metadata[index]} for index, item in enumerate(path)]
            continue
        record = {"role": role, **file_record(path)}
        if role == "ratings": record["authoritative"] = bool(config["inputs"].get("authoritative_ratings", False))
        result[role] = record
    return result


def merge_tables(ratings_path: Path, eeg: pd.DataFrame | None, video: pd.DataFrame | None, config: dict | None = None) -> pd.DataFrame:
    ratings = pd.read_csv(ratings_path, encoding="utf-8-sig").rename(columns={"trigger_sent": "trigger"})
    selected = ratings[["trigger", "stim_id", "category", "valence_rating", "arousal_rating", "valence_rt", "arousal_rt"]].copy()
    if config and config.get("trials", {}).get("allow_incomplete_image_trials", False):
        available = set(map(int, config["trials"]["expected_available_triggers"]))
        selected = selected[selected["trigger"].isin(available)].copy()
    if eeg is not None:
        wide_eeg = eeg.pivot(index="trigger", columns="channel"); wide_eeg.columns = [f"{feature}__{channel}" for feature, channel in wide_eeg.columns]
        selected = selected.merge(wide_eeg.reset_index(), on="trigger", how="left", validate="one_to_one")
    if video is not None: selected = selected.merge(video, on="trigger", how="left", validate="one_to_one")
    return selected


def load_reused_video_features(path: Path) -> pd.DataFrame:
    """Load an existing Image video trial table without regenerating video artifacts."""
    if not path.is_file():
        raise FileNotFoundError(f"Reused video feature table does not exist: {path}")
    frame = pd.read_csv(path)
    video_columns = [column for column in frame if column.startswith("video_")]
    if "trigger" not in frame or not video_columns:
        raise ValueError("Reused video feature table must contain trigger and at least one video_ feature column")
    if frame.trigger.duplicated().any():
        raise ValueError("Reused video feature table has duplicate trigger values")
    return frame.loc[:, ["trigger", *video_columns]].copy()


def run_health_checks(paths: dict[str, Path], config: dict, run_dir: Path, logger) -> None:
    eeg, eeg_rows = eeg_health(paths, config, logger)
    ratings, ratings_rows = ratings_health(paths["ratings"], config, logger)
    video, video_rows = video_health(paths["video"], paths["vision_log"], config, logger)
    sync, pairs = synchronization_health(paths, config, logger)
    domains = {"eeg": eeg, "ratings": ratings, "video": video, "synchronization": sync}
    overall = "MANUAL_REVIEW" if any(item["status"] != "PASS" for item in domains.values()) else "PASS"
    summary = {"participant": config["participant"], "experiment": config["experiment"], "status": overall, **domains}
    prefix = config["participant"].lower()
    write_json(run_dir / f"{prefix}_health_summary.json", summary)
    write_json(run_dir / f"{prefix}_sync_summary.json", sync)
    write_rows(run_dir / f"{prefix}_sync_pairs.csv", pairs)
    status_rows = [{"domain": name, "name": "summary", "status": item["status"], "reasons": " | ".join(item["reasons"])} for name, item in domains.items()]
    status_rows.extend({"domain": row["domain"], "name": row["name"], "status": row["status"], "reasons": ""} for row in eeg_rows + ratings_rows + video_rows)
    write_rows(run_dir / f"{prefix}_health_summary.csv", status_rows)
    (run_dir / f"{prefix}_health_report.md").write_text(markdown_report(config["participant"], summary) + "\n", encoding="utf-8")
    logger.info("Health checks complete: %s", overall)


def main() -> int:
    args = parse_args()
    if args.verbose: args.log_level = "DEBUG"
    legacy_single = args.participant_config is None
    shared_path = args.config.resolve(strict=True) if legacy_single else args.pipeline_config.resolve(strict=True)
    experiment_path = args.config.resolve(strict=True)
    participant_path = args.participant_config.resolve(strict=True) if args.participant_config else None
    config, sources = load_resolved_config(shared_path, experiment_path, participant_path)
    if args.participant and args.participant != config["participant"]: raise ValueError("--participant does not match resolved participant configuration")
    if args.experiment and args.experiment != config["experiment"]: raise ValueError("--experiment does not match resolved experiment configuration")
    root = HERE.parents[1]
    resource_root = args.resource_root.resolve() if "resources" in config else None
    if resource_root is not None:
        # This is installation-specific rather than a scientific setting, but it
        # remains visible in the resolved run configuration and manifest.
        config["resources"]["resource_root"] = str(resource_root)
    paths = resolve_inputs(config, root, resource_root)
    reused_video_features = args.reuse_video_features_from.resolve() if args.reuse_video_features_from else None
    if reused_video_features is not None:
        if args.preprocess_video or args.extract_video_features:
            raise ValueError("--reuse-video-features-from cannot be combined with video preprocessing or extraction flags")
        if not args.merge_modalities:
            raise ValueError("--reuse-video-features-from requires --merge-modalities")
        load_reused_video_features(reused_video_features)
    output_root = args.output_root.resolve(); raw_root = (root / "data").resolve()
    validate_output_root(output_root, root, raw_root)
    run_dir = make_run_dir(output_root, config, args)
    if args.dry_run:
        logger = configure_logging(None, config, args.log_level)
        logger.info("DRY RUN: no files will be created")
        logger.info("Configuration sources: %s", sources)
        logger.info("Resolved configuration:\n%s", yaml.safe_dump(config, sort_keys=False))
        logger.info("Read-only inputs: %s", {key: str(value) for key, value in paths.items() if key != "experiment_dir"})
        if reused_video_features is not None: logger.info("Reused video feature source: %s", reused_video_features)
        logger.info("Planned output directory: %s", run_dir)
        return 0
    stage_flags = [args.crop_preview_only, args.validate_only, args.preprocess_eeg, args.preprocess_video, args.extract_eeg_features, args.extract_video_features, args.merge_modalities]
    if args.health_check_only and any(stage_flags): raise ValueError("--health-check-only cannot be combined with processing-stage flags")
    replacement_logger = configure_logging(None, config, args.log_level)
    replace_run_directory_if_requested(run_dir, output_root, config, args, replacement_logger)
    setup_run(run_dir, config, sources, args, args.health_check_only, paths, reused_video_features)
    logger = configure_logging(run_dir, config, args.log_level)
    logger.info("Activating %s %s pipeline", config["participant"], config["experiment"])
    logger.info("Configuration sources: %s", sources)
    logger.info("Raw inputs are read-only. Derived outputs: %s", run_dir)
    validation = inspect_inputs(paths, config)
    write_json(run_dir / "manifest" / "input_manifest.json", input_manifest(paths, config))
    write_json(run_dir / "manifest" / "validation_summary.json", validation)
    logger.info("Found %d matchable image trials", validation["matchable_image_triggers"])
    if args.health_check_only:
        run_health_checks(paths, config, run_dir, logger); return 0
    if args.crop_preview_only:
        create_crop_preview(paths, config, run_dir / "video" / "crop_preview", logger); return 0
    alignment_rows, alignment_model = build_alignment(paths, config)
    write_rows(run_dir / "alignment" / "trigger_alignment.csv", alignment_rows); write_json(run_dir / "alignment" / "alignment_model.json", alignment_model)
    if args.validate_only:
        logger.info("Validation-only run complete; no EEG or video preprocessing was performed"); return 0
    eeg_features = None
    video_features = load_reused_video_features(reused_video_features) if reused_video_features is not None else None
    if reused_video_features is not None: logger.info("Reusing existing video trial features: %s", reused_video_features)
    output_prefix = f"{config['participant'].lower()}_{config['experiment'].lower()}"
    if args.preprocess_eeg:
        epochs, qc, ica_table, reject_log, autoreject_events = preprocess_eeg(paths, config, logger)
        if args.save_qc:
            write_json(run_dir / "eeg" / "quality_control" / "eeg_qc.json", qc)
            qc_dir = run_dir / "eeg" / "quality_control"
            if len(ica_table):
                ensure_parent(qc_dir / "ica_motion_correlation.csv")
                ica_table.to_csv(qc_dir / "ica_motion_correlation.csv", index=False)
            if reject_log is not None:
                write_rows(qc_dir / "autoreject_epoch_channel_log.csv", autoreject_rows(reject_log, autoreject_events))
                save_autoreject_plot(reject_log, qc_dir / "autoreject_epoch_channel_log.png", config["participant"], args.run_name)
            missing = save_event_related_band_power(
                epochs, config, qc_dir / "event_related_band_power_by_category.png",
                qc_dir / "event_related_band_power_by_category.csv", config["participant"], args.run_name,
            )
            if missing:
                logger.warning("Spectral QC has no retained epochs for categories: %s", ", ".join(missing))
            save_fz_time_frequency(
                epochs, config, qc_dir / "event_related_time_frequency_Fz_by_category.png",
                config["participant"], args.run_name, logger,
            )
        if args.save_clean_epochs:
            ensure_parent(run_dir / "eeg" / "cleaned_epochs" / f"{output_prefix}_cleaned-epo.fif")
            epochs.save(run_dir / "eeg" / "cleaned_epochs" / f"{output_prefix}_cleaned-epo.fif", overwrite=False)
        if args.extract_eeg_features:
            eeg_features = extract_eeg_features(epochs, config)
            ensure_parent(run_dir / "eeg" / "features" / "eeg_epoch_features.csv")
            eeg_features.to_csv(run_dir / "eeg" / "features" / "eeg_epoch_features.csv", index=False)
    elif args.extract_eeg_features: raise ValueError("--extract-eeg-features requires --preprocess-eeg")
    if args.preprocess_video:
        if not args.save_landmarks: raise ValueError("--preprocess-video requires --save-landmarks")
        frames, video_features = process_video(paths, config, run_dir / "video" / "landmarks" / f"{output_prefix}_landmarks.npz", logger)
        ensure_parent(run_dir / "video" / "features" / "video_frame_features.csv")
        frames.to_csv(run_dir / "video" / "features" / "video_frame_features.csv", index=False)
        if args.extract_video_features:
            ensure_parent(run_dir / "video" / "features" / "video_trial_features.csv")
            video_features.to_csv(run_dir / "video" / "features" / "video_trial_features.csv", index=False)
    elif args.extract_video_features: raise ValueError("--extract-video-features requires --preprocess-video")
    if args.merge_modalities:
        merged = merge_tables(paths["ratings"], eeg_features, video_features, config)
        ensure_parent(run_dir / "merged" / f"{output_prefix}_trial_dataset.csv")
        merged.to_csv(run_dir / "merged" / f"{output_prefix}_trial_dataset.csv", index=False)
        logger.info("Wrote merged table with %d trial rows", len(merged))
    logger.info("Pipeline completed successfully"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
