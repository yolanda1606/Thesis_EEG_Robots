#!/usr/bin/env python3
"""Safe, P01-first multimodal Image Experiment pipeline.

All raw files are opened only for reading. Every non-dry-run output is placed in
one new derived run directory; an existing run directory is always an error.
"""
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

# Must be set before imports that otherwise try to write under the user's home.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/multimodal_image_matplotlib")

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.alignment import build_alignment
from src.eeg import preprocess_eeg
from src.export import write_json, write_rows
from src.features import extract_eeg_features
from src.validation import file_record, inspect_inputs, resolve_inputs
from src.video import create_crop_preview, process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--experiment", required=True, choices=["image"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("derived"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--crop-preview-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preprocess-eeg", action="store_true")
    parser.add_argument("--preprocess-video", action="store_true")
    parser.add_argument("--extract-eeg-features", action="store_true")
    parser.add_argument("--extract-video-features", action="store_true")
    parser.add_argument("--merge-modalities", action="store_true")
    parser.add_argument("--save-clean-epochs", action="store_true")
    parser.add_argument("--save-landmarks", action="store_true")
    parser.add_argument("--save-qc", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def make_run_dir(root: Path, args: argparse.Namespace) -> Path:
    return (root / args.participant / "Image_Experiment" / "runs" / args.run_name).resolve()


def configure_logging(run_dir: Path | None, level: str) -> logging.Logger:
    logger = logging.getLogger("multimodal_image")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level))
    console.setFormatter(formatter)
    logger.addHandler(console)
    if run_dir is not None:
        handler = logging.FileHandler(run_dir / "logs" / "pipeline.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_run(run_dir: Path, config_path: Path, config: dict, args: argparse.Namespace) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}. Choose a new --run-name.")
    for folder in ("manifest", "config", "eeg/cleaned_epochs", "eeg/features", "eeg/quality_control", "video/crop_preview", "video/landmarks", "video/features", "video/quality_control", "alignment", "merged", "logs"):
        (run_dir / folder).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config" / config_path.name)
    command = " ".join(map(str, sys.argv))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1], text=True, capture_output=True, check=False).stdout.strip() or None
    versions = {name: importlib.metadata.version(name) for name in ("mne", "autoreject", "mediapipe", "opencv-python", "numpy", "pandas", "scipy", "PyYAML")}
    write_json(run_dir / "manifest" / "run_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "git_commit": git,
        "package_versions": versions, "configuration": config, "flags": vars(args)})


def merge_tables(ratings_path: Path, eeg: pd.DataFrame | None, video: pd.DataFrame | None) -> pd.DataFrame:
    ratings = pd.read_csv(ratings_path, encoding="utf-8-sig").rename(columns={"trigger_sent": "trigger"})
    selected = ratings[["trigger", "stim_id", "category", "valence_rating", "arousal_rating", "valence_rt", "arousal_rt"]].copy()
    if eeg is not None:
        wide_eeg = eeg.pivot(index="trigger", columns="channel")
        wide_eeg.columns = [f"{feature}__{channel}" for feature, channel in wide_eeg.columns]
        selected = selected.merge(wide_eeg.reset_index(), on="trigger", how="left", validate="one_to_one")
    if video is not None:
        selected = selected.merge(video, on="trigger", how="left", validate="one_to_one")
    return selected


def main() -> int:
    args = parse_args()
    if args.verbose:
        args.log_level = "DEBUG"
    config_path = args.config.resolve(strict=True)
    config = load_config(config_path)
    if args.participant != config["participant"] or args.experiment != config["experiment"]:
        raise ValueError("--participant and --experiment must match the approved configuration")
    root = HERE.parents[1]
    paths = resolve_inputs(config, root)
    output_root = args.output_root.resolve()
    raw_root = (root / "data").resolve()
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("--output-root must be outside the raw data directory")
    run_dir = make_run_dir(output_root, args)
    if args.dry_run:
        logger = configure_logging(None, args.log_level)
        logger.info("DRY RUN: no files will be created")
        logger.info("Raw inputs will be read-only: %s", {key: str(value) for key, value in paths.items() if key != "experiment_dir"})
        logger.info("Planned output directory: %s", run_dir)
        logger.info("Requested stages: preview=%s validate=%s eeg=%s video=%s eeg_features=%s video_features=%s merge=%s", args.crop_preview_only, args.validate_only, args.preprocess_eeg, args.preprocess_video, args.extract_eeg_features, args.extract_video_features, args.merge_modalities)
        return 0

    setup_run(run_dir, config_path, config, args)
    logger = configure_logging(run_dir, args.log_level)
    logger.info("Activating P01 Image Experiment pipeline")
    logger.info("Raw inputs are read-only. Derived outputs: %s", run_dir)
    summary = inspect_inputs(paths, config)
    write_json(run_dir / "manifest" / "input_manifest.json", {key: file_record(value) for key, value in paths.items() if key != "experiment_dir"})
    write_json(run_dir / "manifest" / "validation_summary.json", summary)
    logger.info("Found %d matchable image trials", summary["matchable_image_triggers"])
    if args.crop_preview_only:
        create_crop_preview(paths, config, run_dir / "video" / "crop_preview", logger)
        logger.info("Crop preview complete. Full video processing remains blocked until crop approval.")
        return 0
    alignment_rows, alignment_model = build_alignment(paths, config)
    write_rows(run_dir / "alignment" / "trigger_alignment.csv", alignment_rows)
    write_json(run_dir / "alignment" / "alignment_model.json", alignment_model)
    if args.validate_only:
        logger.info("Validation-only run complete; no EEG or video preprocessing was performed")
        return 0

    eeg_features = None
    video_features = None
    if args.preprocess_eeg:
        epochs, qc, ica_table = preprocess_eeg(paths["eeg"], config, logger)
        if args.save_qc:
            write_json(run_dir / "eeg" / "quality_control" / "eeg_qc.json", qc)
            if len(ica_table):
                ica_table.to_csv(run_dir / "eeg" / "quality_control" / "ica_motion_correlation.csv", index=False)
        if args.save_clean_epochs:
            epochs.save(run_dir / "eeg" / "cleaned_epochs" / "p01_image_cleaned-epo.fif", overwrite=False)
        if args.extract_eeg_features:
            eeg_features = extract_eeg_features(epochs, config)
            eeg_features.to_csv(run_dir / "eeg" / "features" / "eeg_epoch_features.csv", index=False)
    elif args.extract_eeg_features:
        raise ValueError("--extract-eeg-features requires --preprocess-eeg in the same run")

    if args.preprocess_video:
        if not args.save_landmarks:
            raise ValueError("--preprocess-video requires --save-landmarks to preserve raw derived landmarks")
        frames, video_features = process_video(paths, config, run_dir / "video" / "landmarks" / "p01_image_landmarks.npz", logger)
        frames.to_csv(run_dir / "video" / "features" / "video_frame_features.csv", index=False)
        if args.extract_video_features:
            video_features.to_csv(run_dir / "video" / "features" / "video_trial_features.csv", index=False)
    elif args.extract_video_features:
        raise ValueError("--extract-video-features requires --preprocess-video in the same run")

    if args.merge_modalities:
        merged = merge_tables(paths["ratings"], eeg_features, video_features)
        merged.to_csv(run_dir / "merged" / "p01_image_trial_dataset.csv", index=False)
        logger.info("Wrote merged table with %d trial rows", len(merged))
    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
