#!/usr/bin/env python3
"""Safe, read-only-input multimodal Image Experiment pipeline."""
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/multimodal_image_matplotlib")

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.alignment import build_alignment
from src.configuration import load_resolved_config
from src.eeg import preprocess_eeg
from src.export import write_json, write_rows
from src.features import extract_eeg_features
from src.health import eeg_health, markdown_report, ratings_health, synchronization_health, video_health
from src.validation import file_record, inspect_inputs, resolve_inputs
from src.video import create_crop_preview, process_video


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
    parser.add_argument("--health-check-only", action="store_true", help="Run read-only EEG, ratings, video, log, and synchronization checks only.")
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


def make_run_dir(root: Path, config: dict, args: argparse.Namespace) -> Path:
    section = "health_checks" if args.health_check_only else "runs"
    return (root / config["participant"] / config["output"]["experiment_output_dir"] / section / args.run_name).resolve()


def configure_logging(run_dir: Path | None, config: dict, level: str) -> logging.Logger:
    logger = logging.getLogger("multimodal_image")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(); console.setLevel(getattr(logging, level)); console.setFormatter(formatter); logger.addHandler(console)
    if run_dir is not None:
        handler = logging.FileHandler(run_dir / "logs" / config["output"]["log_filename"], encoding="utf-8")
        handler.setLevel(logging.DEBUG); handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("mne", "autoreject", "mediapipe", "opencv-python", "numpy", "pandas", "scipy", "PyYAML")}


def setup_run(run_dir: Path, config: dict, sources: list[str], args: argparse.Namespace, health_only: bool) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}. Choose a new --run-name.")
    folders = ("manifest", "config", "logs") if health_only else ("manifest", "config", "eeg/cleaned_epochs", "eeg/features", "eeg/quality_control", "video/crop_preview", "video/landmarks", "video/features", "video/quality_control", "alignment", "merged", "logs")
    for folder in folders: (run_dir / folder).mkdir(parents=True, exist_ok=False)
    with (run_dir / "config" / "resolved_configuration.yaml").open("x", encoding="utf-8") as handle: yaml.safe_dump(config, handle, sort_keys=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1], text=True, capture_output=True, check=False).stdout.strip() or None
    write_json(run_dir / "manifest" / "run_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(map(str, sys.argv)), "configuration_sources": sources, "python": sys.executable, "python_version": sys.version, "platform": platform.platform(), "git_commit": git, "package_versions": package_versions(), "resolved_configuration": config, "flags": vars(args)})


def input_manifest(paths: dict[str, Path], config: dict) -> dict:
    result = {}
    for role, path in paths.items():
        if role == "experiment_dir": continue
        record = {"role": role, **file_record(path)}
        if role == "ratings": record["authoritative"] = bool(config["inputs"].get("authoritative_ratings", False))
        result[role] = record
    return result


def merge_tables(ratings_path: Path, eeg: pd.DataFrame | None, video: pd.DataFrame | None) -> pd.DataFrame:
    ratings = pd.read_csv(ratings_path, encoding="utf-8-sig").rename(columns={"trigger_sent": "trigger"})
    selected = ratings[["trigger", "stim_id", "category", "valence_rating", "arousal_rating", "valence_rt", "arousal_rt"]].copy()
    if eeg is not None:
        wide_eeg = eeg.pivot(index="trigger", columns="channel"); wide_eeg.columns = [f"{feature}__{channel}" for feature, channel in wide_eeg.columns]
        selected = selected.merge(wide_eeg.reset_index(), on="trigger", how="left", validate="one_to_one")
    if video is not None: selected = selected.merge(video, on="trigger", how="left", validate="one_to_one")
    return selected


def run_health_checks(paths: dict[str, Path], config: dict, run_dir: Path, logger) -> None:
    eeg, eeg_rows = eeg_health(paths["eeg"], config, logger)
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
    output_root = args.output_root.resolve(); raw_root = (root / "data").resolve()
    if output_root == raw_root or raw_root in output_root.parents: raise ValueError("--output-root must be outside data/")
    run_dir = make_run_dir(output_root, config, args)
    if args.dry_run:
        logger = configure_logging(None, config, args.log_level)
        logger.info("DRY RUN: no files will be created")
        logger.info("Configuration sources: %s", sources)
        logger.info("Resolved configuration:\n%s", yaml.safe_dump(config, sort_keys=False))
        logger.info("Read-only inputs: %s", {key: str(value) for key, value in paths.items() if key != "experiment_dir"})
        logger.info("Planned output directory: %s", run_dir)
        return 0
    stage_flags = [args.crop_preview_only, args.validate_only, args.preprocess_eeg, args.preprocess_video, args.extract_eeg_features, args.extract_video_features, args.merge_modalities]
    if args.health_check_only and any(stage_flags): raise ValueError("--health-check-only cannot be combined with processing-stage flags")
    setup_run(run_dir, config, sources, args, args.health_check_only)
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
    eeg_features = video_features = None
    output_prefix = f"{config['participant'].lower()}_{config['experiment'].lower()}"
    if args.preprocess_eeg:
        epochs, qc, ica_table = preprocess_eeg(paths["eeg"], config, logger)
        if args.save_qc:
            write_json(run_dir / "eeg" / "quality_control" / "eeg_qc.json", qc)
            if len(ica_table): ica_table.to_csv(run_dir / "eeg" / "quality_control" / "ica_motion_correlation.csv", index=False)
        if args.save_clean_epochs: epochs.save(run_dir / "eeg" / "cleaned_epochs" / f"{output_prefix}_cleaned-epo.fif", overwrite=False)
        if args.extract_eeg_features:
            eeg_features = extract_eeg_features(epochs, config); eeg_features.to_csv(run_dir / "eeg" / "features" / "eeg_epoch_features.csv", index=False)
    elif args.extract_eeg_features: raise ValueError("--extract-eeg-features requires --preprocess-eeg")
    if args.preprocess_video:
        if not args.save_landmarks: raise ValueError("--preprocess-video requires --save-landmarks")
        frames, video_features = process_video(paths, config, run_dir / "video" / "landmarks" / f"{output_prefix}_landmarks.npz", logger)
        frames.to_csv(run_dir / "video" / "features" / "video_frame_features.csv", index=False)
        if args.extract_video_features: video_features.to_csv(run_dir / "video" / "features" / "video_trial_features.csv", index=False)
    elif args.extract_video_features: raise ValueError("--extract-video-features requires --preprocess-video")
    if args.merge_modalities:
        merged = merge_tables(paths["ratings"], eeg_features, video_features); merged.to_csv(run_dir / "merged" / f"{output_prefix}_trial_dataset.csv", index=False)
        logger.info("Wrote merged table with %d trial rows", len(merged))
    logger.info("Pipeline completed successfully"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
