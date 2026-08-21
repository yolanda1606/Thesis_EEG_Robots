#!/usr/bin/env python3
"""Run the existing P19 Image pipeline with its ICA stage explicitly disabled.

This wrapper intentionally delegates all processing to run_image_pipeline.py.
It requests only EEG stages plus the ratings/EEG merged table: face/video
artifacts are not regenerated because no Image classification is run here and
the EEG-only classification schema does not require them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from compare_p19_no_ica import compare_runs


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RUN_NAME = "p19_no_ica"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate the existing pipeline configuration without creating a run.")
    parser.add_argument("--overwrite-run", action="store_true", help="Replace only derived/P19/Image_Experiment/runs/p19_no_ica.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser.parse_args(argv)


def pipeline_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(HERE / "run_image_pipeline.py"),
               "--config", str(HERE / "configs/experiments/image_experiment_no_ica.yaml"),
               "--participant-config", str(HERE / "configs/participants/P19.yaml"),
               "--pipeline-config", str(HERE / "configs/pipeline/eeg_video_defaults.yaml"),
               "--participant", "P19", "--experiment", "image", "--output-root", str(PROJECT_ROOT / "derived"),
               "--run-name", RUN_NAME, "--preprocess-eeg", "--extract-eeg-features", "--merge-modalities",
               "--save-clean-epochs", "--save-qc", "--log-level", args.log_level]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite_run:
        command.append("--overwrite-run")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = subprocess.run(pipeline_command(args), cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        return int(result.returncode)
    if args.dry_run:
        return 0
    passed, messages = compare_runs(PROJECT_ROOT / "derived/P19/Image_Experiment/runs/p19_final",
                                    PROJECT_ROOT / "derived/P19/Image_Experiment/runs/p19_no_ica")
    print("P19 no-ICA comparability check: " + ("PASS" if passed else "INCLUSION_OR_SCHEMA_DIFFERENCE"))
    print("\n".join(messages))
    if not passed:
        print("No classification was run. Review the retained-trigger/schema difference before any downstream use.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
