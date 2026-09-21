#!/usr/bin/env python3
"""Serial, resumable orchestration for the isolated causal Robot replay.

This module deliberately does not implement EEG processing or inference.  It
resolves each participant's established inputs and starts the existing
single-participant replay in a separate Python process.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.multimodal_robot.analysis.causal_streaming import run_causal_robot_replay as replay
from processing.multimodal_robot.prep.continuous import _files
from processing.multimodal_robot.run_robot_pipeline import load, localize_task_paths


PARTICIPANTS = tuple(f"P{number:02d}" for number in range(1, 47))
TASKS = replay.TASKS
REQUIRED_OUTPUT_FILES = (
    "causal_run_manifest.json",
    "causal_filter_design.json",
    "causal_window_predictions.csv",
    "causal_window_consensus.csv",
    "causal_window_latency.csv",
    "causal_stream_timing_chunks.csv",
    "canonical_replay_validation.csv",
    "canonical_causal_feature_agreement.csv",
    "canonical_causal_probability_agreement.csv",
    "canonical_causal_consensus_agreement.csv",
    "canonical_causal_task_consensus.csv",
    "canonical_causal_participant_target_metrics.csv",
)


def participant_paths(participant: str, output_root: Path, raw_root: Path) -> dict[str, Path]:
    """Return only the established participant-specific replay inputs."""
    lower = participant.lower()
    return {
        "participant_config": ROOT / "processing/multimodal_robot/configs/participants" / f"{participant}.yaml",
        "raw_root": raw_root,
        "canonical_eeg_csv": ROOT / "derived" / participant / "Robot_Experiment/runs" / f"{lower}_robot_final_features/features/eeg_window_features.csv",
        "canonical_predictions_csv": ROOT / "outputs/robot_transfer/eeg_only" / participant / f"{participant}_window_predictions.csv",
        "canonical_consensus_csv": ROOT / "outputs/robot_transfer/eeg_only" / participant / f"{participant}_window_consensus.csv",
        "image_reference_csv": ROOT / "derived" / participant / "Image_Experiment/runs" / f"{lower}_no_ica/merged/{lower}_image_trial_dataset.csv",
        "output_dir": output_root / participant,
    }


def output_complete(output_dir: Path) -> bool:
    """A run is resumable only after every required final artifact is present."""
    return output_dir.is_dir() and all((output_dir / name).is_file() for name in REQUIRED_OUTPUT_FILES)


def selected_participants(values: Iterable[str] | None, all_participants: bool) -> tuple[str, ...]:
    if all_participants:
        if values:
            raise ValueError("Use either --all or --participants, not both")
        return PARTICIPANTS
    if not values:
        raise ValueError("Supply --participants P01 ... or --all")
    result = tuple(values)
    invalid = [value for value in result if value not in PARTICIPANTS]
    if invalid:
        raise ValueError(f"Unknown participant IDs: {', '.join(invalid)}")
    if len(set(result)) != len(result):
        raise ValueError("Participant IDs must not be repeated")
    return result


def validate_inputs(participant: str, paths: dict[str, Path]) -> list[Path]:
    """Read configs and check all replay inputs, without EEG processing/inference."""
    missing = [name for name in ("participant_config", "canonical_eeg_csv", "canonical_predictions_csv", "canonical_consensus_csv", "image_reference_csv") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(f"{participant}: missing required inputs: {', '.join(missing)}")
    config = localize_task_paths(load(paths["participant_config"]), paths["raw_root"])
    if config.get("participant") != participant:
        raise ValueError(f"{participant}: config identifies {config.get('participant')!r}")
    eeg_sources: list[Path] = []
    for task in TASKS:
        if task not in config.get("tasks", {}):
            raise ValueError(f"{participant}: config is missing task {task}")
        sources = [Path(path) for path in _files(config["tasks"][task]["eeg"])]
        if not sources:
            raise ValueError(f"{participant}: {task} has no configured EEG source")
        eeg_sources.extend(sources)
    absent_sources = [str(path) for path in eeg_sources if not path.is_file()]
    if absent_sources:
        raise FileNotFoundError(f"{participant}: configured EEG sources are missing: {', '.join(absent_sources)}")
    # This is the canonical frozen-loader validation, with no prediction call.
    replay.transfer.load_frozen_models(replay.model_args(participant))
    return eeg_sources


def runner_command(participant: str, paths: dict[str, Path]) -> list[str]:
    """Build an explicit invocation; never rely on P27 defaults in the runner."""
    return [
        sys.executable, str(Path(replay.__file__).resolve()),
        "--participant", participant,
        "--participant-config", str(paths["participant_config"]),
        "--raw-root", str(paths["raw_root"]),
        "--tasks", *TASKS,
        "--chunk-ms", "100",
        "--canonical-validation-windows", "3",
        "--output-dir", str(paths["output_dir"]),
        "--canonical-eeg-csv", str(paths["canonical_eeg_csv"]),
        "--canonical-predictions-csv", str(paths["canonical_predictions_csv"]),
        "--canonical-consensus-csv", str(paths["canonical_consensus_csv"]),
        "--image-reference-csv", str(paths["image_reference_csv"]),
    ]


def archive_existing(output_dir: Path) -> Path:
    """Preserve an old run before --force starts a replacement at its stable path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = output_dir.with_name(f"{output_dir.name}.archived_{stamp}")
    suffix = 1
    while archived.exists():
        archived = output_dir.with_name(f"{output_dir.name}.archived_{stamp}_{suffix}")
        suffix += 1
    shutil.move(str(output_dir), str(archived))
    return archived


def write_failure_log(output_root: Path, participant: str, command: list[str], error: Exception, started_at: datetime, ended_at: datetime) -> Path:
    """Persist an unabridged subprocess failure record for later diagnosis."""
    log_dir = output_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    stdout = getattr(error, "stdout", None) or getattr(error, "output", None) or ""
    stderr = getattr(error, "stderr", None) or ""
    return_code = getattr(error, "returncode", "")
    path = log_dir / f"{participant}.log"
    path.write_text(
        f"participant: {participant}\nstarted_utc: {started_at.isoformat()}\nended_utc: {ended_at.isoformat()}\n"
        f"return_code: {return_code}\ncommand: {shlex.join(command)}\n\n"
        f"exception:\n{error!r}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}\n",
        encoding="utf-8",
    )
    return path


def concise_failure_reason(error: Exception) -> str:
    """Keep the status table readable while the full traceback remains in logs."""
    stderr = str(getattr(error, "stderr", "") or "")
    detail = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")
    prefix = f"return code {getattr(error, 'returncode', 'n/a')}: "
    text = prefix + (detail or str(error))
    return text[:1000]


def write_status(output_root: Path, rows: list[dict[str, object]]) -> None:
    """Merge a subset rerun into, rather than replace, cohort-wide status history."""
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ["participant", "status", "started_utc", "ended_utc", "elapsed_seconds", "output_directory", "return_code", "log_file", "error_message"]
    path = output_root / "cohort_run_status.csv"
    existing: dict[str, dict[str, object]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["participant"]: row for row in csv.DictReader(handle)}
    existing.update({str(row["participant"]): row for row in rows})
    with (output_root / "cohort_run_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for _, row in sorted(existing.items())])


def run_participants(participants: tuple[str, ...], output_root: Path, raw_root: Path, *, dry_run: bool, force: bool,
                     execute: Callable[..., object] = subprocess.run, writer: Callable[[str], None] = print) -> list[dict[str, object]]:
    """Run serially and retain failure records while continuing the cohort."""
    rows: list[dict[str, object]] = []
    for index, participant in enumerate(participants, start=1):
        paths = participant_paths(participant, output_root, raw_root)
        started_at = datetime.now(timezone.utc); elapsed_start = perf_counter()
        command = runner_command(participant, paths)
        base = {"participant": participant, "started_utc": started_at.isoformat(), "output_directory": str(paths["output_dir"]), "return_code": "", "log_file": "", "error_message": ""}
        try:
            sources = validate_inputs(participant, paths)
            source_text = ", ".join(str(path) for path in sources)
            if dry_run:
                writer(f"[{index}/{len(participants)}] {participant} dry-run OK: {len(sources)} EEG source(s); output={paths['output_dir']}")
                writer(f"[{index}/{len(participants)}] {participant} EEG: {source_text}")
                status = "skipped_existing" if output_complete(paths["output_dir"]) and not force else "success"
            elif output_complete(paths["output_dir"]) and not force:
                writer(f"[{index}/{len(participants)}] {participant} skipped_existing")
                status = "skipped_existing"
            else:
                if paths["output_dir"].exists():
                    if not force:
                        raise FileExistsError(f"{participant}: incomplete output directory exists; use --force to archive and rerun: {paths['output_dir']}")
                    archived = archive_existing(paths["output_dir"])
                    writer(f"[{index}/{len(participants)}] {participant} archived existing output to {archived}")
                writer(f"[{index}/{len(participants)}] {participant} starting")
                execute(command, cwd=ROOT, check=True, capture_output=True, text=True)
                writer(f"[{index}/{len(participants)}] {participant} success")
                status = "success"
        except Exception as error:  # Record and continue; errors remain visible in status/log output.
            ended_at = datetime.now(timezone.utc)
            log = write_failure_log(output_root, participant, command, error, started_at, ended_at)
            base["return_code"] = getattr(error, "returncode", "")
            base["log_file"] = str(log)
            base["error_message"] = concise_failure_reason(error)
            writer(f"[{index}/{len(participants)}] {participant} failed: {base['error_message']} (log: {log})")
            status = "failed"
        else:
            ended_at = datetime.now(timezone.utc)
        rows.append(base | {"status": status, "ended_utc": ended_at.isoformat(), "elapsed_seconds": round(perf_counter() - elapsed_start, 3)})
        write_status(output_root, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participants", nargs="+", metavar="PXX")
    group.add_argument("--all", action="store_true", help="Run P01 through P46 serially.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/robot_transfer/causal_filter_comparison/v1/cohort")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/assets only; do not start replay.")
    parser.add_argument("--force", action="store_true", help="Archive an existing participant output before rerunning it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    participants = selected_participants(args.participants, args.all)
    rows = run_participants(participants, args.output_root.resolve(), args.raw_root.resolve(), dry_run=args.dry_run, force=args.force)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
