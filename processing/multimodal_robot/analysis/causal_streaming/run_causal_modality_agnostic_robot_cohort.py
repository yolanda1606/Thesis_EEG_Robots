#!/usr/bin/env python3
"""Serial cohort orchestration for fixed-offset causal multimodal replay.

Dry-run reads configuration, canonical alignment artifacts, and frozen model
assets only. It never opens a BDF/video stream and never calls inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.multimodal_robot.analysis.causal_streaming import run_causal_modality_agnostic_robot_replay as replay
from processing.multimodal_robot.prep.continuous import _files
from processing.multimodal_robot.run_robot_pipeline import load, localize_task_paths


PARTICIPANTS = tuple(f"P{number:02d}" for number in range(1, 47))
TASKS = replay.TASKS
REQUIRED_OUTPUT_FILES = (
    "run_manifest.json", "calibration_offsets.csv", "calibration_anchors.csv",
    "synchronization_error.csv", "model_window_comparison.csv",
    "causal_window_predictions.csv", "causal_window_consensus.csv",
    "top3_window_completeness.csv", "window_consensus_comparison.csv",
    "task_consensus_comparison.csv", "latency_by_window.csv",
    "latency_summary.csv", "latency_budget_summary.csv", "causal_filter_chunk_timing.csv", "excluded_cases.csv",
)


def participant_paths(participant: str, output_root: Path, raw_root: Path) -> dict[str, Path]:
    """Resolve established participant inputs and the isolated cohort output."""
    paths = replay.participant_paths(participant)
    return paths | {"raw_root": raw_root, "output_dir": output_root / participant}


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


def output_complete(output_dir: Path) -> bool:
    return output_dir.is_dir() and all((output_dir / name).is_file() for name in REQUIRED_OUTPUT_FILES)


def task_eligibility_summary(cases: pd.DataFrame) -> tuple[int, dict[str, int]]:
    """Collapse segment detail into mutually exclusive participant-task outcomes."""
    eligible_count = 0
    reasons = {"canonical_unavailable": 0, "no_status_alignment": 0, "no_early_anchor": 0,
               "missing_source": 0, "other": 0}
    for _, part in cases.groupby(["participant", "task"], sort=False):
        if bool(part["eligible"].any()):
            eligible_count += 1; continue
        text = part["reason"].fillna("").astype(str)
        canonical = part.loc[~text.str.contains("canonical_unavailable", regex=False)]
        if canonical.empty:
            reasons["canonical_unavailable"] += 1; continue
        canonical_text = canonical["reason"].fillna("").astype(str)
        status = canonical.loc[~canonical_text.str.contains("no_status_alignment", regex=False)]
        if status.empty:
            reasons["no_status_alignment"] += 1; continue
        status_text = status["reason"].fillna("").astype(str)
        early = status.loc[~status_text.str.contains("no_early_anchor", regex=False)]
        if early.empty:
            reasons["no_early_anchor"] += 1; continue
        if early["reason"].fillna("").astype(str).str.contains("missing_source", regex=False).all():
            reasons["missing_source"] += 1
        else:
            reasons["other"] += 1
    return eligible_count, reasons


def _anchor_count(anchors: pd.DataFrame, task: str, segment_id: int, segmented: bool) -> int:
    part = anchors.loc[anchors["task"].eq(task)].copy()
    if "segment_id" in part:
        ids = pd.to_numeric(part["segment_id"], errors="coerce")
        part = part.loc[ids.eq(segment_id)] if segmented else part.loc[ids.isna() | ids.eq(segment_id)]
    times = pd.to_numeric(part.get("vision_time_s", pd.Series(dtype=float)), errors="coerce")
    return int(times.le(replay.CALIBRATION_SECONDS + 1e-12).sum())


def preflight_participant(participant: str, paths: dict[str, Path]) -> dict[str, object]:
    """Audit eligibility and frozen assets without processing EEG/video."""
    required = {
        "participant_config": paths["participant_config"],
        "canonical_predictions": paths["canonical_root"] / f"{participant}_window_predictions.csv",
        "canonical_consensus": paths["canonical_root"] / f"{participant}_window_consensus.csv",
        "canonical_tasks": paths["canonical_root"] / f"{participant}_task_consensus_analysis.csv",
        "image_reference": paths["image_reference_csv"],
        "alignment_metadata": paths["alignment_metadata"],
        "synchronization_anchors": paths["synchronization_anchors"],
    }
    missing_artifacts = [name for name, path in required.items() if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(f"{participant}: missing required artifacts: {', '.join(missing_artifacts)}")
    config = localize_task_paths(load(paths["participant_config"]), paths["raw_root"])
    if config.get("participant") != participant:
        raise ValueError(f"{participant}: config identifies {config.get('participant')!r}")
    predictions = pd.read_csv(required["canonical_predictions"], usecols=["task", "segment_id"])
    canonical_cases = set(zip(predictions.task, pd.to_numeric(predictions.segment_id, errors="coerce").fillna(-1).astype(int)))
    anchors = pd.read_csv(required["synchronization_anchors"])
    models = replay.transfer.load_models(replay.model_args(participant))
    routes = sorted({str(model["modality"]) for model in models})
    cases: list[dict[str, object]] = []
    for task_name in TASKS:
        if task_name not in config.get("tasks", {}):
            cases.append({"participant": participant, "task": task_name, "segment_id": 1,
                          "eligible": False, "reason": "missing_source:task_config", "early_anchor_count": 0})
            continue
        task = config["tasks"][task_name]
        segments = replay.configured_segments(task)
        if not segments:
            segments = [(1, None)]
        alignments = {int(item.get("segment_id", 1)): item for item in replay.canonical_alignments(task_name, paths["alignment_metadata"])}
        video_files = [Path(path) for path in _files(task.get("video"))]
        log_files = [Path(path) for path in _files(task.get("vision_log"))]
        video_path = video_files[0] if video_files else None
        log_path = log_files[0] if log_files else None
        segmented = len(segments) > 1 or any(int(item.get("segment_id", 1)) != 1 for item in alignments.values())
        for segment_id, eeg_path in segments:
            canonical = alignments.get(segment_id)
            reasons: list[str] = []
            if canonical is None or not bool(canonical.get("accepted", True)) or (task_name, segment_id) not in canonical_cases:
                reasons.append("canonical_unavailable")
            if canonical is None or not str(canonical.get("method", "")).startswith("status_"):
                reasons.append("no_status_alignment")
            early_count = _anchor_count(anchors, task_name, segment_id, segmented)
            if early_count == 0:
                reasons.append("no_early_anchor")
            missing_sources = [label for label, path in (("EEG", eeg_path), ("video", video_path), ("vision_log", log_path))
                               if path is None or not path.is_file()]
            if missing_sources:
                reasons.append("missing_source:" + ",".join(missing_sources))
            cases.append({"participant": participant, "task": task_name, "segment_id": segment_id,
                          "eligible": not reasons, "reason": ";".join(reasons),
                          "canonical_alignment_type": None if canonical is None else canonical.get("method"),
                          "early_anchor_count": early_count, "eeg_path": str(eeg_path or ""),
                          "video_path": str(video_path or ""), "vision_log_path": str(log_path or "")})
    return {"participant": participant, "cases": cases, "modalities": routes, "model_count": len(models)}


def runner_command(participant: str, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable, str(Path(replay.__file__).resolve()), "--participant", participant,
        "--participant-config", str(paths["participant_config"]), "--raw-root", str(paths["raw_root"]),
        "--tasks", *TASKS, "--output-dir", str(paths["output_dir"]),
        "--canonical-root", str(paths["canonical_root"]),
        "--image-reference-csv", str(paths["image_reference_csv"]),
        "--alignment-metadata", str(paths["alignment_metadata"]),
    ]


def archive_existing(output_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = output_dir.with_name(f"{output_dir.name}.archived_{stamp}")
    suffix = 1
    while archived.exists():
        archived = output_dir.with_name(f"{output_dir.name}.archived_{stamp}_{suffix}"); suffix += 1
    shutil.move(str(output_dir), str(archived))
    return archived


def write_log(output_root: Path, participant: str, command: list[str], started: datetime,
              ended: datetime, *, stdout: str = "", stderr: str = "", error: Exception | None = None) -> Path:
    log_dir = output_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{participant}.log"
    exception_text = repr(error) if error is not None else ""
    path.write_text(
        f"participant: {participant}\nstarted_utc: {started.isoformat()}\nended_utc: {ended.isoformat()}\n"
        f"return_code: {getattr(error, 'returncode', 0)}\ncommand: {shlex.join(command)}\n\n"
        f"exception:\n{exception_text}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}\n",
        encoding="utf-8",
    )
    return path


def write_status(output_root: Path, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ["participant", "status", "eligible_task_cases", "eligible_segments", "excluded_segments",
              "started_utc", "ended_utc", "elapsed_seconds", "output_directory", "return_code", "log_file", "error_message"]
    path = output_root / "cohort_run_status.csv"; existing: dict[str, dict[str, object]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["participant"]: row for row in csv.DictReader(handle)}
    existing.update({str(row["participant"]): row for row in rows})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for _, row in sorted(existing.items())])


def run_participants(participants: tuple[str, ...], output_root: Path, raw_root: Path, *, dry_run: bool,
                     force: bool, execute: Callable[..., object] = subprocess.run,
                     writer: Callable[[str], None] = print) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Preflight and, unless dry-run, run each participant serially."""
    rows: list[dict[str, object]] = []; all_cases: list[dict[str, object]] = []
    for index, participant in enumerate(participants, start=1):
        paths = participant_paths(participant, output_root, raw_root)
        command = runner_command(participant, paths); started = datetime.now(timezone.utc); ended = started; clock = perf_counter()
        base: dict[str, object] = {"participant": participant, "started_utc": started.isoformat(),
            "output_directory": str(paths["output_dir"]), "return_code": "", "log_file": "", "error_message": ""}
        try:
            audit = preflight_participant(participant, paths); cases = list(audit["cases"]); all_cases.extend(cases)
            eligible = [case for case in cases if case["eligible"]]
            base.update({"eligible_task_cases": len({str(case["task"]) for case in eligible}),
                         "eligible_segments": len(eligible), "excluded_segments": len(cases) - len(eligible),
                         "modalities": ",".join(audit["modalities"])})
            if not eligible:
                status = "skipped_no_eligible"
                writer(f"[{index}/{len(participants)}] {participant} no eligible Status-anchored cases")
            elif dry_run:
                status = "dry_run_ok"
                excluded = sorted({str(case["reason"]) for case in cases if not case["eligible"]})
                writer(f"[{index}/{len(participants)}] {participant} dry-run OK: {base['eligible_task_cases']} task(s), {len(eligible)} segment(s), modalities={','.join(audit['modalities'])}, exclusions={excluded or ['none']}, output={paths['output_dir']}")
            elif output_complete(paths["output_dir"]) and not force:
                status = "skipped_existing"; writer(f"[{index}/{len(participants)}] {participant} skipped_existing")
            else:
                if paths["output_dir"].exists():
                    archived = archive_existing(paths["output_dir"])
                    reason = "forced rerun" if force else "incomplete prior run"
                    writer(f"[{index}/{len(participants)}] {participant} archived {reason} to {archived}")
                writer(f"[{index}/{len(participants)}] {participant} starting")
                result = execute(command, cwd=ROOT, check=True, capture_output=True, text=True)
                ended = datetime.now(timezone.utc)
                log = write_log(output_root, participant, command, started, ended,
                                stdout=str(getattr(result, "stdout", "") or ""), stderr=str(getattr(result, "stderr", "") or ""))
                base["log_file"] = str(log); status = "success"
                writer(f"[{index}/{len(participants)}] {participant} success")
        except Exception as error:
            ended = datetime.now(timezone.utc)
            stdout = str(getattr(error, "stdout", None) or getattr(error, "output", None) or "")
            stderr = str(getattr(error, "stderr", None) or "")
            log = write_log(output_root, participant, command, started, ended, stdout=stdout, stderr=stderr, error=error)
            detail = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), str(error))
            base.update({"return_code": getattr(error, "returncode", ""), "log_file": str(log), "error_message": detail[:1000]})
            status = "failed"; writer(f"[{index}/{len(participants)}] {participant} failed: {base['error_message']} (log: {log})")
        else:
            if ended == started:
                ended = datetime.now(timezone.utc)
        rows.append(base | {"status": status, "ended_utc": ended.isoformat(), "elapsed_seconds": round(perf_counter() - clock, 3)})
        if not dry_run:
            write_status(output_root, rows)
    return rows, all_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participants", nargs="+", metavar="PXX")
    group.add_argument("--all", action="store_true", help="Run P01 through P46 serially.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/robot_transfer/causal_modality_agnostic/v1/cohort")
    parser.add_argument("--dry-run", action="store_true", help="Resolve eligibility/assets only; no EEG/video replay or inference.")
    parser.add_argument("--force", action="store_true", help="Archive existing selected participant output before rerunning.")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); participants = selected_participants(args.participants, args.all)
    rows, cases = run_participants(participants, args.output_root.resolve(), args.raw_root.resolve(), dry_run=args.dry_run, force=args.force)
    if args.dry_run:
        frame = pd.DataFrame(cases); eligible = frame.loc[frame.eligible]
        task_count, reasons = task_eligibility_summary(frame)
        segment_reason_mentions = {reason: int(frame.reason.fillna("").str.contains(reason, regex=False).sum())
                                   for reason in ("canonical_unavailable", "no_status_alignment", "no_early_anchor", "missing_source")}
        modalities = sorted({modality for row in rows if row["status"] != "failed"
                             for modality in str(row.get("modalities", "")).split(",") if modality})
        print(f"DRY-RUN SUMMARY participants={len(participants)} eligible_participant_task_cases={task_count} eligible_segments={len(eligible)} task_level_exclusions={reasons} segment_reason_mentions={segment_reason_mentions} expected_modalities={modalities}")
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
