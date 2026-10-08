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
V5_OUTPUT_PARENT = ROOT / "outputs/robot_transfer/causal_modality_agnostic/v5"
REFERENCE_OUTPUT_FILES = ("causal_window_predictions.csv", "causal_window_consensus.csv")
REQUIRED_OUTPUT_SCHEMAS = {
    "run_manifest.json": set(),
    "calibration_offsets.csv": {"participant", "task", "segment_id", "frozen_offset_s"},
    "calibration_anchors.csv": {"participant", "task", "segment_id"},
    "synchronization_error.csv": {"task", "segment_id", "eeg_time_s"},
    "model_window_comparison.csv": {"participant", "task", "window_id", "target", "model_rank"},
    "causal_window_predictions.csv": {"participant", "task", "window_id", "target", "model_rank", "high_probability"},
    "causal_window_consensus.csv": {"participant", "task", "window_id", "target", "top3_consensus_probability"},
    "top3_window_completeness.csv": {"participant", "task", "target", "n_candidate_windows", "n_complete_top3_windows"},
    "window_consensus_comparison.csv": {"participant", "task", "window_id", "target"},
    "task_consensus_comparison.csv": {"participant", "task", "target"},
    "frame_timing.csv": {"participant", "task", "segment_id", "frame_index", "eeg_time_s", "frame_total_compute_ms"},
    "window_timing.csv": {"participant", "task", "segment_id", "window_id", "valid_model_count", "prediction_compute_ms"},
    "model_probabilities.csv": {"participant", "task", "window_id", "target", "model_rank", "high_probability"},
    "consensus_predictions.csv": {"participant", "task", "window_id", "target", "top3_consensus_probability"},
    "realtime_schedule.csv": {"job_type", "participant", "task", "arrival_s", "completion_s", "deadline_met"},
    "latency_by_window.csv": {"participant", "task", "window_id", "total_prediction_path_compute_ms"},
    "latency_summary.csv": {"scope", "task", "metric", "median"},
    "latency_budget_summary.csv": {"scope", "count", "deadline_ms"},
    "deadline_summary.csv": {"participant", "prediction_updates", "deadline_success_rate"},
    "causal_filter_chunk_timing.csv": {"participant", "task", "segment_id", "chunk_start_sample", "causal_car_ms", "causal_filter_ms"},
    "eeg_setup_timing.csv": {"participant", "task", "segment_id", "raw_eeg_extract_ms"},
    "prediction_delivery_by_window.csv": {"participant", "task", "window_id", "scheduled_update", "computational_deadline_met", "valid_valence_consensus", "valid_arousal_consensus", "complete_paired_consensus", "complete_paired_delivered_on_time"},
    "prediction_delivery_summary.csv": {"participant", "task", "scheduled_updates", "complete_paired_delivered_on_time"},
    "excluded_cases.csv": set(),
}
REQUIRED_OUTPUT_FILES = tuple(REQUIRED_OUTPUT_SCHEMAS)
LEGACY_CORE_OUTPUT_FILES = (
    "run_manifest.json", "calibration_offsets.csv", "calibration_anchors.csv",
    "synchronization_error.csv", "model_window_comparison.csv",
    "causal_window_predictions.csv", "causal_window_consensus.csv",
    "top3_window_completeness.csv", "window_consensus_comparison.csv",
    "task_consensus_comparison.csv", "latency_by_window.csv",
    "latency_summary.csv", "latency_budget_summary.csv", "causal_filter_chunk_timing.csv", "excluded_cases.csv",
)


def participant_paths(participant: str, output_root: Path, raw_root: Path, reference_root: Path) -> dict[str, Path]:
    """Resolve established participant inputs and the isolated cohort output."""
    paths = replay.participant_paths(participant)
    return paths | {"raw_root": raw_root, "output_dir": output_root / participant,
                    "reference_causal_output": reference_root / participant}


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


def validate_v5_output_root(output_root: Path, *, resume: bool) -> Path:
    """Require an explicitly named v5 child root and protect older outputs."""
    resolved = output_root.resolve()
    try:
        relative = resolved.relative_to(V5_OUTPUT_PARENT.resolve())
    except ValueError as error:
        raise ValueError(f"--output-root must be a new child of {V5_OUTPUT_PARENT}") from error
    if not relative.parts:
        raise ValueError("--output-root must name a new v5 run directory, not the v5 parent itself")
    if resolved.exists() and not resume and any(resolved.iterdir()):
        raise FileExistsError("--output-root already contains files; use a different new v5 root or --resume")
    return resolved


def validate_output(output_dir: Path, participant: str,
                    expected_tasks: tuple[str, ...] | None = None) -> tuple[bool, str]:
    """Validate a finished v5 participant directory without changing it."""
    if not output_dir.is_dir():
        return False, "output directory does not exist"
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing:
        return False, "missing required artifacts: " + ", ".join(missing)
    try:
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid run manifest: {error}"
    if manifest.get("participant") != participant:
        return False, f"manifest participant is {manifest.get('participant')!r}, expected {participant!r}"
    requested_tasks = manifest.get("tasks_requested")
    if not isinstance(requested_tasks, list) or not requested_tasks:
        return False, "manifest has no tasks_requested list"
    if expected_tasks is not None and tuple(requested_tasks) != expected_tasks:
        return False, f"manifest tasks {requested_tasks!r} do not match requested tasks {list(expected_tasks)!r}"
    if not isinstance(manifest.get("realtime_scheduler"), dict):
        return False, "manifest has no realtime_scheduler record"
    if "frozen_prediction_equivalence" not in manifest:
        return False, "manifest has no frozen_prediction_equivalence record"
    for name, required_columns in REQUIRED_OUTPUT_SCHEMAS.items():
        if not required_columns:
            continue
        try:
            columns = set(pd.read_csv(output_dir / name, nrows=0).columns)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            return False, f"invalid {name}: {error}"
        absent = sorted(required_columns.difference(columns))
        if absent:
            return False, f"{name} missing columns: {', '.join(absent)}"
    return True, "valid v5 participant output"


def output_complete(output_dir: Path, participant: str = "P01") -> bool:
    """Compatibility predicate for callers that only require a boolean."""
    return validate_output(output_dir, participant)[0]


def reference_status(reference_dir: Path) -> str:
    """Return availability without creating or repairing a historical reference."""
    return "available" if reference_dir.is_dir() and all((reference_dir / name).is_file() for name in REFERENCE_OUTPUT_FILES) else "missing"


def validate_frozen_models(models: list[dict[str, object]], participant: str) -> None:
    """Require exactly one frozen record for every target/rank combination."""
    expected = {(target, rank) for target in ("valence", "arousal") for rank in (1, 2, 3)}
    keys = [(str(model.get("target")), int(model.get("model_rank", -1))) for model in models]
    if len(models) != 6 or set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError(f"{participant}: expected exactly six frozen target/rank records; found {keys}")
    for model in models:
        if str(model.get("modality")) not in {"eeg", "face", "multimodal"}:
            raise ValueError(f"{participant}: unsupported frozen modality {model.get('modality')!r}")
        if not model.get("candidate_features") or not model.get("selected_features"):
            raise ValueError(f"{participant}: frozen model {model.get('target')} rank {model.get('model_rank')} has no feature schema")


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


def preflight_participant(participant: str, paths: dict[str, Path], task_names: tuple[str, ...] = TASKS) -> dict[str, object]:
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
    validate_frozen_models(models, participant)
    routes = sorted({str(model["modality"]) for model in models})
    cases: list[dict[str, object]] = []
    for task_name in task_names:
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
        mapping_reason = replay.ambiguous_segment_mapping_reason(task)
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
            if mapping_reason:
                reasons.append(mapping_reason)
            missing_sources = [label for label, path in (("EEG", eeg_path), ("video", video_path), ("vision_log", log_path))
                               if path is None or not path.is_file()]
            if missing_sources:
                reasons.append("missing_source:" + ",".join(missing_sources))
            cases.append({"participant": participant, "task": task_name, "segment_id": segment_id,
                          "eligible": not reasons, "reason": ";".join(reasons),
                          "canonical_alignment_type": None if canonical is None else canonical.get("method"),
                          "early_anchor_count": early_count, "eeg_path": str(eeg_path or ""),
                          "video_path": str(video_path or ""), "vision_log_path": str(log_path or "")})
    return {"participant": participant, "cases": cases, "modalities": routes, "model_count": len(models),
            "reference_status": reference_status(paths["reference_causal_output"]),
            "reference_causal_output": str(paths["reference_causal_output"])}


def runner_command(participant: str, paths: dict[str, Path], task_names: tuple[str, ...], reference: str,
                   validation_mode: str = "strict_historical") -> list[str]:
    command = [
        sys.executable, str(Path(replay.__file__).resolve()), "--participant", participant,
        "--participant-config", str(paths["participant_config"]), "--raw-root", str(paths["raw_root"]),
        "--tasks", *task_names, "--output-dir", str(paths["output_dir"]),
        "--canonical-root", str(paths["canonical_root"]),
        "--image-reference-csv", str(paths["image_reference_csv"]),
        "--alignment-metadata", str(paths["alignment_metadata"]),
        "--validation-mode", validation_mode,
    ]
    if reference == "available":
        command.extend(["--reference-causal-output", str(paths["reference_causal_output"])])
    return command


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
              "started_utc", "ended_utc", "elapsed_seconds", "output_directory", "reference_status", "return_code", "log_file", "error_message"]
    path = output_root / "cohort_run_status.csv"; existing: dict[str, dict[str, object]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["participant"]: row for row in csv.DictReader(handle)}
    existing.update({str(row["participant"]): row for row in rows})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for _, row in sorted(existing.items())])


def run_participants(participants: tuple[str, ...], output_root: Path, raw_root: Path, reference_root: Path,
                     task_names: tuple[str, ...], *, dry_run: bool, resume: bool,
                     validation_mode: str = "strict_historical",
                     execute: Callable[..., object] = subprocess.run,
                     writer: Callable[[str], None] = print) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Preflight and, unless dry-run, run each participant serially."""
    rows: list[dict[str, object]] = []; all_cases: list[dict[str, object]] = []
    for index, participant in enumerate(participants, start=1):
        paths = participant_paths(participant, output_root, raw_root, reference_root)
        started = datetime.now(timezone.utc); ended = started; clock = perf_counter()
        command: list[str] = []
        base: dict[str, object] = {"participant": participant, "started_utc": started.isoformat(),
            "output_directory": str(paths["output_dir"]), "return_code": "", "log_file": "", "error_message": "",
            "reference_status": ""}
        try:
            audit = preflight_participant(participant, paths, task_names); cases = list(audit["cases"]); all_cases.extend(cases)
            eligible = [case for case in cases if case["eligible"]]
            base.update({"eligible_task_cases": len({str(case["task"]) for case in eligible}),
                         "eligible_segments": len(eligible), "excluded_segments": len(cases) - len(eligible),
                         "modalities": ",".join(audit["modalities"]), "reference_status": audit["reference_status"]})
            command = runner_command(participant, paths, task_names, str(audit["reference_status"]), validation_mode)
            if not eligible:
                status = "skipped_no_eligible"
                writer(f"[{index}/{len(participants)}] {participant} no eligible Status-anchored cases")
            elif dry_run:
                status = "dry_run_ok"
                excluded = sorted({str(case["reason"]) for case in cases if not case["eligible"]})
                writer(f"[{index}/{len(participants)}] {participant} dry-run OK: {base['eligible_task_cases']} task(s), {len(eligible)} segment(s), reference={audit['reference_status']}, modalities={','.join(audit['modalities'])}, exclusions={excluded or ['none']}, output={paths['output_dir']}")
            elif paths["output_dir"].exists():
                valid, detail = validate_output(paths["output_dir"], participant, task_names)
                if valid and resume:
                    status = "skipped_existing"; writer(f"[{index}/{len(participants)}] {participant} skipped_existing")
                elif valid:
                    status = "blocked_existing"; base["error_message"] = "existing valid output requires --resume"
                    writer(f"[{index}/{len(participants)}] {participant} blocked_existing: {base['error_message']}")
                else:
                    status = "blocked_existing"; base["error_message"] = f"existing output is incomplete or invalid: {detail}"
                    writer(f"[{index}/{len(participants)}] {participant} blocked_existing: {base['error_message']}")
            else:
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


def write_preflight_report(path: Path, rows: list[dict[str, object]], cases: list[dict[str, object]], output_root: Path) -> None:
    """Write an explicitly requested preflight report only under the new output root."""
    try:
        path.resolve().relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError("--preflight-report must be beneath --output-root") from error
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite preflight report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"participants": rows, "cases": cases}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participants", nargs="+", metavar="PXX")
    group.add_argument("--all", action="store_true", help="Run P01 through P46 serially.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New v5 root. Existing participant outputs are never moved or overwritten.")
    parser.add_argument("--reference-root", type=Path,
                        default=ROOT / "outputs/robot_transfer/causal_modality_agnostic/v1/cohort",
                        help="Read-only historical causal-reference root containing PXX directories.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--dry-run", action="store_true", help="Resolve eligibility/assets only; no EEG/video replay or inference.")
    parser.add_argument("--resume", action="store_true", help="Skip only participant outputs that pass v5 manifest/artifact/schema validation.")
    parser.add_argument("--validation-mode", choices=replay.VALIDATION_MODES, default="strict_historical",
                        help="Replay validation policy; corrected_timestamp permits only membership-derived timestamp differences from v1.")
    parser.add_argument("--preflight-report", type=Path,
                        help="Explicit optional JSON report beneath --output-root; dry-run otherwise writes nothing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); participants = selected_participants(args.participants, args.all)
    output_root = validate_v5_output_root(args.output_root, resume=args.resume)
    rows, cases = run_participants(participants, output_root, args.raw_root.resolve(), args.reference_root.resolve(), tuple(args.tasks), dry_run=args.dry_run, resume=args.resume, validation_mode=args.validation_mode)
    if args.dry_run:
        frame = pd.DataFrame(cases); eligible = frame.loc[frame.eligible]
        task_count, reasons = task_eligibility_summary(frame)
        segment_reason_mentions = {reason: int(frame.reason.fillna("").str.contains(reason, regex=False).sum())
                                   for reason in ("canonical_unavailable", "no_status_alignment", "no_early_anchor", "missing_source")}
        modalities = sorted({modality for row in rows if row["status"] != "failed"
                             for modality in str(row.get("modalities", "")).split(",") if modality})
        print(f"DRY-RUN SUMMARY participants={len(participants)} eligible_participant_task_cases={task_count} eligible_segments={len(eligible)} task_level_exclusions={reasons} segment_reason_mentions={segment_reason_mentions} expected_modalities={modalities}")
        if args.preflight_report is not None:
            write_preflight_report(args.preflight_report, rows, cases, output_root)
    return 1 if any(row["status"] in {"failed", "blocked_existing"} for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
