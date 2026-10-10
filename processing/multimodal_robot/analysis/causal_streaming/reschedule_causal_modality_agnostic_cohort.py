#!/usr/bin/env python3
"""Rebuild causal feasibility schedules from existing timing and consensus CSVs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.multimodal_robot.analysis.causal_streaming.run_causal_modality_agnostic_robot_replay import (
    build_realtime_schedule,
    delivery_accounting,
)


PARTICIPANTS = tuple(f"P{number:02d}" for number in range(1, 47))
SOURCE_FILES = (
    "run_manifest.json", "frame_timing.csv", "causal_filter_chunk_timing.csv",
    "window_timing.csv", "consensus_predictions.csv", "causal_window_predictions.csv",
    "causal_window_consensus.csv",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_directories(roots: list[Path]) -> dict[str, Path]:
    """Resolve exactly one completed source directory for every cohort participant."""
    resolved: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Source root does not exist: {root}")
        for directory in root.glob("P[0-9][0-9]"):
            if directory.name in resolved:
                raise ValueError(f"Duplicate participant source: {directory.name}")
            missing = [name for name in SOURCE_FILES if not (directory / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{directory}: missing required source artifacts: {', '.join(missing)}")
            resolved[directory.name] = directory
    missing = sorted(set(PARTICIPANTS).difference(resolved))
    extra = sorted(set(resolved).difference(PARTICIPANTS))
    if missing or extra:
        raise ValueError(f"Expected P01-P46 exactly; missing={missing}, extra={extra}")
    return resolved


def deadline_summary(schedule: pd.DataFrame, participant: str) -> pd.DataFrame:
    predictions = schedule.loc[schedule.job_type.eq("prediction_update")]
    met = predictions.deadline_met.astype("boolean").fillna(False)
    return pd.DataFrame([{
        "participant": participant,
        "prediction_updates": int(len(predictions)),
        "deadline_s_after_release": 1.0,
        "deadline_met_count": int(met.sum()),
        "deadline_missed_count": int((~met).sum()),
        "deadline_success_rate": float(met.mean()) if len(met) else np.nan,
        "maximum_prediction_waiting_ms": float(predictions.waiting_ms.max()) if len(predictions) else np.nan,
        "maximum_prediction_lateness_ms": float(predictions.lateness_ms.max()) if len(predictions) else np.nan,
        "maximum_shared_worker_backlog_ms": float(schedule.backlog_after_ms.max()) if len(schedule) else np.nan,
    }])


def timeline_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """Summarize utilization and prediction delivery latency per fresh worker timeline."""
    rows = []
    for (participant, task, segment_id), part in schedule.groupby(["participant", "task", "segment_id"], sort=True):
        predictions = part.loc[part.job_type.eq("prediction_update")]
        span_s = float(part.arrival_s.max()) if len(part) else np.nan
        service_s = float(part.service_ms.sum()) / 1000.0
        rows.append({
            "participant": participant, "task": task, "segment_id": segment_id,
            "job_count": int(len(part)), "prediction_windows": int(len(predictions)),
            "timeline_span_s": span_s, "service_s": service_s,
            "worker_utilization": service_s / span_s if span_s > 0 else np.nan,
            "median_prediction_waiting_ms": float(predictions.waiting_ms.median()) if len(predictions) else np.nan,
            "p95_prediction_waiting_ms": float(predictions.waiting_ms.quantile(.95)) if len(predictions) else np.nan,
            "median_prediction_response_ms": float(predictions.response_ms.median()) if len(predictions) else np.nan,
            "p95_prediction_response_ms": float(predictions.response_ms.quantile(.95)) if len(predictions) else np.nan,
            "maximum_backlog_ms": float(part.backlog_after_ms.max()) if len(part) else np.nan,
        })
    return pd.DataFrame(rows)


def reschedule_participant(source: Path, destination: Path) -> dict[str, object]:
    """Write corrected scheduling artifacts without changing source inference artifacts."""
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    participant = str(manifest["participant"])
    source_hashes = {name: file_sha256(source / name) for name in ("causal_window_predictions.csv", "causal_window_consensus.csv")}
    frames = pd.read_csv(source / "frame_timing.csv")
    chunks = pd.read_csv(source / "causal_filter_chunk_timing.csv")
    windows = pd.read_csv(source / "window_timing.csv")
    consensus = pd.read_csv(source / "consensus_predictions.csv")
    schedule = build_realtime_schedule(frames, chunks, windows)
    predictions = schedule.loc[schedule.job_type.eq("prediction_update")]
    if predictions.duplicated(["participant", "task", "segment_id", "window_id"]).any() or len(predictions) != len(windows):
        raise ValueError(f"{participant}: rescheduled prediction identities are not one-to-one with window timing")
    delivery, delivery_summary = delivery_accounting(windows, consensus, schedule)
    if len(delivery) != len(windows):
        raise ValueError(f"{participant}: delivery coverage does not match window timing")
    source_unchanged = {name: file_sha256(source / name) for name in source_hashes}
    if source_hashes != source_unchanged:
        raise RuntimeError(f"{participant}: source prediction or consensus artifact changed during rescheduling")
    destination.mkdir(parents=True)
    schedule.to_csv(destination / "realtime_schedule.csv", index=False)
    delivery.to_csv(destination / "prediction_delivery_by_window.csv", index=False)
    delivery_summary.to_csv(destination / "prediction_delivery_summary.csv", index=False)
    deadline_summary(schedule, participant).to_csv(destination / "deadline_summary.csv", index=False)
    timeline_summary(schedule).to_csv(destination / "timeline_summary.csv", index=False)
    (destination / "rescheduling_manifest.json").write_text(json.dumps({
        "purpose": "CSV-only corrected feasibility scheduling for independent task recordings",
        "participant": participant, "source_output": str(source.resolve()),
        "source_prediction_hashes": source_hashes,
        "source_prediction_hashes_verified_unchanged": True,
        "scheduler": {"worker_scope": "independent participant/task/segment timeline",
                      "nonpreemptive": True, "prediction_release": "window_end_s",
                      "prediction_deadline": "window_end_s + 1.0 s"},
        "prediction_window_count": int(len(windows)),
        "source_validation_mode": manifest.get("validation_mode"),
        "raw_data_or_inference_replayed": False,
    }, indent=2) + "\n", encoding="utf-8")
    return {"participant": participant, "task_cases": int(windows[["participant", "task", "segment_id"]].drop_duplicates().shape[0]),
            "prediction_windows": int(len(windows)), "schedule": schedule, "delivery": delivery}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New v5 child directory; existing results are never overwritten.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    try:
        output.relative_to(ROOT / "outputs/robot_transfer/causal_modality_agnostic/v5")
    except ValueError as error:
        raise ValueError("--output-root must be below outputs/robot_transfer/causal_modality_agnostic/v5") from error
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output}")
    sources = source_directories([root.resolve() for root in args.source_roots])
    results = [reschedule_participant(sources[participant], output / participant) for participant in PARTICIPANTS]
    schedule = pd.concat([result["schedule"] for result in results], ignore_index=True)
    delivery = pd.concat([result["delivery"] for result in results], ignore_index=True)
    predictions = schedule.loc[schedule.job_type.eq("prediction_update")]
    summary = pd.DataFrame([{
        "participants": len(results),
        "task_cases": int(sum(result["task_cases"] for result in results)),
        "prediction_windows": int(len(predictions)),
        "median_prediction_response_ms": float(predictions.response_ms.median()),
        "p95_prediction_response_ms": float(predictions.response_ms.quantile(.95)),
        "deadline_met_count": int(predictions.deadline_met.sum()),
        "deadline_success_rate": float(predictions.deadline_met.mean()),
        "complete_paired_on_time_count": int(delivery.complete_paired_delivered_on_time.sum()),
        "complete_paired_on_time_rate": float(delivery.complete_paired_delivered_on_time.mean()),
    }])
    summary.to_csv(output / "cohort_scheduling_summary.csv", index=False)
    (output / "cohort_rescheduling_manifest.json").write_text(json.dumps({
        "purpose": "CSV-only independent-task feasibility rescheduling",
        "source_roots": [str(root.resolve()) for root in args.source_roots],
        "participant_count": len(results), "task_case_count": int(sum(result["task_cases"] for result in results)),
        "prediction_window_count": int(len(predictions)), "raw_data_or_inference_replayed": False,
        "summary_file": "cohort_scheduling_summary.csv",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
