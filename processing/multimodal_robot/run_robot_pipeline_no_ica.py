#!/usr/bin/env python3
"""Create continuous Robot EEG features with the ICA motion edit disabled.

This pipeline preserves the Robot ICA pipeline's task/segment window identities
and long-format EEG feature schema. It reads no video: saved alignment metadata
from the matching ICA run supplies the established task-inclusion decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import mne
import numpy as np
import pandas as pd
import yaml

from continuous import _extract_eeg_window_features, _metadata, _prepare_raw, robot_eeg_config


HERE = Path(__file__).resolve().parent
TASKS = (
    "pick_place", "shape_sorter_observation", "stack", "sisyphus",
    "shape_sorter_interaction", "shape_sorter_alone",
)
IDENTITY_COLUMNS = (
    "participant", "task", "segment_id", "window_id", "window_start_s",
    "window_end_s", "channel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-config", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS,
                        help="Optional subset of robot tasks; defaults to all six tasks.")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Local root containing the participant's raw BDF files.")
    parser.add_argument("--alignment-metadata", type=Path, required=True,
                        help="Existing ICA alignment_metadata.json used to retain its supported-task decision.")
    parser.add_argument("--output-root", type=Path, default=Path("derived"))
    parser.add_argument("--run-name", help="New Robot run directory name.")
    parser.add_argument("--validate-window-identities", type=Path,
                        help="Compare proposed identities with existing ICA eeg_window_features.csv; writes nothing.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Participant YAML must contain a mapping: {path}")
    return value


def selected_files(item: Any) -> list[Path]:
    """Return selected BDF paths while preserving YAML segment order."""
    if isinstance(item, str):
        return [Path(item)]
    if not isinstance(item, dict):
        return []
    if isinstance(item.get("segments"), list):
        return [Path(segment["file"] if isinstance(segment, dict) else segment) for segment in item["segments"]]
    return [Path(item["file"])] if isinstance(item.get("file"), str) else []


def validate_config(config: dict[str, Any], participant: str) -> None:
    if config.get("participant") != participant:
        raise ValueError("--participant does not match YAML participant")
    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise ValueError("YAML must contain exactly the six expected robot task entries")
    for task_id in TASKS:
        task = tasks[task_id]
        if not isinstance(task, dict) or not isinstance(task.get("category"), str):
            raise ValueError(f"{task_id}: task/category configuration is invalid")
        item = task.get("eeg")
        if not selected_files(item) and not (isinstance(item, dict) and item.get("unavailable") is True):
            raise ValueError(f"{task_id}: EEG must be a selected BDF path/segment list or explicitly unavailable")


def resolve_raws(raw_root: Path, config: dict[str, Any], task_ids: tuple[str, ...]) -> dict[str, list[Path]]:
    """Resolve Windows-configured BDF basenames below one local raw root."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
    resolved: dict[str, list[Path]] = {}
    for task_id in task_ids:
        paths = []
        for configured in selected_files(config["tasks"][task_id].get("eeg")):
            basename = PureWindowsPath(str(configured)).name
            matches = sorted(raw_root.rglob(basename))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{task_id}: expected exactly one BDF named {basename} below {raw_root}; found {len(matches)}"
                )
            paths.append(matches[0])
        resolved[task_id] = paths
    return resolved


def load_alignment(path: Path, task_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {record["task"]: record for record in payload.get("tasks", []) if isinstance(record, dict) and "task" in record}
    missing = set(task_ids).difference(records)
    if missing:
        raise ValueError(f"Alignment metadata is missing selected task(s): {sorted(missing)}")
    return {task_id: records[task_id] for task_id in task_ids}


def task_is_extractable(qc: dict[str, Any]) -> bool:
    method = str(qc.get("method", ""))
    return not method.startswith(("unsupported", "invalid")) and float(qc.get("aligned_overlap_s", 0.0)) >= 2.0


def prepared_raw(path: Path) -> mne.io.BaseRaw:
    """Use ICA-pipeline preparation and its non-ICA signal steps only."""
    raw, _, _ = _prepare_raw({"eeg": path}, robot_eeg_config())
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    raw.filter(1.0, 40.0, method="iir", iir_params={"order": 4, "ftype": "butter", "output": "sos"}, verbose=False)
    return raw


def segment_starts(raw: mne.io.BaseRaw) -> np.ndarray:
    """Mirror the ICA pipeline's complete 2 s / 1 s window loop exactly."""
    duration = raw.n_times / raw.info["sfreq"]
    return np.arange(0.0, duration - 2.0 + 1e-9, 1.0)


def identity_rows(participant: str, raws: dict[str, list[Path]], alignment: dict[str, dict[str, Any]], task_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eeg_names = list(robot_eeg_config()["channels"]["eeg_mapping"].values())
    for task_id in task_ids:
        if not task_is_extractable(alignment[task_id]):
            continue
        for segment_id, path in enumerate(raws[task_id], start=1):
            raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
            duration = raw.n_times / raw.info["sfreq"]
            for start in np.arange(0.0, duration - 2.0 + 1e-9, 1.0):
                for channel in eeg_names:
                    rows.append({
                        "participant": participant, "task": task_id, "segment_id": segment_id,
                        "window_id": f"{task_id}_s{segment_id}_{start:.3f}",
                        "window_start_s": float(start), "window_end_s": float(start + 2.0), "channel": channel,
                    })
    return pd.DataFrame(rows, columns=IDENTITY_COLUMNS)


def validate_window_identities(proposed: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path)
    missing = set(IDENTITY_COLUMNS).difference(reference.columns)
    if missing:
        raise ValueError(f"Identity reference is missing columns: {sorted(missing)}")
    reference = reference.loc[:, IDENTITY_COLUMNS].copy()
    for frame in (proposed, reference):
        frame["segment_id"] = frame["segment_id"].astype(int)
        frame["window_start_s"] = frame["window_start_s"].astype(float).round(9)
        frame["window_end_s"] = frame["window_end_s"].astype(float).round(9)
    proposed_dupes = int(proposed.duplicated().sum())
    reference_dupes = int(reference.duplicated().sum())
    if proposed_dupes or reference_dupes:
        raise RuntimeError(f"Window identity duplicates: proposed={proposed_dupes}, reference={reference_dupes}")
    merged = proposed.merge(reference, on=list(IDENTITY_COLUMNS), how="outer", indicator=True)
    absent_reference = merged.loc[merged["_merge"].eq("left_only"), list(IDENTITY_COLUMNS)]
    absent_proposed = merged.loc[merged["_merge"].eq("right_only"), list(IDENTITY_COLUMNS)]
    if len(absent_reference) or len(absent_proposed):
        details = []
        if len(absent_reference):
            details.append(f"{len(absent_reference)} proposed identities absent from reference; first={absent_reference.iloc[0].to_dict()}")
        if len(absent_proposed):
            details.append(f"{len(absent_proposed)} reference identities absent from proposed; first={absent_proposed.iloc[0].to_dict()}")
        raise RuntimeError("Window identity validation failed: " + " | ".join(details))
    print(f"Window identity validation passed: {len(proposed)} EEG-channel rows match {reference_path}")


def canonical_video_coverage(reference_path: Path) -> dict[tuple[str, int, float, float], float]:
    """Reuse canonical ICA coverage metadata without decoding video again."""
    reference = pd.read_csv(reference_path, usecols=["task", "segment_id", "window_start_s", "window_end_s", "video_coverage"])
    key_columns = ["task", "segment_id", "window_start_s", "window_end_s"]
    coverage = reference.drop_duplicates(key_columns)
    if len(coverage) != reference.groupby(key_columns, dropna=False)["video_coverage"].nunique(dropna=False).sum():
        raise RuntimeError("ICA reference has inconsistent video coverage within a window")
    return {
        (str(row.task), int(row.segment_id), round(float(row.window_start_s), 9), round(float(row.window_end_s), 9)): float(row.video_coverage)
        for row in coverage.itertuples(index=False)
    }


def extract_features(participant: str, config: dict[str, Any], raws: dict[str, list[Path]], alignment: dict[str, dict[str, Any]], coverage: dict[tuple[str, int, float, float], float], task_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cfg = robot_eeg_config()
    eeg_names = list(cfg["channels"]["eeg_mapping"].values())
    for task_id in task_ids:
        qc = alignment[task_id]
        if not task_is_extractable(qc):
            print(f"{task_id}: skipped because canonical ICA alignment is {qc.get('method')}")
            continue
        task = config["tasks"][task_id]
        for segment_id, path in enumerate(raws[task_id], start=1):
            raw = prepared_raw(path)
            sampling_hz = float(raw.info["sfreq"])
            for start in segment_starts(raw):
                first = int(round(start * sampling_hz))
                data = raw.copy().pick(eeg_names).get_data(start=first, stop=first + 500)
                if data.shape != (len(eeg_names), 500):
                    continue
                coverage_key = (task_id, segment_id, round(float(start), 9), round(float(start + 2.0), 9))
                if coverage_key not in coverage:
                    raise RuntimeError(f"Canonical ICA coverage missing for {coverage_key}")
                metadata = _metadata(participant, task_id, task, segment_id, float(start), float(start + 2.0), qc, 1.0, coverage[coverage_key])
                for channel, feature_row in zip(eeg_names, _extract_eeg_window_features(data, sampling_hz)):
                    rows.append({**metadata, "channel": channel, "sample_count": 500, **feature_row})
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No no-ICA EEG feature windows were extracted")
    feature_columns = [column for column in result if column.startswith("eeg_")]
    if not np.isfinite(result[feature_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("No-ICA extraction produced non-finite EEG feature values")
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("Duplicate no-ICA EEG window identities")
    return result


def main() -> int:
    args = parse_args()
    if bool(args.run_name) == bool(args.validate_window_identities):
        raise ValueError("Supply exactly one of --run-name or --validate-window-identities")
    config = load_yaml(args.participant_config.resolve(strict=True))
    validate_config(config, args.participant)
    task_ids = tuple(args.tasks) if args.tasks else TASKS
    raws = resolve_raws(args.raw_root.resolve(strict=True), config, task_ids)
    alignment = load_alignment(args.alignment_metadata.resolve(strict=True), task_ids)
    identities = identity_rows(args.participant, raws, alignment, task_ids)
    if args.validate_window_identities:
        validate_window_identities(identities, args.validate_window_identities.resolve(strict=True))
        return 0
    run = args.output_root.resolve() / args.participant / "Robot_Experiment" / "runs" / args.run_name
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run}")
    reference = args.alignment_metadata.parent.parent / "features" / "eeg_window_features.csv"
    validate_window_identities(identities, reference)
    features = extract_features(args.participant, config, raws, alignment, canonical_video_coverage(reference), task_ids)
    run.mkdir(parents=True)
    (run / "resolved_participant.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    feature_dir = run / "features"
    feature_dir.mkdir()
    features.to_csv(feature_dir / "eeg_window_features.csv", index=False)
    summary = {
        "participant": args.participant,
        "mode": "continuous_features_no_ica",
        "ica_motion_edit": "disabled",
        "alignment_metadata_source": str(args.alignment_metadata.resolve()),
        "window_counts": {"eeg_rows": int(len(features)), "eeg_windows": int(features["window_id"].nunique())},
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.participant} continuous no-ICA features complete: {run}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
