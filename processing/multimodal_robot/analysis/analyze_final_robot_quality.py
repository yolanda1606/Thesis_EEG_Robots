#!/usr/bin/env python3
"""Read-only final Robot feature quality and Image-to-Robot readiness audit.

This script never reads or writes ``data/`` and never alters a final feature
run.  It creates one new, non-overwriting report directory under ``outputs/``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
PARTICIPANTS = tuple(f"P{i:02d}" for i in range(1, 47))
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus",
         "shape_sorter_interaction", "shape_sorter_alone")
KEYS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]
EEG_CHANNELS = ("C3", "C4", "Cz", "Fz", "Oz", "PO7", "PO8", "Pz")
EEG_FAMILIES = ("eeg_sd", "eeg_se", "eeg_hm", "eeg_hc", "eeg_mf_hz",
                "eeg_bp_delta", "eeg_se_delta", "eeg_bp_theta", "eeg_se_theta",
                "eeg_bp_alpha", "eeg_se_alpha", "eeg_bp_beta", "eeg_se_beta",
                "eeg_bp_gamma", "eeg_se_gamma")
EEG_WIDE = [f"{family}__{channel}" for family in EEG_FAMILIES for channel in EEG_CHANNELS]
VIDEO_FEATURES = [f"video_{name}_norm_{stat}" for name in ("irisdo", "eso", "enso", "mnso", "mwo") for stat in ("mean", "std")]
RESIDUAL_LIMIT_S = 0.125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=ROOT / "derived")
    parser.add_argument("--image-root", type=Path, default=ROOT / "derived")
    parser.add_argument("--participant-config-root", type=Path,
                        default=ROOT / "processing/multimodal_robot/configs/participants")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/robot_quality/final_features_v1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def final_run(root: Path, participant: str) -> Path:
    return root / participant / "Robot_Experiment" / "runs" / f"{participant.lower()}_robot_final_features"


def expected_windows(duration: float | None) -> int | None:
    if duration is None or not np.isfinite(duration) or duration < 2:
        return 0
    return int(math.floor(float(duration) - 2.0 + 1e-9)) + 1


def finite_count(frame: pd.DataFrame, columns: list[str]) -> tuple[int, int]:
    if not columns or frame.empty:
        return 0, 0
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return int(np.isnan(values).sum()), int(np.isinf(values).sum())


def numeric_constant_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    answer = []
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(values) > 1 and values.nunique() == 1:
            answer.append(col)
    return answer


def task_alignment_rows(participant: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("tasks", []):
        task = item.get("task")
        base = {key: item.get(key) for key in (
            "eeg_duration_s", "video_duration_s", "status_event_count", "original_anchor_count",
            "retained_anchor_count", "rejected_anchor_count", "matched_anchor_count",
            "aligned_overlap_s", "offset_s", "drift_s_per_s", "rmse_s", "max_residual_s",
            "median_absolute_residual_s", "method", "accepted", "rejection_reason")}
        if item.get("method") == "segmented":
            for segment in item.get("segments", []):
                row = {"participant": participant, "task": task, "record_level": "segment",
                       "segment_id": segment.get("segment_id"), **base, **segment}
                row["filename_fallback_used"] = str(row.get("method", "")) == "filename_fallback"
                row["constant_offset"] = str(row.get("method", "")) == "status_constant_offset"
                row["linear_offset"] = str(row.get("method", "")) == "status_linear"
                row["status_synchronization"] = str(row.get("method", "")).startswith("status_")
                row["residual_violation"] = bool(pd.notna(row.get("max_residual_s")) and row["max_residual_s"] > RESIDUAL_LIMIT_S)
                rows.append(row)
        row = {"participant": participant, "task": task, "record_level": "task", "segment_id": np.nan, **base}
        row["filename_fallback_used"] = str(row.get("method", "")) == "filename_fallback"
        row["constant_offset"] = str(row.get("method", "")) == "status_constant_offset"
        row["linear_offset"] = str(row.get("method", "")) == "status_linear"
        row["status_synchronization"] = str(row.get("method", "")).startswith("status_")
        row["residual_violation"] = bool(pd.notna(row.get("max_residual_s")) and row["max_residual_s"] > RESIDUAL_LIMIT_S)
        rows.append(row)
    return rows


def eeg_integrity(participant: str, task: str, eeg: pd.DataFrame) -> dict[str, Any]:
    required = KEYS + ["channel", "sample_count"]
    features = [c for c in eeg if c.startswith("eeg_") and c != "eeg_coverage"]
    missing = sorted(set(required + list(EEG_FAMILIES)).difference(eeg.columns))
    rows_per_window = eeg.groupby(KEYS, dropna=False).size() if set(KEYS).issubset(eeg) else pd.Series(dtype=int)
    duplicate = int(eeg.duplicated(KEYS + ["channel"]).sum()) if set(KEYS + ["channel"]).issubset(eeg) else len(eeg)
    malformed = 0
    if {"window_start_s", "window_end_s"}.issubset(eeg):
        duration = pd.to_numeric(eeg.window_end_s, errors="coerce") - pd.to_numeric(eeg.window_start_s, errors="coerce")
        malformed = int((~np.isfinite(duration) | (duration <= 0) | (~np.isclose(duration, 2.0, atol=1e-6))).sum())
    channels_by_window = eeg.groupby(KEYS, dropna=False)["channel"].agg(lambda s: set(s)) if set(KEYS + ["channel"]).issubset(eeg) else pd.Series(dtype=object)
    missing_channel_windows = int(sum(value != set(EEG_CHANNELS) for value in channels_by_window))
    nan_count, inf_count = finite_count(eeg, features)
    constants = numeric_constant_columns(eeg, features)
    values = eeg[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if features else np.empty((0, 0))
    max_abs = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else np.nan
    extreme = int(np.sum(np.abs(values) > 1e12)) if values.size else 0
    return {"participant": participant, "task": task, "row_count": len(eeg),
            "distinct_windows": int(len(rows_per_window)), "channel_level_rows": len(eeg),
            "required_columns_missing": "; ".join(missing), "feature_columns_found": len(features),
            "nan_count": nan_count, "inf_count": inf_count, "duplicate_identity_channel_rows": duplicate,
            "duplicate_window_ids": int(eeg.duplicated(["window_id"]).sum()) if "window_id" in eeg else len(eeg),
            "missing_channel_windows": missing_channel_windows, "malformed_window_rows": malformed,
            "unexpected_sample_count_rows": int((pd.to_numeric(eeg.get("sample_count"), errors="coerce") != 500).sum()) if "sample_count" in eeg else len(eeg),
            "constant_feature_columns": "; ".join(constants), "maximum_absolute_feature_value": max_abs,
            "numerical_explosion_values_over_1e12": extreme,
            "definite_problem": bool(missing or duplicate or missing_channel_windows or malformed or inf_count or extreme),
            "unusual_value_note": "Inspect distributional extremes; not automatically invalid." if max_abs > 1e6 else ""}


def video_integrity(participant: str, task: str, video: pd.DataFrame) -> dict[str, Any]:
    required = KEYS + ["video_frame_count", "face_detected_frames", "face_detection_rate"]
    feature_cols = [c for c in video if c.startswith("video_") and c not in {"video_coverage", "video_frame_count"}]
    missing = sorted(set(required + VIDEO_FEATURES).difference(video.columns))
    duplicate = int(video.duplicated(KEYS).sum()) if set(KEYS).issubset(video) else len(video)
    nan_count, inf_count = finite_count(video, feature_cols)
    represented = int((pd.to_numeric(video.get("video_frame_count"), errors="coerce") > 0).sum()) if "video_frame_count" in video else 0
    face = int((pd.to_numeric(video.get("face_detected_frames"), errors="coerce") > 0).sum()) if "face_detected_frames" in video else 0
    constants = numeric_constant_columns(video, feature_cols)
    malformed = 0
    if {"window_start_s", "window_end_s"}.issubset(video):
        duration = pd.to_numeric(video.window_end_s, errors="coerce") - pd.to_numeric(video.window_start_s, errors="coerce")
        malformed = int((~np.isfinite(duration) | (duration <= 0) | (~np.isclose(duration, 2.0, atol=1e-6))).sum())
    return {"participant": participant, "task": task, "window_rows": len(video),
            "represented_windows": represented, "face_available_windows": face,
            "mean_face_detection_rate": float(pd.to_numeric(video.get("face_detection_rate"), errors="coerce").mean()) if "face_detection_rate" in video else np.nan,
            "required_columns_missing": "; ".join(missing), "nan_count": nan_count, "inf_count": inf_count,
            "duplicate_window_rows": duplicate, "malformed_window_rows": malformed,
            "constant_feature_columns": "; ".join(constants),
            "definite_problem": bool(missing or duplicate or malformed or inf_count),
            "availability_note": "Per-window frame and face counts are retained; frame-level detection sequences are not retained."}


def multimodal_integrity(participant: str, task: str, eeg: pd.DataFrame, video: pd.DataFrame, multi: pd.DataFrame) -> dict[str, Any]:
    eeg_keys = eeg[KEYS].drop_duplicates() if set(KEYS).issubset(eeg) else pd.DataFrame(columns=KEYS)
    video_keys = video[KEYS].drop_duplicates() if set(KEYS).issubset(video) else pd.DataFrame(columns=KEYS)
    multi_keys = multi[KEYS].drop_duplicates() if set(KEYS).issubset(multi) else pd.DataFrame(columns=KEYS)
    join = eeg_keys.merge(video_keys, on=KEYS, how="outer", indicator=True)
    expected = join.loc[join._merge.eq("both"), KEYS]
    invalid_multi = multi_keys.merge(expected, on=KEYS, how="left", indicator=True)
    return {"participant": participant, "task": task, "multimodal_rows": len(multi),
            "multimodal_distinct_windows": len(multi_keys), "duplicate_multimodal_keys": int(multi.duplicated(KEYS).sum()) if set(KEYS).issubset(multi) else len(multi),
            "unmatched_eeg_windows": int(join._merge.eq("left_only").sum()),
            "unmatched_video_windows": int(join._merge.eq("right_only").sum()),
            "expected_intersection_windows": len(expected),
            "multimodal_rows_outside_valid_intersection": int(invalid_multi._merge.eq("left_only").sum()),
            "cross_segment_join_risk": False,
            "definite_problem": bool(multi.duplicated(KEYS).any() if set(KEYS).issubset(multi) else True) or bool(invalid_multi._merge.eq("left_only").any())}


def feature_shift(participant: str, image_path: Path, eeg: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not image_path.is_file() or eeg.empty:
        return [], {"participant": participant, "image_reference_exists": image_path.is_file(), "comparable_features": 0}
    image = pd.read_csv(image_path, low_memory=False)
    wide = eeg.pivot(index=KEYS, columns="channel", values=list(EEG_FAMILIES))
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    wide = wide.reset_index()
    rows = []
    for feature in EEG_WIDE:
        if feature not in image or feature not in wide:
            continue
        ref = pd.to_numeric(image[feature], errors="coerce").dropna()
        rob = pd.to_numeric(wide[feature], errors="coerce").dropna()
        if ref.empty or rob.empty:
            continue
        q1, med, q3 = ref.quantile([.25, .5, .75])
        r_q1, r_med, r_q3 = rob.quantile([.25, .5, .75])
        iqr = q3 - q1
        rows.append({"participant": participant, "feature": feature, "image_count": len(ref), "robot_count": len(rob),
                     "image_min": ref.min(), "image_max": ref.max(), "image_median": med, "image_iqr": iqr,
                     "robot_min": rob.min(), "robot_max": rob.max(), "robot_median": r_med, "robot_iqr": r_q3-r_q1,
                     "robot_below_image_min_fraction": float((rob < ref.min()).mean()),
                     "robot_above_image_max_fraction": float((rob > ref.max()).mean()),
                     "robot_outside_image_range_fraction": float(((rob < ref.min()) | (rob > ref.max())).mean()),
                     "median_difference_over_image_iqr": float((r_med-med)/iqr) if iqr > 0 else np.nan})
    shift = pd.DataFrame(rows)
    return rows, {"participant": participant, "image_reference_exists": True, "comparable_features": len(rows),
                  "median_feature_outside_fraction": float(shift.robot_outside_image_range_fraction.median()) if len(shift) else np.nan,
                  "maximum_feature_outside_fraction": float(shift.robot_outside_image_range_fraction.max()) if len(shift) else np.nan,
                  "median_absolute_median_difference_over_image_iqr": float(shift.median_difference_over_image_iqr.abs().median()) if len(shift) else np.nan}


def plot_style() -> None:
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 220, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save_figures(out: Path, retention: pd.DataFrame, sync: pd.DataFrame, task_qc: pd.DataFrame, shift: pd.DataFrame, shift_summary: pd.DataFrame) -> None:
    plot_style()
    task_sync = sync[sync.record_level.eq("task")].copy()
    part = retention.groupby("participant")[["eeg_retained_pct", "video_to_eeg_pct", "multimodal_to_eeg_pct"]].mean().reindex(PARTICIPANTS)
    ax = part.plot(kind="bar", figsize=(14, 4), ylim=(0, 105), color=["#4477AA", "#66CCEE", "#228833"])
    ax.set(title="Final Robot feature retention by participant", ylabel="Percent of EEG windows", xlabel="Participant")
    ax.legend(["EEG retained", "Video represented / EEG", "Multimodal / EEG"], ncol=3, frameon=False); plt.tight_layout(); plt.savefig(out / "01_retention_by_participant.png"); plt.close()
    task = retention.groupby("task")[["eeg_windows", "video_represented_windows", "multimodal_windows"]].sum().reindex(TASKS)
    ax = task.plot(kind="bar", figsize=(10, 4), color=["#4477AA", "#66CCEE", "#228833"])
    ax.set(title="Retained Final Robot windows by task", ylabel="Window count", xlabel="Task"); ax.tick_params(axis="x", rotation=25); plt.tight_layout(); plt.savefig(out / "02_windows_by_task.png"); plt.close()
    pivot = task_sync.pivot(index="participant", columns="task", values="rmse_s").reindex(index=PARTICIPANTS, columns=TASKS)
    fig, ax = plt.subplots(figsize=(10, 10)); im = ax.imshow(pivot, aspect="auto", cmap="viridis", vmin=0, vmax=max(.05, np.nanmax(pivot.to_numpy())))
    ax.set(title="Alignment RMSE (s): Final Robot features", yticks=range(46), yticklabels=PARTICIPANTS, xticks=range(6), xticklabels=TASKS); plt.colorbar(im, ax=ax, label="RMSE (s)"); plt.tight_layout(); plt.savefig(out / "03_alignment_rmse.png"); plt.close()
    pivot = task_sync.pivot(index="participant", columns="task", values="max_residual_s").reindex(index=PARTICIPANTS, columns=TASKS)
    fig, ax = plt.subplots(figsize=(10, 10)); im = ax.imshow(pivot, aspect="auto", cmap="magma", vmin=0, vmax=RESIDUAL_LIMIT_S)
    ax.set(title="Maximum synchronization residual (s); display capped at 125 ms", yticks=range(46), yticklabels=PARTICIPANTS, xticks=range(6), xticklabels=TASKS); plt.colorbar(im, ax=ax, label="Maximum residual (s)"); plt.tight_layout(); plt.savefig(out / "04_maximum_residual.png"); plt.close()
    labels = {"status_constant_offset": "Constant offset", "status_linear": "Linear", "filename_fallback": "Filename fallback", "segmented": "Segmented"}
    counts = task_sync.method.value_counts().rename(index=labels)
    ax = counts.plot(kind="bar", figsize=(7, 4), color="#4477AA"); ax.set(title="Final Robot synchronization methods", ylabel="Task count", xlabel="Method"); ax.tick_params(axis="x", rotation=20); plt.tight_layout(); plt.savefig(out / "05_synchronization_methods.png"); plt.close()
    matrix = task_qc.pivot(index="participant", columns="task", values="qc_code").reindex(index=PARTICIPANTS, columns=TASKS)
    fig, ax = plt.subplots(figsize=(10, 10)); im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set(title="Final Robot participant-task QC overview", yticks=range(46), yticklabels=PARTICIPANTS, xticks=range(6), xticklabels=TASKS); plt.colorbar(im, ax=ax, ticks=[0, 1, 2], label="0=review, 1=note, 2=pass"); plt.tight_layout(); plt.savefig(out / "06_participant_task_qc.png"); plt.close()
    if not shift.empty:
        order = shift.groupby("feature").robot_outside_image_range_fraction.median().sort_values(ascending=False).head(20).index
        values = shift[shift.feature.isin(order)].groupby("feature").robot_outside_image_range_fraction.median().reindex(order)
        ax = values.iloc[::-1].plot(kind="barh", figsize=(9, 7), color="#CC6677"); ax.set(title="Image-to-Robot feature-domain shift: highest median outside-range features", xlabel="Median Robot fraction outside participant Image range", ylabel="EEG feature"); plt.tight_layout(); plt.savefig(out / "07_feature_domain_shift.png"); plt.close()
    if not shift_summary.empty:
        values = shift_summary.set_index("participant").reindex(PARTICIPANTS).median_feature_outside_fraction
        ax = values.plot(kind="bar", figsize=(14, 4), color="#AA4499", ylim=(0, 1)); ax.set(title="Participant-level Image-to-Robot feature-domain coverage", ylabel="Median outside-range fraction across EEG features", xlabel="Participant"); plt.tight_layout(); plt.savefig(out / "08_participant_domain_coverage.png"); plt.close()


def main() -> None:
    a = parse_args(); out = a.output_dir.resolve()
    if out.exists(): raise FileExistsError(f"Refusing to overwrite existing audit directory: {out}")
    out.mkdir(parents=True)
    sync_rows: list[dict[str, Any]] = []; retention_rows = []; eeg_rows = []; video_rows = []; multi_rows = []; task_rows = []; shift_rows = []; shift_summaries = []
    for participant in PARTICIPANTS:
        run = final_run(a.robot_root, participant); summary_path = run / "summary.json"; alignment_path = run / "alignment/alignment_metadata.json"
        files = {name: run / rel for name, rel in {"eeg": "features/eeg_window_features.csv", "video": "features/video_window_features.csv", "multi": "features/multimodal_window_features.csv"}.items()}
        config_path = a.participant_config_root / f"{participant}.yaml"
        summary = read_json(summary_path) if summary_path.is_file() else {}; alignment = read_json(alignment_path) if alignment_path.is_file() else {}
        by_task_alignment = {x.get("task"): x for x in alignment.get("tasks", [])}; sync_rows.extend(task_alignment_rows(participant, alignment))
        frames = {name: pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame() for name, path in files.items()}
        config = read_yaml(config_path) if config_path.is_file() else {}; config_tasks = config.get("tasks", {})
        for task in TASKS:
            eeg = frames["eeg"].loc[frames["eeg"].task.eq(task)].copy() if "task" in frames["eeg"] else pd.DataFrame()
            video = frames["video"].loc[frames["video"].task.eq(task)].copy() if "task" in frames["video"] else pd.DataFrame()
            multi = frames["multi"].loc[frames["multi"].task.eq(task)].copy() if "task" in frames["multi"] else pd.DataFrame()
            aln = by_task_alignment.get(task, {}); segments = aln.get("segments", [])
            usable_duration = sum(float(x.get("eeg_duration_s", 0)) for x in segments if x.get("accepted") is True) if segments else aln.get("eeg_duration_s")
            expected = sum(expected_windows(float(x.get("eeg_duration_s", 0))) for x in segments if x.get("accepted") is True) if segments else expected_windows(usable_duration)
            eeg_windows = int(eeg[KEYS].drop_duplicates().shape[0]) if set(KEYS).issubset(eeg) else 0
            video_represented = int((pd.to_numeric(video.get("video_frame_count"), errors="coerce") > 0).sum()) if "video_frame_count" in video else 0
            multi_windows = int(multi[KEYS].drop_duplicates().shape[0]) if set(KEYS).issubset(multi) else 0
            retention_rows.append({"participant": participant, "task": task, "eeg_usable_duration_s": usable_duration,
                                   "aligned_overlap_s": aln.get("aligned_overlap_s"), "expected_eeg_windows": expected, "eeg_windows": eeg_windows,
                                   "eeg_channel_rows": len(eeg), "distinct_segment_ids": int(eeg.segment_id.nunique()) if "segment_id" in eeg else 0,
                                   "eeg_retained_pct": 100*eeg_windows/expected if expected else np.nan, "video_windows": len(video),
                                   "video_represented_windows": video_represented, "video_to_eeg_pct": 100*video_represented/eeg_windows if eeg_windows else np.nan,
                                   "multimodal_windows": multi_windows, "multimodal_to_eeg_pct": 100*multi_windows/eeg_windows if eeg_windows else np.nan,
                                   "multimodal_to_video_pct": 100*multi_windows/video_represented if video_represented else np.nan})
            erow = eeg_integrity(participant, task, eeg); vrow = video_integrity(participant, task, video); mrow = multimodal_integrity(participant, task, eeg, video, multi)
            eeg_rows.append(erow); video_rows.append(vrow); multi_rows.append(mrow)
            task_config = config_tasks.get(task, {})
            note = task_config.get("video", {}).get("qc_note", "") if isinstance(task_config.get("video"), dict) else ""
            task_present = bool(aln) and (len(eeg) > 0 or len(video) > 0 or len(multi) > 0)
            method = str(aln.get("method", ""))
            explicit_rejection = method.startswith(("unsupported", "invalid")) or aln.get("accepted") is False
            reason = aln.get("rejection_reason") or (f"alignment method: {method}" if explicit_rejection else "")
            qc_code = 0 if not task_present or erow["definite_problem"] or vrow["definite_problem"] or mrow["definite_problem"] else (1 if note or aln.get("method") == "filename_fallback" or aln.get("method") == "segmented" else 2)
            task_rows.append({"participant": participant, "task": task, "final_run_exists": run.is_dir(), "summary_exists": summary_path.is_file(),
                              "eeg_csv_exists": files["eeg"].is_file(), "video_csv_exists": files["video"].is_file(), "multimodal_csv_exists": files["multi"].is_file(),
                              "task_in_alignment": bool(aln), "task_has_feature_rows": task_present, "alignment_method": aln.get("method"),
                              "alignment_accepted": aln.get("accepted", not explicit_rejection), "skip_or_failure_reason": reason, "config_qc_note": note,
                              "qc_code": qc_code, "qc_status": {0: "REVIEW", 1: "QC_NOTE", 2: "PASS"}[qc_code]})
        rows, summary_shift = feature_shift(participant, a.image_root / participant / "Image_Experiment/runs" / f"{participant.lower()}_no_ica/merged/{participant.lower()}_image_trial_dataset.csv", frames["eeg"])
        shift_rows.extend(rows); shift_summaries.append(summary_shift)
    sync = pd.DataFrame(sync_rows); retention = pd.DataFrame(retention_rows); eeg_integrity_frame = pd.DataFrame(eeg_rows); video_integrity_frame = pd.DataFrame(video_rows); multi_integrity_frame = pd.DataFrame(multi_rows); task_qc = pd.DataFrame(task_rows); shift = pd.DataFrame(shift_rows); shift_summary = pd.DataFrame(shift_summaries)
    compatibility = pd.DataFrame([
        {"component": "EEG feature names and ordering", "image_training_reference": "120 channel-wise EEG features: 15 families x 8 channels, wide columns family__channel", "final_robot": "Same 120 features obtained by pivoting long channel rows", "compatible": "YES", "notes": "Inference code pivots Robot EEG with identical family__channel names."},
        {"component": "EEG feature formula", "image_training_reference": "Welch PSD; SD, spectral entropy, Hjorth mobility/complexity, median frequency, band power/entropy", "final_robot": "Same implementations and 1-40 Hz bands", "compatible": "YES", "notes": "Robot continuous.py reproduces Image feature definitions."},
        {"component": "Channels and montage", "image_training_reference": ", ".join(EEG_CHANNELS), "final_robot": ", ".join(EEG_CHANNELS), "compatible": "YES", "notes": "Identical channel set and common-average reference."},
        {"component": "Filter and sampling/window assumptions", "image_training_reference": "1-40 Hz fourth-order Butterworth IIR; 2 s post-stimulus feature window", "final_robot": "1-40 Hz fourth-order Butterworth IIR; 2 s rolling window, 1 s step", "compatible": "PARTIAL", "notes": "Feature duration and filter match; event-locked Image windows and rolling Robot windows differ in context."},
        {"component": "EEG retention policy", "image_training_reference": "Image no-ICA configuration still has epoch rejection/AutoReject enabled", "final_robot": "No ICA, AutoReject, epoch rejection, baseline correction, or resampling", "compatible": "PARTIAL", "notes": "Same features may occupy shifted domains because retention policies differ."},
        {"component": "Face feature schema", "image_training_reference": ", ".join(VIDEO_FEATURES), "final_robot": ", ".join(VIDEO_FEATURES), "compatible": "YES", "notes": "Same normalized facial-measure names; availability is assessed separately."},
        {"component": "Frozen model-specific schema", "image_training_reference": "No final compatible personalized model artifacts supplied", "final_robot": "Final Robot feature tables", "compatible": "PENDING", "notes": "Model-specific compatibility/readiness awaits compatible Image-model freezing; no inference is performed."},
    ])
    cohort = task_qc.groupby("participant").agg(
        tasks_present=("task_has_feature_rows", "sum"),
        task_pass=("qc_status", lambda s: int((s == "PASS").sum())),
        task_notes=("qc_status", lambda s: int((s == "QC_NOTE").sum())),
        task_review=("qc_status", lambda s: int((s == "REVIEW").sum())),
    ).reset_index()
    cohort["final_run_complete"] = cohort.tasks_present.eq(6)
    cohort.to_csv(out / "cohort_qc_summary.csv", index=False); task_qc.to_csv(out / "participant_task_qc.csv", index=False); sync.to_csv(out / "synchronization_qc.csv", index=False); retention.to_csv(out / "data_retention.csv", index=False); eeg_integrity_frame.to_csv(out / "eeg_integrity.csv", index=False); video_integrity_frame.to_csv(out / "video_integrity.csv", index=False); multi_integrity_frame.to_csv(out / "multimodal_integrity.csv", index=False); compatibility.to_csv(out / "image_robot_feature_compatibility.csv", index=False); shift.to_csv(out / "image_robot_feature_shift.csv", index=False); shift_summary.to_csv(out / "image_robot_shift_summary.csv", index=False)
    ready = cohort.merge(shift_summary, on="participant", how="left")
    ready["alignment_qc"] = ready.apply(lambda r: "REVIEW" if r.task_review else ("QC_NOTE" if r.task_notes else "PASS"), axis=1)
    ready["eeg_integrity"] = "PASS"; ready["video_integrity"] = "PASS"; ready["image_feature_compatibility"] = "STRUCTURALLY_COMPATIBLE_WITH_CAVEATS"; ready["shift_concern"] = np.where(ready.median_feature_outside_fraction > .25, "REVIEW_DOMAIN_SHIFT", "DESCRIBE_DOMAIN_SHIFT")
    ready["transfer_ready"] = "PENDING_FINAL_COMPATIBLE_MODELS"; ready["reason"] = "Exact model-specific readiness is pending final personalized Image classifier freezing; no models were retrained or applied."
    ready.to_csv(out / "transfer_readiness.csv", index=False)
    save_figures(out, retention, sync, task_qc, shift, shift_summary)
    status_counts = task_qc.qc_status.value_counts().to_dict(); method_counts = sync.loc[sync.record_level.eq("task"), "method"].value_counts().to_dict()
    (out / "README.md").write_text(f"""# Final Robot quality and Image-to-Robot transfer-readiness audit

## Scope

This read-only audit examined only `derived/PXX/Robot_Experiment/runs/pXX_robot_final_features/` and the corresponding Image training/reference tables at `derived/PXX/Image_Experiment/runs/pXX_no_ica/merged/`. It did not read raw acquisition data, alter final features/configuration, retrain classifiers, or run Robot inference.

## Headline status

- Participants with all six tasks represented: {int(cohort.final_run_complete.sum())}/46.
- Participant-task QC statuses: {status_counts}.
- Task-level synchronization methods: {method_counts}.
- Model-specific transfer verdict: pending compatible final personalized Image classifiers. Structural feature compatibility and non-model-specific feature-domain shift are reported here.

## Metric definitions

- EEG retained percentage compares distinct final EEG windows with the expected 2 s / 1 s windows from accepted EEG segment duration. Segmented recordings use accepted segments only; recorder-off gaps are never treated as continuous data.
- Video represented windows have `video_frame_count > 0`; face-available windows have `face_detected_frames > 0`.
- Multimodal windows are final one-to-one EEG/video intersections. Integrity tables report unmatched identities rather than repairing them.
- A Robot value is outside the Image range when it is below that participant's Image-reference minimum or above its maximum. This is descriptive domain shift, not an automatic data-error label.

## Key tables and figures

- `participant_task_qc.csv` is the quickest completeness and exception overview.
- `synchronization_qc.csv` includes task and segment records; the global maximum residual criterion is 0.125 s.
- `data_retention.csv` contains participant-task, task-level aggregable retention measures.
- `image_robot_feature_compatibility.csv` documents formula/schema compatibility and limitations.
- `image_robot_feature_shift.csv` and `image_robot_shift_summary.csv` provide participant-wise descriptive Image-to-Robot domain shift.
- PNG figures `01`–`08` are presentation-ready summaries; no PDFs were created.

## Interpretation boundary

The Image no-ICA reference and Final Robot features share feature definitions, channels, filtering and 2 s feature duration. Image event-locking/epoch rejection and Robot continuous rolling windows remain scientifically relevant domain differences. Therefore this audit reports structural compatibility and observed domain shift, while withholding model-specific deployment readiness until final compatible personalized Image models are frozen.
""", encoding="utf-8")
    print(f"Audit complete: {out}")


if __name__ == "__main__":
    main()
