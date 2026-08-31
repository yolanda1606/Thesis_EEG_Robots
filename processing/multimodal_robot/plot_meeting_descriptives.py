#!/usr/bin/env python3
"""Create Robot Experiment processing-retention figures from saved outputs only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multimodal_robot_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PARTICIPANTS = ("P01", "P15", "P19", "P27")
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")
TASK_LABELS = ("Pick &\nPlace", "Shape Sorter\nObservation", "Stack", "Sisyphus", "Shape Sorter\nInteraction", "Shape Sorter\nAlone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/meeting_01_09")
    return parser.parse_args()


def robot_path(participant: str, run_suffix: str, filename: str) -> Path:
    return ROOT / "derived" / participant / "Robot_Experiment" / "runs" / f"{participant.lower()}_robot_{run_suffix}" / "features" / filename


def retained_eeg_windows(frame: pd.DataFrame, expected_channels: int) -> set[str]:
    """Return windows with complete finite no-ICA features for every EEG channel."""
    feature_columns = [column for column in frame if column.startswith("eeg_")]
    if not feature_columns or "sample_count" not in frame:
        raise ValueError("No-ICA EEG CSV lacks feature or sample-count columns")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    complete = frame["sample_count"].eq(500) & np.isfinite(numeric).all(axis=1)
    checked = frame.assign(_complete=complete).groupby("window_id", observed=True).agg(channel_count=("channel", "nunique"), all_complete=("_complete", "all"))
    return set(checked.index[(checked["channel_count"] == expected_channels) & checked["all_complete"]])


def eeg_rows(participant: str) -> list[dict[str, object]]:
    expected_path = robot_path(participant, "continuous", "eeg_window_features.csv")
    if not expected_path.is_file():
        raise FileNotFoundError(f"Missing continuous EEG window source: {expected_path}")
    expected = pd.read_csv(expected_path)
    required = {"task", "window_id", "channel"}
    if missing := required.difference(expected.columns):
        raise ValueError(f"{expected_path} is missing {sorted(missing)}")
    expected_channels = int(expected["channel"].nunique())
    no_ica_path = robot_path(participant, "continuous_no_ica", "eeg_window_features.csv")
    available = no_ica_path.is_file()
    retained: set[str] = set()
    status = "Unavailable: no processed Robot no-ICA EEG output" if not available else "Available; no AutoReject stage recorded in Robot no-ICA pipeline"
    if available:
        retained = retained_eeg_windows(pd.read_csv(no_ica_path), expected_channels)
    rows = []
    for task in TASKS:
        task_expected = set(expected.loc[expected["task"].eq(task), "window_id"])
        task_retained = task_expected.intersection(retained) if available else set()
        rows.append({"participant": participant, "scope": "task", "task": task,
                     "eeg_expected_windows": len(task_expected), "eeg_retained_valid_windows": len(task_retained) if available else np.nan,
                     "eeg_retention_percent": 100 * len(task_retained) / len(task_expected) if available and task_expected else np.nan,
                     "eeg_status": status})
    all_expected = set(expected["window_id"])
    rows.append({"participant": participant, "scope": "participant_total", "task": "",
                 "eeg_expected_windows": len(all_expected), "eeg_retained_valid_windows": len(all_expected.intersection(retained)) if available else np.nan,
                 "eeg_retention_percent": 100 * len(all_expected.intersection(retained)) / len(all_expected) if available and all_expected else np.nan,
                 "eeg_status": status})
    return rows


def video_rows(participant: str) -> list[dict[str, object]]:
    path = robot_path(participant, "continuous", "video_window_features.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing video window source: {path}")
    video = pd.read_csv(path)
    required = {"task", "video_frame_count", "face_detected_frames"}
    if missing := required.difference(video.columns):
        raise ValueError(f"{path} is missing {sorted(missing)}")
    rows = []
    for task in (*TASKS, None):
        subset = video if task is None else video.loc[video["task"].eq(task)]
        frames, detected = int(subset["video_frame_count"].sum()), int(subset["face_detected_frames"].sum())
        if detected > frames:
            raise ValueError(f"{path}: face-detected frames exceed video frames")
        rows.append({"participant": participant, "scope": "participant_total" if task is None else "task", "task": "" if task is None else task,
                     "face_expected_frames": frames, "face_detected_frames": detected,
                     "face_detection_percent": 100 * detected / frames if frames else np.nan,
                     "face_status": "Available; face_detected_frames is the count of frames with valid face landmarks/features"})
    return rows


def barplot(frame: pd.DataFrame, column: str, title: str, ylabel: str, filename: Path, color: str, count_label: str) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    indexed, x = frame.set_index("participant"), np.arange(len(PARTICIPANTS))
    values = indexed[column].reindex(PARTICIPANTS)
    available = values.notna()
    axis.bar(x[available], values[available], color=color, width=0.62)
    for position, participant in enumerate(PARTICIPANTS):
        row = indexed.loc[participant]
        if available.iloc[position]:
            axis.text(position, min(values.iloc[position] - 4.0, 95.0), f"{values.iloc[position]:.1f}%", ha="center", color="white", fontsize=15, weight="bold")
        else:
            axis.text(position, 5, "N/A\nno no-ICA output", ha="center", va="bottom", fontsize=11, color="#555555")
    axis.set(xticks=x, xticklabels=PARTICIPANTS, ylim=(0, 100), ylabel=ylabel, xlabel="Participant", title=title)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.tick_params(axis="both", labelsize=14)
    axis.xaxis.label.set_size(16); axis.yaxis.label.set_size(16); axis.title.set_size(20); axis.title.set_weight("bold")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8); axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(filename, dpi=220, facecolor="white")
    plt.close(fig)


def task_face_coverage(frame: pd.DataFrame, filename: Path) -> None:
    grouped = frame.groupby("task", observed=True)[["face_expected_frames", "face_detected_frames"]].sum().reindex(TASKS)
    percent = 100 * grouped["face_detected_frames"] / grouped["face_expected_frames"]
    fig, axis = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    bars = axis.bar(range(len(TASKS)), percent, color="#66AA55", width=0.68)
    for bar, value in zip(bars, percent):
        axis.text(bar.get_x() + bar.get_width() / 2, min(value - 4.0, 95.0), f"{value:.1f}%", ha="center", color="white", fontsize=13, weight="bold")
    axis.set(xticks=range(len(TASKS)), xticklabels=TASK_LABELS, ylim=(0, 100), ylabel="Face detection (%)", xlabel="Robot task", title="Robot Experiment: Face Detection Coverage by Task (n=4)")
    axis.set_yticks(np.arange(0, 101, 20)); axis.tick_params(axis="both", labelsize=13)
    axis.xaxis.label.set_size(16); axis.yaxis.label.set_size(16); axis.title.set_size(20); axis.title.set_weight("bold")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8); axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(filename, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eeg = pd.DataFrame([row for participant in PARTICIPANTS for row in eeg_rows(participant)])
    video = pd.DataFrame([row for participant in PARTICIPANTS for row in video_rows(participant)])
    summary = eeg.merge(video, on=["participant", "scope", "task"], how="outer", validate="one_to_one")
    summary.to_csv(args.output_dir / "robot_data_quality_summary.csv", index=False, na_rep="NA")
    totals = summary.loc[summary["scope"].eq("participant_total")].copy()
    barplot(totals, "eeg_retention_percent", "EEG Data Retention - Robot Experiment", "EEG windows retained (%)", args.output_dir / "robot_eeg_data_retention.png", "#4477AA", "{eeg_retained_valid_windows:.0f}/{eeg_expected_windows:.0f}")
    barplot(totals, "face_detection_percent", "Face Detection Data - Robot Experiment", "Face detection (%)", args.output_dir / "robot_face_detection_retention.png", "#66AA55", "{face_detected_frames:.0f}/{face_expected_frames:.0f} frames")
    task_face_coverage(summary.loc[summary["scope"].eq("task")], args.output_dir / "robot_task_data_coverage.png")


if __name__ == "__main__":
    main()
