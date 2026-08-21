#!/usr/bin/env python3
"""Plot existing matched P19 no-ICA Image-to-C0 Robot inference outputs only.

No models, features, preprocessing, diagnostics, or prediction values are
calculated or changed. The script reads the completed window prediction table
and optionally overlays already validated Status-anchored event times.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INFERENCE_DIR = Path("outputs/robot_personalized_inference/P19/no_ica_c0_inference")
PREDICTIONS = INFERENCE_DIR / "P19_no_ica_C0_window_predictions.csv"
EVENT_TIMELINE = Path("outputs/robot_personalized_inference/P19/P19_robot_event_timeline.csv")
FIGURE_DIR = INFERENCE_DIR / "figures"
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction")
TASK_LABELS = {"pick_place": "Pick & Place", "shape_sorter_observation": "Shape Sorter Observation", "stack": "Stack", "sisyphus": "Sisyphus", "shape_sorter_interaction": "Shape Sorter Interaction"}
FILENAMES = {"pick_place": "P19_pick_and_place_no_ica_C0_inference.png", "shape_sorter_observation": "P19_shape_sorter_observation_no_ica_C0_inference.png", "stack": "P19_stack_no_ica_C0_inference.png", "sisyphus": "P19_sisyphus_no_ica_C0_inference.png", "shape_sorter_interaction": "P19_shape_sorter_interaction_no_ica_C0_inference.png"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--event-timeline", type=Path, default=EVENT_TIMELINE)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the five named no-ICA/C0 figure files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate plots and event times without writing figures.")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file(): raise FileNotFoundError(f"Matched no-ICA prediction table missing: {path}")
    frame = pd.read_csv(path)
    required = {"participant", "target", "task", "window_id", "window_index", "window_start_s", "window_end_s", "high_class_score", "feature_domain_reliability"}
    if missing := sorted(required.difference(frame.columns)): raise ValueError(f"Prediction table missing columns: {missing}")
    if set(frame.participant.astype(str)) != {"P19"} or set(frame.target) != {"valence", "arousal"} or set(frame.task) != set(TASKS):
        raise ValueError("Prediction table does not contain exactly the approved P19 targets/tasks")
    if frame.duplicated(["target", "window_id"]).any(): raise ValueError("Prediction table has duplicate target/window rows")
    counts = frame.groupby("target").size()
    if not counts.eq(337).all() or len(frame) != 674: raise ValueError("Prediction table is not the expected 337 windows per target")
    scores = frame.high_class_score.to_numpy(float)
    if not np.isfinite(scores).all() or not np.logical_and(scores >= 0, scores <= 1).all(): raise ValueError("HIGH-class scores are not finite probabilities in [0, 1]")
    if not np.isfinite(frame[["window_start_s", "window_end_s"]].to_numpy(float)).all(): raise ValueError("Window times are non-finite")
    valid = {"within_reference", "above_image_95pct", "beyond_image_reference"}
    if not set(frame.feature_domain_reliability).issubset(valid): raise ValueError("Unknown feature-domain status")
    return frame.sort_values(["task", "target", "window_index"]).reset_index(drop=True)


def load_events(path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    """Read existing direct Status-anchor events; no event extraction is performed."""
    if not path.is_file(): return pd.DataFrame(columns=["task", "event_type", "event_label", "task_relative_time_s", "plot_color", "plot_linestyle"])
    events = pd.read_csv(path)
    required = {"task", "event_type", "event_label", "task_relative_time_s", "plot_color", "plot_linestyle", "alignment_method", "alignment_qc"}
    if missing := sorted(required.difference(events.columns)): raise ValueError(f"Event timeline missing columns: {missing}")
    reliable = events[events.alignment_method.eq("direct_retained_status_anchor_eeg_time") & events.alignment_qc.astype(str).str.startswith("PASS")].copy()
    for task, group in reliable.groupby("task"):
        if task not in TASKS: continue
        duration = float(predictions.loc[predictions.task.eq(task), "window_end_s"].max())
        if not group.task_relative_time_s.between(0, duration, inclusive="both").all(): raise ValueError(f"{task}: retained event lies outside C0 task window range")
    return reliable[reliable.task.isin(TASKS)].copy()


def overlay_events(axis, events: pd.DataFrame, labels: bool) -> None:
    seen: set[str] = set()
    for event in events.itertuples(index=False):
        label = event.event_label if labels and event.event_type not in seen else None
        axis.axvline(event.task_relative_time_s, color=event.plot_color, linestyle=event.plot_linestyle, linewidth=1.0, alpha=.8, label=label, zorder=0)
        seen.add(event.event_type)


def plot_task(task: str, predictions: pd.DataFrame, events: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.4, 5.8), sharex=True, constrained_layout=True)
    task_events = events[events.task.eq(task)]
    for index, (axis, target, ylabel) in enumerate(zip(axes, ("valence", "arousal"), ("High-valence probability", "High-arousal probability"))):
        part = predictions[predictions.task.eq(task) & predictions.target.eq(target)].sort_values("window_index")
        axis.plot(part.window_start_s, part.high_class_score, color="#1f77b4", marker="o", markersize=2.8, linewidth=1.15, label="Raw sliding-window score", zorder=2)
        outside = part.feature_domain_reliability.eq("beyond_image_reference")
        if outside.any():
            axis.scatter(part.loc[outside, "window_start_s"], part.loc[outside, "high_class_score"], marker="x", s=24, linewidths=1.05, color="#b22222", label="Outside Image calibration domain", zorder=4)
        axis.axhline(.5, color="black", linestyle="--", linewidth=.9, label="0.5 LOW/HIGH boundary", zorder=1)
        overlay_events(axis, task_events, labels=index == 0)
        axis.set(ylim=(0, 1), ylabel=ylabel)
        axis.legend(loc="best", frameon=False, fontsize=7.5)
    axes[-1].set_xlabel("Task-relative time (s)")
    suffix = "; Status-anchored events" if len(task_events) else ""
    figure.suptitle(f"P19 - {TASK_LABELS[task]}: matched no-ICA personalized EEG inference{suffix}")
    figure.text(.5, .005, "Scores are HIGH-class probabilities from P19 Image-calibrated personalized classifiers, not continuous emotion ground truth.", ha="center", fontsize=7)
    figure.savefig(path, dpi=160); plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_hash = sha256(args.predictions)
    predictions = load_predictions(args.predictions); events = load_events(args.event_timeline, predictions)
    paths = [args.figure_dir / FILENAMES[task] for task in TASKS]
    if args.dry_run:
        print(f"Dry-run validation passed: five tasks / 337 windows per target; {len(events)} existing reliable Status-anchored events available for overlay.")
        for task in TASKS: print(f"{task}: {int((predictions.task == task).sum() / 2)} windows; {len(events[events.task == task])} event markers")
        return 0
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite: raise FileExistsError("Refusing to overwrite no-ICA/C0 figures without --overwrite")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for task, path in zip(TASKS, paths): plot_task(task, predictions, events, path)
    if sha256(args.predictions) != predictions_hash: raise RuntimeError("Prediction CSV changed during visualization")
    print("P19 matched no-ICA/C0 trajectory figures written:")
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
