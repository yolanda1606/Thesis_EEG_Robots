#!/usr/bin/env python3
"""Overlay reliably aligned robot events on frozen P19 inference trajectory figures.

This visualization-only entry point reads the existing prediction table, robot
metrics, and Status-anchor metadata.  It never fits models or changes scores,
windows, task summaries, raw data, or derived data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = Path("outputs/robot_personalized_inference/P19")
DOMAIN_RELIABILITY_PATH = INFERENCE_DIR / "domain_shift_audit/P19_window_domain_reliability.csv"
ALIGNMENT_PATH = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/alignment/alignment_metadata.json")
ANCHORS_PATH = Path("derived/P19/Robot_Experiment/runs/p19_robot_continuous/alignment/synchronization_anchors.csv")
SISYPHUS_METRICS = Path("data/P19_2026-06-11/Sisyphus/robot_metrics_16-06-07_SISYPHUS_INTERACTION.csv")
SHAPE_SORTER_INTERACTION_METRICS = Path("data/P19_2026-06-11/Shape Sorter Interaction/robot_metrics_16-09-27_HRI_INTERACTION.csv")
TASK_ORDER = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction")
TASK_LABELS = {
    "pick_place": "Pick and Place",
    "shape_sorter_observation": "Shape Sorter Observation",
    "stack": "Stack",
    "sisyphus": "Sisyphus",
    "shape_sorter_interaction": "Shape Sorter Interaction",
}
SISYPHUS_EVENT_TYPES = {
    1: {"event_type": "task_start", "event_label": "Task start", "color": "#2f4f4f", "linestyle": ":"},
    12: {"event_type": "cycle_start", "event_label": "Cycle start", "color": "#1f77b4", "linestyle": "-"},
    21: {"event_type": "release", "event_label": "Release", "color": "#2ca02c", "linestyle": "--"},
    50: {"event_type": "hri_collision_pause", "event_label": "HRI collision / pause", "color": "#d62728", "linestyle": "-"},
}
INTERACTION_FORCE_RELEASE_EVENTS = {
    15: {"event_type": "force_threshold_release", "event_label": "Force threshold / release", "color": "#d62728", "linestyle": "-"},
    25: {"event_type": "force_threshold_release", "event_label": "Force threshold / release", "color": "#d62728", "linestyle": "-"},
    35: {"event_type": "force_threshold_release", "event_label": "Force threshold / release", "color": "#d62728", "linestyle": "-"},
    45: {"event_type": "force_threshold_release", "event_label": "Force threshold / release", "color": "#d62728", "linestyle": "-"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--alignment-path", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument("--anchors-path", type=Path, default=ANCHORS_PATH)
    parser.add_argument("--sisyphus-metrics", type=Path, default=SISYPHUS_METRICS)
    parser.add_argument("--shape-sorter-interaction-metrics", type=Path, default=SHAPE_SORTER_INTERACTION_METRICS)
    parser.add_argument("--domain-reliability-csv", type=Path, default=DOMAIN_RELIABILITY_PATH,
                        help="Read-only optional feature-domain reliability table produced by analyze_feature_transferability.py.")
    parser.add_argument("--show-domain-reliability", action="store_true",
                        help="Mark windows outside the Image feature-domain reference; probabilities remain unchanged.")
    parser.add_argument("--overwrite-figures", action="store_true", help="Replace only the five existing trajectory PNG files.")
    parser.add_argument("--overwrite-event-timeline", action="store_true", help="Replace an existing event timeline CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Validate event provenance and time mapping without writing files.")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task_durations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Alignment metadata missing: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    output = {row["task"]: row for row in metadata.get("tasks", [])}
    missing = set(TASK_ORDER).difference(output)
    if missing:
        raise ValueError(f"Alignment metadata missing required tasks: {sorted(missing)}")
    return output


def extract_status_anchored_events(task: str, metrics_path: Path, anchors_path: Path, alignment: dict[str, Any], event_types: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Join explicit robot metric events to retained Status anchors by code and occurrence.

    The returned ``task_relative_time_s`` is the anchor's direct EEG time, not
    a re-estimated robot-metrics-to-video clock conversion. This is exactly the
    time base used by the existing EEG window start/end fields.
    """
    if not metrics_path.is_file() or not anchors_path.is_file():
        raise FileNotFoundError(f"{task}: robot metrics or synchronization-anchor file is missing")
    metrics = pd.read_csv(metrics_path)
    required_metrics = {"Experiment_Time", "Event_Trigger", "Controller_Type", "Waypoint_Stage"}
    if missing := sorted(required_metrics.difference(metrics.columns)):
        raise ValueError(f"{task}: robot metrics missing columns: {missing}")
    events = metrics[metrics["Event_Trigger"].fillna(0).astype(int).isin(event_types)].copy()
    events["event_code"] = events["Event_Trigger"].astype(int)
    events["occurrence"] = events.groupby("event_code").cumcount() + 1
    anchors = pd.read_csv(anchors_path)
    required_anchors = {"task", "event_identity", "occurrence", "eeg_time_s", "vision_time_s", "retained"}
    if missing := sorted(required_anchors.difference(anchors.columns)):
        raise ValueError(f"Synchronization anchors missing columns: {missing}")
    retained = anchors["retained"].astype(str).str.lower().eq("true")
    anchors = anchors[(anchors["task"] == task) & retained].copy()
    anchors["event_code"] = anchors["event_identity"].astype(int)
    joined = events.merge(anchors[["event_code", "occurrence", "eeg_time_s", "vision_time_s"]], on=["event_code", "occurrence"], how="left", validate="one_to_one")
    if joined["eeg_time_s"].isna().any():
        missing = joined.loc[joined["eeg_time_s"].isna(), ["event_code", "occurrence"]].to_dict(orient="records")
        raise ValueError(f"{task}: selected metric events lack retained Status anchors: {missing}")
    rows: list[dict[str, Any]] = []
    for item in joined.itertuples(index=False):
        style = event_types[int(item.event_code)]
        rows.append({
            "participant": "P19", "task": task, "event_type": style["event_type"], "event_label": style["event_label"],
            "event_code": int(item.event_code), "event_occurrence": int(item.occurrence), "task_relative_time_s": float(item.eeg_time_s),
            "source_file": str(metrics_path), "source_time": float(item.Experiment_Time), "source_timebase": "robot_metrics.Experiment_Time (s)",
            "source_controller_type": str(item.Controller_Type), "source_waypoint_stage": str(item.Waypoint_Stage),
            "status_anchor_vision_time_s": float(item.vision_time_s), "alignment_method": "direct_retained_status_anchor_eeg_time",
            "alignment_qc": f"PASS; robot/vision Status alignment={alignment['method']}; RMSE={alignment['rmse_s']:.3f}s; max residual={alignment['max_residual_s']:.3f}s",
            "plot_color": style["color"], "plot_linestyle": style["linestyle"],
        })
    timeline = pd.DataFrame(rows).sort_values("task_relative_time_s").reset_index(drop=True)
    duration = float(alignment["eeg_duration_s"])
    if not timeline["task_relative_time_s"].between(0.0, duration, inclusive="both").all():
        raise ValueError(f"{task}: a selected event falls outside the EEG task duration")
    return timeline


def overlay_events(axis, events: pd.DataFrame, *, show_labels: bool) -> None:
    """Draw event lines with one legend entry per event type."""
    seen: set[str] = set()
    for event in events.itertuples(index=False):
        label = event.event_label if show_labels and event.event_type not in seen else None
        axis.axvline(event.task_relative_time_s, color=event.plot_color, linestyle=event.plot_linestyle,
                    linewidth=1.05, alpha=0.85, label=label, zorder=0)
        seen.add(event.event_type)


def load_domain_reliability(path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature-domain reliability table missing: {path}")
    reliability = pd.read_csv(path)
    required = {"target", "window_id", "feature_domain_reliability"}
    if missing := sorted(required.difference(reliability.columns)):
        raise ValueError(f"Feature-domain reliability table missing columns: {missing}")
    if reliability.duplicated(["target", "window_id"]).any():
        raise ValueError("Feature-domain reliability table has duplicate target/window rows")
    expected = set(predictions["window_id"])
    for target in ("valence", "arousal"):
        actual = set(reliability.loc[reliability["target"] == target, "window_id"])
        if actual != expected:
            raise ValueError(f"Feature-domain reliability windows do not match predictions for {target}")
    valid = {"within_reference", "above_image_95pct", "beyond_image_reference"}
    if not set(reliability["feature_domain_reliability"]).issubset(valid):
        raise ValueError("Feature-domain reliability table contains an unknown label")
    return reliability


def overlay_domain_reliability(axis, part: pd.DataFrame, reliability: pd.DataFrame | None, target: str) -> None:
    """Show feature-domain distance status without modifying model scores."""
    if reliability is None:
        return
    lookup = reliability[reliability["target"] == target].set_index("window_id")["feature_domain_reliability"]
    labels = part["window_id"].map(lookup)
    styles = (("above_image_95pct", "^", "#d98600", "Above Image 95th percentile"),
              ("beyond_image_reference", "x", "#b22222", "Beyond Image reference"))
    for label, marker, color, legend in styles:
        mask = labels.eq(label)
        if mask.any():
            axis.scatter(part.loc[mask, "window_start_s"], part.loc[mask, f"{target}_high_score"], marker=marker,
                         s=25, linewidths=1.1, color=color, label=legend, zorder=4)


def save_figures(predictions: pd.DataFrame, events: pd.DataFrame, output_dir: Path, reliability: pd.DataFrame | None = None) -> list[Path]:
    paths: list[Path] = []
    for task in TASK_ORDER:
        part = predictions[predictions["task"] == task].sort_values("window_index")
        task_events = events[events["task"] == task]
        figure, axes = plt.subplots(2, 1, figsize=(9.3, 5.7), sharex=True, constrained_layout=True)
        for index, (axis, target, ylabel) in enumerate(zip(axes, ("valence", "arousal"), ("High-valence probability", "High-arousal probability"))):
            axis.plot(part["window_start_s"], part[f"{target}_high_score"], marker="o", markersize=3, linewidth=1.2,
                      label="Raw sliding-window score", zorder=2)
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 classification threshold", zorder=1)
            overlay_events(axis, task_events, show_labels=index == 0)
            overlay_domain_reliability(axis, part, reliability, target)
            axis.set_ylim(-0.03, 1.03)
            axis.set_ylabel(ylabel)
            axis.legend(loc="best", frameon=False, fontsize=8)
        axes[-1].set_xlabel("Task-relative EEG window start (s)")
        event_note = "; Status-anchored robot events overlaid" if len(task_events) else ""
        reliability_note = "; feature-domain distance markers shown" if reliability is not None else ""
        figure.suptitle(f"P19 {TASK_LABELS[task]}: Image-calibrated robot EEG transfer pilot{event_note}{reliability_note}")
        path = output_dir / f"P19_{task}_high_class_probabilities.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def validate_predictions(predictions: pd.DataFrame, task_durations: dict[str, dict[str, Any]]) -> None:
    required = {"task", "window_id", "window_index", "window_start_s", "window_end_s", "valence_high_score", "arousal_high_score"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"Prediction table missing required plotting columns: {missing}")
    if predictions.duplicated("window_id").any():
        raise ValueError("Prediction table has duplicate window IDs")
    if set(predictions["task"]) != set(TASK_ORDER):
        raise ValueError("Prediction table task set does not match the five approved tasks")
    for task, part in predictions.groupby("task"):
        duration = float(task_durations[task]["eeg_duration_s"])
        if not part["window_start_s"].between(0.0, duration, inclusive="both").all():
            raise ValueError(f"{task}: prediction window start lies outside EEG duration")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_path = args.inference_dir / "P19_robot_window_predictions.csv"
    timeline_path = args.inference_dir / "P19_robot_event_timeline.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Frozen prediction table missing: {prediction_path}")
    task_durations = load_task_durations(args.alignment_path)
    predictions_before = sha256(prediction_path)
    input_hashes = {str(path): sha256(path) for path in (prediction_path, args.alignment_path, args.anchors_path, args.sisyphus_metrics, args.shape_sorter_interaction_metrics)}
    predictions = pd.read_csv(prediction_path)
    validate_predictions(predictions, task_durations)
    reliability = load_domain_reliability(args.domain_reliability_csv, predictions) if args.show_domain_reliability else None
    sisyphus_events = extract_status_anchored_events("sisyphus", args.sisyphus_metrics, args.anchors_path,
                                                     task_durations["sisyphus"], SISYPHUS_EVENT_TYPES)
    interaction_events = extract_status_anchored_events("shape_sorter_interaction", args.shape_sorter_interaction_metrics,
                                                       args.anchors_path, task_durations["shape_sorter_interaction"],
                                                       INTERACTION_FORCE_RELEASE_EVENTS)
    events = pd.concat([sisyphus_events, interaction_events], ignore_index=True).sort_values(["task", "task_relative_time_s"]).reset_index(drop=True)
    if args.dry_run:
        print(f"Dry-run validation passed: {len(sisyphus_events)} Status-anchored Sisyphus events and {len(interaction_events)} force-threshold/release events.")
        print(events[["event_label", "event_occurrence", "task_relative_time_s", "source_time", "alignment_method"]].to_string(index=False))
        return 0
    if not args.overwrite_figures:
        raise ValueError("Refusing to replace existing trajectory figures without --overwrite-figures")
    if timeline_path.exists() and not args.overwrite_event_timeline:
        raise FileExistsError(f"Refusing to overwrite existing event timeline without --overwrite-event-timeline: {timeline_path}")
    events.to_csv(timeline_path, index=False)
    save_figures(predictions, events, args.inference_dir, reliability)
    if sha256(prediction_path) != predictions_before:
        raise RuntimeError("Frozen prediction table changed during visualization; stopping")
    for raw_path, before_hash in input_hashes.items():
        if sha256(Path(raw_path)) != before_hash:
            raise RuntimeError(f"Read-only input changed during visualization: {raw_path}")
    print(f"P19 event-overlay figures written; event timeline: {timeline_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
