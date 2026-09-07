#!/usr/bin/env python3
"""Extend a completed frozen Image-to-Robot transfer with descriptive reports.

This script reads saved transfer predictions.  It does not load, fit, calibrate,
select, or otherwise execute any Image model or Robot inference pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_final_frozen_eeg_transfer import DISPLAY_TASKS, TASK_ORDER, event_table, plot_target


ROOT = Path(__file__).resolve().parents[3]
KEYS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path,
                        help="Optional historical transfer directory for a read-only comparison.")
    parser.add_argument("--participant-config", type=Path, required=True)
    return parser.parse_args()


def robot_conditions(config_path: Path) -> dict[str, str]:
    """Read observation-task condition labels from the authoritative Robot config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    conditions: dict[str, str] = {}
    for task, specification in config["tasks"].items():
        if specification.get("category") != "observation":
            conditions[task] = "N/A"
            continue
        condition = specification.get("condition", {})
        actual = str(condition.get("actual", "")).upper()
        speed = str(condition.get("speed_actual", "")).upper()
        # A faulty observation remains FAULTY even when it also has a speed.
        conditions[task] = "FAULTY" if actual == "FAULTY" else speed if speed in {"FAST", "SLOW"} else "N/A"
    return conditions


def attach_frozen_plot_metadata(predictions: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Add frozen top-three metadata in memory for timeline labels only."""
    metadata = pd.DataFrame([{
        "target": model["target"],
        "model_rank": model["model_rank"],
        "selected_feature_count": len(model["selected_features"]),
        "frozen_image_balanced_accuracy": model["mean_outer_cv_balanced_accuracy"],
    } for model in manifest["models"]])
    return predictions.merge(metadata, on=["target", "model_rank"], how="left", validate="many_to_one")


def task_statistics(predictions: pd.DataFrame, ratings: pd.DataFrame, conditions: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (participant, target, task), part in predictions.groupby(["participant", "target", "task"], sort=False):
        probabilities = part.pivot(index=KEYS, columns="model_rank", values="high_probability").dropna().sort_index()
        hard = part.pivot(index=KEYS, columns="model_rank", values="original_hard_prediction").reindex(probabilities.index)
        if sorted(probabilities.columns) != [1, 2, 3]:
            raise ValueError(f"{participant} {target} {task}: expected exactly ranks 1, 2, 3")
        consensus = probabilities.median(axis=1)
        ranges = probabilities.max(axis=1) - probabilities.min(axis=1)
        pair_diffs = pd.DataFrame({
            "r1_r2": (probabilities[1] - probabilities[2]).abs(),
            "r1_r3": (probabilities[1] - probabilities[3]).abs(),
            "r2_r3": (probabilities[2] - probabilities[3]).abs(),
        })
        rating = ratings[(ratings.target.eq(target)) & (ratings.task.eq(task))]
        if len(rating) != 1:
            raise ValueError(f"Expected one saved rating for {target} {task}; found {len(rating)}")
        rating_row = rating.iloc[0]
        median_probability = float(consensus.median())
        rows.append({
            "participant": participant, "target": target, "task": task, "task_display": DISPLAY_TASKS[task],
            "robot_condition": conditions.get(task, "N/A"),
            "n_consensus_windows": int(len(consensus)),
            "median_top3_consensus_probability": median_probability,
            "mean_top3_consensus_probability": float(consensus.mean()),
            "fraction_consensus_windows_ge_0_5": float((consensus >= 0.5).mean()),
            "descriptive_task_verdict": "HIGH" if median_probability >= 0.5 else "LOW",
            "mean_top3_probability_range": float(ranges.mean()),
            "median_top3_probability_range": float(ranges.median()),
            "mean_pairwise_absolute_probability_difference": float(pair_diffs.mean(axis=1).mean()),
            "fraction_unanimous_low_windows": float((hard == 0).all(axis=1).mean()),
            "fraction_unanimous_high_windows": float((hard == 1).all(axis=1).mean()),
            "fraction_unanimous_top3_windows": float((hard.nunique(axis=1) == 1).mean()),
            "majority_vote_high_fraction": float((hard.sum(axis=1) >= 2).mean()),
            "probability_correlation_r1_r2": float(probabilities[1].corr(probabilities[2])),
            "probability_correlation_r1_r3": float(probabilities[1].corr(probabilities[3])),
            "probability_correlation_r2_r3": float(probabilities[2].corr(probabilities[3])),
            "robot_rating_1_to_7": float(rating_row.robot_rating),
            "robot_rating_class": rating_row.robot_rating_class,
            "verdict_matches_robot_rating": bool(("HIGH" if median_probability >= 0.5 else "LOW") == rating_row.robot_rating_class),
        })
    return pd.DataFrame(rows).sort_values(["target", "task"]).reset_index(drop=True)


def event_style(label: str) -> tuple[str, str]:
    upper = label.upper()
    if "START" in upper or "END" in upper:
        return "#4d4d4d", "--"
    if any(word in upper for word in ["COLLISION", "WRONG", "MID-AIR"]):
        return "#b2182b", "-"
    return "#2166ac", "-"


def plot_task_trigger_consensus(predictions: pd.DataFrame, statistics: pd.DataFrame, events: pd.DataFrame, participant: str, figures: Path) -> None:
    """Save one wide annotated trigger figure for every target/task combination."""
    for target, data in predictions.groupby("target", sort=False):
        tasks = [task for task in TASK_ORDER if task in set(data.task)]
        for task in tasks:
            part = data[data.task.eq(task)].copy()
            part["window_center_s"] = (part.window_start_s + part.window_end_s) / 2
            probabilities = part.pivot(index="window_center_s", columns="model_rank", values="high_probability").dropna().sort_index()
            task_events = events[events.task.eq(task)]
            event_types = list(task_events.event_label.drop_duplicates())
            # Sisyphus has only a few event classes but many repeated markers, so
            # it needs the same additional horizontal space as a dense legend task.
            figure_width = 16 if len(event_types) >= 5 or task in {"stack", "sisyphus"} else 13
            fig, axis = plt.subplots(figsize=(figure_width, 5.2))
            axis.fill_between(probabilities.index, probabilities.min(axis=1), probabilities.max(axis=1), color="0.35", alpha=0.18, label="Top-3 range")
            axis.plot(probabilities.index, probabilities.median(axis=1), color="0.15", lw=1.8, label="Top-3 median")
            labels_seen: set[str] = set()
            for event in task_events.itertuples(index=False):
                # Repeated event types use one legend label only. This preserves
                # their timing without adding repeated text to dense tasks.
                label = event.event_label if event.event_label not in labels_seen else None
                color, linestyle = event_style(event.event_label)
                axis.axvline(event.time_s, color=color, ls=linestyle, lw=0.9, alpha=0.8, label=label)
                labels_seen.add(event.event_label)
            stat = statistics[(statistics.target.eq(target)) & (statistics.task.eq(task))].iloc[0]
            annotation = (
                f"Self-report: {stat.robot_rating_1_to_7:.0f}/7 ({stat.robot_rating_class})\n"
                f"Task median P(HIGH): {stat.median_top3_consensus_probability:.2f} → {stat.descriptive_task_verdict}\n"
                f"Unanimous top-3 windows: {stat.fraction_unanimous_top3_windows:.0%}"
            )
            axis.text(0.995, 0.04, annotation, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9})
            axis.axhline(0.5, color="black", ls=":", lw=0.8)
            axis.set_ylim(0, 1); axis.set_ylabel("P(HIGH)"); axis.set_xlabel("Time within task (s)")
            condition = stat.robot_condition
            task_title = DISPLAY_TASKS[task] if condition == "N/A" else f"{DISPLAY_TASKS[task]} ({condition})"
            fig.suptitle(f"{participant} {target} {task_title}: descriptive top-3 consensus with semantic Robot events", y=0.97, fontsize=13)
            # The legend sits in its own band between the title and axes.
            axis.legend(fontsize=7, ncol=min(4, max(1, len(labels_seen) + 2)), loc="lower center", bbox_to_anchor=(0.5, 1.03), frameon=True)
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.70)
            fig.savefig(figures / f"{participant}_{target}_{DISPLAY_TASKS[task]}_consensus_triggers.png", dpi=170)
            plt.close(fig)


def feature_text(features: list[str]) -> str:
    if len(features) > 24:
        return f"All {len(features)} frozen EEG features (full list retained in source manifest)."
    return "; ".join(features)


def historical_comparison(final_manifest: dict, historical_dir: Path) -> str:
    historical_selection = json.loads((historical_dir / "selected_models_v1.json").read_text(encoding="utf-8"))
    historical_models = json.loads((historical_dir / "final_image_models_v1/model_manifest.json").read_text(encoding="utf-8"))
    historical_ica = json.loads((historical_dir / "ica_inference_v2/inference_manifest.json").read_text(encoding="utf-8"))
    historical_no_ica = json.loads((historical_dir / "no_ica_inference_v2/inference_manifest.json").read_text(encoding="utf-8"))
    old_by_id = {model["identifier"]: model for model in historical_models["models"]}
    new_by_key = {(model["target"], model["model_rank"]): model for model in final_manifest["models"]}
    lines = ["## Historical versus final P27 transfer", "", "This is a read-only comparison. It explains documented design and provenance differences; it does not seek to reproduce historical trajectories.", "", "### Model selection and selected features", "", "The historical run selected Stage-B representatives and then prepared/refit deployment models. The final run uses the frozen focused EEG-only top-3 table and does not select or refit transfer models.", ""]
    for old in historical_selection["selection"]:
        key = (old["target"], old["rank"])
        old_model = old_by_id[f"{old['target']}__{old['modality']}__{old['classifier']}__{old['feature_count_request']}"]
        new = new_by_key[key]
        lines.extend([
            f"- {old['target']} R{old['rank']}: historical `{old['modality']} {old['classifier']} {old['feature_count_request']}` → final `EEG {new['classifier']} {new['feature_family']} ({len(new['selected_features'])} selected)`.",
            f"  - Historical selected features ({len(old_model['selected_features'])}): {feature_text(old_model['selected_features'])}",
            f"  - Final selected features ({len(new['selected_features'])}): {feature_text(new['selected_features'])}",
        ])
    lines.extend([
        "", "### Robot input provenance and preprocessing", "",
        f"- Historical ICA input: `{historical_ica['robot_eeg_csv']}`; it also supplied `{historical_ica['robot_video_csv']}` for Multimodal models.",
        f"- Historical no-ICA input: `{historical_no_ica['robot_eeg_csv']}`; it also supplied `{historical_no_ica['robot_video_csv']}` for Multimodal models.",
        f"- Final input: `{final_manifest['robot_eeg_input']}` only. It is the canonical final no-ICA Robot EEG feature table; no video features or Face/Multimodal models are used.",
        "- The final output has 422 windows for every model. Historical Multimodal P27 rows had 371 windows per condition, while its EEG-only rows had 422, because Multimodal inference required matching video windows.",
        "", "### SVM probability handling", "",
        "- Historical P27 valence models were SVC artifacts serialized with `probability=True`, after the historical deployment-model preparation/refit workflow.",
        "- Final P27 contains one SVM: arousal R3. Its original frozen artifact is `SVC(probability=False)` and retains its hard class. Its P(HIGH) comes from the separate Image-only cross-validated Platt-calibrated wrapper; Robot data and ratings did not participate in calibration.",
        "", "These differences in model families, modalities, fixed selected features, source Image run, Robot input condition, available windows, and SVM probability construction make trajectory changes expected. They are not evidence that either result should be transformed to reproduce the other.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    transfer_dir = args.transfer_dir.resolve()
    historical_dir = args.historical_dir.resolve() if args.historical_dir else None
    participant = transfer_dir.name
    predictions = pd.read_csv(transfer_dir / f"{participant}_window_predictions.csv")
    ratings = pd.read_csv(transfer_dir / f"{participant}_robot_rating_comparison.csv")
    manifest = json.loads((transfer_dir / f"{participant}_transfer_manifest.json").read_text(encoding="utf-8"))
    predictions = attach_frozen_plot_metadata(predictions, manifest)
    conditions = robot_conditions(args.participant_config.resolve())
    ratings = ratings.assign(robot_condition=ratings.task.map(conditions).fillna("N/A"))
    ratings.to_csv(transfer_dir / f"{participant}_robot_rating_comparison.csv", index=False)
    statistics = task_statistics(predictions, ratings, conditions)
    statistics.to_csv(transfer_dir / f"{participant}_task_consensus_analysis.csv", index=False)
    events, _ = event_table(args.participant_config.resolve())
    figures = transfer_dir / "figures"
    for target in ["valence", "arousal"]:
        plot_target(predictions, target, participant, figures)
    for target in sorted(predictions.target.unique()):
        obsolete = figures / f"{participant}_{target}_probability_consensus_triggers.png"
        if obsolete.exists():
            obsolete.unlink()
    plot_task_trigger_consensus(predictions, statistics, events, participant, figures)
    compact = statistics[["task_display", "robot_condition", "target", "median_top3_consensus_probability", "mean_top3_consensus_probability", "fraction_consensus_windows_ge_0_5", "descriptive_task_verdict", "mean_top3_probability_range", "median_top3_probability_range", "mean_pairwise_absolute_probability_difference", "fraction_unanimous_low_windows", "fraction_unanimous_high_windows", "fraction_unanimous_top3_windows", "majority_vote_high_fraction", "robot_rating_1_to_7", "robot_rating_class", "verdict_matches_robot_rating"]].copy()
    compact.columns = ["Task", "Robot condition", "Target", "Median P(HIGH)", "Mean P(HIGH)", "Consensus windows ≥0.5", "Verdict", "Mean range", "Median range", "Mean pairwise |ΔP|", "Unanimous LOW", "Unanimous HIGH", "Unanimous", "Majority HIGH", "Self-report (1–7)", "Rating class", "Verdict match"]
    historical_section = (historical_comparison(manifest, historical_dir)
                          if historical_dir is not None
                          else "## Historical comparison\n\nNo historical comparison directory was supplied.\n")
    report = [f"# {participant} final transfer extended descriptive report", "", "All quantities are post-inference summaries of saved predictions. Robot ratings are retained on their original 1–7 scale and used only for descriptive HIGH/LOW comparison; P(HIGH) is not mapped to that scale.", "", "## Task-level consensus and agreement", "", compact.to_markdown(index=False), "", historical_section]
    (transfer_dir / f"{participant}_extended_transfer_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Extended post-transfer report complete: {transfer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
