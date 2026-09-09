#!/usr/bin/env python3
"""Build a read-only descriptive comparison of completed final Robot transfers.

This script reads already-generated final Image-to-Robot transfer outputs.  It
does not load Robot EEG, fit models, calibrate probabilities, or change the
frozen Image model selection.  Task probability and verdict follow the frozen
consensus-first definition: median over Robot windows of the per-window median
across ranks 1--3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRANSFER_ROOT = ROOT / "outputs/robot_transfer/focused_personalized_binary_v1"
DEFAULT_TOP3 = ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv"
DEFAULT_OUTPUT = DEFAULT_TRANSFER_ROOT / "pilot_comparison_v1"
TASK_ORDER = (
    "pick_place",
    "shape_sorter_observation",
    "stack",
    "sisyphus",
    "shape_sorter_interaction",
    "shape_sorter_alone",
)
TASK_DISPLAY = {
    "pick_place": "PnP",
    "shape_sorter_observation": "SSObs",
    "stack": "St",
    "sisyphus": "Sisyphus",
    "shape_sorter_interaction": "SSInt",
    "shape_sorter_alone": "SSAlone",
}
TARGET_ORDER = ("valence", "arousal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default="P19,P27,P36", help="Ordered comma-separated participant IDs.")
    parser.add_argument("--transfer-root", type=Path, default=DEFAULT_TRANSFER_ROOT)
    parser.add_argument("--top3-csv", type=Path, default=DEFAULT_TOP3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_participants(value: str) -> list[str]:
    participants = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not participants or len(set(participants)) != len(participants):
        raise ValueError("--participants must be a non-empty, unique list")
    return participants


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required final-transfer output is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return frame


def frozen_image_ba(top3: pd.DataFrame, participant: str, target: str) -> dict[str, float]:
    rows = top3[(top3["participant"].eq(participant)) & (top3["target"].eq(target))].copy()
    if len(rows) != 3 or sorted(rows["transfer_rank"].tolist()) != [1, 2, 3]:
        raise ValueError(f"{participant} {target}: expected exactly frozen EEG-only ranks 1--3")
    selected = rows["actual_selected_feature_names"].fillna("")
    if selected.str.contains("video_", regex=False).any() or selected.str.contains("face", case=False, regex=False).any():
        raise ValueError(f"{participant} {target}: top-3 rows are not EEG-only")
    values = rows.set_index("transfer_rank")["mean_outer_cv_balanced_accuracy"].astype(float)
    return {
        "rank1_image_ba": float(values.loc[1]),
        "rank2_image_ba": float(values.loc[2]),
        "rank3_image_ba": float(values.loc[3]),
        "median_top3_image_ba": float(values.median()),
    }


def strict_unanimous_agreement(predictions: pd.DataFrame, participant: str, target: str, task: str) -> float:
    subset = predictions[(predictions["target"].eq(target)) & (predictions["task"].eq(task))]
    keys = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]
    wide = subset.pivot(index=keys, columns="model_rank", values="original_hard_prediction")
    if sorted(wide.columns.tolist()) != [1, 2, 3]:
        raise ValueError(f"{participant} {target} {task}: missing a frozen top-3 hard-prediction rank")
    wide = wide.dropna()
    if wide.empty:
        raise ValueError(f"{participant} {target} {task}: no windows have all three hard predictions")
    return float((wide.nunique(axis=1) == 1).mean())


def resolved_robot_ratings(directory: Path, participant: str) -> pd.DataFrame:
    """Load existing descriptive ratings and derive their fixed LOW/HIGH class."""
    ratings = read_csv(
        directory / f"{participant}_robot_rating_comparison.csv",
        {"target", "task", "robot_rating"},
    ).copy()
    ratings = ratings.loc[ratings["target"].isin(TARGET_ORDER) & ratings["task"].isin(TASK_ORDER)].copy()
    expected = {(target, task) for target in TARGET_ORDER for task in TASK_ORDER}
    observed = set(zip(ratings["target"], ratings["task"]))
    if observed != expected or ratings.duplicated(["target", "task"]).any():
        raise ValueError(f"{participant}: Robot rating coverage does not match the six-task/two-target final design")
    ratings["robot_rating"] = pd.to_numeric(ratings["robot_rating"], errors="raise")
    if not np.isfinite(ratings["robot_rating"].to_numpy(dtype=float)).all():
        raise ValueError(f"{participant}: Robot ratings contain non-finite values")
    ratings["robot_rating_class"] = np.where(ratings["robot_rating"] >= 4.0, "HIGH", "LOW")
    if "robot_condition" not in ratings:
        ratings["robot_condition"] = pd.NA
    return ratings[["target", "task", "robot_rating", "robot_rating_class", "robot_condition"]]


def task_rows_for_participant(participant: str, transfer_root: Path, top3: pd.DataFrame) -> pd.DataFrame:
    directory = transfer_root / participant
    consensus = read_csv(
        directory / f"{participant}_task_consensus_analysis.csv",
        {"participant", "target", "task", "median_top3_consensus_probability", "median_top3_probability_range"},
    )
    ratings = resolved_robot_ratings(directory, participant)
    if "robot_condition" not in consensus:
        consensus["robot_condition"] = pd.NA
    consensus = consensus.loc[:, [
        "participant", "target", "task", "robot_condition",
        "median_top3_consensus_probability", "median_top3_probability_range",
    ]]
    consensus = consensus.merge(
        ratings,
        on=["target", "task"],
        how="left",
        validate="one_to_one",
        suffixes=("_consensus", "_rating"),
    )
    consensus["robot_condition"] = consensus["robot_condition_consensus"].combine_first(
        consensus["robot_condition_rating"]
    ).fillna("N/A")
    predictions = read_csv(
        directory / f"{participant}_window_predictions.csv",
        {"participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s", "target",
         "model_rank", "original_hard_prediction"},
    )
    shift = read_csv(
        directory / f"{participant}_selected_feature_shift_task_summary.csv",
        {"target", "model_rank", "task", "median_nearest_image_distance", "fraction_feature_values_outside"},
    )
    expected = {(target, task) for target in TARGET_ORDER for task in TASK_ORDER}
    observed = set(zip(consensus["target"], consensus["task"]))
    if observed != expected or consensus.duplicated(["target", "task"]).any():
        raise ValueError(f"{participant}: consensus task coverage does not match the six-task/two-target final design")
    expected_shift = {(target, rank, task) for target in TARGET_ORDER for rank in (1, 2, 3) for task in TASK_ORDER}
    observed_shift = set(zip(shift["target"], shift["model_rank"], shift["task"]))
    if observed_shift != expected_shift or shift.duplicated(["target", "model_rank", "task"]).any():
        raise ValueError(f"{participant}: selected-feature shift coverage does not match ranks 1--3 for all tasks")

    rows: list[dict[str, object]] = []
    for item in consensus.itertuples(index=False):
        if item.target not in TARGET_ORDER or item.task not in TASK_ORDER:
            continue
        task_shift = shift[(shift["target"].eq(item.target)) & (shift["task"].eq(item.task))].set_index("model_rank")
        nearest = task_shift["median_nearest_image_distance"].astype(float)
        outside = task_shift["fraction_feature_values_outside"].astype(float)
        p_high = float(item.median_top3_consensus_probability)
        verdict = "HIGH" if p_high >= 0.5 else "LOW"
        rating_class = str(item.robot_rating_class).upper()
        if rating_class not in {"LOW", "HIGH"}:
            raise ValueError(f"{participant} {item.target} {item.task}: invalid Robot rating class {rating_class!r}")
        rows.append({
            "participant": participant,
            "target": item.target,
            "task": item.task,
            "task_display": TASK_DISPLAY[item.task],
            "robot_condition": item.robot_condition,
            "robot_rating": float(item.robot_rating),
            "robot_rating_class": rating_class,
            "task_p_high": p_high,
            "task_verdict": verdict,
            "rating_match": bool(verdict == rating_class),
            "unanimous_agreement_rate": strict_unanimous_agreement(predictions, participant, item.target, item.task),
            "median_probability_range": float(item.median_top3_probability_range),
            "nearest_image_distance_rank1": float(nearest.loc[1]),
            "nearest_image_distance_rank2": float(nearest.loc[2]),
            "nearest_image_distance_rank3": float(nearest.loc[3]),
            "median_nearest_image_distance_across_models": float(nearest.median()),
            "outside_range_fraction_rank1": float(outside.loc[1]),
            "outside_range_fraction_rank2": float(outside.loc[2]),
            "outside_range_fraction_rank3": float(outside.loc[3]),
            "outside_range_fraction_across_models": float(outside.median()),
            **frozen_image_ba(top3, participant, item.target),
        })
    return pd.DataFrame(rows)


def summary_rows(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (participant, target), part in tasks.groupby(["participant", "target"], sort=False):
        part = part.set_index("task").loc[list(TASK_ORDER)].reset_index()
        n_compared = int(part["rating_match"].notna().sum())
        n_matched = int(part["rating_match"].sum())
        rows.append({
            "participant": participant,
            "target": target,
            "rank1_image_ba": float(part["rank1_image_ba"].iloc[0]),
            "rank2_image_ba": float(part["rank2_image_ba"].iloc[0]),
            "rank3_image_ba": float(part["rank3_image_ba"].iloc[0]),
            "median_top3_image_ba": float(part["median_top3_image_ba"].iloc[0]),
            "robot_match_rate": n_matched / n_compared if n_compared else np.nan,
            "n_tasks_compared": n_compared,
            "n_tasks_matched": n_matched,
            "median_task_p_high": float(part["task_p_high"].median()),
            "median_unanimous_agreement": float(part["unanimous_agreement_rate"].median()),
            "median_probability_range": float(part["median_probability_range"].median()),
            "median_nearest_image_distance": float(part["median_nearest_image_distance_across_models"].median()),
            "median_outside_range_fraction": float(part["outside_range_fraction_across_models"].median()),
        })
    return pd.DataFrame(rows).sort_values(["participant", "target"], key=lambda col: col.map({"P19": 0, "P27": 1, "P36": 2, "valence": 0, "arousal": 1}).fillna(99)).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def report(summary: pd.DataFrame, tasks: pd.DataFrame) -> str:
    compact = summary[[
        "participant", "target", "median_top3_image_ba", "robot_match_rate", "n_tasks_matched",
        "n_tasks_compared", "median_task_p_high", "median_unanimous_agreement", "median_probability_range",
        "median_nearest_image_distance", "median_outside_range_fraction",
    ]].copy()
    compact.columns = [
        "Participant", "Target", "Image BA", "Robot match", "Matched tasks", "Compared tasks",
        "Median task P(HIGH)", "Median unanimous agreement", "Median prob. range",
        "Nearest-Image distance", "Outside-range fraction",
    ]
    detail = tasks[["participant", "target", "task_display", "task_p_high", "task_verdict", "robot_rating_class", "rating_match"]].copy()
    detail.columns = ["Participant", "Target", "Task", "Task P(HIGH)", "Verdict", "Rating class", "Match"]
    return "\n".join([
        "# Pilot final Image-to-Robot transfer comparison",
        "",
        "Descriptive comparison of existing final frozen EEG-only Image-to-Robot transfer outputs for P19, P27, and P36. No inference, retraining, calibration, threshold optimization, or Robot-based model selection was performed.",
        "",
        "Task probability is defined consistently as `median_windows(median_models(P(HIGH)))`; task HIGH is `P(HIGH) >= 0.5`. Robot ratings are external descriptive comparisons only.",
        "",
        "## Participant × target summary",
        "",
        markdown_table(compact),
        "",
        "## Task-level verdict comparison",
        "",
        markdown_table(detail),
        "",
        "## Interpretation limits",
        "",
        "Image BA is the median of the three frozen EEG-only outer-CV balanced accuracies. Agreement is the median across tasks of the fraction of windows with all three frozen hard predictions equal. Shift quantities remain model-specific until their prescribed median aggregation. These descriptive pilot values do not establish that Image BA, model disagreement, or feature shift causes Robot-rating agreement or disagreement.",
        "",
    ])


def main() -> int:
    args = parse_args()
    participants = parse_participants(args.participants)
    transfer_root = args.transfer_root.resolve()
    top3_path = args.top3_csv.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison output directory: {output}")
    top3 = read_csv(top3_path, {"participant", "target", "transfer_rank", "mean_outer_cv_balanced_accuracy", "actual_selected_feature_names"})
    task_frames = [task_rows_for_participant(participant, transfer_root, top3) for participant in participants]
    tasks = pd.concat(task_frames, ignore_index=True)
    tasks["target"] = pd.Categorical(tasks["target"], TARGET_ORDER, ordered=True)
    tasks["task"] = pd.Categorical(tasks["task"], TASK_ORDER, ordered=True)
    tasks = tasks.sort_values(["participant", "target", "task"]).reset_index(drop=True)
    summary = summary_rows(tasks)
    output.mkdir(parents=True)
    summary.to_csv(output / "pilot_transfer_comparison_summary.csv", index=False)
    tasks.to_csv(output / "pilot_transfer_comparison_by_task.csv", index=False)
    (output / "pilot_transfer_comparison.md").write_text(report(summary, tasks), encoding="utf-8")
    print(f"Wrote descriptive pilot comparison: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}")
