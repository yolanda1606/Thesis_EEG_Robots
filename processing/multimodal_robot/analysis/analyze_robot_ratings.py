#!/usr/bin/env python3
"""Read-only descriptive and paired analysis of canonical Robot self-ratings.

This script uses only the canonical Robot ratings CSV and each participant's
authoritative task metadata.  It does not load EEG, Robot windows, predictions,
or frozen models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.multimodal_robot.transfer.run_final_frozen_eeg_transfer import (
    DISPLAY_TASKS,
    robot_ratings,
)


PARTICIPANTS = [f"P{number:02d}" for number in range(1, 47)]
TASK_ORDER = [
    "pick_place",
    "shape_sorter_observation",
    "stack",
    "sisyphus",
    "shape_sorter_interaction",
    "shape_sorter_alone",
]
TASK_TYPE_ORDER = ["Observation", "Interaction", "Alone"]
OUTCOMES = ["valence", "arousal"]
CONFIG_DIR = ROOT / "processing/multimodal_robot/configs/participants"
RATINGS_PATH = ROOT / "data/Robot Ratings - Robot Ratings.csv"
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_RESAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/robot_ratings",
        help="New directory for ratings-only derived outputs; must not already exist.",
    )
    parser.add_argument(
        "--revised-general-figures-only",
        action="store_true",
        help=("Create new trajectory-free versions of the two general distribution "
              "figures in an existing ratings-analysis directory."),
    )
    return parser.parse_args()


def load_canonical_ratings() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load ratings via the canonical loader and attach YAML actual metadata."""
    raw = pd.read_csv(RATINGS_PATH)
    if len(raw) != 276:
        raise ValueError(f"Expected 276 raw rating rows; found {len(raw)}")
    if raw["Participant ID"].astype(str).nunique() != 46:
        raise ValueError("Raw ratings do not contain exactly 46 participants")
    if ((raw[["Valence", "Arousal"]] < 1) | (raw[["Valence", "Arousal"]] > 7)).any().any():
        raise ValueError("Raw valence/arousal ratings must both be within 1–7")

    rows: list[pd.DataFrame] = []
    validation_notes: list[dict[str, object]] = []
    for participant in PARTICIPANTS:
        config_path = CONFIG_DIR / f"{participant}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing participant metadata: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("participant") != participant:
            raise ValueError(f"{config_path}: participant identifier does not match filename")
        tasks = config.get("tasks", {})
        if set(tasks) != set(TASK_ORDER):
            raise ValueError(f"{config_path}: canonical task set does not match expected tasks")

        # Reuse the existing canonical raw-label mapping and rating loader.
        ratings = robot_ratings(participant)
        if len(ratings) != 12 or ratings.duplicated(["task", "target"]).any():
            raise ValueError(f"{participant}: canonical rating loader did not yield 6 tasks × 2 outcomes")
        metadata = []
        for task in TASK_ORDER:
            specification = tasks[task]
            category = str(specification.get("category", "")).lower()
            expected_category = (
                "observation" if task in TASK_ORDER[:3]
                else "interaction" if task in TASK_ORDER[3:5]
                else "alone"
            )
            if category != expected_category:
                raise ValueError(f"{participant} {task}: expected category {expected_category!r}; found {category!r}")
            condition = specification.get("condition", {})
            actual = str(condition.get("actual", "")).upper()
            speed_actual = str(condition.get("speed_actual", "")).upper()
            if category == "observation":
                if actual not in {"CONTROL", "FAULTY"} or speed_actual not in {"FAST", "SLOW"}:
                    raise ValueError(f"{participant} {task}: observation task lacks valid actual condition/speed")
                condition_label = f"{actual}_{speed_actual}"
            else:
                if actual or speed_actual:
                    raise ValueError(f"{participant} {task}: non-observation task unexpectedly has a condition")
                condition_label = "N/A"
            metadata.append({
                "task": task,
                "task_display": DISPLAY_TASKS[task],
                "task_type": category.title(),
                "actual_condition": actual or pd.NA,
                "actual_speed": speed_actual or pd.NA,
                "condition_label": condition_label,
                "planned_condition": str(condition.get("planned", "")).upper() or pd.NA,
                "planned_speed": str(condition.get("speed_planned", "")).upper() or pd.NA,
            })
        participant_rows = ratings.merge(pd.DataFrame(metadata), on="task", validate="many_to_one")
        participant_rows.insert(0, "participant", participant)
        rows.append(participant_rows)

        planned_actual = pd.DataFrame(metadata).query("task_type == 'Observation'")
        deviations = planned_actual.loc[
            (planned_actual.actual_condition != planned_actual.planned_condition)
            | (planned_actual.actual_speed != planned_actual.planned_speed),
            ["task", "planned_condition", "planned_speed", "actual_condition", "actual_speed"],
        ]
        for item in deviations.itertuples(index=False):
            validation_notes.append({"participant": participant, **item._asdict()})

    ratings = pd.concat(rows, ignore_index=True)
    if len(ratings) != 552 or ratings.groupby(["participant", "task"]).size().ne(2).any():
        raise ValueError("Canonical long ratings table is not 46 participants × 6 tasks × 2 outcomes")
    return ratings, validation_notes


def descriptive(values: pd.Series) -> dict[str, float | int]:
    """Return transparent rating descriptives, including a t-based mean CI."""
    values = pd.to_numeric(values, errors="raise").dropna()
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    sem = stats.sem(values) if n > 1 else np.nan
    margin = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
    q1, q3 = values.quantile([0.25, 0.75]) if n else (np.nan, np.nan)
    return {
        "n": n, "mean": mean, "sd": sd, "median": float(values.median()) if n else np.nan,
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(values.min()) if n else np.nan, "max": float(values.max()) if n else np.nan,
        "mean_ci_95_low": mean - margin, "mean_ci_95_high": mean + margin,
    }


def paired_statistics(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    """Analyze paired left-minus-right ratings using Wilcoxon and rank-biserial r."""
    pairs = pd.DataFrame({"left": left, "right": right}).dropna()
    delta = pairs.left - pairs.right
    nonzero = delta[delta.ne(0)]
    result: dict[str, float | int] = {
        "paired_n": len(delta), "mean_delta": float(delta.mean()), "median_delta": float(delta.median()),
        "positive_delta_count": int(delta.gt(0).sum()), "negative_delta_count": int(delta.lt(0).sum()),
        "zero_delta_count": int(delta.eq(0).sum()),
    }
    if nonzero.empty:
        result.update({"wilcoxon_statistic": np.nan, "wilcoxon_p_value": np.nan, "rank_biserial_correlation": np.nan,
                       "median_delta_ci_95_low": np.nan, "median_delta_ci_95_high": np.nan})
        return result
    test = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
    ranks = stats.rankdata(nonzero.abs())
    positive_ranks = float(ranks[nonzero.to_numpy() > 0].sum())
    negative_ranks = float(ranks[nonzero.to_numpy() < 0].sum())
    denominator = positive_ranks + negative_ranks
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
    bootstrap_medians = np.median(delta.to_numpy()[bootstrap_indices], axis=1)
    ci_low, ci_high = np.quantile(bootstrap_medians, [0.025, 0.975])
    result.update({
        "wilcoxon_statistic": float(test.statistic), "wilcoxon_p_value": float(test.pvalue),
        "rank_biserial_correlation": (positive_ranks - negative_ranks) / denominator,
        "median_delta_ci_95_low": float(ci_low), "median_delta_ci_95_high": float(ci_high),
    })
    return result


def task_level_summary(ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        for target in OUTCOMES:
            data = ratings[(ratings.task == task) & (ratings.target == target)]
            rows.append({"task": task, "task_display": DISPLAY_TASKS[task], "target": target, **descriptive(data.robot_rating)})
    return pd.DataFrame(rows)


def participant_task_types(ratings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each participant before any task-type comparison."""
    rows = []
    task_groups = {
        "Observation": TASK_ORDER[:3],
        "Interaction": TASK_ORDER[3:5],
        "Alone": TASK_ORDER[5:],
    }
    for participant in PARTICIPANTS:
        for target in OUTCOMES:
            data = ratings[(ratings.participant == participant) & (ratings.target == target)]
            for task_type, tasks in task_groups.items():
                values = data.set_index("task").loc[tasks, "robot_rating"]
                if len(values) != len(tasks):
                    raise ValueError(f"{participant} {target}: missing task-type ratings for {task_type}")
                rows.append({"participant": participant, "target": target, "task_type": task_type, "rating": float(values.mean()), "n_tasks_aggregated": len(tasks)})
    return pd.DataFrame(rows)


def task_type_summary(aggregates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = [], []
    for target in OUTCOMES:
        pivot = aggregates[aggregates.target == target].pivot(index="participant", columns="task_type", values="rating").reindex(columns=TASK_TYPE_ORDER)
        test = stats.friedmanchisquare(*(pivot[column] for column in TASK_TYPE_ORDER))
        tests.append({"target": target, "paired_n": len(pivot), "friedman_chi_square": float(test.statistic), "friedman_p_value": float(test.pvalue)})
        for task_type in TASK_TYPE_ORDER:
            rows.append({"target": target, "task_type": task_type, **descriptive(pivot[task_type]), "friedman_paired_n": len(pivot), "friedman_chi_square": float(test.statistic), "friedman_p_value": float(test.pvalue)})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def condition_outputs(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two pre-specified within-participant actual-condition contrasts."""
    observation = ratings[ratings.task_type.eq("Observation")].copy()
    wide = observation.pivot(index=["participant", "target"], columns="condition_label", values="robot_rating")
    for condition in ["FAULTY_FAST", "CONTROL_FAST", "CONTROL_SLOW", "FAULTY_SLOW"]:
        if condition not in wide:
            wide[condition] = np.nan
    delta_rows = []
    comparisons = []
    specifications = [
        ("fault_effect", "FAULTY_FAST", "CONTROL_FAST", "FAULTY_FAST minus CONTROL_FAST"),
        ("speed_effect", "CONTROL_FAST", "CONTROL_SLOW", "CONTROL_FAST minus CONTROL_SLOW"),
    ]
    for target in OUTCOMES:
        values = wide.xs(target, level="target").reindex(PARTICIPANTS)
        record = pd.DataFrame({"participant": PARTICIPANTS, "target": target})
        for name, left_name, right_name, label in specifications:
            left, right = values[left_name], values[right_name]
            record[f"{name}_left_condition"] = left_name
            record[f"{name}_right_condition"] = right_name
            record[f"{name}_left_rating"] = left.to_numpy()
            record[f"{name}_right_rating"] = right.to_numpy()
            record[f"{name}_delta"] = (left - right).to_numpy()
            record[f"{name}_included"] = left.notna().to_numpy() & right.notna().to_numpy()
            comparison = {
                "comparison": name, "comparison_label": label, "target": target,
                "left_condition": left_name, "right_condition": right_name,
                **{f"left_{key}": value for key, value in descriptive(left.dropna()).items()},
                **{f"right_{key}": value for key, value in descriptive(right.dropna()).items()},
                **paired_statistics(left, right),
            }
            comparisons.append(comparison)
        delta_rows.append(record)
    deltas = pd.concat(delta_rows, ignore_index=True)
    deltas["exclusion_note"] = np.where(
        deltas.participant.eq("P14") & deltas.target.notna(),
        "P14 Stack is FAULTY_SLOW; excluded only from CONTROL_FAST vs CONTROL_SLOW.",
        pd.NA,
    )
    return pd.DataFrame(comparisons), deltas


def save_figures(output: Path, ratings: pd.DataFrame, aggregates: pd.DataFrame, deltas: pd.DataFrame) -> None:
    """Save concise publication-style figures for the planned ratings analyses."""
    colors = {"valence": "#377eb8", "arousal": "#e41a1c"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        data = ratings[ratings.target == target]
        values = [data[data.task == task].robot_rating for task in TASK_ORDER]
        axis.boxplot(values, tick_labels=[DISPLAY_TASKS[task] for task in TASK_ORDER], showfliers=False)
        for _, participant_data in data.groupby("participant"):
            series = participant_data.set_index("task").loc[TASK_ORDER, "robot_rating"]
            axis.plot(range(1, 7), series, color="0.55", alpha=0.18, linewidth=0.7, zorder=1)
        for position, series in enumerate(values, start=1):
            axis.scatter(np.full(len(series), position) + rng.normal(0, 0.045, len(series)), series, s=10, color=colors[target], alpha=0.42, zorder=2)
        axis.set_title(target.title()); axis.set_xlabel("Task"); axis.set_ylim(0.75, 7.25); axis.set_yticks(range(1, 8)); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Self-reported rating (1–7)")
    fig.suptitle("Robot ratings by task; faint lines connect each participant", y=1.02)
    fig.tight_layout(); fig.savefig(output / "task_rating_distributions.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        data = aggregates[aggregates.target == target]
        values = [data[data.task_type == task_type].rating for task_type in TASK_TYPE_ORDER]
        axis.boxplot(values, tick_labels=TASK_TYPE_ORDER, showfliers=False)
        pivot = data.pivot(index="participant", columns="task_type", values="rating").reindex(columns=TASK_TYPE_ORDER)
        for _, row in pivot.iterrows(): axis.plot(range(1, 4), row, color="0.55", alpha=0.25, linewidth=0.8)
        axis.set_title(target.title()); axis.set_ylim(0.75, 7.25); axis.set_yticks(range(1, 8)); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Participant-level task-type rating (1–7)")
    fig.suptitle("Participant-level task-type aggregates", y=1.02)
    fig.tight_layout(); fig.savefig(output / "task_type_distributions.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    for comparison, filename, title in [
        ("fault_effect", "fault_effect_paired.png", "Fault effect: FAULTY_FAST versus CONTROL_FAST"),
        ("speed_effect", "speed_effect_paired.png", "Speed effect: CONTROL_FAST versus CONTROL_SLOW"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), sharey=True)
        for axis, target in zip(axes, OUTCOMES):
            data = deltas[(deltas.target == target) & deltas[f"{comparison}_included"]]
            left, right = f"{comparison}_left_rating", f"{comparison}_right_rating"
            for row in data.itertuples(index=False): axis.plot([1, 2], [getattr(row, left), getattr(row, right)], color="0.55", alpha=0.35, linewidth=0.8)
            axis.scatter(np.ones(len(data)), data[left], color=colors[target], s=18, label=data[f"{comparison}_left_condition"].iloc[0])
            axis.scatter(np.full(len(data), 2), data[right], color="white", edgecolor=colors[target], s=25, label=data[f"{comparison}_right_condition"].iloc[0])
            axis.set_xlim(.7, 2.3); axis.set_xticks([1, 2], [data[f"{comparison}_left_condition"].iloc[0], data[f"{comparison}_right_condition"].iloc[0]])
            axis.set_title(f"{target.title()} (n={len(data)})"); axis.set_ylim(.75, 7.25); axis.set_yticks(range(1, 8)); axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Self-reported rating (1–7)")
        fig.suptitle(title, y=1.02); fig.tight_layout(); fig.savefig(output / filename, dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    for task in TASK_ORDER:
        data = ratings[ratings.task == task].pivot(index="participant", columns="target", values="robot_rating")
        axis.scatter(data.valence + rng.normal(0, .04, len(data)), data.arousal + rng.normal(0, .04, len(data)), alpha=.20, s=20)
        centroid = data[["valence", "arousal"]].mean()
        axis.scatter(centroid.valence, centroid.arousal, s=115, edgecolor="black", linewidth=.6, label=DISPLAY_TASKS[task])
    axis.set(xlim=(.75, 7.25), ylim=(.75, 7.25), xticks=range(1, 8), yticks=range(1, 8), xlabel="Valence rating (1–7)", ylabel="Arousal rating (1–7)", title="Robot valence–arousal space: task centroids and raw ratings")
    axis.grid(alpha=.25); axis.legend(title="Task", loc="best")
    fig.tight_layout(); fig.savefig(output / "valence_arousal_space.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def save_revised_general_distribution_figures(output: Path, ratings: pd.DataFrame, aggregates: pd.DataFrame) -> None:
    """Save new distribution figures without participant trajectory lines."""
    destinations = [
        output / "task_rating_distributions_no_trajectories.png",
        output / "task_type_distributions_no_trajectories.png",
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite revised distribution figures: " + ", ".join(map(str, existing)))
    colors = {"valence": "#377eb8", "arousal": "#e41a1c"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        data = ratings[ratings.target == target]
        values = [data[data.task == task].robot_rating for task in TASK_ORDER]
        axis.boxplot(values, tick_labels=[DISPLAY_TASKS[task] for task in TASK_ORDER], showfliers=False)
        for position, series in enumerate(values, start=1):
            axis.scatter(np.full(len(series), position) + rng.normal(0, 0.045, len(series)), series, s=12, color=colors[target], alpha=0.48, zorder=2)
        axis.set_title(target.title()); axis.set_xlabel("Task"); axis.set_ylim(0.75, 7.25); axis.set_yticks(range(1, 8)); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Self-reported rating (1–7)")
    fig.suptitle("Robot ratings by task", y=1.02)
    fig.tight_layout(); fig.savefig(destinations[0], dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        data = aggregates[aggregates.target == target]
        values = [data[data.task_type == task_type].rating for task_type in TASK_TYPE_ORDER]
        axis.boxplot(values, tick_labels=TASK_TYPE_ORDER, showfliers=False)
        for position, series in enumerate(values, start=1):
            axis.scatter(np.full(len(series), position) + rng.normal(0, 0.045, len(series)), series, s=12, color=colors[target], alpha=0.48, zorder=2)
        axis.set_title(target.title()); axis.set_ylim(0.75, 7.25); axis.set_yticks(range(1, 8)); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Participant-level task-type rating (1–7)")
    fig.suptitle("Participant-level task-type aggregates", y=1.02)
    fig.tight_layout(); fig.savefig(destinations[1], dpi=220, bbox_inches="tight"); plt.close(fig)


def markdown_summary(
    task_summary: pd.DataFrame,
    task_type_summary_frame: pd.DataFrame,
    friedman: pd.DataFrame,
    comparisons: pd.DataFrame,
    condition_counts: pd.Series,
    deviations: list[dict[str, object]],
) -> str:
    validation = [
        "- Raw ratings: 276 rows (46 participants × 6 tasks).",
        "- All valence and arousal ratings are within the original 1–7 scale.",
        "- Task type mapping: Observation = PnP/SSObs/St; Interaction = Sisyphus/SSInt; Alone = SSAlone.",
        "- Actual observation-condition counts: " + ", ".join(f"{label}={count}" for label, count in condition_counts.items()) + ".",
        "- P14 Stack is retained as FAULTY_SLOW in descriptive outputs and is excluded only from the CONTROL_FAST versus CONTROL_SLOW contrast.",
        "- P20 has planned-versus-actual speed deviations; actual metadata is used throughout.",
    ]
    contrast_columns = ["comparison", "target", "left_condition", "right_condition", "paired_n", "mean_delta", "median_delta", "median_delta_ci_95_low", "median_delta_ci_95_high", "positive_delta_count", "negative_delta_count", "zero_delta_count", "wilcoxon_statistic", "wilcoxon_p_value", "rank_biserial_correlation"]
    return "\n".join([
        "# Robot participant-ratings analysis", "",
        "This is a ratings-only, read-only analysis of the original 1–7 Robot valence and arousal self-reports. It does not use EEG, model predictions, or transfer outputs.", "",
        "## Validation", "", *validation, "",
        "## Task-level ratings", "", task_summary.to_markdown(index=False), "",
        "## Participant-level task-type aggregates", "", task_type_summary_frame.to_markdown(index=False), "",
        "## Repeated-measures omnibus tests", "", friedman.to_markdown(index=False), "",
        "## Pre-specified actual-condition contrasts", "", comparisons.loc[:, contrast_columns].to_markdown(index=False), "",
        "## Interpretation boundary", "", "The Wilcoxon tests are the two pre-specified paired contrasts separately for valence and arousal. The task-type results are Friedman omnibus tests; no unplanned post-hoc pairwise testing was performed. Effect size is rank-biserial correlation for non-zero paired differences. Paired-difference confidence intervals are deterministic percentile-bootstrap 95% intervals for the median difference.", "",
        "## Metadata deviations recorded", "", pd.DataFrame(deviations).to_markdown(index=False), "",
    ])


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.revised_general_figures_only:
        if not output.is_dir():
            raise FileNotFoundError(f"Missing existing ratings-analysis directory: {output}")
        ratings, _ = load_canonical_ratings()
        aggregates = participant_task_types(ratings)
        save_revised_general_distribution_figures(output, ratings, aggregates)
        print(json.dumps({"output": str(output), "artifacts": [
            str(output / "task_rating_distributions_no_trajectories.png"),
            str(output / "task_type_distributions_no_trajectories.png"),
        ]}, indent=2))
        return 0
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing ratings analysis directory: {output}")
    ratings, deviations = load_canonical_ratings()
    condition_counts = ratings.loc[
        ratings.task_type.eq("Observation") & ratings.target.eq("valence"), "condition_label"
    ].value_counts().reindex(["FAULTY_FAST", "CONTROL_FAST", "CONTROL_SLOW", "FAULTY_SLOW"], fill_value=0)
    if condition_counts.to_dict() != {"FAULTY_FAST": 46, "CONTROL_FAST": 46, "CONTROL_SLOW": 45, "FAULTY_SLOW": 1}:
        raise ValueError(f"Unexpected participant-task actual-condition counts: {condition_counts.to_dict()}")
    p14 = ratings[(ratings.participant == "P14") & (ratings.task == "stack")]
    if set(p14.condition_label) != {"FAULTY_SLOW"}:
        raise ValueError("P14 Stack must remain FAULTY_SLOW")
    if not any(note["participant"] == "P20" for note in deviations):
        raise ValueError("Expected planned-versus-actual P20 deviation was not found")

    task_summary = task_level_summary(ratings)
    aggregates = participant_task_types(ratings)
    type_summary, friedman = task_type_summary(aggregates)
    comparisons, deltas = condition_outputs(ratings)
    paired_counts = comparisons.groupby("comparison").paired_n.first().to_dict()
    if paired_counts != {"fault_effect": 46, "speed_effect": 45}:
        raise ValueError("Primary contrast paired sample sizes do not match the validated design")

    output.mkdir(parents=True)
    task_summary.to_csv(output / "task_level_summary.csv", index=False)
    type_summary.to_csv(output / "task_type_summary.csv", index=False)
    comparisons.to_csv(output / "condition_comparison.csv", index=False)
    deltas.to_csv(output / "participant_condition_deltas.csv", index=False)
    save_figures(output, ratings, aggregates, deltas)
    (output / "robot_ratings_summary.md").write_text(markdown_summary(task_summary, type_summary, friedman, comparisons, condition_counts, deviations) + "\n", encoding="utf-8")
    manifest = {
        "purpose": "Read-only Robot participant ratings analysis",
        "raw_ratings_source": str(RATINGS_PATH),
        "participant_metadata_source": str(CONFIG_DIR),
        "uses_actual_condition_fields": ["condition.actual", "condition.speed_actual"],
        "uses_predictions": False,
        "participant_count": 46,
        "raw_rating_rows": 276,
        "long_rating_rows": len(ratings),
        "actual_observation_condition_counts": condition_counts.to_dict(),
        "primary_contrast_paired_n": paired_counts,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "condition_comparison_paired_n": manifest["primary_contrast_paired_n"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
