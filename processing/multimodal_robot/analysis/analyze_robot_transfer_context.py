#!/usr/bin/env python3
"""Read-only task-type and actual-condition analysis of frozen Robot transfer."""
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
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
COHORT_PATH = ROOT / "outputs/robot_transfer/cohort_analysis/cohort_task_level.csv"
CONFIG_DIR = ROOT / "processing/multimodal_robot/configs/participants"
PARTICIPANTS = [f"P{i:02d}" for i in range(1, 47)]
BRANCHES = ["eeg_only", "modality_agnostic"]
TARGETS = ["valence", "arousal"]
TASK_TYPES = {"Observation": ["pick_place", "shape_sorter_observation", "stack"], "Interaction": ["sisyphus", "shape_sorter_interaction"], "Alone": ["shape_sorter_alone"]}
METRICS = {
    "task_consensus_p_high": "predicted_probability",
    "absolute_margin": "absolute_margin",
    "unanimous_agreement_rate": "unanimous_agreement_rate",
    "disagreement_rate": "disagreement_rate",
    "mean_top3_probability_range": "mean_top3_probability_range",
    "median_top3_probability_range": "median_top3_probability_range",
    "complete_top3_coverage": "fraction_complete_top3_windows",
    "nearest_image_distance": "top3_median_median_nearest_image_distance",
    "outside_selected_fraction": "top3_median_fraction_feature_values_outside",
    "median_abs_image_z": "top3_median_median_abs_image_z",
}
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_RESAMPLES = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_transfer/context_analysis")
    return parser.parse_args()


def bh(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float); valid = p_values.dropna()
    if valid.empty: return result
    ordered = valid.sort_values(); values = ordered.to_numpy() * len(ordered) / np.arange(1, len(ordered) + 1)
    result.loc[ordered.index] = np.minimum(np.minimum.accumulate(values[::-1])[::-1], 1)
    return result


def descriptive(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna(); q1, q3 = values.quantile([.25, .75]) if len(values) else (np.nan, np.nan)
    return {"n_participants": len(values), "mean": float(values.mean()) if len(values) else np.nan, "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan, "median": float(values.median()) if len(values) else np.nan, "q1": float(q1), "q3": float(q3), "iqr": float(q3-q1)}


def canonical_conditions() -> pd.DataFrame:
    rows = []
    for participant in PARTICIPANTS:
        path = CONFIG_DIR / f"{participant}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for task, specification in config["tasks"].items():
            if specification["category"] != "observation": continue
            condition = specification["condition"]
            actual = f"{str(condition['actual']).upper()}_{str(condition['speed_actual']).upper()}"
            rows.append({"participant": participant, "task": task, "actual_condition": actual})
    conditions = pd.DataFrame(rows)
    expected = {"FAULTY_FAST": 46, "CONTROL_FAST": 46, "CONTROL_SLOW": 45, "FAULTY_SLOW": 1}
    if conditions.actual_condition.value_counts().to_dict() != expected: raise ValueError("Canonical actual-condition assignments do not match validated design")
    p14 = conditions[(conditions.participant == "P14") & (conditions.task == "stack")]
    if p14.actual_condition.tolist() != ["FAULTY_SLOW"]: raise ValueError("P14 Stack special case is absent")
    return conditions


def load_cohort(conditions: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_PATH)
    needed = {"participant", "branch", "target", "task", *METRICS.values()}
    missing = needed - set(cohort)
    if missing: raise ValueError(f"Cohort table missing fields: {sorted(missing)}")
    if len(cohort) != 1072 or cohort.duplicated(["branch", "participant", "target", "task"]).any(): raise ValueError("Unexpected cohort task-level row structure")
    cohort = cohort.merge(conditions, on=["participant", "task"], how="left", validate="many_to_one")
    cohort["task_type"] = cohort.task.map({task: task_type for task_type, tasks in TASK_TYPES.items() for task in tasks})
    if cohort.task_type.isna().any(): raise ValueError("Task type metadata missing")
    return cohort


def task_type_tables(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = cohort.groupby(["branch", "participant", "target", "task_type"], as_index=False)[list(METRICS.values())].mean().rename(columns={value: key for key, value in METRICS.items()})
    summary_rows, test_rows = [], []
    for (branch, target), data in participant.groupby(["branch", "target"], sort=False):
        for task_type in TASK_TYPES:
            values = data[data.task_type == task_type]
            for metric in METRICS: summary_rows.append({"branch": branch, "target": target, "task_type": task_type, "metric": metric, **descriptive(values[metric])})
        for metric in METRICS:
            pivot = data.pivot(index="participant", columns="task_type", values=metric).reindex(columns=list(TASK_TYPES))
            complete = pivot.dropna()
            values = complete.to_numpy(dtype=float)
            if len(complete) < 2:
                statistic, p_value, test_status = np.nan, np.nan, "undefined_fewer_than_two_complete_participants"
            elif np.isclose(values.max() - values.min(), 0.0):
                # Friedman's tie correction is zero when every input is identical.
                statistic, p_value, test_status = np.nan, np.nan, "undefined_all_values_identical"
            else:
                test = stats.friedmanchisquare(*(complete[column] for column in TASK_TYPES))
                statistic, p_value, test_status = float(test.statistic), float(test.pvalue), "computed"
            test_rows.append({"branch": branch, "target": target, "metric": metric, "paired_n": len(complete), "friedman_chi_square": statistic, "p_value_raw": p_value, "test_status": test_status})
    tests = pd.DataFrame(test_rows); tests["p_value_fdr_bh"] = tests.groupby(["branch", "target"], group_keys=False).p_value_raw.apply(bh); tests["fdr_significant_0_05"] = tests.p_value_fdr_bh.lt(.05)
    return pd.DataFrame(summary_rows), tests


def paired_stats(delta: pd.Series) -> dict[str, float | int]:
    delta = delta.dropna(); nonzero = delta[delta.ne(0)]
    result = {"paired_n": len(delta), "mean_delta": float(delta.mean()), "median_delta": float(delta.median()), "positive_delta_count": int(delta.gt(0).sum()), "negative_delta_count": int(delta.lt(0).sum()), "zero_delta_count": int(delta.eq(0).sum())}
    if nonzero.empty:
        return result | {"wilcoxon_statistic": np.nan, "p_value_raw": np.nan, "rank_biserial_correlation": np.nan, "median_delta_ci_95_low": np.nan, "median_delta_ci_95_high": np.nan}
    test = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
    ranks = stats.rankdata(nonzero.abs()); positive = ranks[nonzero.to_numpy() > 0].sum(); negative = ranks[nonzero.to_numpy() < 0].sum()
    rng = np.random.default_rng(BOOTSTRAP_SEED); indices = rng.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta))); medians = np.median(delta.to_numpy()[indices], axis=1); low, high = np.quantile(medians, [.025, .975])
    return result | {"wilcoxon_statistic": float(test.statistic), "p_value_raw": float(test.pvalue), "rank_biserial_correlation": float((positive-negative)/(positive+negative)), "median_delta_ci_95_low": float(low), "median_delta_ci_95_high": float(high)}


def condition_tables(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observation = cohort[cohort.actual_condition.notna()].copy().rename(columns={value: key for key, value in METRICS.items()})
    summary_rows = []
    for (branch, target, condition), data in observation.groupby(["branch", "target", "actual_condition"], sort=False):
        for metric in METRICS: summary_rows.append({"branch": branch, "target": target, "actual_condition": condition, "metric": metric, **descriptive(data[metric])})
    delta_rows, test_rows = [], []
    comparisons = [("fault_effect", "FAULTY_FAST", "CONTROL_FAST"), ("speed_effect", "CONTROL_FAST", "CONTROL_SLOW")]
    for (branch, target), data in observation.groupby(["branch", "target"], sort=False):
        for comparison, left, right in comparisons:
            for metric in METRICS:
                wide = data.pivot(index="participant", columns="actual_condition", values=metric).reindex(columns=[left, right])
                for participant, row in wide.iterrows(): delta_rows.append({"branch": branch, "target": target, "participant": participant, "comparison": comparison, "left_condition": left, "right_condition": right, "metric": metric, "left_value": row[left], "right_value": row[right], "delta_left_minus_right": row[left]-row[right], "included": bool(pd.notna(row[left]) and pd.notna(row[right]))})
                paired = wide.dropna(); expected_n = 46 if comparison == "fault_effect" else 45
                if len(paired) != expected_n: raise ValueError(f"{branch} {target} {comparison} {metric}: expected n={expected_n}; found {len(paired)}")
                test_rows.append({"branch": branch, "target": target, "comparison": comparison, "left_condition": left, "right_condition": right, "metric": metric, **paired_stats(paired[left]-paired[right])})
    tests = pd.DataFrame(test_rows); tests["p_value_fdr_bh"] = tests.groupby(["branch", "target", "comparison"], group_keys=False).p_value_raw.apply(bh); tests["fdr_significant_0_05"] = tests.p_value_fdr_bh.lt(.05)
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows), tests


def box_task_types(path: Path, cohort: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    participant = cohort.groupby(["branch", "participant", "target", "task_type"], as_index=False).predicted_probability.mean()
    for axis, branch, target in zip(axes.flat, ["eeg_only", "eeg_only", "modality_agnostic", "modality_agnostic"], ["valence", "arousal", "valence", "arousal"]):
        data = participant[(participant.branch == branch) & (participant.target == target)]; values = [data[data.task_type == kind].predicted_probability for kind in TASK_TYPES]
        axis.boxplot(values, tick_labels=list(TASK_TYPES), showfliers=False)
        for _, row in data.pivot(index="participant", columns="task_type", values="predicted_probability").reindex(columns=list(TASK_TYPES)).iterrows(): axis.plot([1,2,3], row, color="0.65", alpha=.25, linewidth=.7)
        axis.axhline(.5, color="0.35", linestyle="--", linewidth=.8); axis.set(title=f"{branch.replace('_',' ').title()}: {target.title()}", ylim=(0,1)); axis.grid(axis="y", alpha=.25)
    axes[0,0].set_ylabel("Participant task-type mean P(HIGH)"); axes[1,0].set_ylabel("Participant task-type mean P(HIGH)"); fig.suptitle("Frozen predictor P(HIGH) by task type", y=.995); fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def paired_figure(path: Path, deltas: pd.DataFrame, comparison: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
    for axis, branch, target in zip(axes.flat, ["eeg_only", "eeg_only", "modality_agnostic", "modality_agnostic"], ["valence", "arousal", "valence", "arousal"]):
        data = deltas[(deltas.comparison == comparison) & (deltas.branch == branch) & (deltas.target == target) & (deltas.metric == "task_consensus_p_high") & deltas.included]
        left, right = data.left_condition.iloc[0], data.right_condition.iloc[0]
        for row in data.itertuples(index=False): axis.plot([1,2], [row.left_value, row.right_value], color="0.65", alpha=.45, linewidth=.7)
        axis.scatter(np.ones(len(data)), data.left_value, color="#d95f02", s=18); axis.scatter(np.full(len(data),2), data.right_value, color="#1b9e77", s=18)
        axis.axhline(.5, color="0.35", linestyle="--", linewidth=.8); axis.set(xticks=[1,2], xticklabels=[left,right], title=f"{branch.replace('_',' ').title()}: {target.title()} (n={len(data)})", ylim=(0,1)); axis.grid(axis="y", alpha=.25)
    axes[0,0].set_ylabel("Task consensus P(HIGH)"); axes[1,0].set_ylabel("Task consensus P(HIGH)"); fig.suptitle(title, y=.995); fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def report(task_summary: pd.DataFrame, task_tests: pd.DataFrame, condition_tests: pd.DataFrame) -> str:
    type_p = task_summary[task_summary.metric.eq("task_consensus_p_high")]
    type_tests_p = task_tests[task_tests.metric.eq("task_consensus_p_high")]
    condition_p = condition_tests[condition_tests.metric.eq("task_consensus_p_high")]
    return "\n".join(["# Frozen Robot-transfer context analysis", "", "Read-only analysis of saved final cohort task-level outputs. Participant-level aggregation precedes every task-type or condition test; no overlapping windows were treated as independent observations.", "", "## Task-type P(HIGH) summaries", "", type_p.to_markdown(index=False), "", "## Task-type Friedman tests: P(HIGH)", "", type_tests_p.to_markdown(index=False), "", "## Observation-condition paired tests: P(HIGH)", "", condition_p.to_markdown(index=False), "", "## Ratings-only directional reference", "", "Ratings-only results found FAULTY_FAST lower valence, higher arousal, and no clear CONTROL_FAST versus CONTROL_SLOW effect. Predictor-versus-rating direction is descriptive only; no merged inferential test was performed.", "", "## Interpretation boundary", "", "Friedman tests compare Observation, Interaction, and Alone without automated post-hoc tests. Wilcoxon tests are targeted FAULTY_FAST−CONTROL_FAST and CONTROL_FAST−CONTROL_SLOW comparisons. FDR correction is within each branch × target × metric family for task types and within each branch × target × contrast family for condition metrics.", ""])


def main() -> int:
    args = parse_args(); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing context analysis directory: {output}")
    conditions = canonical_conditions(); cohort = load_cohort(conditions)
    task_summary, task_tests = task_type_tables(cohort); condition_summary, deltas, condition_tests = condition_tables(cohort)
    output.mkdir(parents=True)
    task_summary.to_csv(output / "task_type_predictor_summary.csv", index=False); task_tests.to_csv(output / "task_type_predictor_tests.csv", index=False); condition_summary.to_csv(output / "observation_condition_predictor_summary.csv", index=False); deltas.to_csv(output / "observation_condition_predictor_deltas.csv", index=False); condition_tests.to_csv(output / "observation_condition_predictor_tests.csv", index=False)
    box_task_types(output / "task_type_p_high.png", cohort); paired_figure(output / "fault_contrast_p_high.png", deltas, "fault_effect", "Fault contrast: FAULTY_FAST versus CONTROL_FAST"); paired_figure(output / "speed_contrast_p_high.png", deltas, "speed_effect", "Speed contrast: CONTROL_FAST versus CONTROL_SLOW")
    (output / "context_analysis.md").write_text(report(task_summary, task_tests, condition_tests) + "\n", encoding="utf-8")
    manifest = {"purpose": "Read-only frozen Robot-transfer context analysis", "cohort_input": str(COHORT_PATH), "condition_metadata": str(CONFIG_DIR), "uses_saved_frozen_predictor_outputs": True, "runs_inference_or_loads_models": False, "condition_counts": conditions.actual_condition.value_counts().to_dict(), "fault_paired_n": 46, "speed_paired_n": 45, "participant_level_aggregation": True, "window_independence_used": False}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "task_type_tests": len(task_tests), "condition_tests": len(condition_tests)}, indent=2)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error: print(f"ERROR: {error}"); raise SystemExit(2)
