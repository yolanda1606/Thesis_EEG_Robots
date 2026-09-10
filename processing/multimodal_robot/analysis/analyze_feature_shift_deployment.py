#!/usr/bin/env python3
"""Read-only association analysis of frozen Image-to-Robot feature shift and deployment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from processing.visualization.thesis_style import apply_thesis_style, load_thesis_style


COHORT = ROOT / "outputs/robot_transfer/cohort_analysis/cohort_task_level.csv"
PALETTE = ROOT / "processing/visualization/thesis_palette.yaml"
BRANCHES = ["eeg_only", "modality_agnostic"]
TARGETS = ["valence", "arousal"]
TASK_ORDER = ["pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone"]
TASK_LABELS = {"pick_place": "PnP", "shape_sorter_observation": "SSObs", "stack": "St", "sisyphus": "Sisyphus", "shape_sorter_interaction": "SSInt", "shape_sorter_alone": "SSAlone"}
SHIFT = {
    "nearest_image_distance": "top3_median_median_nearest_image_distance",
    "outside_selected_fraction": "top3_median_fraction_feature_values_outside",
    "median_abs_image_z": "top3_median_median_abs_image_z",
    "max_abs_image_z": "top3_median_maximum_abs_image_z",
}
PRIMARY_SHIFT = ["nearest_image_distance", "outside_selected_fraction", "median_abs_image_z"]
CONFIDENCE = {"absolute_margin": "absolute_margin", "unanimous_agreement": "unanimous_agreement_rate", "probability_range": "median_top3_probability_range"}
SEED = 20260910
# Cluster resampling is computationally heavier than row bootstrap because each
# resample refits the task/target-adjusted logistic model.
BOOTSTRAPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_transfer/feature_shift_analysis")
    return parser.parse_args()


def describe(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {"n_task_cases": len(values), "mean": float(values.mean()), "median": float(values.median()), "iqr": float(values.quantile(.75) - values.quantile(.25))}


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return result
    order = valid.sort_values().index
    adjusted = valid.loc[order].to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def participant_bootstrap(frame: pd.DataFrame, statistic, rng: np.random.Generator) -> tuple[float, float, float]:
    participants = frame.participant.drop_duplicates().to_numpy()
    estimates = []
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(participants, size=len(participants), replace=True)
        # Retain duplicated clusters: they are the bootstrap resampling unit.
        pieces = [frame.loc[frame.participant.eq(person)] for person in sample]
        value = statistic(pd.concat(pieces, ignore_index=True))
        if np.isfinite(value):
            estimates.append(value)
    estimates = np.asarray(estimates)
    if len(estimates) < 30:
        return np.nan, np.nan, np.nan
    low, high = np.quantile(estimates, [.025, .975])
    # Add-one correction avoids reporting an impossible zero Monte-Carlo p-value.
    p = min(1.0, 2 * min(((estimates <= 0).sum() + 1) / (len(estimates) + 1), ((estimates >= 0).sum() + 1) / (len(estimates) + 1)))
    return float(low), float(high), float(p)


def standardized_logistic(frame: pd.DataFrame, metric: str) -> tuple[float, float, int]:
    """Fit correctness ~ z(shift) + target + task with a small ridge for stability."""
    data = frame.dropna(subset=[metric, "correct", "target", "task"]).copy()
    y = data.correct.astype(int).to_numpy()
    x_shift = data[metric].to_numpy(dtype=float)
    scale = x_shift.std(ddof=0)
    if len(data) < 30 or scale == 0 or y.min() == y.max():
        return np.nan, np.nan, len(data)
    x_shift = (x_shift - x_shift.mean()) / scale
    covariates = pd.get_dummies(data[["target", "task"]], drop_first=True, dtype=float)
    design = np.column_stack([x_shift, covariates.to_numpy(dtype=float)])
    # A negligible L2 penalty provides stable repeated fits without materially
    # changing the task/target-adjusted maximum-likelihood estimate.
    fit = LogisticRegression(C=1e6, penalty="l2", solver="lbfgs", max_iter=500).fit(design, y)
    if not np.isfinite(fit.coef_).all():
        return np.nan, np.nan, len(data)
    return float(fit.coef_[0, 0]), float(np.exp(fit.coef_[0, 0])), len(data)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(COHORT)
    required = {"branch", "participant", "target", "task", "correct", "predicted_probability", "predicted_class", "actual_class", "absolute_margin", "unanimous_agreement_rate", "median_top3_probability_range", "fraction_complete_top3_windows", "median_top3_image_ba", *SHIFT.values()}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Cohort is missing required frozen fields: {sorted(missing)}")
    data = data[data.branch.isin(BRANCHES) & data.target.isin(TARGETS) & data.task.isin(TASK_ORDER)].copy()
    if data.duplicated(["branch", "participant", "target", "task"]).any() or len(data) != 1072:
        raise ValueError("Unexpected task-level cohort structure")
    data.correct = data.correct.astype(bool)
    data["task_display"] = data.task.map(TASK_LABELS)
    data.rename(columns={column: name for name, column in SHIFT.items()}, inplace=True)
    return data


def correctness_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for branch in BRANCHES:
        for target in TARGETS:
            subset = data[(data.branch == branch) & (data.target == target)]
            for metric in SHIFT:
                for correct in [True, False]:
                    group = subset[subset.correct.eq(correct)]
                    rows.append({"row_type": "group_summary", "branch": branch, "target": target, "shift_metric": metric, "outcome": "correct" if correct else "incorrect", "n_participants": group.participant.nunique(), **describe(group[metric])})
                correct_values = subset.loc[subset.correct, metric].dropna()
                incorrect_values = subset.loc[~subset.correct, metric].dropna()
                pooled_sd = np.sqrt(((len(correct_values) - 1) * correct_values.var(ddof=1) + (len(incorrect_values) - 1) * incorrect_values.var(ddof=1)) / (len(correct_values) + len(incorrect_values) - 2))
                mean_difference = float(incorrect_values.mean() - correct_values.mean())
                effect = mean_difference / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
                rows.append({"row_type": "incorrect_minus_correct_descriptive", "branch": branch, "target": target, "shift_metric": metric, "outcome": "incorrect_minus_correct", "n_participants": subset.participant.nunique(), "n_correct_task_cases": len(correct_values), "n_incorrect_task_cases": len(incorrect_values), "mean_difference": mean_difference, "standardized_mean_difference": effect})
    return pd.DataFrame(rows)


def association_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 1)
    for branch in BRANCHES:
        branch_data = data[data.branch.eq(branch)]
        for metric in PRIMARY_SHIFT:
            coefficient, odds_ratio, n = standardized_logistic(branch_data, metric)
            def fit_boot(sample: pd.DataFrame) -> float:
                return standardized_logistic(sample, metric)[0]
            low, high, p = participant_bootstrap(branch_data, fit_boot, rng)
            rows.append({"analysis_type": "participant_cluster_bootstrap_logistic_adjusted_task_target", "branch": branch, "outcome": "correct", "predictor": metric, "n_task_cases": n, "n_participants": branch_data.participant.nunique(), "coefficient_per_1sd": coefficient, "odds_ratio_per_1sd": odds_ratio, "coefficient_ci_95_low": low, "coefficient_ci_95_high": high, "p_value_bootstrap_sign": p})
        for metric in PRIMARY_SHIFT:
            for outcome, column in CONFIDENCE.items():
                subset = branch_data.groupby("participant", as_index=False)[[metric, column]].mean().dropna()
                observed, p = stats.spearmanr(subset[metric], subset[column])
                rows.append({"analysis_type": "participant_mean_spearman", "branch": branch, "outcome": outcome, "predictor": metric, "n_participants": len(subset), "spearman_rho": observed, "p_value_raw": p})
        for target in TARGETS:
            for task in TASK_ORDER:
                subset = branch_data[(branch_data.target == target) & (branch_data.task == task)]
                for metric in PRIMARY_SHIFT:
                    rho, p = stats.spearmanr(subset[metric], subset.correct.astype(int))
                    rows.append({"analysis_type": "within_task_target_spearman_descriptive", "branch": branch, "target": target, "task": task, "outcome": "correct", "predictor": metric, "n_task_cases": len(subset), "n_participants": subset.participant.nunique(), "spearman_rho": rho, "p_value_raw": p})
    result = pd.DataFrame(rows)
    mask = result.analysis_type.eq("participant_cluster_bootstrap_logistic_adjusted_task_target")
    result.loc[mask, "p_value_fdr_bh"] = result.loc[mask].groupby("branch", group_keys=False).p_value_bootstrap_sign.apply(bh)
    return result


def task_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (branch, target, task), group in data.groupby(["branch", "target", "task"], sort=False):
        actual_high = group.actual_class.eq("HIGH")
        actual_low = ~actual_high
        sensitivity = group.loc[actual_high, "correct"].mean() if actual_high.any() else np.nan
        specificity = group.loc[actual_low, "correct"].mean() if actual_low.any() else np.nan
        rows.append({"branch": branch, "target": target, "task": task, "task_display": TASK_LABELS[task], "n_valid": len(group), "balanced_accuracy": np.nanmean([sensitivity, specificity]), "accuracy": group.correct.mean(), "actual_high_count": int(actual_high.sum()), "actual_low_count": int(actual_low.sum()), "median_top3_image_ba": group.median_top3_image_ba.median(), "median_nearest_image_distance": group.nearest_image_distance.median(), "median_outside_selected_fraction": group.outside_selected_fraction.median(), "median_abs_image_z": group.median_abs_image_z.median(), "median_max_abs_image_z": group.max_abs_image_z.median(), "mean_unanimous_agreement": group.unanimous_agreement_rate.mean(), "mean_absolute_margin": group.absolute_margin.mean(), "median_probability_range": group.median_top3_probability_range.median(), "mean_complete_top3_coverage": group.fraction_complete_top3_windows.mean()})
    result = pd.DataFrame(rows)
    for branch in BRANCHES:
        subset = result[result.branch.eq(branch)]
        rho, p = stats.spearmanr(subset.median_nearest_image_distance, subset.balanced_accuracy)
        result.loc[result.branch.eq(branch), "task_level_ba_nearest_distance_spearman_rho"] = rho
        result.loc[result.branch.eq(branch), "task_level_ba_nearest_distance_p_value_raw"] = p
    return result


def figures(data: pd.DataFrame, tasks: pd.DataFrame, output: Path) -> None:
    style = load_thesis_style(PALETTE); apply_thesis_style(style)
    colors = style["roles"]["emotion"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rng = np.random.default_rng(SEED)
    for axis, branch in zip(axes, BRANCHES):
        subset = data[data.branch.eq(branch)]
        for position, correct in enumerate([True, False], start=1):
            values = subset.loc[subset.correct.eq(correct), "nearest_image_distance"].dropna()
            axis.boxplot(values, positions=[position], widths=.55, showfliers=False)
            jitter = rng.normal(position, .045, len(values))
            axis.scatter(jitter, values, s=10, alpha=.28, color=style["roles"]["neutral"]["connector"])
        axis.set_xticks([1, 2], ["Correct", "Incorrect"]); axis.set_title(branch.replace("_", " ").title()); axis.set_yscale("log"); axis.grid(axis="y")
    axes[0].set_ylabel("Median top-3 nearest-Image distance (log scale)")
    fig.suptitle("Feature-space shift by deployment correctness")
    fig.tight_layout(); fig.savefig(output / "shift_by_correctness_nearest_distance.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, branch in zip(axes, BRANCHES):
        subset = tasks[tasks.branch.eq(branch)]
        for target in TARGETS:
            points = subset[subset.target.eq(target)]
            axis.scatter(points.median_nearest_image_distance, points.balanced_accuracy, s=42, color=colors[target]["main"], label=target.title())
            for row in points.itertuples(): axis.annotate(row.task_display, (row.median_nearest_image_distance, row.balanced_accuracy), xytext=(4, 3), textcoords="offset points", fontsize=8)
        axis.set_title(branch.replace("_", " ").title()); axis.set_xlabel("Median top-3 nearest-Image distance"); axis.grid()
    axes[0].set_ylabel("Deployment balanced accuracy"); axes[1].legend(title="Target")
    fig.suptitle("Task-level deployment BA versus Image-to-Robot shift")
    fig.tight_layout(); fig.savefig(output / "task_ba_vs_nearest_distance.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, branch in zip(axes, BRANCHES):
        subset = data[data.branch.eq(branch)]
        for position, correct in enumerate([True, False], start=1):
            values = subset.loc[subset.correct.eq(correct), "outside_selected_fraction"].dropna()
            axis.boxplot(values, positions=[position], widths=.55, showfliers=False)
            axis.scatter(rng.normal(position, .045, len(values)), values, s=10, alpha=.28, color=style["roles"]["neutral"]["connector"])
        axis.set_xticks([1, 2], ["Correct", "Incorrect"]); axis.set_title(branch.replace("_", " ").title()); axis.grid(axis="y")
    axes[0].set_ylabel("Median top-3 outside-selected fraction")
    fig.suptitle("Outside-range feature fraction by deployment correctness")
    fig.tight_layout(); fig.savefig(output / "shift_by_correctness_outside_fraction.png", dpi=220); plt.close(fig)


def report(tasks: pd.DataFrame, models: pd.DataFrame) -> str:
    main = models[models.analysis_type.eq("participant_cluster_bootstrap_logistic_adjusted_task_target")]
    return "\n".join([
        "# Image-to-Robot feature-shift and frozen deployment", "",
        "## Methods: frozen shift definition", "",
        "Each frozen Image model contributes only its selected features. Robot and finite Image-reference rows are mapped with that model pipeline's fitted `scale.mean_` and `scale.scale_` values. For each Robot window, nearest-Image distance is the minimum Euclidean distance to a finite Image row in this selected, scaled feature space; all selected features have equal weight. Outside-selected fraction is calculated in the original feature scale against finite Image-reference minima and maxima. Median and maximum absolute Image z are calculated from the same frozen scaler. Non-finite Image rows are removed before diagnostics; inference accepts only Robot windows finite across candidate features. Per-model task summaries use window median nearest-distance, mean outside fraction, window median median-|z|, and window maximum max-|z|. The cohort then takes the median of each metric across the frozen top three ranks; this analysis uses those saved task values and never treats windows as independent.", "",
        "## Association method", "",
        "Correct and incorrect task cases are descriptively unpaired, so their group summaries and standardized mean differences are not treated as independent-case hypothesis tests. Because a logistic mixed-effects implementation is unavailable in the project environment, correctness associations use participant-cluster bootstrap logistic regressions, separately by branch, with task and target fixed effects. Coefficients are per one within-branch SD greater shift. These are associations with frozen deployment correctness, not causal effects.", "",
        "## Correctness models", "", main.to_markdown(index=False), "",
        "## Task-level nearest-distance versus BA", "", tasks[["branch", "task_display", "target", "balanced_accuracy", "median_nearest_image_distance", "median_outside_selected_fraction", "median_abs_image_z", "mean_unanimous_agreement", "mean_absolute_margin", "mean_complete_top3_coverage", "task_level_ba_nearest_distance_spearman_rho", "task_level_ba_nearest_distance_p_value_raw"]].to_markdown(index=False), "",
        "## Interpretation boundary", "", "Feature shift is measured relative to the frozen Image training domain; Robot data were not used for adaptation. Metrics are model-specific because selected features differ across frozen models. An association between shift and performance cannot establish that feature shift caused a deployment error.", ""
    ])


def main() -> int:
    args = parse_args(); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    data = load_data()
    correct_summary = correctness_summary(data)
    models = association_models(data)
    tasks = task_summary(data)
    output.mkdir(parents=True)
    data.to_csv(output / "task_level_shift_deployment.csv", index=False)
    correct_summary.to_csv(output / "shift_correctness_summary.csv", index=False)
    models.to_csv(output / "shift_association_models.csv", index=False)
    tasks.to_csv(output / "task_shift_performance_summary.csv", index=False)
    figures(data, tasks, output)
    (output / "feature_shift_analysis.md").write_text(report(tasks, models) + "\n", encoding="utf-8")
    manifest = {"purpose": "Read-only frozen Image-to-Robot feature-shift deployment analysis", "cohort_input": str(COHORT), "runs_inference_or_loads_models": False, "uses_saved_frozen_cohort_and_shift_outputs": True, "n_task_rows": len(data), "branches": BRANCHES, "participant_cluster_bootstrap_resamples": BOOTSTRAPS, "window_independence_used": False}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "n_task_rows": len(data), "n_models": int(models.analysis_type.eq("participant_cluster_bootstrap_logistic_adjusted_task_target").sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
