#!/usr/bin/env python3
"""Read-only cohort analysis of final EEG-only and modality-agnostic transfer.

The deployment observation is one participant × target × task row.  This
script never reads windows for metric calculations, reruns inference, refits
models, or writes into participant transfer folders.
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
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
PARTICIPANTS = [f"P{number:02d}" for number in range(1, 47)]
TASK_ORDER = ["pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone"]
TASK_LABELS = {"pick_place": "PnP", "shape_sorter_observation": "SSObs", "stack": "St", "sisyphus": "Sisyphus", "shape_sorter_interaction": "SSInt", "shape_sorter_alone": "SSAlone"}
BRANCHES = {
    "eeg_only": {
        "root": ROOT / "outputs/robot_transfer/eeg_only",
        "top3": ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv",
    },
    "modality_agnostic": {
        "root": ROOT / "outputs/robot_transfer/modality_agnostic",
        "top3": ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target_modality.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_transfer/cohort_analysis")
    return parser.parse_args()


def safe_corr(frame: pd.DataFrame, left: str, right: str) -> float:
    values = frame[[left, right]].dropna()
    return float(values[left].corr(values[right])) if len(values) >= 3 and values[left].nunique() > 1 and values[right].nunique() > 1 else np.nan


def image_quality(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"participant", "target", "transfer_rank", "mean_outer_cv_balanced_accuracy", "mean_accuracy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    if frame.duplicated(["participant", "target", "transfer_rank"]).any():
        raise ValueError(f"{path}: duplicate participant/target/rank rows")
    rows = []
    for (participant, target), group in frame.groupby(["participant", "target"], sort=False):
        group = group.set_index("transfer_rank").reindex([1, 2, 3])
        if group["mean_outer_cv_balanced_accuracy"].isna().any():
            raise ValueError(f"{path}: incomplete top-3 Image rows for {participant} {target}")
        row = {"participant": participant, "target": target}
        for rank in (1, 2, 3):
            row[f"rank{rank}_image_ba"] = float(group.loc[rank, "mean_outer_cv_balanced_accuracy"])
            row[f"rank{rank}_image_accuracy"] = float(group.loc[rank, "mean_accuracy"])
        row["mean_top3_image_ba"] = float(group["mean_outer_cv_balanced_accuracy"].mean())
        row["median_top3_image_ba"] = float(group["mean_outer_cv_balanced_accuracy"].median())
        row["mean_top3_image_accuracy"] = float(group["mean_accuracy"].mean())
        row["median_top3_image_accuracy"] = float(group["mean_accuracy"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_from_consensus(path: Path) -> pd.DataFrame:
    """Derive unavailable task agreement fields from already-saved complete windows."""
    frame = pd.read_csv(path)
    probability_columns = [
        [f"rank{rank}_high_probability", f"rank{rank}_p_high"]
        for rank in (1, 2, 3)
    ]
    probabilities = [next((name for name in names if name in frame), None) for names in probability_columns]
    hard = [f"rank{rank}_hard_prediction" for rank in (1, 2, 3)]
    if any(name is None for name in probabilities) or any(name not in frame for name in hard):
        raise ValueError(f"{path}: cannot recover complete-window agreement columns")
    rows = []
    for (target, task), group in frame.groupby(["target", "task"], sort=False):
        row = {"target": target, "task": task, "disagreement_rate": float((~group.all_three_hard_agree).mean())}
        for left, right in ((1, 2), (1, 3), (2, 3)):
            row[f"class_agreement_{left}_{right}"] = float((group[hard[left - 1]] == group[hard[right - 1]]).mean())
            row[f"probability_correlation_{left}_{right}"] = float(group[probabilities[left - 1]].corr(group[probabilities[right - 1]]))
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_branch(branch: str, source: dict[str, Path]) -> pd.DataFrame:
    """Normalize branch-specific files immediately into one task-level schema."""
    image = image_quality(source["top3"])
    all_rows = []
    for participant in PARTICIPANTS:
        folder = source["root"] / participant
        task_path = folder / f"{participant}_task_consensus_analysis.csv"
        agreement_path = folder / f"{participant}_model_agreement.csv"
        shift_path = folder / f"{participant}_selected_feature_shift_task_summary.csv"
        for path in (task_path, agreement_path, shift_path):
            if not path.is_file():
                raise FileNotFoundError(f"{branch} {participant}: missing {path.name}")
        task = pd.read_csv(task_path)
        required = {"participant", "target", "task", "robot_rating", "robot_rating_class", "median_top3_consensus_probability", "n_candidate_windows", "n_complete_top3_windows", "n_incomplete_top3_windows", "fraction_complete_top3_windows", "unanimous_agreement_rate", "unanimous_high_rate", "unanimous_low_rate", "majority_vote_high_fraction", "median_top3_probability_range", "mean_top3_probability_range", "mean_pairwise_absolute_probability_difference"}
        missing = required - set(task.columns)
        if missing:
            raise ValueError(f"{task_path}: missing {sorted(missing)}")
        agreement = pd.read_csv(agreement_path)
        agreement_columns = ["participant", "target", "task", *[column for column in ("disagreement_rate", "class_agreement_1_2", "class_agreement_1_3", "class_agreement_2_3", "probability_correlation_1_2", "probability_correlation_1_3", "probability_correlation_2_3") if column in agreement]]
        task = task.merge(agreement.loc[:, agreement_columns], on=["participant", "target", "task"], how="left", validate="one_to_one")
        needed_agreement = ["disagreement_rate", "class_agreement_1_2", "class_agreement_1_3", "class_agreement_2_3", "probability_correlation_1_2", "probability_correlation_1_3", "probability_correlation_2_3"]
        unavailable = [column for column in needed_agreement if column not in task or task[column].isna().all()]
        if unavailable:
            derived = agreement_from_consensus(folder / f"{participant}_window_consensus.csv")
            task = task.merge(derived.loc[:, ["target", "task", *unavailable]], on=["target", "task"], how="left", suffixes=("", "_derived"), validate="many_to_one")
            for column in unavailable:
                derived_column = f"{column}_derived"
                if column in task and derived_column in task:
                    task[column] = task[column].combine_first(task.pop(derived_column))
                elif derived_column in task:
                    task.rename(columns={derived_column: column}, inplace=True)
        shift = pd.read_csv(shift_path)
        shift_metrics = [column for column in ("median_nearest_image_distance", "fraction_windows_with_outside", "fraction_feature_values_outside", "median_abs_image_z", "maximum_abs_image_z") if column in shift]
        shift = shift.groupby(["target", "task"], as_index=False)[shift_metrics].median().rename(columns={metric: f"top3_median_{metric}" for metric in shift_metrics})
        task = task.merge(shift, on=["target", "task"], how="left", validate="many_to_one")
        task = task.merge(image, on=["participant", "target"], how="left", validate="many_to_one")
        task["branch"] = branch
        task["task_display"] = task.get("task_display", task["task"].map(TASK_LABELS)).fillna(task["task"].map(TASK_LABELS))
        task["actual_class"] = np.where(pd.to_numeric(task["robot_rating"], errors="coerce") >= 4, "HIGH", "LOW")
        task["actual_high"] = task["actual_class"].eq("HIGH").astype(int)
        task["predicted_probability"] = pd.to_numeric(task["median_top3_consensus_probability"], errors="coerce")
        task["predicted_class"] = np.where(task["predicted_probability"] >= .5, "HIGH", "LOW")
        task.loc[task["predicted_probability"].isna(), "predicted_class"] = pd.NA
        task["predicted_high"] = task["predicted_class"].eq("HIGH").astype("Int64")
        task["correct"] = (task["predicted_class"] == task["actual_class"]).where(task["predicted_class"].notna())
        task["absolute_margin"] = (task["predicted_probability"] - .5).abs()
        all_rows.append(task)
    cohort = pd.concat(all_rows, ignore_index=True)
    valid = cohort[cohort["predicted_probability"].notna() & cohort["robot_rating"].notna()].copy()
    if valid.duplicated(["branch", "participant", "target", "task"]).any():
        raise ValueError(f"{branch}: duplicate normalized deployment rows")
    return valid


def metric_row(frame: pd.DataFrame, branch: str, scope: str, group_value: str) -> dict[str, object]:
    y = frame.actual_high.to_numpy(dtype=int)
    predicted = frame.predicted_high.astype(int).to_numpy()
    tp = int(((y == 1) & (predicted == 1)).sum()); tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum()); fn = int(((y == 1) & (predicted == 0)).sum())
    high_count, low_count = int((y == 1).sum()), int((y == 0).sum())
    sensitivity = tp / high_count if high_count else np.nan
    specificity = tn / low_count if low_count else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if pd.notna(precision) and pd.notna(sensitivity) and precision + sensitivity else np.nan
    probability = frame.predicted_probability.to_numpy(float)
    return {"branch": branch, "scope": scope, "group": group_value, "n_valid": len(frame), "correct_count": tp + tn, "wrong_count": fp + fn, "actual_high_count": high_count, "actual_low_count": low_count, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(frame) if len(frame) else np.nan, "balanced_accuracy": np.nanmean([sensitivity, specificity]) if high_count and low_count else np.nan, "sensitivity_high": sensitivity, "specificity_low": specificity, "precision_high": precision, "f1_high": f1, "brier_score": float(np.mean((probability - y) ** 2)), "mean_absolute_margin": float(frame.absolute_margin.mean()), "median_absolute_margin": float(frame.absolute_margin.median()), "mean_margin_correct": float(frame.loc[frame.correct, "absolute_margin"].mean()), "mean_margin_wrong": float(frame.loc[~frame.correct, "absolute_margin"].mean()), "roc_auc": float(roc_auc_score(y, probability)) if high_count and low_count else np.nan}


def metric_tables(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, target, task = [], [], []
    for branch, data in cohort.groupby("branch"):
        overall.append(metric_row(data, branch, "overall", "all"))
        for value, group in data.groupby("target"): target.append(metric_row(group, branch, "target", value))
        for value, group in data.groupby("task"): task.append(metric_row(group, branch, "task", value))
    return pd.DataFrame(overall), pd.DataFrame(target), pd.DataFrame(task)


def grouped_mean(cohort: pd.DataFrame, metrics: list[str], dimensions: list[str]) -> pd.DataFrame:
    return cohort.groupby(dimensions, dropna=False)[metrics].agg(["count", "mean", "median"]).stack(0).reset_index().rename(columns={"level_" + str(len(dimensions)): "metric"})


def coverage_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimensions, scope in [(["branch"], "overall"), (["branch", "target"], "target"), (["branch", "task"], "task")]:
        for keys, frame in cohort.groupby(dimensions, dropna=False):
            values = frame.fraction_complete_top3_windows
            key_values = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(dimensions, key_values)) | {"scope": scope, "n_cases": len(frame), "mean_fraction_complete": values.mean(), "median_fraction_complete": values.median(), "iqr_fraction_complete": values.quantile(.75) - values.quantile(.25), "min_fraction_complete": values.min(), "complete_100_count": int((values == 1).sum()), "complete_ge_90_count": int((values >= .9).sum()), "complete_ge_75_count": int((values >= .75).sum()), "complete_lt_75_count": int((values < .75).sum()), "zero_complete_count": int((frame.n_complete_top3_windows == 0).sum())}
            rows.append(row)
    return pd.DataFrame(rows)


def paired_table(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["participant", "target", "task", "correct", "predicted_probability", "absolute_margin", "unanimous_agreement_rate", "fraction_complete_top3_windows", "top3_median_median_nearest_image_distance"]
    eeg = cohort[cohort.branch.eq("eeg_only")].loc[:, columns].rename(columns={column: f"eeg_{column}" for column in columns if column not in ("participant", "target", "task")})
    modality = cohort[cohort.branch.eq("modality_agnostic")].loc[:, columns].rename(columns={column: f"modality_{column}" for column in columns if column not in ("participant", "target", "task")})
    paired = eeg.merge(modality, on=["participant", "target", "task"], validate="one_to_one")
    paired["outcome"] = np.select([paired.eeg_correct & paired.modality_correct, paired.eeg_correct & ~paired.modality_correct, ~paired.eeg_correct & paired.modality_correct], ["both_correct", "eeg_correct_modality_wrong", "eeg_wrong_modality_correct"], default="both_wrong")
    summary = paired.groupby("outcome").size().rename("count").reset_index()
    return paired, summary


def save_figures(output: Path, cohort: pd.DataFrame, overall: pd.DataFrame, target: pd.DataFrame, task: pd.DataFrame, paired_summary: pd.DataFrame) -> None:
    figures = output / "figures"; figures.mkdir()
    branches = ["eeg_only", "modality_agnostic"]
    # 1: accuracy / balanced accuracy by target.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, metric in zip(axes, ["accuracy", "balanced_accuracy"]):
        pivot = target.pivot(index="group", columns="branch", values=metric).reindex(columns=branches)
        pivot.plot.bar(ax=axis, color=["#377eb8", "#e41a1c"], rot=0)
        axis.set_title(metric.replace("_", " ").title()); axis.set_ylim(0, 1); axis.set_ylabel("Task-level metric")
    fig.tight_layout(); fig.savefig(figures / "deployment_accuracy_by_target.png", dpi=180); plt.close(fig)
    # 2: branch × target confusion matrices.
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for axis, (_, row) in zip(axes.flat, target.iterrows()):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        axis.imshow(matrix, cmap="Blues"); axis.set_title(f"{row.branch}: {row.group}")
        axis.set_xticks([0, 1], ["Pred LOW", "Pred HIGH"]); axis.set_yticks([0, 1], ["Actual LOW", "Actual HIGH"])
        for i in range(2):
            for j in range(2): axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrices_by_target.png", dpi=180); plt.close(fig)
    # 3: task accuracy.
    fig, axis = plt.subplots(figsize=(10, 4.5)); pivot = task.assign(label=task.group.map(TASK_LABELS)).pivot(index="label", columns="branch", values="accuracy").reindex([TASK_LABELS[item] for item in TASK_ORDER])
    pivot.plot.bar(ax=axis, color=["#377eb8", "#e41a1c"], rot=0); axis.set_ylim(0, 1); axis.set_ylabel("Accuracy"); axis.set_title("Robot deployment accuracy by task")
    fig.tight_layout(); fig.savefig(figures / "deployment_accuracy_by_task.png", dpi=180); plt.close(fig)
    # 4: agreement and range by correctness.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4));
    for axis, metric, title in zip(axes, ["unanimous_agreement_rate", "mean_top3_probability_range"], ["Unanimous agreement", "Mean top-3 probability range"]):
        data = [cohort.loc[cohort.correct.eq(value), metric].dropna() for value in (True, False)]
        axis.boxplot(data, tick_labels=["Correct", "Wrong"]); axis.set_title(title); axis.set_ylabel(metric.replace("_", " "))
    fig.tight_layout(); fig.savefig(figures / "agreement_vs_correctness.png", dpi=180); plt.close(fig)
    # 5: Image BA vs deployment result.
    fig, axis = plt.subplots(figsize=(6.5, 4.5));
    for branch, data in cohort.groupby("branch"):
        axis.scatter(data.median_top3_image_ba, data.correct.astype(int) + np.where(data.target.eq("arousal"), .035, -.035), alpha=.35, label=branch)
    axis.set_xlabel("Median top-3 Image nested-CV BA"); axis.set_ylabel("Robot task correctness (jittered)"); axis.legend(); axis.set_title("Image-domain BA versus Robot deployment")
    fig.tight_layout(); fig.savefig(figures / "image_ba_vs_robot_correctness.png", dpi=180); plt.close(fig)
    # 6: feature shift vs correctness.
    fig, axis = plt.subplots(figsize=(7, 4.5)); groups = [cohort.loc[(cohort.branch == branch) & (cohort.correct == correct), "top3_median_median_nearest_image_distance"].dropna() for branch in branches for correct in (True, False)]
    axis.boxplot(groups, tick_labels=["EEG correct", "EEG wrong", "Modality correct", "Modality wrong"]); axis.set_ylabel("Median across top-3 nearest-Image distance"); axis.set_title("Feature shift versus task correctness")
    fig.tight_layout(); fig.savefig(figures / "feature_shift_vs_correctness.png", dpi=180); plt.close(fig)
    # 7: paired outcomes.
    fig, axis = plt.subplots(figsize=(7, 4)); paired_summary.set_index("outcome")["count"].plot.bar(ax=axis, color="#6a3d9a", rot=25); axis.set_ylabel("Matched participant-target-task cases"); axis.set_title("Paired EEG-only versus modality-agnostic outcomes")
    fig.tight_layout(); fig.savefig(figures / "paired_branch_outcomes.png", dpi=180); plt.close(fig)


def main() -> int:
    args = parse_args(); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing cohort analysis directory: {output}")
    cohort = pd.concat([normalize_branch(branch, source) for branch, source in BRANCHES.items()], ignore_index=True)
    overall, target, task = metric_tables(cohort)
    agreement = grouped_mean(cohort, ["unanimous_agreement_rate", "unanimous_high_rate", "unanimous_low_rate", "disagreement_rate", "majority_vote_high_fraction", "class_agreement_1_2", "class_agreement_1_3", "class_agreement_2_3", "probability_correlation_1_2", "probability_correlation_1_3", "probability_correlation_2_3", "mean_top3_probability_range", "median_top3_probability_range", "mean_pairwise_absolute_probability_difference"], ["branch", "correct"])
    coverage = coverage_summary(cohort)
    image_summary_rows = []
    for branch, data in cohort.groupby("branch"):
        participant_accuracy = data.groupby(["participant", "target"], as_index=False).agg(robot_deployment_accuracy=("correct", "mean"), median_top3_image_ba=("median_top3_image_ba", "first"))
        image_summary_rows.append({"branch": branch, "level": "task", "n": len(data), "image_ba_vs_correctness_pearson": safe_corr(data, "median_top3_image_ba", "correct"), "image_ba_vs_probability_pearson": safe_corr(data, "median_top3_image_ba", "predicted_probability"), "image_ba_vs_margin_pearson": safe_corr(data, "median_top3_image_ba", "absolute_margin")})
        image_summary_rows.append({"branch": branch, "level": "participant_target", "n": len(participant_accuracy), "image_ba_vs_correctness_pearson": safe_corr(participant_accuracy, "median_top3_image_ba", "robot_deployment_accuracy"), "image_ba_vs_probability_pearson": np.nan, "image_ba_vs_margin_pearson": np.nan})
    image_summary = pd.DataFrame(image_summary_rows)
    shift_summary_rows = []
    shift_metric = "top3_median_median_nearest_image_distance"
    for branch, data in cohort.groupby("branch"):
        for correct, group in data.groupby("correct"):
            shift_summary_rows.append({"branch": branch, "correct": bool(correct), "n": len(group), "median_nearest_image_distance_mean": group[shift_metric].mean(), "median_nearest_image_distance_median": group[shift_metric].median(), "outside_selected_fraction_mean": group.get("top3_median_fraction_feature_values_outside", pd.Series(dtype=float)).mean(), "median_abs_image_z_mean": group.get("top3_median_median_abs_image_z", pd.Series(dtype=float)).mean(), "max_abs_image_z_mean": group.get("top3_median_maximum_abs_image_z", pd.Series(dtype=float)).mean()})
    shift_summary = pd.DataFrame(shift_summary_rows)
    paired, paired_summary = paired_table(cohort)
    participant_summary = cohort.groupby(["branch", "participant", "target"], as_index=False).agg(n_valid_tasks=("task", "size"), robot_deployment_accuracy=("correct", "mean"), mean_task_probability=("predicted_probability", "mean"), median_complete_coverage=("fraction_complete_top3_windows", "median"), median_top3_image_ba=("median_top3_image_ba", "first"))
    output.mkdir(parents=True); cohort.to_csv(output / "cohort_task_level.csv", index=False); overall.to_csv(output / "branch_overall_metrics.csv", index=False); target.to_csv(output / "branch_target_metrics.csv", index=False); task.to_csv(output / "branch_task_metrics.csv", index=False); agreement.to_csv(output / "agreement_correctness_summary.csv", index=False); coverage.to_csv(output / "coverage_summary.csv", index=False); image_summary.to_csv(output / "image_quality_transfer_summary.csv", index=False); shift_summary.to_csv(output / "shift_correctness_summary.csv", index=False); paired.to_csv(output / "paired_branch_comparison.csv", index=False); participant_summary.to_csv(output / "participant_summary.csv", index=False); paired_summary.to_csv(output / "paired_branch_outcome_counts.csv", index=False)
    save_figures(output, cohort, overall, target, task, paired_summary)
    coverage_overall = coverage[coverage.scope.eq("overall")]
    agreement_focus = agreement[agreement.metric.isin(["unanimous_agreement_rate", "mean_top3_probability_range", "disagreement_rate"])]
    lines = [
        "# Final Robot-transfer cohort analysis", "",
        "All results are descriptive participant × target × task deployment summaries; overlapping Robot windows were not treated as independent observations.", "",
        "## Overall deployment metrics", "", overall.to_markdown(index=False), "",
        "## By target", "", target.to_markdown(index=False), "",
        "## By Robot task", "", task.assign(group=task.group.map(TASK_LABELS)).to_markdown(index=False), "",
        "## Top-3 agreement by task correctness", "", agreement_focus.to_markdown(index=False), "",
        "## Complete-window coverage", "", coverage_overall.to_markdown(index=False), "",
        "## Image-domain quality versus deployment", "", image_summary.to_markdown(index=False), "",
        "## Feature shift by task correctness", "", shift_summary.to_markdown(index=False), "",
        "## Paired branch outcomes", "", paired_summary.to_markdown(index=False), "",
        "## Interpretation boundary", "", "Image-domain nested-CV BA/accuracy and Robot deployment metrics are distinct. Correlations and branch differences are descriptive; no statistical-significance claim is made from task/window overlap.",
    ]
    (output / "cohort_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"purpose": "Read-only cohort analysis of existing final transfer outputs", "deployment_observation": "participant × target × task", "participant_count": 46, "branch_case_counts": cohort.groupby("branch").size().to_dict(), "sources": {key: {name: str(value) for name, value in source.items()} for key, source in BRANCHES.items()}, "window_independence_used": False}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "valid_cases": cohort.groupby("branch").size().to_dict()}, indent=2)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error:
        print(f"ERROR: {error}"); raise SystemExit(2)
