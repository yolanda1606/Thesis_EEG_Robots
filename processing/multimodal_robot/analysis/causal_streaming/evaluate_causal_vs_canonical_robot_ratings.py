#!/usr/bin/env python3
"""Post-hoc Robot-rating evaluation of saved frozen canonical and causal outputs.

This is deliberately an analysis-only layer: it reads the paired complete-top-3
window consensus files produced by completed replay runs, calculates the existing
Definition-B task verdicts, then joins the established Robot ratings.  It never
starts replay, loads a model, or uses ratings to generate/select predictions.
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
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.multimodal_robot.transfer import run_final_frozen_eeg_transfer as transfer


TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")
KEYS = ["participant", "task", "target"]
PROBABILITY_COLUMNS = ("top3_consensus_probability_canonical", "top3_consensus_probability_causal")


def complete_window_task_predictions(consensus: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen Definition-B task aggregation to already paired top-3 windows."""
    required = set(KEYS + ["window_id", *PROBABILITY_COLUMNS])
    missing = sorted(required.difference(consensus.columns))
    if missing:
        raise ValueError(f"Consensus input is missing required columns: {', '.join(missing)}")
    if consensus.duplicated(KEYS + ["segment_id", "window_id"]).any():
        raise ValueError("Duplicate paired complete-top-3 window keys found in consensus inputs")
    if consensus[list(PROBABILITY_COLUMNS)].isna().any().any():
        raise ValueError("Missing canonical or causal paired complete-top-3 probabilities")
    task = consensus.groupby(KEYS, as_index=False).agg(
        canonical_P_task=("top3_consensus_probability_canonical", "median"),
        causal_P_task=("top3_consensus_probability_causal", "median"),
        number_complete_top3_windows=("window_id", "size"),
    )
    if task.duplicated(KEYS).any() or (task.number_complete_top3_windows <= 0).any():
        raise ValueError("Definition-B task aggregation did not produce unique nonempty task rows")
    task["canonical_verdict"] = np.where(task.canonical_P_task >= 0.5, "HIGH", "LOW")
    task["causal_verdict"] = np.where(task.causal_P_task >= 0.5, "HIGH", "LOW")
    return task


def load_paired_task_predictions(cohort_root: Path) -> pd.DataFrame:
    """Load every saved participant agreement file without consulting ratings."""
    paths = sorted(cohort_root.glob("P??/canonical_causal_consensus_agreement.csv"))
    if not paths:
        raise FileNotFoundError(f"No saved consensus agreement files found below {cohort_root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        participant = path.parent.name
        if "participant" not in frame or not frame.participant.astype(str).eq(participant).all():
            raise ValueError(f"{path}: participant column does not match its output directory")
        frames.append(frame)
    return complete_window_task_predictions(pd.concat(frames, ignore_index=True))


def join_posthoc_ratings(task_predictions: pd.DataFrame) -> pd.DataFrame:
    """Join canonical project ratings after, and only after, task predictions exist."""
    participants = sorted(task_predictions.participant.unique())
    ratings = pd.concat([transfer.robot_ratings(participant).assign(participant=participant) for participant in participants], ignore_index=True)
    ratings = ratings.rename(columns={"robot_rating_class": "robot_label"})
    if ratings.duplicated(KEYS).any():
        raise ValueError("Established Robot rating loader returned duplicate participant/task/target rows")
    paired = task_predictions.merge(ratings, on=KEYS, how="inner", validate="one_to_one")
    if len(paired) != len(task_predictions):
        missing = task_predictions.merge(ratings[KEYS], on=KEYS, how="left", indicator=True).query("_merge != 'both'")
        raise ValueError(f"Missing Robot ratings for {len(missing)} existing paired task predictions")
    paired["canonical_correct"] = paired.canonical_verdict.eq(paired.robot_label)
    paired["causal_correct"] = paired.causal_verdict.eq(paired.robot_label)
    paired["verdict_changed"] = ~paired.canonical_verdict.eq(paired.causal_verdict)
    paired["probability_difference"] = paired.causal_P_task - paired.canonical_P_task
    paired["absolute_probability_difference"] = paired.probability_difference.abs()
    complete = paired.groupby("participant").task.nunique().eq(len(TASKS))
    paired["complete_participant"] = paired.participant.map(complete).astype(bool)
    if paired.duplicated(KEYS).any():
        raise ValueError("Rating join did not produce one row per participant/task/target")
    return paired.sort_values(KEYS).reset_index(drop=True)


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def performance(frame: pd.DataFrame, pipeline: str, scope: str, task: str | None = None, target: str | None = None) -> dict[str, object]:
    """Frozen binary performance, with HIGH explicitly the positive class."""
    actual_high = frame.robot_label.eq("HIGH")
    predicted_high = frame[f"{pipeline}_verdict"].eq("HIGH")
    tp = int((actual_high & predicted_high).sum()); tn = int((~actual_high & ~predicted_high).sum())
    fp = int((~actual_high & predicted_high).sum()); fn = int((actual_high & ~predicted_high).sum())
    recall = _safe_divide(tp, tp + fn); specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    return {
        "scope": scope, "task": task if task is not None else "", "target": target if target is not None else "",
        "pipeline": pipeline, "positive_class": "HIGH", "N": len(frame), "robot_LOW_count": int((~actual_high).sum()),
        "robot_HIGH_count": int(actual_high.sum()), "accuracy": _safe_divide(tp + tn, len(frame)),
        "balanced_accuracy": np.nan if np.isnan(recall) or np.isnan(specificity) else float((recall + specificity) / 2),
        "precision": precision, "recall": recall, "F1": np.nan if np.isnan(precision) or np.isnan(recall) or precision + recall == 0 else float(2 * precision * recall / (precision + recall)),
        "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
    }


def performance_table(frame: pd.DataFrame, include_task: bool) -> pd.DataFrame:
    """Compute each canonical/causal metric on the exact same already paired rows."""
    rows = []
    groups: list[tuple[str, str | None, str | None, pd.DataFrame]] = [("overall", None, None, frame)]
    groups += [("target", None, target, part) for target, part in frame.groupby("target", sort=True)]
    if include_task:
        groups += [("task_target", task, target, part) for (task, target), part in frame.groupby(["task", "target"], sort=True)]
    for scope, task, target, part in groups:
        for pipeline in ("canonical", "causal"):
            rows.append(performance(part, pipeline, scope, task, target))
    return pd.DataFrame(rows)


def correctness_transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.select(
        [frame.canonical_correct & frame.causal_correct, frame.canonical_correct & ~frame.causal_correct,
         ~frame.canonical_correct & frame.causal_correct],
        ["both_correct", "canonical_correct_causal_wrong", "canonical_wrong_causal_correct"], default="both_wrong",
    )
    detail = frame.copy(); detail["correctness_transition"] = labels
    rows = []
    groups: list[tuple[str, str, str, pd.DataFrame]] = [("overall", "", "", detail)]
    groups += [("target", "", target, part) for target, part in detail.groupby("target", sort=True)]
    groups += [("task", task, "", part) for task, part in detail.groupby("task", sort=True)]
    groups += [("task_target", task, target, part) for (task, target), part in detail.groupby(["task", "target"], sort=True)]
    for scope, task, target, part in groups:
        counts = part.correctness_transition.value_counts()
        rows.append({"scope": scope, "task": task, "target": target, "N": len(part),
                     **{name: int(counts.get(name, 0)) for name in ("both_correct", "canonical_correct_causal_wrong", "canonical_wrong_causal_correct", "both_wrong")}})
    return pd.DataFrame(rows), detail


def spearman_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, str, str, pd.DataFrame]] = [("overall", "", "", frame)]
    groups += [("target", "", target, part) for target, part in frame.groupby("target", sort=True)]
    groups += [("task_target", task, target, part) for (task, target), part in frame.groupby(["task", "target"], sort=True)]
    for scope, task, target, part in groups:
        for pipeline in ("canonical", "causal"):
            values = part[f"{pipeline}_P_task"]
            if len(part) < 5 or values.nunique() < 2 or part.robot_rating.nunique() < 2:
                rho = p_value = np.nan
            else:
                rho, p_value = spearmanr(values, part.robot_rating)
            rows.append({"scope": scope, "task": task, "target": target, "pipeline": pipeline, "N": len(part), "spearman_rho": rho, "p_value": p_value, "analysis": "descriptive_secondary"})
    return pd.DataFrame(rows)


def mcnemar_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, target, part in [("overall", "", detail), *[("target", name, sub) for name, sub in detail.groupby("target", sort=True)]]:
        canonical_only = int(part.correctness_transition.eq("canonical_correct_causal_wrong").sum())
        causal_only = int(part.correctness_transition.eq("canonical_wrong_causal_correct").sum())
        discordant = canonical_only + causal_only
        p_value = float(binomtest(min(canonical_only, causal_only), n=discordant, p=0.5).pvalue) if discordant else np.nan
        rows.append({"scope": scope, "target": target, "N": len(part), "both_correct": int(part.correctness_transition.eq("both_correct").sum()),
                     "canonical_only_correct": canonical_only, "causal_only_correct": causal_only, "both_wrong": int(part.correctness_transition.eq("both_wrong").sum()),
                     "method": "exact two-sided McNemar binomial test", "discordant_pairs": discordant, "p_value": p_value})
    return pd.DataFrame(rows)


def create_figures(output: Path, overall: pd.DataFrame, by_task: pd.DataFrame, transitions: pd.DataFrame, changed: pd.DataFrame, paired: pd.DataFrame) -> None:
    target = overall.query("scope == 'target'").pivot(index="target", columns="pipeline", values="balanced_accuracy").reindex(columns=["canonical", "causal"])
    ax = target.plot.bar(figsize=(6, 4), color=["#4c78a8", "#f58518"], ylim=(0, 1)); ax.set(xlabel="Target", ylabel="Balanced accuracy"); plt.tight_layout(); plt.savefig(output / "balanced_accuracy_by_target.png", dpi=180); plt.close()
    plot = by_task.query("scope == 'task_target'").copy(); plot["task_target"] = plot.task + "\n" + plot.target
    pivot = plot.pivot(index="task_target", columns="pipeline", values="balanced_accuracy").reindex(columns=["canonical", "causal"])
    ax = pivot.plot.bar(figsize=(11, 4), color=["#4c78a8", "#f58518"], ylim=(0, 1)); ax.set(xlabel="Task / target", ylabel="Balanced accuracy"); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(output / "balanced_accuracy_by_task_target.png", dpi=180); plt.close()
    transition = transitions.query("scope == 'target'").set_index("target")[["both_correct", "canonical_correct_causal_wrong", "canonical_wrong_causal_correct", "both_wrong"]]
    ax = transition.plot.bar(stacked=True, figsize=(7, 4), color=["#59a14f", "#e15759", "#76b7b2", "#79706e"]); ax.set(xlabel="Target", ylabel="Paired rows"); plt.tight_layout(); plt.savefig(output / "correctness_transition_counts.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4));
    if len(changed):
        values = changed.reset_index(drop=True)
        for index, row in values.iterrows(): plt.plot([row.canonical_P_task, row.causal_P_task], [index, index], color="#888888", alpha=.55)
        plt.scatter(values.canonical_P_task, values.index, label="Canonical", color="#4c78a8", s=22); plt.scatter(values.causal_P_task, values.index, label="Causal", color="#f58518", s=22)
    plt.axvline(.5, color="black", ls=":"); plt.xlabel("Task P(HIGH)"); plt.ylabel("Changed-verdict case"); plt.legend(); plt.tight_layout(); plt.savefig(output / "changed_verdict_probabilities.png", dpi=180); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, target_name in zip(axes, ("valence", "arousal")):
        part = paired.loc[paired.target.eq(target_name)]
        axis.scatter(part.robot_rating, part.canonical_P_task, color="#4c78a8", alpha=.55, label="Canonical")
        axis.scatter(part.robot_rating, part.causal_P_task, color="#f58518", alpha=.45, label="Causal")
        axis.set(title=target_name.title(), xlabel="Robot rating (1–7)", ylim=(0, 1))
    axes[0].set_ylabel("Task P(HIGH)"); axes[1].legend(); fig.tight_layout(); fig.savefig(output / "robot_rating_vs_task_probability.png", dpi=180); plt.close(fig)


def metric_value(table: pd.DataFrame, pipeline: str, target: str = "") -> dict[str, float]:
    row = table.loc[(table.pipeline.eq(pipeline)) & (table.target.eq(target)) & (table.scope.eq("overall") if not target else table.scope.eq("target"))]
    return row.iloc[0].to_dict() if len(row) == 1 else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=ROOT / "outputs/robot_transfer/causal_filter_comparison/v1/cohort")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_transfer/causal_filter_comparison/v1/robot_rating_evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Replace only prior evaluation artifacts in --output-dir.")
    args = parser.parse_args(); output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation output directory: {output}")
    task_predictions = load_paired_task_predictions(args.cohort_root.resolve())
    paired = join_posthoc_ratings(task_predictions)
    if paired.groupby(KEYS).size().ne(1).any(): raise ValueError("Paired denominator integrity check failed")
    overall = performance_table(paired, include_task=False); by_task = performance_table(paired, include_task=True).query("scope == 'task_target'").reset_index(drop=True)
    transitions, detailed = correctness_transitions(paired)
    changed = detailed.loc[detailed.verdict_changed].copy()
    changed["canonical_distance_from_0_5"] = (changed.canonical_P_task - .5).abs(); changed["causal_distance_from_0_5"] = (changed.causal_P_task - .5).abs()
    correlation = spearman_table(paired); mcnemar = mcnemar_table(detailed)
    complete_paired = paired.loc[paired.complete_participant].copy(); complete_performance = performance_table(complete_paired, include_task=True)
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_robot_rating_evaluation.csv", index=False); overall.to_csv(output / "robot_rating_performance_overall.csv", index=False); by_task.to_csv(output / "robot_rating_performance_by_task.csv", index=False)
    transitions.to_csv(output / "canonical_vs_causal_correctness_transitions.csv", index=False); changed.to_csv(output / "changed_verdict_cases.csv", index=False); correlation.to_csv(output / "spearman_probability_rating_descriptive.csv", index=False); mcnemar.to_csv(output / "canonical_vs_causal_mcnemar.csv", index=False); complete_performance.to_csv(output / "robot_rating_performance_complete_participants.csv", index=False)
    create_figures(output, overall, by_task, transitions, changed, paired)
    overall_canonical = metric_value(overall, "canonical"); overall_causal = metric_value(overall, "causal")
    target_metrics = {target: {pipeline: metric_value(overall, pipeline, target) for pipeline in ("canonical", "causal")} for target in ("valence", "arousal")}
    changed_directions = {"canonical_wrong_to_causal_right": int(detailed.correctness_transition.eq("canonical_wrong_causal_correct").sum()), "canonical_right_to_causal_wrong": int(detailed.correctness_transition.eq("canonical_correct_causal_wrong").sum())}
    unavailable = overall.loc[overall.balanced_accuracy.isna(), ["scope", "target", "pipeline"]].to_dict("records")
    summary = {"analysis": "post-hoc Robot-rating evaluation of saved frozen canonical versus causal Definition-B outputs", "ratings_used_posthoc_only": True,
               "task_prediction_definition": "median complete-top-3 consensus P(HIGH) across paired windows; HIGH if >= 0.5", "total_paired_rows": int(len(paired)), "unique_participants": int(paired.participant.nunique()), "valid_participant_task_cases": int(paired[["participant", "task"]].drop_duplicates().shape[0]),
               "complete_participant_sensitivity": {"participants": int(complete_paired.participant.nunique()), "rows": int(len(complete_paired)), "task_cases": int(complete_paired[["participant", "task"]].drop_duplicates().shape[0])}, "overall": {"canonical": overall_canonical, "causal": overall_causal}, "target_metrics": target_metrics,
               "changed_verdicts": int(len(changed)), **changed_directions, "mcnemar_overall_p_value": mcnemar.loc[mcnemar.scope.eq("overall"), "p_value"].iloc[0], "unavailable_or_na_metrics": unavailable}
    (output / "robot_rating_evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: None if pd.isna(x) else x) + "\n", encoding="utf-8")
    task_delta = by_task.pivot(index=["task", "target"], columns="pipeline", values="balanced_accuracy").assign(delta=lambda d: d.causal-d.canonical).delta.abs().sort_values(ascending=False)
    complete_overall = complete_performance.query("scope == 'overall'").set_index("pipeline")
    primary_delta = overall_causal["balanced_accuracy"] - overall_canonical["balanced_accuracy"]; sensitivity_delta = complete_overall.loc["causal", "balanced_accuracy"] - complete_overall.loc["canonical", "balanced_accuracy"]
    conclusion = "same direction" if np.sign(primary_delta) == np.sign(sensitivity_delta) else "different direction"
    print(f"1. paired evaluation rows: {len(paired)}")
    print(f"2. class balance: LOW={(paired.robot_label == 'LOW').sum()}, HIGH={(paired.robot_label == 'HIGH').sum()}")
    print(f"3. overall canonical vs causal accuracy/BA: {overall_canonical['accuracy']:.3f}/{overall_canonical['balanced_accuracy']:.3f} vs {overall_causal['accuracy']:.3f}/{overall_causal['balanced_accuracy']:.3f}")
    print(f"4. valence canonical vs causal BA: {target_metrics['valence']['canonical']['balanced_accuracy']:.3f} vs {target_metrics['valence']['causal']['balanced_accuracy']:.3f}")
    print(f"5. arousal canonical vs causal BA: {target_metrics['arousal']['canonical']['balanced_accuracy']:.3f} vs {target_metrics['arousal']['causal']['balanced_accuracy']:.3f}")
    print(f"6. largest task-level BA differences: {', '.join(f'{task}/{target}={value:.3f}' for (task, target), value in task_delta.head(3).items())}")
    print(f"7. changed verdicts: {len(changed)}; canonical wrong→causal right={changed_directions['canonical_wrong_to_causal_right']}; canonical right→causal wrong={changed_directions['canonical_right_to_causal_wrong']}")
    print(f"8. complete-participant sensitivity ({len(complete_paired)} rows): canonical/causal BA={complete_overall.loc['canonical', 'balanced_accuracy']:.3f}/{complete_overall.loc['causal', 'balanced_accuracy']:.3f}")
    print(f"9. qualitative conclusion changes: {conclusion} (primary BA delta={primary_delta:.3f}; sensitivity delta={sensitivity_delta:.3f})")
    print(f"10. files created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
