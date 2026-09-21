#!/usr/bin/env python3
"""Read-only descriptive aggregation of completed causal Robot replay outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")
REQUIRED = ("causal_window_predictions.csv", "canonical_causal_probability_agreement.csv", "canonical_causal_consensus_agreement.csv", "causal_window_latency.csv", "causal_stream_timing_chunks.csv", "canonical_replay_validation.csv")


def q95(values: pd.Series) -> float:
    return float(values.quantile(.95)) if len(values) else np.nan


def numeric_summary(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    result = frame.groupby(groups, dropna=False)[value].agg(n="size", mean="mean", median="median", std="std", minimum="min", maximum="max").reset_index()
    result["p25"] = frame.groupby(groups, dropna=False)[value].quantile(.25).to_numpy()
    result["p75"] = frame.groupby(groups, dropna=False)[value].quantile(.75).to_numpy()
    result["p95"] = frame.groupby(groups, dropna=False)[value].quantile(.95).to_numpy()
    return result


def participant_complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in REQUIRED)


def completion(status: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for record in status.itertuples(index=False):
        participant = record.participant; directory = root / participant
        row = {"participant": participant, "status": record.status, "tasks_expected": len(TASKS), "tasks_completed": 0,
               "targets_completed": 0, "canonical_windows": 0, "causal_windows": 0,
               "complete_top3_windows": 0, "missing_or_unavailable_task": "", "warning_or_exception": getattr(record, "error_message", "") or ""}
        if participant_complete(directory):
            prediction = pd.read_csv(directory / "canonical_causal_probability_agreement.csv")
            consensus = pd.read_csv(directory / "canonical_causal_consensus_agreement.csv")
            latency = pd.read_csv(directory / "causal_window_latency.csv")
            row.update({
                "tasks_completed": prediction.task.nunique(), "targets_completed": prediction.target.nunique(),
                "canonical_windows": prediction.drop_duplicates(["task", "segment_id", "window_id"]).shape[0],
                "causal_windows": latency.shape[0], "complete_top3_windows": consensus.drop_duplicates(["task", "segment_id", "window_id", "target"]).shape[0],
            })
            missing = sorted(set(TASKS).difference(prediction.task.unique()))
            row["missing_or_unavailable_task"] = ";".join(missing)
            if record.status != "success": row["warning_or_exception"] = f"status={record.status}; complete files present"
        else:
            row["missing_or_unavailable_task"] = ";".join(TASKS)
            if not row["warning_or_exception"]: row["warning_or_exception"] = "required participant output files are absent or incomplete"
        rows.append(row)
    return pd.DataFrame(rows)


def timing_summary(participants: list[str], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per, chunks, windows = [], [], []
    for participant in participants:
        directory = root / participant
        chunk = pd.read_csv(directory / "causal_stream_timing_chunks.csv"); latency = pd.read_csv(directory / "causal_window_latency.csv")
        chunks.append(chunk); windows.append(latency)
        prediction = latency.loc[~latency.causal_warmup_excluded].copy()
        per.append({"participant": participant, "number_chunks": len(chunk), "causal_filter_median_ms": chunk.causal_filter_ms.median(),
                    "causal_filter_p95_ms": q95(chunk.causal_filter_ms), "causal_filter_max_ms": chunk.causal_filter_ms.max(),
                    "proportion_chunks_over_100ms": (chunk.causal_filter_ms > 100).mean(), "number_prediction_windows": len(prediction),
                    "eeg_feature_median_ms": prediction.eeg_feature_ms.median(), "eeg_feature_p95_ms": q95(prediction.eeg_feature_ms), "eeg_feature_max_ms": prediction.eeg_feature_ms.max(),
                    "inference_amortized_ms": prediction.frozen_inference_amortized_ms.median(), "total_causal_compute_median_ms": prediction.total_causal_compute_amortized_ms.median(),
                    "total_causal_compute_p95_ms": q95(prediction.total_causal_compute_amortized_ms), "total_causal_compute_max_ms": prediction.total_causal_compute_amortized_ms.max()})
    chunk_all = pd.concat(chunks, ignore_index=True); window_all = pd.concat(windows, ignore_index=True)
    active = window_all.loc[~window_all.causal_warmup_excluded]
    overall = pd.DataFrame([{"scope": "offline_replay_only", "total_chunks": len(chunk_all), "total_prediction_windows": len(active),
        "filter_median_ms": chunk_all.causal_filter_ms.median(), "filter_p95_ms": q95(chunk_all.causal_filter_ms), "filter_max_ms": chunk_all.causal_filter_ms.max(),
        "filter_chunks_over_100ms": int((chunk_all.causal_filter_ms > 100).sum()), "filter_proportion_over_100ms": (chunk_all.causal_filter_ms > 100).mean(),
        "eeg_feature_median_ms": active.eeg_feature_ms.median(), "eeg_feature_p95_ms": q95(active.eeg_feature_ms), "eeg_feature_max_ms": active.eeg_feature_ms.max(),
        "total_compute_median_ms": active.total_causal_compute_amortized_ms.median(), "total_compute_p95_ms": q95(active.total_causal_compute_amortized_ms), "total_compute_max_ms": active.total_causal_compute_amortized_ms.max(),
        "participant_filter_median_min_ms": min(row["causal_filter_median_ms"] for row in per), "participant_filter_median_max_ms": max(row["causal_filter_median_ms"] for row in per)}])
    return pd.DataFrame(per), overall, chunk_all, active


def stability(participants: list[str], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction = pd.concat([pd.read_csv(root / p / "canonical_causal_probability_agreement.csv") for p in participants], ignore_index=True)
    consensus = pd.concat([pd.read_csv(root / p / "canonical_causal_consensus_agreement.csv") for p in participants], ignore_index=True)
    def pred_group(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        return frame.groupby(groups, dropna=False).agg(n_paired_predictions=("hard_agree", "size"), mean_absolute_probability_difference=("absolute_probability_difference", "mean"), median_absolute_probability_difference=("absolute_probability_difference", "median"), p95_absolute_probability_difference=("absolute_probability_difference", q95), hard_prediction_agreement=("hard_agree", "mean"), canonical_high_fraction=("original_hard_prediction_canonical", "mean"), causal_high_fraction=("original_hard_prediction_causal", "mean")).reset_index()
    model = pred_group(prediction, ["target", "classifier_canonical", "model_rank"])
    participant_model = pred_group(prediction, ["participant", "target", "classifier_canonical", "model_rank"])
    consensus["absolute_consensus_probability_difference"] = consensus.signed_consensus_probability_difference.abs()
    return model, participant_model, prediction, consensus


def consensus_summary(consensus: pd.DataFrame) -> pd.DataFrame:
    def aggregate(frame: pd.DataFrame, group: list[str], label: str) -> pd.DataFrame:
        output = frame.groupby(group, dropna=False).agg(n_paired_complete_top3_windows=("consensus_class_agree", "size"), mean_absolute_consensus_probability_difference=("absolute_consensus_probability_difference", "mean"), median_absolute_consensus_probability_difference=("absolute_consensus_probability_difference", "median"), consensus_hard_class_agreement=("consensus_class_agree", "mean"), canonical_high_fraction=("top3_consensus_class_canonical", lambda x: x.eq("HIGH").mean()), causal_high_fraction=("top3_consensus_class_causal", lambda x: x.eq("HIGH").mean())).reset_index()
        output["p95_absolute_consensus_probability_difference"] = frame.groupby(group)["absolute_consensus_probability_difference"].quantile(.95).to_numpy(); output.insert(0, "summary_scope", label)
        return output
    return pd.concat([aggregate(consensus, ["target"], "target"), aggregate(consensus, ["target", "task"], "target_task")], ignore_index=True)


def task_verdicts(consensus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = consensus.groupby(["participant", "task", "target"], as_index=False).agg(canonical_P_task=("top3_consensus_probability_canonical", "median"), causal_P_task=("top3_consensus_probability_causal", "median"), number_of_complete_windows=("window_id", "size"))
    grouped["signed_difference"] = grouped.causal_P_task - grouped.canonical_P_task
    grouped["absolute_difference"] = grouped.signed_difference.abs()
    grouped["canonical_verdict"] = np.where(grouped.canonical_P_task >= .5, "HIGH", "LOW")
    grouped["causal_verdict"] = np.where(grouped.causal_P_task >= .5, "HIGH", "LOW")
    grouped["verdict_agreement"] = grouped.canonical_verdict.eq(grouped.causal_verdict)
    grouped["change_direction"] = np.where(grouped.verdict_agreement, "unchanged", grouped.canonical_verdict + "_to_" + grouped.causal_verdict)
    grouped["canonical_distance_to_threshold"] = (grouped.canonical_P_task - .5).abs()
    def summary(frame: pd.DataFrame, names: list[str], scope: str) -> pd.DataFrame:
        result = frame.groupby(names, dropna=False).agg(n_tasks=("verdict_agreement", "size"), verdict_agreement=("verdict_agreement", "mean"), verdict_changes=("verdict_agreement", lambda x: int((~x).sum())), high_to_low=("change_direction", lambda x: int(x.eq("HIGH_to_LOW").sum())), low_to_high=("change_direction", lambda x: int(x.eq("LOW_to_HIGH").sum())), median_canonical_distance_changed=("canonical_distance_to_threshold", lambda x: x[~frame.loc[x.index, "verdict_agreement"]].median()), median_canonical_distance_unchanged=("canonical_distance_to_threshold", lambda x: x[frame.loc[x.index, "verdict_agreement"]].median())).reset_index()
        result["proportion_verdict_changes"] = result.verdict_changes / result.n_tasks; result.insert(0, "summary_scope", scope); return result
    verdict_summary = pd.concat([
        summary(grouped.assign(overall="overall"), ["overall"], "overall"),
        summary(grouped, ["target"], "target"),
        summary(grouped, ["task", "target"], "task_target"),
    ], ignore_index=True)
    shifts = pd.concat([numeric_summary(grouped, "signed_difference", ["target"]).assign(summary_scope="target"), numeric_summary(grouped, "signed_difference", ["task", "target"]).assign(summary_scope="task_target")], ignore_index=True)
    shifts["proportion_positive"] = shifts.apply(lambda r: (grouped.loc[(grouped.target == r.target) & ((r.summary_scope == "target") | (grouped.task == r.task)), "signed_difference"] > 0).mean(), axis=1)
    shifts["proportion_negative"] = shifts.apply(lambda r: (grouped.loc[(grouped.target == r.target) & ((r.summary_scope == "target") | (grouped.task == r.task)), "signed_difference"] < 0).mean(), axis=1)
    return grouped, verdict_summary, shifts


def validation_summary(participants: list[str], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(root / p / "canonical_replay_validation.csv") for p in participants]
    all_checks = pd.concat(frames, ignore_index=True)
    per = all_checks.groupby("participant").agg(number_checks=("pass", "size"), number_passed=("pass", "sum"), number_failed=("pass", lambda x: int((~x).sum())), max_absolute_difference=("absolute_difference", "max"), median_absolute_difference=("absolute_difference", "median")).reset_index()
    overall = pd.DataFrame([{"participant": "COHORT", "number_checks": len(all_checks), "number_passed": int(all_checks["pass"].sum()), "number_failed": int((~all_checks["pass"]).sum()), "max_absolute_difference": all_checks.absolute_difference.max(), "median_absolute_difference": all_checks.absolute_difference.median()}])
    return pd.concat([per, overall], ignore_index=True), all_checks


def figures(output: Path, chunks: pd.DataFrame, windows: pd.DataFrame, verdicts: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4)); plt.hist(chunks.causal_filter_ms, bins=80, color="#4c78a8"); plt.xlabel("Causal filter chunk compute time (ms)"); plt.ylabel("Chunks"); plt.tight_layout(); plt.savefig(output / "filter_chunk_latency_distribution.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4)); plt.hist(windows.total_causal_compute_amortized_ms, bins=80, color="#72b7b2"); plt.xlabel("Window compute time (ms; feature + amortized frozen inference)"); plt.ylabel("Prediction windows"); plt.tight_layout(); plt.savefig(output / "window_compute_latency_distribution.png", dpi=180); plt.close()
    plt.figure(figsize=(5, 5));
    for target, part in verdicts.groupby("target"): plt.scatter(part.canonical_P_task, part.causal_P_task, s=22, alpha=.7, label=target.title())
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("Canonical task P(HIGH)"); plt.ylabel("Causal task P(HIGH)"); plt.legend(); plt.tight_layout(); plt.savefig(output / "task_probability_identity.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4)); plt.boxplot([verdicts.loc[verdicts.target.eq(t), "absolute_difference"] for t in ("valence", "arousal")], labels=["Valence", "Arousal"]); plt.ylabel("Absolute task P(HIGH) difference"); plt.tight_layout(); plt.savefig(output / "task_absolute_probability_difference_by_target.png", dpi=180); plt.close()
    pivot = verdicts.pivot_table(index="task", columns="target", values="verdict_agreement", aggfunc="mean").reindex(columns=["valence", "arousal"]); ax = pivot.plot.bar(figsize=(8,4), color=["#4c78a8", "#f58518"]); ax.set(ylabel="Verdict agreement", xlabel="Robot task", ylim=(0,1)); plt.tight_layout(); plt.savefig(output / "task_verdict_agreement.png", dpi=180); plt.close()
    plt.figure(figsize=(7,4)); [plt.hist(part.signed_difference, bins=40, alpha=.55, label=target.title()) for target, part in verdicts.groupby("target")]; plt.axvline(0, color="k", ls=":"); plt.xlabel("Causal − canonical task P(HIGH)"); plt.ylabel("Tasks"); plt.legend(); plt.tight_layout(); plt.savefig(output / "task_probability_shift_distribution.png", dpi=180); plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cohort-root", type=Path, default=ROOT / "outputs/robot_transfer/causal_filter_comparison/v1/cohort"); parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_transfer/causal_filter_comparison/v1/cohort_summary"); args = parser.parse_args()
    root = args.cohort_root.resolve(); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing summary directory: {output}")
    status = pd.read_csv(root / "cohort_run_status.csv"); complete = completion(status, root); participants = complete.loc[complete.tasks_completed.eq(len(TASKS)), "participant"].tolist()
    timing, timing_overall, chunks, windows = timing_summary(participants, root); model, participant_model, prediction, consensus = stability(participants, root); window_summary = consensus_summary(consensus); verdicts, verdict_summary, shifts = task_verdicts(consensus); validation, validation_all = validation_summary(participants, root)
    output.mkdir(parents=True)
    complete.to_csv(output / "cohort_completion_summary.csv", index=False); timing.to_csv(output / "cohort_computational_feasibility.csv", index=False); timing_overall.to_csv(output / "cohort_computational_feasibility_overall.csv", index=False); model.to_csv(output / "cohort_model_prediction_stability.csv", index=False); participant_model.to_csv(output / "cohort_model_prediction_stability_by_participant.csv", index=False); window_summary.to_csv(output / "cohort_window_consensus_stability.csv", index=False); verdicts.to_csv(output / "cohort_task_verdict_comparison.csv", index=False); verdict_summary.to_csv(output / "cohort_task_verdict_summary.csv", index=False); shifts.to_csv(output / "cohort_task_probability_shift_summary.csv", index=False); validation.to_csv(output / "cohort_canonical_validation_summary.csv", index=False)
    figures(output, chunks, windows, verdicts)
    overview = {"analysis": "descriptive aggregation of frozen canonical versus causal replay outputs", "robot_ratings_used": False, "participant_status_counts": complete.status.value_counts().to_dict(), "complete_participants": participants, "incomplete_participants": complete.loc[complete.tasks_completed.ne(len(TASKS)), "participant"].tolist(), "total_prediction_windows": int(len(prediction.drop_duplicates(["participant", "task", "segment_id", "window_id"]))), "total_model_probability_pairs": int(len(prediction)), "total_complete_top3_window_target_pairs": int(len(consensus)), "canonical_validation_failures": int((~validation_all["pass"]).sum()), "timing_scope": "offline replay computation only; excludes hardware acquisition, Bluetooth, camera, driver, and synchronization latency", "task_verdict_definition": "median of three frozen model probabilities per complete window, then median across complete windows; HIGH if >= 0.5", "files": {"completion": "cohort_completion_summary.csv", "computational": "cohort_computational_feasibility.csv", "model_stability": "cohort_model_prediction_stability.csv", "window_consensus": "cohort_window_consensus_stability.csv", "task_verdicts": "cohort_task_verdict_comparison.csv", "validation": "cohort_canonical_validation_summary.csv"}}
    (output / "cohort_summary.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    print(f"Cohort summary complete for {len(participants)} complete participants: {output}"); return 0


if __name__ == "__main__": raise SystemExit(main())
