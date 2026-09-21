#!/usr/bin/env python3
"""Run an isolated, stateful causal-filter replay against frozen EEG models.

This experiment never edits the canonical Robot branch, raw data, frozen
models, or established transfer outputs. It is intentionally explicit and
single-participant; all task processing requires a user command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
ROBOT_ROOT = ROOT / "processing/multimodal_robot"
for value in (ROOT, ROBOT_ROOT):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from processing.multimodal_image.src.eeg import _prepare_raw
from processing.multimodal_robot.analysis.causal_streaming.causal_filter import (
    SAMPLE_RATE_HZ, SENSITIVITY_WARMUP_SECONDS, WARMUP_SECONDS, causal_sos,
    causal_warmup_excluded, filter_design_metadata, filter_stream,
)
from processing.multimodal_robot.analysis.causal_streaming.compare_canonical_causal import (
    KEYS, canonical_eeg_wide, compare_features, compare_predictions,
)
from processing.multimodal_robot.prep.continuous import _extract_eeg_window_features, _files, robot_eeg_config
from processing.multimodal_robot.run_robot_pipeline import load, localize_task_paths
from processing.multimodal_robot.transfer import run_final_frozen_eeg_transfer as transfer

TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")
CANONICAL_VALIDATION_ATOL = 1e-9
CANONICAL_VALIDATION_RTOL = 1e-15


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participant", default="P27")
    p.add_argument("--participant-config", type=Path, default=ROOT / "processing/multimodal_robot/configs/participants/P27.yaml")
    p.add_argument("--raw-root", type=Path, default=ROOT / "data")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--chunk-ms", type=float, default=100.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--canonical-eeg-csv", type=Path, default=ROOT / "derived/P27/Robot_Experiment/runs/p27_robot_final_features/features/eeg_window_features.csv")
    p.add_argument("--canonical-predictions-csv", type=Path, default=ROOT / "outputs/robot_transfer/eeg_only/P27/P27_window_predictions.csv")
    p.add_argument("--canonical-consensus-csv", type=Path, default=ROOT / "outputs/robot_transfer/eeg_only/P27/P27_window_consensus.csv")
    p.add_argument("--image-reference-csv", type=Path, default=ROOT / "derived/P27/Image_Experiment/runs/p27_no_ica/merged/p27_image_trial_dataset.csv")
    p.add_argument("--canonical-validation-windows", type=int, default=3, help="Per-task zero-phase replay validation sample; 0 disables validation.")
    return p.parse_args()


def model_args(participant: str) -> SimpleNamespace:
    return SimpleNamespace(participant=participant,
        top3_csv=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv",
        model_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_models",
        calibration_root=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_probability_calibration_v1")


def metadata(participant: str, task: str, segment: int, start: float) -> dict[str, object]:
    return {"participant": participant, "task": task, "segment_id": segment,
            "window_id": f"{task}_s{segment}_{start:.3f}", "window_start_s": start, "window_end_s": start + 2.0}


def causal_features(participant: str, task_name: str, segment: int, raw, chunk_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """CAR then causal-filter one independent BDF and retain the canonical grid."""
    eeg_names = list(robot_eeg_config()["channels"]["eeg_mapping"].values())
    if not np.isclose(raw.info["sfreq"], SAMPLE_RATE_HZ):
        raise ValueError(f"{task_name}: expected {SAMPLE_RATE_HZ:g} Hz, found {raw.info['sfreq']:g}")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    data = raw.copy().pick(eeg_names).get_data()
    causal, chunks = filter_stream(data, causal_sos(), chunk_samples)
    long, wide, latency = [], [], []
    duration = causal.shape[1] / SAMPLE_RATE_HZ
    for start in np.arange(0.0, duration - 2.0 + 1e-9, 1.0):
        first = int(round(start * SAMPLE_RATE_HZ)); segment_data = causal[:, first:first + 500]
        if segment_data.shape[1] != 500: continue
        started = perf_counter_ns(); values = _extract_eeg_window_features(segment_data, SAMPLE_RATE_HZ); feature_ms = (perf_counter_ns()-started)/1e6
        meta = metadata(participant, task_name, segment, float(start)); excluded = causal_warmup_excluded(float(start))
        row = {**meta, "causal_warmup_excluded": excluded}
        for channel, features in zip(eeg_names, values, strict=True):
            long.append({**row, "channel": channel, "sample_count": 500, **features})
            row.update({f"{key}__{channel}": value for key, value in features.items()})
        wide.append(row)
        latency.append({**meta, "causal_warmup_excluded": excluded, "eeg_feature_ms": feature_ms})
    chunk_rows = [{"participant": participant, "task": task_name, "segment_id": segment, **row} for row in chunks]
    return pd.DataFrame(long), pd.DataFrame(wide), chunk_rows, latency


def canonical_validation(participant: str, task_name: str, segment: int, raw, canonical: pd.DataFrame, count: int) -> pd.DataFrame:
    """Sample the canonical zero-phase function and compare saved feature rows."""
    if count <= 0: return pd.DataFrame()
    from processing.multimodal_robot.prep.continuous import prepare_continuous_robot_eeg
    eeg_names = list(robot_eeg_config()["channels"]["eeg_mapping"].values())
    replay = raw.copy(); prepare_continuous_robot_eeg(replay)
    rows = []
    for start in np.arange(0.0, replay.n_times / SAMPLE_RATE_HZ - 2.0 + 1e-9, 1.0)[:count]:
        data = replay.copy().pick(eeg_names).get_data(start=int(round(start*SAMPLE_RATE_HZ)), stop=int(round(start*SAMPLE_RATE_HZ))+500)
        item = metadata(participant, task_name, segment, float(start))
        for channel, features in zip(eeg_names, _extract_eeg_window_features(data, SAMPLE_RATE_HZ), strict=True):
            rows.append({**item, "channel": channel, **features})
    check = pd.DataFrame(rows).merge(canonical, on=KEYS + ["channel"], suffixes=("_replay", "_canonical"), validate="one_to_one")
    features = [name for name in rows[0] if name.startswith("eeg_")] if rows else []
    output = []
    for name in features:
        diff = (check[f"{name}_replay"] - check[f"{name}_canonical"]).abs()
        output.extend({**{key: check.at[index, key] for key in KEYS + ["channel"]}, "feature": name,
                       "absolute_difference": value,
                       "pass": bool(np.isclose(check.at[index, f"{name}_replay"], check.at[index, f"{name}_canonical"],
                                                 rtol=CANONICAL_VALIDATION_RTOL, atol=CANONICAL_VALIDATION_ATOL, equal_nan=False))}
                      for index, value in diff.items())
    return pd.DataFrame(output)


def figures(output: Path, family: pd.DataFrame, probability: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4)); values = family.set_index("feature_family").median_pearson_r.sort_values()
    values.plot.barh(ax=ax, color="#4c78a8"); ax.set(xlabel="Median Pearson r", title="Canonical vs causal feature-family correlation", xlim=(-1, 1)); fig.tight_layout(); fig.savefig(output/"feature_family_correlation_heatmap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); values = family.set_index("feature_family").median_symmetric_relative_difference.sort_values()
    values.plot.barh(ax=ax, color="#e45756"); ax.set(xlabel="Median symmetric relative difference", title="Canonical vs causal feature-family difference"); fig.tight_layout(); fig.savefig(output/"feature_family_relative_difference_heatmap.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, (target, part) in zip(axes, comparison.groupby("target")):
        ax.scatter(part.high_probability_canonical, part.high_probability_causal, s=14, alpha=.7); ax.plot([0,1],[0,1],"k:"); ax.set(title=target.title(), xlabel="Canonical P(HIGH)", ylabel="Causal P(HIGH)")
    fig.tight_layout(); fig.savefig(output/"probability_scatter.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    for (target, rank), part in comparison.groupby(["target", "model_rank"]):
        part=part.sort_values("window_start_s")
        ax.plot(part.window_start_s, part.high_probability_canonical, lw=1, ls="--", label=f"{target} r{rank} canonical")
        ax.plot(part.window_start_s, part.high_probability_causal, lw=1, label=f"{target} r{rank} causal")
    ax.set(title="Canonical and causal P(HIGH) trajectories", xlabel="Window start (s)", ylabel="P(HIGH)"); ax.legend(ncol=3, fontsize=6.5); fig.tight_layout(); fig.savefig(output/"probability_trajectories.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4)); ax.hist(pd.to_numeric(probability.causal_filter_ms), bins="auto", color="#72b7b2"); ax.set(title="Causal streaming filter chunk latency", xlabel="Filtering time (ms)", ylabel="Chunks"); fig.tight_layout(); fig.savefig(output/"causal_filter_latency_distribution.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    disagreement = comparison.loc[~comparison.hard_agree]
    ax.scatter(comparison.window_start_s, comparison.model_rank, color="0.75", s=12, label="Agreement")
    if len(disagreement): ax.scatter(disagreement.window_start_s, disagreement.model_rank, color="#e45756", s=22, label="Disagreement")
    ax.set(title="Canonical versus causal hard-prediction agreement", xlabel="Window start (s)", ylabel="Model rank"); ax.legend(); fig.tight_layout(); fig.savefig(output/"hard_prediction_disagreement_timeline.png", dpi=180); plt.close(fig)


def participant_target_metrics(causal_task: pd.DataFrame, canonical_task: pd.DataFrame) -> pd.DataFrame:
    """Compute participant-by-target BA only when the six task labels contain both classes."""
    rows=[]
    for target, causal in causal_task.groupby("target", sort=False):
        canonical=canonical_task[canonical_task.target.eq(target)]
        for branch, frame in (("canonical", canonical), ("causal", causal)):
            rating=frame.robot_rating_class.astype(str); predicted=frame.descriptive_task_verdict.astype(str)
            high=rating.eq("HIGH"); low=rating.eq("LOW")
            ba=np.nan if not high.any() or not low.any() else .5*((predicted[high].eq("HIGH").mean())+(predicted[low].eq("LOW").mean()))
            rows.append({"participant":frame.participant.iloc[0] if len(frame) else pd.NA,"target":target,"branch":branch,"n_tasks":len(frame),"actual_high_count":int(high.sum()),"actual_low_count":int(low.sum()),"balanced_accuracy":ba})
    return pd.DataFrame(rows)


def main() -> int:
    a = args(); output = a.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    if a.chunk_ms <= 0 or not np.isclose(a.chunk_ms * SAMPLE_RATE_HZ / 1000, round(a.chunk_ms * SAMPLE_RATE_HZ / 1000)):
        raise ValueError("--chunk-ms must correspond to a positive integer number of 250-Hz samples")
    config = localize_task_paths(load(a.participant_config.resolve()), a.raw_root.resolve())
    canonical_long = pd.read_csv(a.canonical_eeg_csv); canonical_wide = canonical_eeg_wide(a.canonical_eeg_csv)
    chunk_samples = int(round(a.chunk_ms * SAMPLE_RATE_HZ / 1000)); all_long=[]; all_wide=[]; chunks=[]; latency=[]; validation=[]
    for task_name in a.tasks:
        task = config["tasks"][task_name]
        for segment, path in enumerate(_files(task["eeg"]), start=1):
            raw, _, _ = _prepare_raw({"eeg": path}, robot_eeg_config())
            validation.append(canonical_validation(a.participant, task_name, segment, raw, canonical_long, a.canonical_validation_windows))
            long, wide, chunk_rows, latency_rows = causal_features(a.participant, task_name, segment, raw, chunk_samples)
            all_long.append(long); all_wide.append(wide); chunks.extend(chunk_rows); latency.extend(latency_rows)
    causal_long=pd.concat(all_long,ignore_index=True); causal_wide=pd.concat(all_wide,ignore_index=True); timing=pd.DataFrame(chunks); latency=pd.DataFrame(latency)
    validation=pd.concat([x for x in validation if not x.empty],ignore_index=True) if any(not x.empty for x in validation) else pd.DataFrame()
    if not validation.empty and not validation["pass"].all(): raise RuntimeError("Canonical zero-phase replay validation failed; causal comparison is not interpretable")
    models=transfer.load_frozen_models(model_args(a.participant)); causal_for_inference=causal_wide.loc[~causal_wide.causal_warmup_excluded].drop(columns="causal_warmup_excluded")
    inference_started = perf_counter_ns()
    predictions, _, shift_tasks = transfer.run_inference(models, causal_for_inference, a.image_reference_csv)
    inference_ms = (perf_counter_ns() - inference_started) / 1e6
    complete, completeness=transfer.complete_top3_windows(predictions); agreement=transfer.agreement(complete, completeness)
    ratings=transfer.robot_ratings(a.participant); task_stats=transfer.task_statistics(complete, agreement, ratings, {}, shift_tasks)
    feature_detail, family=compare_features(canonical_wide, causal_wide)
    probability_detail, probability_summary, disagreements=compare_predictions(a.canonical_predictions_csv, predictions)
    output.mkdir(parents=True)
    causal_long.to_csv(output/"causal_window_features_long.csv",index=False); causal_wide.to_csv(output/"causal_window_features_wide.csv",index=False)
    predictions.to_csv(output/"causal_window_predictions.csv",index=False); complete.to_csv(output/"causal_window_consensus.csv",index=False)
    feature_detail.to_csv(output/"canonical_causal_feature_agreement.csv",index=False); family.to_csv(output/"canonical_causal_feature_family_summary.csv",index=False)
    probability_detail.to_csv(output/"canonical_causal_probability_agreement.csv",index=False); probability_summary.to_csv(output/"canonical_causal_probability_summary.csv",index=False)
    probability_detail[[*KEYS,"target","model_rank","original_hard_prediction_canonical","original_hard_prediction_causal","hard_agree"]].to_csv(output/"canonical_causal_hard_prediction_agreement.csv",index=False); disagreements.to_csv(output/"canonical_causal_disagreement_windows.csv",index=False)
    canonical_consensus = pd.read_csv(a.canonical_consensus_csv)
    consensus_compare = canonical_consensus.merge(complete, on=KEYS + ["target"], suffixes=("_canonical", "_causal"), validate="one_to_one")
    consensus_compare["signed_consensus_probability_difference"] = consensus_compare.top3_consensus_probability_causal - consensus_compare.top3_consensus_probability_canonical
    consensus_compare["consensus_class_agree"] = consensus_compare.top3_consensus_class_causal.eq(consensus_compare.top3_consensus_class_canonical)
    canonical_task_path = a.canonical_predictions_csv.parent / f"{a.participant}_task_consensus_analysis.csv"
    canonical_task = pd.read_csv(canonical_task_path)
    metrics = participant_target_metrics(task_stats, canonical_task[canonical_task.task.isin(a.tasks)])
    task_stats.to_csv(output/"canonical_causal_task_consensus.csv",index=False); consensus_compare.to_csv(output/"canonical_causal_consensus_agreement.csv",index=False)
    metrics.to_csv(output/"canonical_causal_participant_target_metrics.csv",index=False)
    timing.to_csv(output/"causal_stream_timing_chunks.csv",index=False)
    latency["frozen_inference_amortized_ms"] = inference_ms / len(predictions) if len(predictions) else np.nan
    latency["total_causal_compute_amortized_ms"] = latency.eeg_feature_ms + latency.frozen_inference_amortized_ms
    latency.to_csv(output/"causal_window_latency.csv",index=False); validation.to_csv(output/"canonical_replay_validation.csv",index=False)
    json.dump(filter_design_metadata(causal_sos()) | {"chunk_ms":a.chunk_ms,"chunk_samples":chunk_samples,"tasks":a.tasks}, (output/"causal_filter_design.json").open("w"),indent=2)
    json.dump({"participant":a.participant,"tasks":a.tasks,"canonical_validation_pass":bool(validation.empty or validation["pass"].all()),"canonical_validation_atol":CANONICAL_VALIDATION_ATOL,"canonical_validation_rtol":CANONICAL_VALIDATION_RTOL,"warmup_seconds":WARMUP_SECONDS,"warmup_sensitivity_seconds":SENSITIVITY_WARMUP_SECONDS,"frozen_inference_total_ms":inference_ms,"no_training_or_refitting":True}, (output/"causal_run_manifest.json").open("w"),indent=2)
    figures(output, family, timing, probability_detail)
    print(f"Causal replay complete: {output}"); return 0

if __name__ == "__main__": raise SystemExit(main())
