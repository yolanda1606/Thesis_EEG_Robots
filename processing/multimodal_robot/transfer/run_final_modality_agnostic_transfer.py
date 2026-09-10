#!/usr/bin/env python3
"""Apply final frozen modality-agnostic Image top-3 models to Robot windows.

This is a separate branch from the historical EEG-only transfer entry point.
It performs no model fitting, model selection, calibration, threshold tuning,
or Robot-based optimization. Robot ratings are descriptive post-inference data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Support direct execution from the repository root, matching the established
# bootstrap used by run_personalized_inference.py.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from processing.multimodal_robot.transfer.run_final_frozen_eeg_transfer import (
    DISPLAY_TASKS,
    KEYS,
    TASK_ORDER,
    event_style,
    event_table,
    image_reference,
    robot_conditions,
    robot_ratings,
    shift_diagnostics,
)


KEY_WITH_TARGET = [*KEYS, "target"]
MODALITIES = {"eeg", "face", "multimodal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--top3-csv", type=Path, default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target_modality.csv")
    parser.add_argument("--image-model-root", type=Path, default=ROOT / "outputs/image_classification/focused_personalized_binary_v1")
    parser.add_argument("--calibration-root", type=Path, default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/modality_agnostic_probability_calibration_v1")
    parser.add_argument("--robot-eeg-csv", type=Path, required=True)
    parser.add_argument("--robot-video-csv", type=Path, required=True)
    parser.add_argument("--robot-multimodal-csv", type=Path, required=True)
    parser.add_argument("--image-reference-csv", type=Path, required=True)
    parser.add_argument("--participant-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate frozen inputs and schema without writing outputs.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = pd.read_csv(args.top3_csv)
    rows = rows[(rows.participant == args.participant) & rows.target.isin(["valence", "arousal"])].copy()
    if len(rows) != 6 or rows.groupby("target").size().to_dict() != {"arousal": 3, "valence": 3}:
        raise ValueError(f"Expected exactly three authoritative frozen rows per target for {args.participant}")
    if set(rows.modality) - MODALITIES or set(rows.selection_scope) != {"modality_agnostic_image_domain_top3"}:
        raise ValueError("Top-3 input is not the authoritative modality-agnostic Image-domain selection")
    records: list[dict[str, Any]] = []
    for row in rows.sort_values(["target", "transfer_rank"]).itertuples(index=False):
        path = Path(row.final_model_path).resolve()
        if not path.is_file() or not path.is_relative_to(args.image_model_root):
            raise ValueError(f"Frozen Image model is absent or outside --image-model-root: {path}")
        pipeline = joblib.load(path)
        if not hasattr(pipeline, "feature_names_in_"):
            raise ValueError(f"{path.name}: not a fitted frozen pipeline")
        candidates = list(pipeline.feature_names_in_)
        selector = pipeline.named_steps.get("selector")
        selected = list(selector.get_feature_names_out(pipeline.feature_names_in_)) if selector is not None else candidates.copy()
        if selected != json.loads(row.actual_selected_feature_names):
            raise ValueError(f"{path.name}: authoritative selected features do not match frozen pipeline")
        classifier = pipeline.named_steps["classifier"]
        classes = [int(value) for value in classifier.classes_]
        if 1 not in classes:
            raise ValueError(f"{path.name}: HIGH class absent")
        record = {
            "participant": row.participant, "target": row.target, "model_rank": int(row.transfer_rank),
            "modality": row.modality, "classifier": row.classifier, "feature_family": row.feature_family,
            "mean_outer_cv_balanced_accuracy": float(row.mean_outer_cv_balanced_accuracy),
            "authoritative_final_hyperparameters": row.final_refit_best_parameters,
            "model_path": path, "model_sha256": sha256(path), "pipeline": pipeline,
            "candidate_features": candidates, "selected_features": selected,
            "high_probability_column": classes.index(1), "probability_source": "original_frozen_pipeline",
        }
        if str(row.classifier).lower() == "svm":
            artifact = args.calibration_root / "artifacts" / f"{row.participant}_{row.target}_rank{int(row.transfer_rank)}_calibrated.joblib"
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing Image-only SVM calibration artifact: {artifact}")
            payload = joblib.load(artifact)
            metadata, probability_model = payload.get("deployment_metadata", {}), payload.get("calibrated_probability_model")
            if metadata.get("original_frozen_model_sha256") != record["model_sha256"]:
                raise ValueError(f"{artifact.name}: frozen-model SHA-256 mismatch")
            if metadata.get("modality") != row.modality or metadata.get("selected_features") != selected:
                raise ValueError(f"{artifact.name}: modality or selected-feature identity mismatch")
            if metadata.get("robot_data_used") is not False or not hasattr(probability_model, "predict_proba"):
                raise ValueError(f"{artifact.name}: not a valid Image-only calibrated probability artifact")
            calibrated_classes = [int(value) for value in probability_model.classes_]
            if list(probability_model.feature_names_in_) != selected or 1 not in calibrated_classes:
                raise ValueError(f"{artifact.name}: calibrated input schema or HIGH class mismatch")
            record.update({"probability_model": probability_model, "probability_source": "image_only_platt_calibration",
                           "calibrated_artifact_path": artifact.resolve(), "calibrated_artifact_sha256": sha256(artifact),
                           "calibrated_high_probability_column": calibrated_classes.index(1)})
        elif not hasattr(pipeline, "predict_proba"):
            raise ValueError(f"{path.name}: frozen non-SVM pipeline lacks predict_proba()")
        else:
            record["probability_model"] = pipeline
        records.append(record)
    return records


def robot_eeg_wide(path: Path) -> pd.DataFrame:
    long = pd.read_csv(path)
    required = set(KEYS + ["channel"])
    if missing := sorted(required - set(long.columns)):
        raise ValueError(f"Robot EEG table missing: {missing}")
    features = [name for name in long.columns if name.startswith("eeg_")]
    if not features or long.duplicated(KEYS + ["channel"]).any():
        raise ValueError("Robot EEG table has no EEG features or duplicate window/channel rows")
    wide = long.pivot(index=KEYS, columns="channel", values=features)
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    return wide.reset_index()


def robot_wide(path: Path, modality: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if missing := sorted(set(KEYS) - set(frame.columns)):
        raise ValueError(f"Robot {modality} table missing: {missing}")
    prefix = "video_" if modality == "face" else None
    feature_columns = [column for column in frame if (column.startswith(prefix) if prefix else column.startswith(("eeg_", "video_")))]
    if not feature_columns or frame.duplicated(KEYS).any():
        raise ValueError(f"Robot {modality} table has no expected features or duplicate window rows")
    return frame


def load_robot_tables(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return {"eeg": robot_eeg_wide(args.robot_eeg_csv), "face": robot_wide(args.robot_video_csv, "face"),
            "multimodal": robot_wide(args.robot_multimodal_csv, "multimodal")}


def infer(models: list[dict[str, Any]], robot_tables: dict[str, pd.DataFrame], image_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, shift_windows, shift_tasks = [], [], []
    for record in models:
        robot = robot_tables[record["modality"]]
        missing = sorted(set(record["candidate_features"]) - set(robot.columns))
        if missing:
            raise ValueError(f"{record['model_path'].name}: Robot {record['modality']} schema missing {missing}")
        values = robot.loc[:, record["candidate_features"]].apply(pd.to_numeric, errors="raise")
        valid = np.isfinite(values.to_numpy()).all(axis=1)
        if not valid.any():
            raise ValueError(f"{record['model_path'].name}: no finite Robot windows")
        frame, values = robot.loc[valid].reset_index(drop=True), values.loc[valid].reset_index(drop=True)
        hard = record["pipeline"].predict(values)
        output = frame.loc[:, KEYS].copy()
        output["target"], output["model_rank"], output["modality"] = record["target"], record["model_rank"], record["modality"]
        output["classifier"], output["feature_family"] = record["classifier"], record["feature_family"]
        output["selected_feature_count"], output["frozen_image_balanced_accuracy"] = len(record["selected_features"]), record["mean_outer_cv_balanced_accuracy"]
        output["original_hard_prediction"] = hard
        if str(record["classifier"]).lower() == "svm":
            calibrated = frame.loc[:, record["selected_features"]].apply(pd.to_numeric, errors="raise")
            output["high_probability"] = record["probability_model"].predict_proba(calibrated)[:, record["calibrated_high_probability_column"]]
        else:
            output["high_probability"] = record["probability_model"].predict_proba(values)[:, record["high_probability_column"]]
        predictions.append(output)
        reference = image_reference(image_path, record["selected_features"])
        windows, tasks = shift_diagnostics(record["pipeline"], record["candidate_features"], record["selected_features"], frame, reference)
        for item in (windows, tasks):
            item.insert(0, "modality", record["modality"]); item.insert(0, "model_rank", record["model_rank"]); item.insert(0, "target", record["target"])
        shift_windows.append(windows); shift_tasks.append(tasks)
    return pd.concat(predictions, ignore_index=True), pd.concat(shift_windows, ignore_index=True), pd.concat(shift_tasks, ignore_index=True)


def consensus(predictions: pd.DataFrame, models: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    modalities = {(item["target"], item["model_rank"]): item["modality"] for item in models}
    complete_rows, counts = [], []
    for (participant, target, task), part in predictions.groupby(["participant", "target", "task"], sort=False):
        prob = part.pivot(index=KEYS, columns="model_rank", values="high_probability").reindex(columns=[1, 2, 3])
        hard = part.pivot(index=KEYS, columns="model_rank", values="original_hard_prediction").reindex(columns=[1, 2, 3])
        mask = prob.notna().all(axis=1) & hard.notna().all(axis=1)
        total, n_complete = len(prob), int(mask.sum())
        counts.append({"participant": participant, "target": target, "task": task, "n_candidate_windows": total, "n_complete_top3_windows": n_complete, "n_incomplete_top3_windows": total - n_complete, "fraction_complete_top3_windows": n_complete / total if total else np.nan})
        if not n_complete:
            continue
        frame = prob.loc[mask].index.to_frame(index=False); frame["target"] = target
        frame["window_center_s"] = (frame.window_start_s + frame.window_end_s) / 2
        for rank in (1, 2, 3):
            frame[f"rank{rank}_modality"] = modalities[target, rank]
            frame[f"rank{rank}_p_high"] = prob.loc[mask, rank].to_numpy()
            frame[f"rank{rank}_hard_prediction"] = hard.loc[mask, rank].astype(int).to_numpy()
        p = prob.loc[mask]; frame["top3_consensus_probability"] = p.median(axis=1).to_numpy()
        frame["top3_probability_min"], frame["top3_probability_max"] = p.min(axis=1).to_numpy(), p.max(axis=1).to_numpy()
        frame["top3_probability_range"] = frame.top3_probability_max - frame.top3_probability_min
        frame["top3_consensus_class"] = np.where(frame.top3_consensus_probability >= .5, "HIGH", "LOW")
        frame["all_three_hard_agree"] = hard.loc[mask].nunique(axis=1).eq(1).to_numpy()
        complete_rows.append(frame)
    columns = [*KEYS, "target", "window_center_s", *[f"rank{rank}_{field}" for rank in (1,2,3) for field in ("modality", "p_high", "hard_prediction")], "top3_consensus_probability", "top3_probability_min", "top3_probability_max", "top3_probability_range", "top3_consensus_class", "all_three_hard_agree"]
    complete = pd.concat(complete_rows, ignore_index=True) if complete_rows else pd.DataFrame(columns=columns)
    return complete.sort_values(["target", "task", "segment_id", "window_center_s"], kind="stable").reset_index(drop=True), pd.DataFrame(counts)


def agreement_and_tasks(complete: pd.DataFrame, counts: pd.DataFrame, models: list[dict[str, Any]], participant: str, config: Path, shift: pd.DataFrame) -> pd.DataFrame:
    modalities = {(item["target"], item["model_rank"]): item["modality"] for item in models}
    ratings, conditions = robot_ratings(participant), robot_conditions(config)
    rows = []
    for item in counts.itertuples(index=False):
        part = complete[(complete.target == item.target) & (complete.task == item.task)]
        rating = ratings[(ratings.target == item.target) & (ratings.task == item.task)].iloc[0]
        row = {**item._asdict(), "task_display": DISPLAY_TASKS[item.task], "robot_condition": conditions.get(item.task, "N/A"), "robot_rating": float(rating.robot_rating), "robot_rating_1_to_7": float(rating.robot_rating), "robot_rating_class": rating.robot_rating_class,
               **{f"rank{rank}_modality": modalities[item.target, rank] for rank in (1,2,3)}}
        if part.empty:
            row.update({key: np.nan for key in ("median_top3_consensus_probability", "mean_top3_consensus_probability", "fraction_consensus_windows_ge_0_5", "unanimous_agreement_rate", "unanimous_high_rate", "unanimous_low_rate", "majority_vote_high_fraction", "median_top3_probability_range", "mean_top3_probability_range", "mean_pairwise_absolute_probability_difference", "probability_correlation_1_2", "probability_correlation_1_3", "probability_correlation_2_3")}); row["descriptive_task_verdict"] = pd.NA; row["verdict_matches_robot_rating"] = pd.NA
            row["consensus_verdict"] = pd.NA; row["consensus_matches_robot_rating"] = pd.NA
        else:
            hard = part[["rank1_hard_prediction", "rank2_hard_prediction", "rank3_hard_prediction"]]
            median = float(part.top3_consensus_probability.median()); verdict = "HIGH" if median >= .5 else "LOW"
            pairs = pd.concat([(part[f"rank{a}_p_high"] - part[f"rank{b}_p_high"]).abs() for a,b in ((1,2),(1,3),(2,3))], axis=1)
            row.update({"median_top3_consensus_probability": median, "mean_top3_consensus_probability": float(part.top3_consensus_probability.mean()), "fraction_consensus_windows_ge_0_5": float((part.top3_consensus_probability >= .5).mean()), "descriptive_task_verdict": verdict, "consensus_verdict": verdict, "verdict_matches_robot_rating": verdict == rating.robot_rating_class, "consensus_matches_robot_rating": verdict == rating.robot_rating_class, "unanimous_agreement_rate": float(part.all_three_hard_agree.mean()), "unanimous_high_rate": float((hard == 1).all(axis=1).mean()), "unanimous_low_rate": float((hard == 0).all(axis=1).mean()), "majority_vote_high_fraction": float((hard.sum(axis=1) >= 2).mean()), "median_top3_probability_range": float(part.top3_probability_range.median()), "mean_top3_probability_range": float(part.top3_probability_range.mean()), "mean_pairwise_absolute_probability_difference": float(pairs.mean(axis=1).mean())})
            for a,b in ((1,2),(1,3),(2,3)): row[f"probability_correlation_{a}_{b}"] = float(part[f"rank{a}_p_high"].corr(part[f"rank{b}_p_high"]))
        row["models_with_shift_warning"] = ";".join(f"R{rank}" for rank in shift[(shift.target == item.target) & (shift.task == item.task) & (shift.median_nearest_image_distance > 10)].model_rank)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "task"]).reset_index(drop=True)


def segment_parts(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [part.sort_values("window_center_s", kind="stable") for _, part in frame.groupby("segment_id", sort=True)]


def model_legend_label(rank_data: pd.DataFrame, rank: int) -> str:
    """Match the EEG-only legend while exposing the frozen model modality."""
    classifier = {"svm": "SVM", "logreg": "LogReg", "gnb": "GNB"}.get(
        str(rank_data.classifier.iloc[0]).lower(), str(rank_data.classifier.iloc[0]).upper()
    )
    family = str(rank_data.feature_family.iloc[0])
    feature_count = int(rank_data.selected_feature_count.iloc[0])
    image_ba = float(rank_data.frozen_image_balanced_accuracy.iloc[0])
    return f"R{rank} {str(rank_data.modality.iloc[0]).title()} | {classifier} | {family} | k={feature_count} | BA={image_ba:.3f}"


def plot_outputs(predictions: pd.DataFrame, complete: pd.DataFrame, statistics: pd.DataFrame, events: pd.DataFrame, participant: str, figures: Path) -> None:
    colors = {1: "#1b9e77", 2: "#377eb8", 3: "#984ea3"}
    for target in ("valence", "arousal"):
        tasks = [task for task in TASK_ORDER if task in set(predictions[predictions.target == target].task)]
        fig, axes = plt.subplots(len(tasks), 1, figsize=(14.5, 2.15 * len(tasks)))
        for axis, task in zip(np.atleast_1d(axes), tasks):
            part = predictions[(predictions.target == target) & (predictions.task == task)].copy()
            part["window_center_s"] = (part.window_start_s + part.window_end_s) / 2
            for (rank, modality), model in part.groupby(["model_rank", "modality"]):
                for index, segment in enumerate(segment_parts(model)):
                    axis.plot(segment.window_center_s, segment.high_probability, color=colors[rank], lw=.9, marker="o", ms=2.1, alpha=.75, label=model_legend_label(model, rank) if index == 0 else None)
            axis.axhline(.5, color="black", ls=":", lw=.8); axis.set_ylim(0, 1)
            axis.set_ylabel(DISPLAY_TASKS[task]); axis.set_title(DISPLAY_TASKS[task], loc="left", fontsize=9)
        axes[0].legend(ncol=1, fontsize=6.7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        axes[-1].set_xlabel("Time within task (s)")
        fig.suptitle(f"{participant} {target}: frozen-model P(HIGH)")
        fig.tight_layout(rect=(0, 0, .68, .98)); fig.savefig(figures / f"{participant}_{target}_probability_timeline.png", dpi=170); plt.close(fig)
        fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 1.75 * len(tasks)))
        for axis, task in zip(np.atleast_1d(axes), tasks):
            part = complete[(complete.target == target) & (complete.task == task)]
            for index, segment in enumerate(segment_parts(part)):
                axis.fill_between(segment.window_center_s, segment.top3_probability_min, segment.top3_probability_max, color=".35", alpha=.18, label="Top-3 range" if index == 0 else None)
                axis.plot(segment.window_center_s, segment.top3_consensus_probability, color=".15", lw=1.8, label="Top-3 median" if index == 0 else None)
            axis.axhline(.5, color="black", ls=":", lw=.8); axis.set_ylim(0, 1)
            axis.set_ylabel(DISPLAY_TASKS[task]); axis.set_title(DISPLAY_TASKS[task], loc="left", fontsize=9)
        axes[-1].set_xlabel("Time within task (s)"); axes[0].legend(fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, .98)); fig.suptitle(f"{participant} {target}: descriptive top-3 median P(HIGH)")
        fig.savefig(figures / f"{participant}_{target}_probability_consensus.png", dpi=170); plt.close(fig)
        for task in tasks:
            part = complete[(complete.target == target) & (complete.task == task)]; pieces = segment_parts(part)
            task_events = events[events.task == task]; fig, axis = plt.subplots(figsize=(16 if len(task_events.event_label.drop_duplicates()) >= 5 or task in {"stack", "sisyphus"} else 13, 5.2))
            for index, segment in enumerate(pieces):
                axis.fill_between(segment.window_center_s, segment.top3_probability_min, segment.top3_probability_max, color=".35", alpha=.18, label="Top-3 range" if index == 0 else None)
                axis.plot(segment.window_center_s, segment.top3_consensus_probability, color=".15", lw=1.8, label="Top-3 median" if index == 0 else None)
            seen: set[str] = set()
            if len(pieces) == 1:
                for event in task_events.itertuples(index=False):
                    label = event.event_label if event.event_label not in seen else None
                    color, style = event_style(event.event_label); axis.axvline(event.time_s, color=color, ls=style, lw=.9, alpha=.8, label=label); seen.add(event.event_label)
            else:
                axis.text(.01, .03, "Trigger overlay omitted: segment-local times lack a verified task-global mapping.", transform=axis.transAxes, ha="left", va="bottom", fontsize=6.5, bbox={"boxstyle":"round,pad=0.25", "facecolor":"white", "edgecolor":"0.6", "alpha":.9})
            stat = statistics[(statistics.target == target) & (statistics.task == task)].iloc[0]
            annotation = f"Self-report: {stat.robot_rating_1_to_7:.0f}/7 ({stat.robot_rating_class})\nTask median P(HIGH): {stat.median_top3_consensus_probability:.2f} → {stat.descriptive_task_verdict}\nUnanimous top-3 windows: {stat.unanimous_agreement_rate:.0%}"
            axis.text(.995, .04, annotation, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5, bbox={"boxstyle":"round,pad=0.25", "facecolor":"white", "edgecolor":"0.6", "alpha":.9})
            axis.axhline(.5, color="black", ls=":", lw=.8); axis.set_ylim(0,1); axis.set_xlabel("Time within task (s)"); axis.set_ylabel("P(HIGH)")
            condition = stat.robot_condition; title = DISPLAY_TASKS[task] if condition == "N/A" else f"{DISPLAY_TASKS[task]} ({condition})"
            fig.suptitle(f"{participant} {target} {title}: descriptive top-3 consensus with semantic Robot events", y=.97, fontsize=13)
            axis.legend(fontsize=7, ncol=min(4, max(1, len(seen) + 2)), loc="lower center", bbox_to_anchor=(.5, 1.03), frameon=True)
            fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.70); fig.savefig(figures / f"{participant}_{target}_{DISPLAY_TASKS[task]}_consensus_triggers.png", dpi=170); plt.close(fig)


def main() -> int:
    args = parse_args()
    for name in ("top3_csv", "image_model_root", "calibration_root", "robot_eeg_csv", "robot_video_csv", "robot_multimodal_csv", "image_reference_csv", "participant_config", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    models, robot_tables = load_models(args), load_robot_tables(args)
    compatibility = {"models": [{"target": m["target"], "rank": m["model_rank"], "modality": m["modality"], "model_sha256": m["model_sha256"], "missing_candidate_features": sorted(set(m["candidate_features"]) - set(robot_tables[m["modality"]].columns))} for m in models]}
    if any(item["missing_candidate_features"] for item in compatibility["models"]): raise ValueError(f"Robot schema incompatible: {compatibility}")
    events, event_sources = event_table(args.participant_config)
    if args.dry_run:
        print(json.dumps({"status":"preflight_ok", "participant":args.participant, "compatibility":compatibility, "robot_data_used_for_fitting":False, "event_count":len(events)}, indent=2)); return 0
    if args.output_dir.exists(): raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    predictions, shift_windows, shift_tasks = infer(models, robot_tables, args.image_reference_csv)
    complete, completeness = consensus(predictions, models)
    statistics = agreement_and_tasks(complete, completeness, models, args.participant, args.participant_config, shift_tasks)
    probability_summary = predictions.groupby(["target", "model_rank", "modality", "classifier", "feature_family", "task"], dropna=False).agg(
        n_windows=("high_probability", "size"), mean_high_probability=("high_probability", "mean"),
        median_high_probability=("high_probability", "median"), sd_high_probability=("high_probability", "std"),
        fraction_predicted_high=("original_hard_prediction", "mean"),
    ).reset_index()
    verdict_data = statistics[[
        "task", "task_display", "target", "n_candidate_windows", "n_complete_top3_windows",
        "n_incomplete_top3_windows", "fraction_complete_top3_windows",
        "median_top3_consensus_probability", "consensus_verdict", "unanimous_agreement_rate",
        "robot_rating", "robot_rating_class", "consensus_matches_robot_rating",
        "models_with_shift_warning", "rank1_modality", "rank2_modality", "rank3_modality",
    ]].copy()
    ratings = robot_ratings(args.participant)
    conditions = robot_conditions(args.participant_config)
    ratings["robot_condition"] = ratings.task.map(conditions).fillna("N/A")
    args.output_dir.mkdir(parents=True); figures = args.output_dir / "figures"; figures.mkdir()
    predictions.to_csv(args.output_dir / f"{args.participant}_window_predictions.csv", index=False); complete.to_csv(args.output_dir / f"{args.participant}_window_consensus.csv", index=False); completeness.to_csv(args.output_dir / f"{args.participant}_top3_window_completeness.csv", index=False); probability_summary.to_csv(args.output_dir / f"{args.participant}_task_probability_summary.csv", index=False); statistics.to_csv(args.output_dir / f"{args.participant}_task_consensus_analysis.csv", index=False); statistics.to_csv(args.output_dir / f"{args.participant}_model_agreement.csv", index=False); verdict_data.to_csv(args.output_dir / f"{args.participant}_task_verdict_summary.csv", index=False); ratings.to_csv(args.output_dir / f"{args.participant}_robot_rating_comparison.csv", index=False); shift_windows.to_csv(args.output_dir / f"{args.participant}_selected_feature_shift_window_diagnostics.csv", index=False); shift_tasks.to_csv(args.output_dir / f"{args.participant}_selected_feature_shift_task_summary.csv", index=False)
    semantic_events_path = args.output_dir / f"{args.participant}_semantic_events.csv"
    events.to_csv(semantic_events_path, index=False)
    plot_outputs(predictions, complete, statistics, events, args.participant, figures)
    manifest = {"participant":args.participant, "purpose":"Final frozen modality-agnostic Image-to-Robot transfer", "command":" ".join([sys.executable,*sys.argv]), "top3_csv":str(args.top3_csv), "top3_csv_sha256":sha256(args.top3_csv), "robot_eeg_input":str(args.robot_eeg_csv), "robot_eeg_sha256":sha256(args.robot_eeg_csv), "robot_video_input":str(args.robot_video_csv), "robot_video_sha256":sha256(args.robot_video_csv), "robot_multimodal_input":str(args.robot_multimodal_csv), "robot_multimodal_sha256":sha256(args.robot_multimodal_csv), "image_reference_input":str(args.image_reference_csv), "image_reference_sha256":sha256(args.image_reference_csv), "participant_config":str(args.participant_config), "participant_config_sha256":sha256(args.participant_config), "calibration_root":str(args.calibration_root), "models":[{k:v for k,v in item.items() if k not in {"pipeline","probability_model","model_path"}} | {"model_path":str(item["model_path"])} for item in models], "feature_compatibility":compatibility, "event_sources":event_sources, "semantic_event_count":int(len(events)), "semantic_events_csv":str(semantic_events_path.resolve()), "semantic_events_csv_sha256":sha256(semantic_events_path), "semantic_events_csv_row_count":int(len(events)), "task_probability_definition":"median_windows(median_models(P(HIGH))) over complete ranks 1-3 windows only", "robot_data_used_for_fitting":False, "model_selection_or_refitting":False, "ratings_use":"Descriptive post-inference comparison only"}
    (args.output_dir / f"{args.participant}_transfer_manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n", encoding="utf-8")
    compact = statistics[["task_display", "target", "median_top3_consensus_probability", "descriptive_task_verdict", "unanimous_agreement_rate", "robot_rating", "robot_rating_class", "verdict_matches_robot_rating", "models_with_shift_warning"]].copy()
    compact.columns = ["Task", "Target", "Median P(HIGH)", "Verdict", "Agreement", "Robot rating", "Rating class", "Match", "Shift warnings"]
    model_rows = pd.DataFrame([{
        "Rank": f"R{item['model_rank']}", "Target": item["target"], "Modality": item["modality"].title(),
        "Classifier": item["classifier"], "Feature family": item["feature_family"],
        "Selected feature count": len(item["selected_features"]), "Image BA": item["mean_outer_cv_balanced_accuracy"],
        "Hyperparameters": item["authoritative_final_hyperparameters"],
        "Selected features": json.dumps(item["selected_features"]), "Frozen model path": str(item["model_path"]),
    } for item in models]).sort_values(["Target", "Rank"])
    (args.output_dir / f"{args.participant}_transfer_summary.md").write_text(
        f"# {args.participant} final frozen modality-agnostic Image-to-Robot transfer\n\n"
        "Predictions use frozen Image pipelines. Robot ratings are descriptive post-inference comparisons only.\n\n"
        "## Frozen Image models\n\n" + model_rows.to_markdown(index=False) + "\n\n"
        "## Definition-B predicted versus actual Robot ratings\n\n" + compact.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"Final modality-agnostic frozen transfer complete: {args.output_dir}"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
