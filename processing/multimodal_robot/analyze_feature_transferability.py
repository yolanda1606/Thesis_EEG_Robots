#!/usr/bin/env python3
"""Feature-level, read-only transfer diagnostics for frozen P19 Image-to-Robot models.

This script loads the already frozen P19 Image calibration pipelines and existing
robot windows.  It does not fit, adapt, scale, select, predict, or overwrite
the primary inference outputs.  It writes only new diagnostics below the
existing domain-shift audit directory when explicitly invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.multimodal_robot.audit_personalized_transfer import (
    AUDIT_DIR,
    DERIVED_ROOT,
    INFERENCE_DIR,
    ROBOT_FEATURE_PATH,
    TARGETS,
    distance_audit,
    load_frozen_inputs,
    percentile,
    transform_selected,
)
from processing.multimodal_robot.run_personalized_inference import EXCLUDED_TASKS, PARTICIPANT, TASK_LABELS, TASK_ORDER


OUTPUTS = (
    "P19_feature_transferability.csv",
    "P19_window_domain_reliability.csv",
    "P19_distance_driver_summary.csv",
    "P19_feature_transferability_summary.md",
    "P19_feature_transferability_overview.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default=PARTICIPANT, choices=(PARTICIPANT,))
    parser.add_argument("--derived-root", type=Path, default=DERIVED_ROOT)
    parser.add_argument("--robot-features", type=Path, default=ROBOT_FEATURE_PATH)
    parser.add_argument("--inference-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's five named diagnostics.")
    parser.add_argument("--dry-run", action="store_true", help="Validate frozen inputs without calculating or writing diagnostics.")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median_iqr(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def interval_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether two robust IQR intervals intersect (including at an endpoint)."""
    return max(np.percentile(left, 25), np.percentile(right, 25)) <= min(np.percentile(left, 75), np.percentile(right, 75))


def overlap_label(robot: np.ndarray, low: np.ndarray, high: np.ndarray) -> str:
    matches_low, matches_high = interval_overlap(robot, low), interval_overlap(robot, high)
    if matches_low and matches_high:
        return "both"
    if matches_low:
        return "Image_LOW"
    if matches_high:
        return "Image_HIGH"
    return "neither"


def transfer_category(row: dict[str, float | str]) -> tuple[str, int]:
    """Exploratory multi-indicator classification, not a scientific threshold."""
    indicators = [
        float(row["robot_z_median_abs"]) > 2.0,
        float(row["robot_z_p95_abs"]) > 3.0,
        float(row["robot_z_pct_abs_gt_3"]) >= 25.0,
        float(row["robot_z_pct_abs_gt_5"]) >= 10.0,
        float(row["robot_pct_outside_image_min_max"]) >= 25.0,
        row["iqr_overlap"] == "neither",
    ]
    count = int(sum(indicators))
    if count >= 4:
        return "SEVERELY_SHIFTED", count
    if count >= 2:
        return "MODERATELY_SHIFTED", count
    return "TRANSFER_COMPATIBLE", count


def transformed_spaces(state: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    spaces, references = {}, {}
    for target in TARGETS:
        image = transform_selected(state["models"][target], state["image"][target]["features"])
        robot = transform_selected(state["models"][target], state["robot"][state["image"][target]["features"].columns])
        reference, _ = distance_audit(image, robot, k=11)
        spaces[target] = (image, robot)
        references[target] = reference
    return spaces, references


def feature_transferability(state: dict, driver: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    robot = state["robot"]
    for target in TARGETS:
        image = state["image"][target]["features"]
        labels = state["image"][target]["labels"]
        scaler = state["models"][target].named_steps["scale"]
        selected = state["image"][target]["selected_features"]
        scaled_robot = pd.DataFrame(scaler.transform(robot[image.columns]), columns=image.columns, index=robot.index)
        for task in TASK_ORDER:
            if task in EXCLUDED_TASKS:
                continue
            part = robot[robot["task"].astype(str) == task]
            for feature in selected:
                raw, low, high = (part[feature].to_numpy(float), image.loc[labels == 0, feature].to_numpy(float), image.loc[labels == 1, feature].to_numpy(float))
                overall = image[feature].to_numpy(float)
                z = scaled_robot.loc[part.index, feature].to_numpy(float)
                imed, iiqr = median_iqr(overall)
                lmed, liqr = median_iqr(low)
                hmed, hiqr = median_iqr(high)
                rmed, riqr = median_iqr(raw)
                row = {
                    "participant": PARTICIPANT, "target": target, "task": task, "task_label": TASK_LABELS[task], "feature": feature,
                    "image_low_median": lmed, "image_low_iqr": liqr, "image_high_median": hmed, "image_high_iqr": hiqr,
                    "image_overall_median": imed, "image_overall_iqr": iiqr, "image_min": float(np.min(overall)), "image_max": float(np.max(overall)),
                    "robot_median": rmed, "robot_iqr": riqr, "robot_min": float(np.min(raw)), "robot_max": float(np.max(raw)),
                    "robot_z_median_abs": float(np.median(np.abs(z))), "robot_z_p95_abs": percentile(np.abs(z), 95), "robot_z_max_abs": float(np.max(np.abs(z))),
                    "robot_z_pct_abs_gt_2": float(np.mean(np.abs(z) > 2) * 100), "robot_z_pct_abs_gt_3": float(np.mean(np.abs(z) > 3) * 100),
                    "robot_z_pct_abs_gt_5": float(np.mean(np.abs(z) > 5) * 100),
                    "robot_pct_outside_image_min_max": float(np.mean((raw < np.min(overall)) | (raw > np.max(overall))) * 100),
                    "iqr_overlap": overlap_label(raw, low, high),
                }
                driven = driver[(driver["target"] == target) & (driver["task"] == task) & (driver["feature"] == feature)]
                if len(driven):
                    row.update({"distance_median_contribution": float(driven["median_contribution"].iloc[0]),
                                "distance_p95_contribution": float(driven["p95_contribution"].iloc[0]),
                                "distance_largest_contributor_pct": float(driven["largest_contributor_pct"].iloc[0])})
                else:
                    row.update({"distance_median_contribution": 0.0, "distance_p95_contribution": 0.0, "distance_largest_contributor_pct": 0.0})
                row["transferability_category"], row["shift_indicator_count"] = transfer_category(row)
                rows.append(row)
    return pd.DataFrame(rows)


def arousal_driver_rows(state: dict, image: np.ndarray, robot: np.ndarray) -> pd.DataFrame:
    """Nearest-neighbour and 11-neighbour squared-distance decompositions."""
    selected = state["image"]["arousal"]["selected_features"]
    table = state["robot"]
    full = pairwise_distances(robot, image, metric="euclidean")
    nearest = np.argmin(full, axis=1)
    neighbor_11 = np.argsort(full, axis=1)[:, :11]
    nearest_sq = (robot - image[nearest]) ** 2
    nearest_fraction = nearest_sq / np.maximum(nearest_sq.sum(axis=1, keepdims=True), np.finfo(float).eps)
    neighbor_sq = (robot[:, None, :] - image[neighbor_11]) ** 2
    neighbor_fraction = neighbor_sq / np.maximum(neighbor_sq.sum(axis=2, keepdims=True), np.finfo(float).eps)
    rows: list[dict] = []
    for task in TASK_ORDER:
        if task in EXCLUDED_TASKS:
            continue
        positions = np.flatnonzero(table["task"].astype(str).to_numpy() == task)
        for index, feature in enumerate(selected):
            values, values11 = nearest_fraction[positions, index], neighbor_fraction[positions, :, index].mean(axis=1)
            rows.append({"participant": PARTICIPANT, "target": "arousal", "task": task, "task_label": TASK_LABELS[task], "feature": feature,
                         "geometry": "nearest_image_neighbor", "median_contribution": float(np.median(values)), "p95_contribution": percentile(values, 95),
                         "largest_contributor_pct": float(np.mean(np.argmax(nearest_fraction[positions], axis=1) == index) * 100)})
            rows.append({"participant": PARTICIPANT, "target": "arousal", "task": task, "task_label": TASK_LABELS[task], "feature": feature,
                         "geometry": "mean_of_11_image_neighbors", "median_contribution": float(np.median(values11)), "p95_contribution": percentile(values11, 95),
                         "largest_contributor_pct": float(np.mean(np.argmax(neighbor_fraction[positions].mean(axis=1), axis=1) == index) * 100)})
    return pd.DataFrame(rows)


def valence_driver_rows(state: dict, selected_robot: np.ndarray) -> pd.DataFrame:
    pipeline = state["models"]["valence"]
    classifier = pipeline.named_steps["classifier"]
    features = state["image"]["valence"]["selected_features"]
    logpdf = -0.5 * (np.log(2 * np.pi * classifier.var_[None, :, :]) + ((selected_robot[:, None, :] - classifier.theta_[None, :, :]) ** 2) / classifier.var_[None, :, :])
    difference = logpdf[:, 1, :] - logpdf[:, 0, :]
    table, rows = state["robot"], []
    for task in TASK_ORDER:
        if task in EXCLUDED_TASKS:
            continue
        positions = np.flatnonzero(table["task"].astype(str).to_numpy() == task)
        for index, feature in enumerate(features):
            values = difference[positions, index]
            rows.append({"participant": PARTICIPANT, "target": "valence", "task": task, "task_label": TASK_LABELS[task], "feature": feature,
                         "geometry": "GaussianNB_HIGH_minus_LOW_log_likelihood", "median_abs_contribution": float(np.median(np.abs(values))),
                         "median_signed_contribution": float(np.median(values)), "direction": "toward_HIGH" if np.median(values) > 0 else "toward_LOW",
                         "dominant_contributor_pct": float(np.mean(np.argmax(np.abs(difference[positions]), axis=1) == index) * 100)})
    return pd.DataFrame(rows)


def reliability_rows(state: dict, spaces: dict, references: dict) -> pd.DataFrame:
    rows: list[dict] = []
    robot = state["robot"]
    for target in TARGETS:
        image, robot_space = spaces[target]
        _, distance = distance_audit(image, robot_space, k=11)
        reference = references[target]
        p95, maximum = percentile(reference["nearest"], 95), float(np.max(reference["nearest"]))
        scores = state["predictions"].set_index("window_id")[f"{target}_high_score"]
        for index, window in enumerate(robot.itertuples(index=False)):
            nearest = float(distance["nearest"][index])
            label = "beyond_image_reference" if nearest > maximum else ("above_image_95pct" if nearest > p95 else "within_reference")
            rows.append({"participant": PARTICIPANT, "target": target, "task": str(window.task), "window_id": window.window_id,
                         "window_index": int(window.window_index), "window_start_s": float(window.window_start_s), "high_class_score": float(scores[window.window_id]),
                         "nearest_image_distance": nearest, "image_reference_distance_percentile": float(np.mean(reference["nearest"] <= nearest) * 100),
                         "image_reference_nearest_distance_p95": p95, "image_reference_nearest_distance_max": maximum,
                         "nearest_distance_exceeds_image_p95": bool(nearest > p95), "nearest_distance_exceeds_image_max": bool(nearest > maximum),
                         "mean_11_image_neighbor_distance": float(distance["mean_k"][index]), "feature_domain_reliability": label})
    return pd.DataFrame(rows)


def assess_scale_pattern(frame: pd.DataFrame) -> str:
    """Conservative descriptive assessment; no corrective transform is proposed."""
    severe = frame[frame["transferability_category"] == "SEVERELY_SHIFTED"]
    if severe.empty:
        return "unclear: no repeated severe-shift pattern was found."
    signs = severe.assign(sign=np.sign(severe["robot_median"] - severe["image_overall_median"])).groupby(["target", "feature"])["sign"].agg(lambda v: abs(float(np.mean(v))))
    coherent = float(np.mean(signs >= 0.8) * 100)
    varied = severe.groupby("task")["robot_z_median_abs"].median().max() / max(severe.groupby("task")["robot_z_median_abs"].median().min(), np.finfo(float).eps)
    return (f"mixture/unclear: {coherent:.1f}% of severely shifted target-feature pairs have a consistent median-offset direction across tasks, "
            f"but task-level median |z| varies by {varied:.1f}x. This does not support a single global additive or multiplicative correction.")


def plot_overview(frame: pd.DataFrame, output_dir: Path) -> Path:
    tasks = [task for task in TASK_ORDER if task not in EXCLUDED_TASKS]
    labels = list(dict.fromkeys(frame.apply(lambda r: f"{r.target}:{r.feature}", axis=1)))
    values = np.array([[frame[(frame["task"] == task) & (frame["target"] == label.split(":", 1)[0]) & (frame["feature"] == label.split(":", 1)[1])]["robot_z_median_abs"].iloc[0] for task in tasks] for label in labels])
    figure, axis = plt.subplots(figsize=(10, max(4, 0.42 * len(labels))))
    heatmap = axis.imshow(values, aspect="auto", cmap="magma")
    axis.set(xticks=range(len(tasks)), xticklabels=[TASK_LABELS[t] for t in tasks], yticks=range(len(labels)), yticklabels=labels,
             title="P19 selected-feature median |z| under frozen Image scalers")
    axis.tick_params(axis="x", rotation=35)
    figure.colorbar(heatmap, ax=axis, label="Median |z|")
    figure.tight_layout()
    path = output_dir / "P19_feature_transferability_overview.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_summary(frame: pd.DataFrame, arousal: pd.DataFrame, valence: pd.DataFrame, reliability: pd.DataFrame, output_dir: Path) -> None:
    nearest = arousal[arousal["geometry"] == "nearest_image_neighbor"]
    lines = ["# P19 Feature Transferability and Feature-Domain Reliability", "",
             "This is a read-only diagnostic of frozen P19 Image-calibrated models. It does not alter probabilities, fit a model, use robot ratings, establish confidence, or establish emotion validity.", "",
             "## Methods", "",
             "- Distribution overlap means intersection of robot and Image-class IQR intervals; it is a robust descriptive convention, not a test of equivalence.",
             "- `TRANSFER_COMPATIBLE`, `MODERATELY_SHIFTED`, and `SEVERELY_SHIFTED` are exploratory multi-indicator labels. They use several z/range/overlap indicators and do not justify deleting a feature.",
             "- Window labels describe feature-domain proximity to leave-self-out Image neighbours only: `within_reference`, `above_image_95pct`, and `beyond_image_reference`.", "",
             "## Per-task feature transferability", ""]
    for target in TARGETS:
        lines.append(f"### {target.capitalize()}")
        for task in TASK_ORDER:
            part = frame[(frame.target == target) & (frame.task == task)]
            if part.empty:
                continue
            categories = ", ".join(f"{r.feature}: {r.transferability_category}" for r in part.itertuples(index=False))
            lines.append(f"- {TASK_LABELS[task]} — {categories}.")
        lines.append("")
    lines.extend(["## Distance and likelihood drivers", ""])
    for task in TASK_ORDER:
        ar = nearest[nearest.task == task].sort_values("median_contribution", ascending=False)
        va = valence[valence.task == task].sort_values("median_abs_contribution", ascending=False)
        if ar.empty:
            continue
        lines.append(f"- {TASK_LABELS[task]}: arousal nearest-neighbour median distance is led by `{ar.iloc[0].feature}` ({ar.iloc[0].median_contribution:.1%}; largest in {ar.iloc[0].largest_contributor_pct:.1f}% of windows). Valence likelihood is led by `{va.iloc[0].feature}` (median |HIGH−LOW| log contribution {va.iloc[0].median_abs_contribution:.3g}; dominant in {va.iloc[0].dominant_contributor_pct:.1f}% of windows).")
    counts = reliability.groupby(["target", "feature_domain_reliability"]).size().unstack(fill_value=0)
    count_lines = ["| Target | within_reference | above_image_95pct | beyond_image_reference |", "|---|---:|---:|---:|"]
    for target in TARGETS:
        current = counts.loc[target] if target in counts.index else pd.Series(dtype=int)
        count_lines.append(f"| {target} | {int(current.get('within_reference', 0))} | {int(current.get('above_image_95pct', 0))} | {int(current.get('beyond_image_reference', 0))} |")
    lines.extend(["", "## Window feature-domain reliability", "", *count_lines, "", "## Scaling-mismatch investigation", "", assess_scale_pattern(frame),
                  "The extremely large Image-scaled z-scores require preprocessing/provenance investigation, but this analysis cannot distinguish a unit mismatch from context effects conclusively and applies no correction.",
                  "", "## Interpretation boundary", "", "Shape Sorter Interaction may be closer for a subset of selected features, but arousal distance diagnostics still place most of its windows beyond the Image reference maximum. No feature subset, rescaling, or reduced model is recommended or implemented here."])
    (output_dir / "P19_feature_transferability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = load_frozen_inputs(args)
    prediction = args.inference_dir / "P19_robot_window_predictions.csv"
    prediction_hash = sha256(prediction)
    if args.dry_run:
        print(f"Dry-run validation passed: {len(state['robot'])} robot windows, frozen P19 models, and Image calibration tables are compatible.")
        print("No diagnostics or figures were written.")
        return 0
    existing = [args.output_dir / name for name in OUTPUTS if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite transferability diagnostics without --overwrite: " + ", ".join(str(path) for path in existing))
    spaces, references = transformed_spaces(state)
    arousal = arousal_driver_rows(state, *spaces["arousal"])
    valence = valence_driver_rows(state, spaces["valence"][1])
    driver = arousal[arousal["geometry"] == "nearest_image_neighbor"].rename(columns={"geometry": "driver_geometry"})
    transfer = feature_transferability(state, driver)
    reliability = reliability_rows(state, spaces, references)
    for table in (transfer, arousal, valence, reliability):
        numeric = table.select_dtypes(include=[np.number]).to_numpy(float)
        if not np.isfinite(numeric).all():
            raise RuntimeError("Diagnostic calculation produced NaN or infinity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transfer.to_csv(args.output_dir / "P19_feature_transferability.csv", index=False)
    reliability.to_csv(args.output_dir / "P19_window_domain_reliability.csv", index=False)
    pd.concat([arousal, valence], ignore_index=True, sort=False).to_csv(args.output_dir / "P19_distance_driver_summary.csv", index=False)
    plot_overview(transfer, args.output_dir)
    write_summary(transfer, arousal, valence, reliability, args.output_dir)
    if sha256(prediction) != prediction_hash:
        raise RuntimeError("Frozen prediction CSV changed during read-only diagnostics")
    print(f"P19 transferability diagnostics written under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
