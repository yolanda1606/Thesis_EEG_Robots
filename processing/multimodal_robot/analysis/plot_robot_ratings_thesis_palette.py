#!/usr/bin/env python3
"""Plot-only thesis-palette copies of existing Robot-ratings figures.

No statistics are calculated or written. Saved ratings CSVs supply summaries,
class balance, paired deltas, and demographic associations; canonical ratings
and YAML metadata are read only for raw task observations and VA coordinates.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.multimodal_robot.analysis.analyze_robot_ratings import (
    DISPLAY_TASKS, OUTCOMES, TASK_ORDER, TASK_TYPE_ORDER, load_canonical_ratings, participant_task_types,
)
from processing.visualization.thesis_style import apply_thesis_style, load_thesis_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-config", type=Path, default=ROOT / "processing/visualization/thesis_palette.yaml")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/robot_ratings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_ratings/figures_thesis_palette_test")
    parser.add_argument("--class-balance-only", action="store_true", help="Create only the three class-balance figures from the saved balance CSV.")
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)


def colored_boxplot(axis: plt.Axes, values: list[pd.Series], labels: list[str], main: str, light: str, edge: str) -> None:
    plot = axis.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False, widths=.55,
                        medianprops={"color": main, "linewidth": 1.4}, boxprops={"edgecolor": edge, "linewidth": .8},
                        whiskerprops={"color": edge, "linewidth": .8}, capprops={"color": edge, "linewidth": .8})
    for box in plot["boxes"]:
        box.set_facecolor(light)
        box.set_alpha(.55)


def jitter(axis: plt.Axes, values: list[pd.Series], main: str, seed: int = 20260910) -> None:
    rng = np.random.default_rng(seed)
    for position, series in enumerate(values, 1):
        axis.scatter(np.full(len(series), position) + rng.normal(0, .045, len(series)), series, s=15, color=main, alpha=.50, linewidths=0)


def distribution_figures(output: Path, ratings: pd.DataFrame, aggregates: pd.DataFrame, roles: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        semantic = roles["emotion"][target]; values = [ratings[(ratings.target == target) & (ratings.task == task)].robot_rating for task in TASK_ORDER]
        colored_boxplot(axis, values, [DISPLAY_TASKS[t] for t in TASK_ORDER], semantic["main"], semantic["light"], roles["neutral"]["edge"])
        jitter(axis, values, semantic["main"]); axis.set(title=target.title(), xlabel="Task", ylim=(.75, 7.25), yticks=range(1, 8)); axis.grid(axis="y")
    axes[0].set_ylabel("Self-reported rating (1–7)"); fig.suptitle("Robot ratings by task", y=1.02); save(fig, output / "task_rating_distributions_no_trajectories.png")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        semantic = roles["emotion"][target]; values = [aggregates[(aggregates.target == target) & (aggregates.task_type == kind)].rating for kind in TASK_TYPE_ORDER]
        colored_boxplot(axis, values, TASK_TYPE_ORDER, semantic["main"], semantic["light"], roles["neutral"]["edge"])
        jitter(axis, values, semantic["main"]); axis.set(title=target.title(), ylim=(.75, 7.25), yticks=range(1, 8)); axis.grid(axis="y")
    axes[0].set_ylabel("Participant-level task-type rating (1–7)"); fig.suptitle("Participant-level task-type aggregates", y=1.02); save(fig, output / "task_type_distributions_no_trajectories.png")


def paired_figure(output: Path, deltas: pd.DataFrame, roles: dict, effect: str, filename: str, title: str, left_role: str, right_role: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), sharey=True)
    for axis, target in zip(axes, OUTCOMES):
        data = deltas[(deltas.target == target) & deltas[f"{effect}_included"]]
        left, right = f"{effect}_left_rating", f"{effect}_right_rating"; left_label = data[f"{effect}_left_condition"].iloc[0]; right_label = data[f"{effect}_right_condition"].iloc[0]
        for row in data.itertuples(index=False): axis.plot([1, 2], [getattr(row, left), getattr(row, right)], color=roles["neutral"]["connector"], alpha=.5, linewidth=.75, zorder=1)
        axis.scatter(np.ones(len(data)), data[left], color=roles["condition"][left_role]["main"], s=26, marker="o", label=left_label, zorder=2)
        axis.scatter(np.full(len(data), 2), data[right], color=roles["condition"][right_role]["main"], s=32, marker="s", label=right_label, zorder=2)
        axis.set(xlim=(.7, 2.3), xticks=[1, 2], xticklabels=[left_label, right_label], title=f"{target.title()} (n={len(data)})", ylim=(.75, 7.25), yticks=range(1, 8)); axis.grid(axis="y")
    axes[0].set_ylabel("Self-reported rating (1–7)"); axes[1].legend(loc="best"); fig.suptitle(title, y=1.02); save(fig, output / filename)


def va_figure(output: Path, ratings: pd.DataFrame, roles: dict) -> None:
    fig, axis = plt.subplots(figsize=(7, 6)); markers = ["o", "s", "^", "D", "P", "X"]
    for task, marker in zip(TASK_ORDER, markers):
        data = ratings[ratings.task == task].pivot(index="participant", columns="target", values="robot_rating")
        color = roles["task_identity"][task]
        axis.scatter(data.valence, data.arousal, color=color, marker=marker, alpha=.25, s=24)
        centroid = data[["valence", "arousal"]].mean(); axis.scatter(centroid.valence, centroid.arousal, color=color, marker=marker, edgecolor=roles["neutral"]["edge"], linewidth=.8, s=115, label=DISPLAY_TASKS[task])
    axis.set(xlim=(.75, 7.25), ylim=(.75, 7.25), xticks=range(1, 8), yticks=range(1, 8), xlabel="Valence rating (1–7)", ylabel="Arousal rating (1–7)", title="Robot valence–arousal space")
    axis.grid(); axis.legend(title="Task", loc="best"); save(fig, output / "valence_arousal_space.png")


def stack(axis: plt.Axes, data: pd.DataFrame, labels: list[str], roles: dict) -> None:
    data = data.set_index("group").reindex(labels); pos = np.arange(len(data)); low, high = data.proportion_low.to_numpy(), data.proportion_high.to_numpy()
    axis.bar(pos, low, color=roles["class"]["low"]["main"], edgecolor=roles["neutral"]["edge"], linewidth=.45, label="LOW (<4)"); axis.bar(pos, high, bottom=low, color=roles["class"]["high"]["main"], edgecolor=roles["neutral"]["edge"], linewidth=.45, label="HIGH (≥4)")
    for i, (a, b) in enumerate(zip(low, high)):
        axis.text(i, a / 2, f"{a:.0%}", ha="center", va="center", fontsize=8); axis.text(i, a + b / 2, f"{b:.0%}", ha="center", va="center", fontsize=8)
    axis.set(xticks=pos, xticklabels=labels, ylim=(0, 1)); axis.grid(axis="y")


def class_figures(output: Path, balance: pd.DataFrame, roles: dict) -> None:
    overall = balance[balance.scope == "overall"].copy(); fig, axis = plt.subplots(figsize=(6, 4.5)); stack(axis, overall.assign(group=overall.target.str.title()), ["Valence", "Arousal"], roles)
    axis.set(ylabel="Proportion of all participant-task ratings", title="Overall Robot-rating validation-label balance"); axis.legend(loc="upper right"); save(fig, output / "class_balance_overall.png")
    for scope, groups, filename, title in [("task", [DISPLAY_TASKS[t] for t in TASK_ORDER], "class_balance_by_task.png", "Robot-rating validation-label balance by task"), ("task_type", TASK_TYPE_ORDER, "class_balance_by_task_type.png", "Robot-rating validation-label balance by task type")]:
        fig, axes = plt.subplots(1, 2, figsize=(11 if scope == "task" else 8.5, 4.7), sharey=True)
        for axis, target in zip(axes, OUTCOMES): stack(axis, balance[(balance.scope == scope) & (balance.target == target)], groups, roles); axis.set(title=target.title())
        axes[0].set_ylabel("Proportion"); axes[1].legend(loc="upper right"); fig.suptitle(title, y=1.02); save(fig, output / filename)


def demographic_figures(output: Path, participants: pd.DataFrame, associations: pd.DataFrame, roles: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for axis, target in zip(axes, OUTCOMES):
        measure = f"fault_{target}_delta"; data = participants.dropna(subset=["age", measure]); semantic = roles["emotion"][target]
        axis.scatter(data.age, data[measure], color=semantic["main"], alpha=.65); slope, intercept, *_ = stats.linregress(data.age, data[measure]); x = np.array([data.age.min(), data.age.max()]); axis.plot(x, intercept + slope * x, color=roles["neutral"]["reference"], linewidth=1)
        result = associations[(associations.demographic == "age") & (associations.measure == measure)].iloc[0]; axis.axhline(0, color=roles["neutral"]["reference"], linestyle="--", linewidth=.8)
        axis.set(xlabel="Age (years)", ylabel="FAULTY_FAST − CONTROL_FAST", title=f"{target.title()}: ρ={result.effect_size:.2f}, FDR p={result.p_value_fdr_bh:.3f}"); axis.grid()
    fig.suptitle("Exploratory age association with fault-response deltas", y=1.02); save(fig, output / "age_vs_fault_deltas.png")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4)); rng = np.random.default_rng(20260910)
    for axis, target in zip(axes, OUTCOMES):
        measure = f"fault_{target}_delta"; semantic = roles["emotion"][target]; values = [participants.loc[participants.prior_robot_experience == label, measure].dropna() for label in ["No", "Yes"]]
        colored_boxplot(axis, values, ["No", "Yes"], semantic["main"], semantic["light"], roles["neutral"]["edge"]); jitter(axis, values, semantic["main"], seed=20260911); result = associations[(associations.demographic == "prior_robot_experience") & (associations.measure == measure)].iloc[0]
        axis.axhline(0, color=roles["neutral"]["reference"], linestyle="--", linewidth=.8); axis.set(xlabel="Prior robot experience", ylabel="FAULTY_FAST − CONTROL_FAST", title=f"{target.title()}: FDR p={result.p_value_fdr_bh:.3f}"); axis.grid(axis="y")
    fig.suptitle("Exploratory prior-experience comparison of fault-response deltas", y=1.02); save(fig, output / "experience_vs_fault_deltas.png")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args(); analysis = args.analysis_dir.resolve(); output = args.output_dir.resolve(); style = load_thesis_style(args.style_config.resolve()); apply_thesis_style(style)
    needed = ["class_balance_summary.csv"] if args.class_balance_only else ["participant_condition_deltas.csv", "class_balance_summary.csv", "participant_demographic_rating_summary.csv", "demographic_rating_associations.csv"]
    missing = [name for name in needed if not (analysis / name).is_file()]
    if missing: raise FileNotFoundError(f"Missing saved ratings-only inputs: {missing}")
    if output.exists(): raise FileExistsError(f"Refusing to overwrite palette-test directory: {output}")
    output.mkdir(parents=True)
    original_figure_dir = analysis / "figures"
    original_hashes = {path.name: sha256(path) for path in original_figure_dir.glob("*.png")}
    balance = pd.read_csv(analysis / "class_balance_summary.csv")
    if args.class_balance_only:
        class_figures(output, balance, style["roles"])
    else:
        ratings, _ = load_canonical_ratings(); aggregates = participant_task_types(ratings)
        deltas = pd.read_csv(analysis / "participant_condition_deltas.csv")
        participants = pd.read_csv(analysis / "participant_demographic_rating_summary.csv"); associations = pd.read_csv(analysis / "demographic_rating_associations.csv")
        distribution_figures(output, ratings, aggregates, style["roles"]); paired_figure(output, deltas, style["roles"], "fault_effect", "fault_effect_paired.png", "Fault effect: FAULTY_FAST versus CONTROL_FAST", "faulty", "control_fast"); paired_figure(output, deltas, style["roles"], "speed_effect", "speed_effect_paired.png", "Speed effect: CONTROL_FAST versus CONTROL_SLOW", "control_fast", "control_slow")
        va_figure(output, ratings, style["roles"]); class_figures(output, balance, style["roles"]); demographic_figures(output, participants, associations, style["roles"])
    manifest = {"purpose": "Plot-only thesis-palette test for Robot ratings", "mode": "class_balance_only" if args.class_balance_only else "all_robot_ratings_figures", "style_config": style["source"], "reads_predictions_or_models": False, "statistics_recomputed": False, "original_robot_ratings_figure_dir": str(original_figure_dir), "original_robot_ratings_png_sha256_before": original_hashes, "new_figures": sorted(path.name for path in output.glob("*.png"))}
    (output / "palette_test_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": manifest["new_figures"]}, indent=2)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error: print(f"ERROR: {error}"); raise SystemExit(2)
