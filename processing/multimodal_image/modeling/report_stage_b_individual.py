#!/usr/bin/env python3
"""Create thesis-ready figures from completed Stage B individual CSV results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ("valence", "arousal")
MODALITIES = ("eeg", "face", "multimodal")
CLASSIFIERS = ("knn", "svm", "gnb")
FEATURE_LABELS = {"all": "All", "5": "Top 5", "10": "Top 10", "20": "Top 20"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual-dir", type=Path, required=True)
    parser.add_argument("--general-dir", type=Path)
    return parser.parse_args()


def best_rows(summary: pd.DataFrame, parameters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one deterministic row per participant-target and all exact winners."""
    winners = summary.loc[summary.groupby(["participant", "target"])["balanced_accuracy_mean"].transform("max") == summary["balanced_accuracy_mean"]].copy()
    winners["feature_count_request"] = winners["feature_count_request"].astype(str)
    order = {value: index for index, value in enumerate(FEATURE_LABELS)}
    winners["_feature_order"] = winners.feature_count_request.map(order)
    chosen = winners.sort_values(["participant", "target", "modality", "classifier", "_feature_order"], kind="stable").groupby(["participant", "target"], as_index=False).first()
    tie_counts = winners.groupby(["participant", "target"]).size().rename("winner_tie_count")
    chosen = chosen.merge(tie_counts, on=["participant", "target"], validate="one_to_one")

    frequency = parameters.groupby(["participant", "target", "modality", "classifier", "feature_count_request", "best_parameters"]).size().rename("count").reset_index()
    frequency["feature_count_request"] = frequency.feature_count_request.astype(str)
    frequency = frequency.sort_values(["participant", "target", "modality", "classifier", "feature_count_request", "count", "best_parameters"], ascending=[True, True, True, True, True, False, True], kind="stable")
    most_frequent = frequency.groupby(["participant", "target", "modality", "classifier", "feature_count_request"], as_index=False).first()
    most_frequent["most_frequent_hyperparameters"] = most_frequent.apply(lambda row: f"{row.best_parameters} ({row['count']}/5 folds)", axis=1)
    chosen = chosen.merge(most_frequent.drop(columns=["best_parameters", "count"]), on=["participant", "target", "modality", "classifier", "feature_count_request"], how="left", validate="one_to_one")
    return chosen, winners


def winner_counts(winners: pd.DataFrame, column: str, categories: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        target_winners = winners[winners.target == target]
        for participant, group in target_winners.groupby("participant"):
            values = set(group[column])
            category = next(iter(values)) if len(values) == 1 else "Tie"
            rows.append({"target": target, "participant": participant, "category": category})
    counts = pd.DataFrame(rows).groupby(["target", "category"]).size().unstack(fill_value=0)
    return counts.reindex(columns=[*categories, "Tie"], fill_value=0)


def save_best_figure(best: pd.DataFrame, out: Path) -> None:
    participants = sorted(best.participant.unique())
    x = np.arange(len(participants)); width = .38
    fig, ax = plt.subplots(figsize=(16, 6))
    for offset, target, color in ((-width / 2, "valence", "#4477AA"), (width / 2, "arousal", "#CC6677")):
        values = best[best.target == target].set_index("participant").loc[participants, "balanced_accuracy_mean"]
        ax.bar(x + offset, values, width, label=target.capitalize(), color=color)
    ax.axhline(.5, color="black", linestyle="--", linewidth=1.2, label="Chance (BA = 0.50)")
    ax.set(xlim=(-.7, len(participants)-.3), ylim=(0, 1), xticks=x, xticklabels=participants, ylabel="Best nested-CV balanced accuracy", title="Stage B: best validated individual classification performance")
    ax.legend(ncol=3, frameon=False); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def save_distribution(best: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6)); values = [best.loc[best.target == target, "balanced_accuracy_mean"] for target in TARGETS]
    box = ax.boxplot(values, tick_labels=[target.capitalize() for target in TARGETS], patch_artist=True, widths=.5)
    for patch, color in zip(box["boxes"], ("#4477AA", "#CC6677")): patch.set_facecolor(color); patch.set_alpha(.45)
    for index, series in enumerate(values, 1):
        ax.scatter(np.full(len(series), index) + np.linspace(-.08, .08, len(series)), series, color="#333333", s=20, zorder=3)
        stats = series.agg(["mean", "median", "std", "min", "max"])
        ax.text(index, .04, f"mean {stats['mean']:.3f}\nmedian {stats['median']:.3f}\nSD {stats['std']:.3f}\nrange {stats['min']:.3f}–{stats['max']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(.5, color="black", linestyle="--", linewidth=1.2); ax.set(ylim=(0, 1), ylabel="Best nested-CV balanced accuracy", title="Distribution of participant-best performance")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def save_counts(counts: pd.DataFrame, title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(counts.columns)); width = .36
    for offset, target, color in ((-width/2, "valence", "#4477AA"), (width/2, "arousal", "#CC6677")):
        bars = ax.bar(x + offset, counts.loc[target], width, label=target.capitalize(), color=color)
        ax.bar_label(bars, padding=2, fontsize=9)
    ax.set(xticks=x, xticklabels=[value.upper() if value != "Tie" else value for value in counts.columns], ylim=(0, 28), ylabel=ylabel, title=title)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def save_thresholds(best: pd.DataFrame, out: Path) -> None:
    thresholds = (.50, .55, .60, .65); labels = ("> 0.50", "≥ 0.55", "≥ 0.60", "≥ 0.65")
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(thresholds)); width = .36
    for offset, target, color in ((-width/2, "valence", "#4477AA"), (width/2, "arousal", "#CC6677")):
        values = best[best.target == target].balanced_accuracy_mean
        counts = [int((values > .5).sum()) if threshold == .5 else int((values >= threshold).sum()) for threshold in thresholds]
        bars = ax.bar(x + offset, counts, width, label=target.capitalize(), color=color)
        ax.bar_label(bars, labels=[f"{count}/28" for count in counts], padding=2, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 30), ylabel="Participants (n = 28)", title="Participants meeting cumulative best-BA thresholds")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def format_stats(values: pd.Series) -> str:
    return f"mean {values.mean():.3f}; median {values.median():.3f}; SD {values.std():.3f}; range {values.min():.3f}–{values.max():.3f}"


def report(best: pd.DataFrame, modality: pd.DataFrame, classifier: pd.DataFrame, general_dir: Path | None, out: Path) -> None:
    cohort = ", ".join(sorted(best.participant.unique()))
    lines = ["# Stage B Individual Classification", "", "## Cohort", "", f"The completed individual analysis includes 28 participants: {cohort}.", "", "## Overall individualized performance", ""]
    for target in TARGETS:
        values = best.loc[best.target == target, "balanced_accuracy_mean"]
        lines.append(f"- {target.capitalize()}: {format_stats(values)}; ≥0.55: {(values >= .55).sum()}/28; ≥0.60: {(values >= .60).sum()}/28; ≥0.65: {(values >= .65).sum()}/28.")
    lines += ["", "## Best participants", ""]
    for target in TARGETS:
        lines.append(f"### {target.capitalize()}")
        lines.append("")
        for row in best[best.target == target].nlargest(5, "balanced_accuracy_mean").itertuples():
            lines.append(f"- {row.participant}: BA {row.balanced_accuracy_mean:.3f}; {row.modality}; {row.classifier}; {FEATURE_LABELS[str(row.feature_count)]}.")
        lines.append("")
    lines += ["## Modality patterns", "", "Exact ties are defined as configurations with exactly equal best mean BA. A participant is counted as `Tie` only when those exact winners span multiple modalities; otherwise the shared modality receives the count."]
    for target in TARGETS: lines.append(f"- {target.capitalize()}: " + "; ".join(f"{name} {int(modality.loc[target, name])}" for name in modality.columns) + ".")
    lines += ["", "These counts describe winning configurations and do not show that multimodal input is intrinsically better; multimodal models contain more predictors.", "", "## Classifier patterns", "", "The same exact-tie rule is used for classifier counts."]
    for target in TARGETS: lines.append(f"- {target.capitalize()}: " + "; ".join(f"{name} {int(classifier.loc[target, name])}" for name in classifier.columns) + ".")
    feature = best.feature_count.astype(str).map(FEATURE_LABELS).value_counts().reindex(FEATURE_LABELS.values(), fill_value=0)
    lines += ["", "## Feature-count patterns", ""] + [f"- {label}: {int(count)} participant-target winners." for label, count in feature.items()]
    lines += ["", "## Interpretation", "", "These are exploratory nested-CV results. Selecting the best of many configurations per participant is useful for identifying calibration potential, but it is not an independent estimate of a pre-specified deployed model."]
    if general_dir and (general_dir / "results_summary.csv").is_file():
        general = pd.read_csv(general_dir / "results_summary.csv")
        lines += ["", "The completed Stage B subject-independent LOSO analysis was near chance at its best configuration: " + "; ".join(f"{target} BA {general[general.target == target].balanced_accuracy_mean.max():.3f}" for target in TARGETS) + "."]
    lines += ["Participant-best individual values that are higher than these LOSO results are consistent with participant-specific calibration being more promising than subject-independent transfer. They are not final thesis conclusions, and no significance tests were performed here."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); out = args.individual_dir
    summary = pd.read_csv(out / "results_summary.csv"); parameters = pd.read_csv(out / "best_hyperparameters.csv")
    best, winners = best_rows(summary, parameters)
    best = best.rename(columns={"balanced_accuracy_mean": "best_balanced_accuracy", "accuracy_mean": "accuracy", "precision_mean": "precision", "recall_mean": "recall", "f1_mean": "f1", "balanced_accuracy_sd": "BA_SD", "balanced_accuracy_median": "BA_median", "balanced_accuracy_min": "BA_min", "balanced_accuracy_max": "BA_max", "feature_count_request": "feature_count"})
    best["feature_count"] = best.feature_count.astype(str)
    columns = ["participant", "target", "best_balanced_accuracy", "accuracy", "precision", "recall", "f1", "modality", "classifier", "feature_count", "BA_SD", "BA_median", "BA_min", "BA_max", "most_frequent_hyperparameters", "winner_tie_count"]
    best[columns].sort_values(["participant", "target"]).to_csv(out / "stage_b_best_per_participant.csv", index=False)
    plot_best = best.rename(columns={"best_balanced_accuracy": "balanced_accuracy_mean"})
    modality = winner_counts(winners, "modality", MODALITIES); classifier = winner_counts(winners, "classifier", CLASSIFIERS)
    save_best_figure(plot_best, out / "stage_b_best_ba_per_participant.png")
    save_distribution(plot_best, out / "stage_b_best_ba_distribution.png")
    save_counts(modality, "Winning modality among participant-best configurations", "Participants (n = 28)", out / "stage_b_winning_modality_counts.png")
    save_counts(classifier, "Winning classifier among participant-best configurations", "Participants (n = 28)", out / "stage_b_winning_classifier_counts.png")
    save_thresholds(plot_best, out / "stage_b_participants_above_threshold.png")
    report(plot_best, modality, classifier, args.general_dir, out / "STAGE_B_INDIVIDUAL_SUMMARY.md")


if __name__ == "__main__":
    main()
