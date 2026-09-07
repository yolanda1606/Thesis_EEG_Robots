#!/usr/bin/env python3
"""Create a read-only participant-best comparison of old and focused EEG searches."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OLD_PATH = PROJECT_ROOT / "outputs/image_classification/no_ica/results_summary.csv"
NEW_PATH = PROJECT_ROOT / "outputs/image_classification/focused_personalized_binary_v1/configuration_summary_all_modalities.csv"
OUTPUT = PROJECT_ROOT / "outputs/image_classification/focused_vs_old_eeg_comparison"
PARTICIPANTS = [f"P{number:02d}" for number in range(1, 47)]
TARGETS = ("valence", "arousal")
MODALITIES = ("eeg", "face", "multimodal")


def load_eeg(path: Path, score_column: str) -> pd.DataFrame:
    """Read one search and retain its EEG configuration-level observations."""
    frame = pd.read_csv(path)
    required = {"participant", "target", "modality", score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return frame.loc[frame.modality.eq("eeg"), ["participant", "target", score_column]].copy()


def participant_best(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Reduce each search to its best configuration per participant-target."""
    old_best = old.groupby(["participant", "target"], as_index=False)["balanced_accuracy_mean"].max().rename(columns={"balanced_accuracy_mean": "old_best_ba"})
    new_best = new.groupby(["participant", "target"], as_index=False)["mean_outer_cv_balanced_accuracy"].max().rename(columns={"mean_outer_cv_balanced_accuracy": "new_best_ba"})
    output = old_best.merge(new_best, on=["participant", "target"], how="outer", validate="one_to_one")
    if len(output) != 92 or output[["old_best_ba", "new_best_ba"]].isna().any().any():
        raise ValueError("Expected complete 46-participant by two-target EEG comparison")
    output["delta_ba"] = output.new_best_ba - output.old_best_ba
    output["improved"] = output.delta_ba > 0
    for prefix in ("old", "new"):
        output[f"{prefix}_ge_060"] = output[f"{prefix}_best_ba"] >= .60
        output[f"{prefix}_ge_065"] = output[f"{prefix}_best_ba"] >= .65
    return output.assign(participant=pd.Categorical(output.participant, PARTICIPANTS, ordered=True)).sort_values(["target", "participant"]).reset_index(drop=True)


def summary_rows(participant: pd.DataFrame, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Create participant-level and explicitly labeled configuration-level context rows."""
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        group = participant.loc[participant.target.eq(target)]
        rows.append({
            "summary_type": "participant_level_best", "target": target,
            "old_mean_participant_best_ba": group.old_best_ba.mean(), "new_mean_participant_best_ba": group.new_best_ba.mean(),
            "mean_difference_new_minus_old": group.delta_ba.mean(), "old_median_participant_best_ba": group.old_best_ba.median(),
            "new_median_participant_best_ba": group.new_best_ba.median(), "median_difference_new_minus_old": group.delta_ba.median(),
            "improved_count": int((group.delta_ba > 0).sum()), "worsened_count": int((group.delta_ba < 0).sum()), "tied_count": int((group.delta_ba == 0).sum()),
            "old_ge_060_count": int(group.old_ge_060.sum()), "new_ge_060_count": int(group.new_ge_060.sum()),
            "old_ge_065_count": int(group.old_ge_065.sum()), "new_ge_065_count": int(group.new_ge_065.sum()),
            "old_maximum_participant_best_ba": group.old_best_ba.max(), "new_maximum_participant_best_ba": group.new_best_ba.max(),
        })
        old_config = old.loc[old.target.eq(target), "balanced_accuracy_mean"]
        new_config = new.loc[new.target.eq(target), "mean_outer_cv_balanced_accuracy"]
        rows.append({
            "summary_type": "configuration_level_context", "target": target,
            "old_configuration_rows": len(old_config), "new_configuration_rows": len(new_config),
            "old_configuration_mean_ba": old_config.mean(), "new_configuration_mean_ba": new_config.mean(),
            "old_configuration_max_ba": old_config.max(), "new_configuration_max_ba": new_config.max(),
            "old_configuration_ge_060_count": int((old_config >= .60).sum()), "new_configuration_ge_060_count": int((new_config >= .60).sum()),
            "old_configuration_ge_065_count": int((old_config >= .65).sum()), "new_configuration_ge_065_count": int((new_config >= .65).sum()),
        })
    return pd.DataFrame(rows)


def save_line_plot(frame: pd.DataFrame, target: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 5.2))
    axis.plot(frame.participant.astype(str), frame.old_best_ba, marker="o", linewidth=1.6, label="Old broad search")
    axis.plot(frame.participant.astype(str), frame.new_best_ba, marker="o", linewidth=1.6, label="New focused search")
    axis.axhline(.50, color="gray", linewidth=1, linestyle="--", label="BA = 0.50")
    axis.axhline(.60, color="black", linewidth=1, linestyle=":", label="BA = 0.60")
    axis.set(title=f"{target.title()}: participant-best EEG balanced accuracy", xlabel="Participant", ylabel="Balanced accuracy", ylim=(.35, .75))
    axis.tick_params(axis="x", rotation=90); axis.legend(ncol=4, fontsize=9); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)


def save_delta_plot(frame: pd.DataFrame, target: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 4.8))
    colors = ["#2a7f62" if value > 0 else "#b24a4a" if value < 0 else "#808080" for value in frame.delta_ba]
    axis.bar(frame.participant.astype(str), frame.delta_ba, color=colors)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(title=f"{target.title()}: focused minus old participant-best EEG BA", xlabel="Participant", ylabel="Δ balanced accuracy", ylim=(-.14, .14))
    axis.tick_params(axis="x", rotation=90); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)


def save_scatter_plot(frame: pd.DataFrame, target: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 5.8))
    axis.scatter(frame.old_best_ba, frame.new_best_ba, color="#356a9a", alpha=.8)
    axis.plot([.35, .75], [.35, .75], color="black", linestyle="--", linewidth=1, label="y = x")
    axis.set(title=f"{target.title()}: old vs focused participant-best EEG BA", xlabel="Old broad search", ylabel="New focused search", xlim=(.35, .75), ylim=(.35, .75))
    axis.legend(); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)


def write_report(path: Path, summary: pd.DataFrame) -> None:
    participant = summary.loc[summary.summary_type.eq("participant_level_best")].set_index("target")
    config = summary.loc[summary.summary_type.eq("configuration_level_context")].set_index("target")
    lines = ["# Old broad vs focused personalized EEG search", "", "This is a read-only comparison. Participant-level results select one best outer-CV mean BA per participant × target from each search. Configuration-level rows are context only because the searches differ in size and candidate space.", ""]
    for target in TARGETS:
        value = participant.loc[target]
        lines += [f"## {target.title()}", "", f"Participant-best mean BA: old {value.old_mean_participant_best_ba:.3f}; focused {value.new_mean_participant_best_ba:.3f}; Δ {value.mean_difference_new_minus_old:+.3f}.", f"Participant-best median BA: old {value.old_median_participant_best_ba:.3f}; focused {value.new_median_participant_best_ba:.3f}; Δ {value.median_difference_new_minus_old:+.3f}.", f"Improved/worsened/tied: {int(value.improved_count)}/{int(value.worsened_count)}/{int(value.tied_count)}. BA ≥0.60: {int(value.old_ge_060_count)} → {int(value.new_ge_060_count)}; BA ≥0.65: {int(value.old_ge_065_count)} → {int(value.new_ge_065_count)}.", f"Configuration-level context: old {int(config.loc[target, 'old_configuration_rows'])} rows, mean {config.loc[target, 'old_configuration_mean_ba']:.3f}, max {config.loc[target, 'old_configuration_max_ba']:.3f}; focused {int(config.loc[target, 'new_configuration_rows'])} rows, mean {config.loc[target, 'new_configuration_mean_ba']:.3f}, max {config.loc[target, 'new_configuration_max_ba']:.3f}.", ""]
    valence, arousal = participant.loc["valence"], participant.loc["arousal"]
    lines += ["## Verdict", "", f"Yes. The focused search improved personalized EEG classification across the cohort: participant-best mean BA increased by {valence.mean_difference_new_minus_old:+.3f} for valence and {arousal.mean_difference_new_minus_old:+.3f} for arousal; median BA increased for both targets; and the number reaching BA ≥0.60 increased from {int(valence.old_ge_060_count)} to {int(valence.new_ge_060_count)} for valence and from {int(arousal.old_ge_060_count)} to {int(arousal.new_ge_060_count)} for arousal. The old search retained neither target's highest participant-best BA in this comparison, but even if it had, a single maximum would not outweigh the cohort-wide participant-level improvement."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_all_modalities(path: Path, score_column: str) -> pd.DataFrame:
    """Read the three established modalities without altering source rows."""
    frame = pd.read_csv(path)
    required = {"participant", "target", "modality", score_column}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    metadata = [column for column in ("classifier", "feature_family", "channel_subset", "feature_count_request", "feature_count_resolved") if column in frame]
    output = frame.loc[frame.modality.isin(MODALITIES), ["participant", "target", "modality", score_column, *metadata]].copy()
    if set(output.modality) != set(MODALITIES):
        raise ValueError(f"{path} lacks one or more required modalities")
    return output


def overall_participant_best(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Select one highest-BA configuration per participant-target across modalities.

    Exact BA ties use the fixed modality order EEG, Face, Multimodal and then
    the original stable row order. This only makes winner metadata deterministic;
    it never changes the selected BA.
    """
    def select(frame: pd.DataFrame, score: str, prefix: str) -> pd.DataFrame:
        ordered = frame.copy()
        ordered["_modality_order"] = pd.Categorical(ordered.modality, MODALITIES, ordered=True)
        winners = ordered.sort_values(
            ["participant", "target", score, "_modality_order"],
            ascending=[True, True, False, True], kind="stable",
        ).groupby(["participant", "target"], as_index=False, sort=False).head(1).copy()
        rename = {score: f"{prefix}_best_ba", "modality": f"{prefix}_best_modality", "classifier": f"{prefix}_best_classifier"}
        for column in ("feature_family", "channel_subset", "feature_count_request", "feature_count_resolved"):
            if column in winners:
                rename[column] = f"{prefix}_{column}"
        return winners.rename(columns=rename).drop(columns="_modality_order")

    old_winners = select(old, "balanced_accuracy_mean", "old")
    new_winners = select(new, "mean_outer_cv_balanced_accuracy", "new")
    old_columns = [column for column in old_winners if column.startswith("old_")] + ["participant", "target"]
    new_columns = [column for column in new_winners if column.startswith("new_")] + ["participant", "target"]
    output = old_winners[old_columns].merge(new_winners[new_columns], on=["participant", "target"], how="outer", validate="one_to_one")
    if len(output) != 92 or output[["old_best_ba", "new_best_ba"]].isna().any().any():
        raise ValueError("Expected complete 46-participant × two-target overall-modality comparison")
    output["delta_ba"] = output.new_best_ba - output.old_best_ba
    output["comparison_outcome"] = output.delta_ba.map(lambda value: "improved" if value > 0 else "worsened" if value < 0 else "tied")
    return output.assign(participant=pd.Categorical(output.participant, PARTICIPANTS, ordered=True)).sort_values(["target", "participant"]).reset_index(drop=True)


def overall_summary(participant: pd.DataFrame) -> pd.DataFrame:
    """Summarize the modality-agnostic participant-best comparison by target."""
    rows = []
    for target, group in participant.groupby("target", sort=True, observed=True):
        rows.append({
            "target": target, "participants": len(group),
            "old_mean_participant_best_ba": group.old_best_ba.mean(), "new_mean_participant_best_ba": group.new_best_ba.mean(),
            "old_median_participant_best_ba": group.old_best_ba.median(), "new_median_participant_best_ba": group.new_best_ba.median(),
            "mean_delta_ba": group.delta_ba.mean(), "median_delta_ba": group.delta_ba.median(),
            "improved_count": int((group.comparison_outcome == "improved").sum()), "worsened_count": int((group.comparison_outcome == "worsened").sum()), "tied_count": int((group.comparison_outcome == "tied").sum()),
            "old_ge_060_count": int((group.old_best_ba >= .60).sum()), "old_ge_060_fraction": (group.old_best_ba >= .60).mean(),
            "new_ge_060_count": int((group.new_best_ba >= .60).sum()), "new_ge_060_fraction": (group.new_best_ba >= .60).mean(),
            "old_ge_065_count": int((group.old_best_ba >= .65).sum()), "old_ge_065_fraction": (group.old_best_ba >= .65).mean(),
            "new_ge_065_count": int((group.new_best_ba >= .65).sum()), "new_ge_065_fraction": (group.new_best_ba >= .65).mean(),
            "old_maximum_participant_best_ba": group.old_best_ba.max(), "new_maximum_participant_best_ba": group.new_best_ba.max(),
            "winning_modality_changed_count": int((group.old_best_modality != group.new_best_modality).sum()),
        })
    return pd.DataFrame(rows)


def save_overall_line_figure(frame: pd.DataFrame, target: str, path: Path) -> None:
    """Plot one modality-agnostic participant-best comparison."""
    def statistics(column: str) -> dict[str, float | int]:
        values = frame[column]
        return {
            "mean": float(values.mean()), "median": float(values.median()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "ge_060": int((values >= .60).sum()),
        }

    old, focused = statistics("old_best_ba"), statistics("new_best_ba")
    summary_text = (
        "Best configuration per participant,\nregardless of modality\n\n"
        f"Old\nmean {old['mean']:.3f} | median {old['median']:.3f}\n"
        f"min {old['minimum']:.3f} | max {old['maximum']:.3f}\n"
        f"BA ≥ 0.60: {old['ge_060']}/46\n\n"
        f"Focused\nmean {focused['mean']:.3f} | median {focused['median']:.3f}\n"
        f"min {focused['minimum']:.3f} | max {focused['maximum']:.3f}\n"
        f"BA ≥ 0.60: {focused['ge_060']}/46"
    )
    figure, axis = plt.subplots(figsize=(13, 5.2))
    axis.plot(frame.participant.astype(str), frame.old_best_ba, marker="o", linewidth=1.6, label="Old best across modalities")
    axis.plot(frame.participant.astype(str), frame.new_best_ba, marker="o", linewidth=1.6, label="Focused best across modalities")
    axis.axhline(.50, color="gray", linewidth=1, linestyle="--", label="BA = 0.50")
    axis.axhline(.60, color="black", linewidth=1, linestyle=":", label="BA = 0.60")
    axis.set(title=f"{target.title()}: overall participant-best balanced accuracy", xlabel="Participant", ylabel="Balanced accuracy", ylim=(.35, .75))
    axis.tick_params(axis="x", rotation=90); axis.legend(ncol=4, fontsize=9)
    axis.text(1.02, .98, summary_text, transform=axis.transAxes, va="top", ha="left", fontsize=8.5,
              bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#666666", "alpha": .96}, clip_on=False)
    figure.tight_layout(rect=(0, 0, .76, 1)); figure.savefig(path, dpi=200); plt.close(figure)


def save_overall_delta_figure(frame: pd.DataFrame, target: str, path: Path) -> None:
    """Plot focused-minus-old best BA across all modalities."""
    figure, axis = plt.subplots(figsize=(13, 4.8))
    colors = ["#2a7f62" if value > 0 else "#b24a4a" if value < 0 else "#808080" for value in frame.delta_ba]
    axis.bar(frame.participant.astype(str), frame.delta_ba, color=colors); axis.axhline(0, color="black", linewidth=1)
    axis.set(title=f"{target.title()}: focused minus old overall participant-best BA", xlabel="Participant", ylabel="Δ balanced accuracy", ylim=(-.14, .14))
    axis.tick_params(axis="x", rotation=90); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)


def participant_best_all_modalities(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Reduce each search to one participant-best BA per target and modality."""
    keys = ["participant", "target", "modality"]
    old_best = old.groupby(keys, as_index=False).balanced_accuracy_mean.max().rename(columns={"balanced_accuracy_mean": "old_best_ba"})
    new_best = new.groupby(keys, as_index=False).mean_outer_cv_balanced_accuracy.max().rename(columns={"mean_outer_cv_balanced_accuracy": "new_best_ba"})
    output = old_best.merge(new_best, on=keys, how="outer", validate="one_to_one")
    if len(output) != 276 or output[["old_best_ba", "new_best_ba"]].isna().any().any():
        raise ValueError("Expected complete 46-participant × two-target × three-modality comparison")
    output["delta_ba"] = output.new_best_ba - output.old_best_ba
    output["improved"] = output.delta_ba > 0
    output["old_ge_060"] = output.old_best_ba >= .60; output["new_ge_060"] = output.new_best_ba >= .60
    output["old_ge_065"] = output.old_best_ba >= .65; output["new_ge_065"] = output.new_best_ba >= .65
    return output.assign(participant=pd.Categorical(output.participant, PARTICIPANTS, ordered=True)).sort_values(["target", "modality", "participant"]).reset_index(drop=True)


def all_modality_summary(participant: pd.DataFrame) -> pd.DataFrame:
    """Return the six requested participant-level target-modality summaries."""
    rows = []
    for (target, modality), group in participant.groupby(["target", "modality"], sort=True, observed=True):
        rows.append({"target": target, "modality": modality, "participants": len(group),
                     "old_mean_participant_best_ba": group.old_best_ba.mean(), "new_mean_participant_best_ba": group.new_best_ba.mean(), "mean_delta_ba": group.delta_ba.mean(),
                     "old_median_participant_best_ba": group.old_best_ba.median(), "new_median_participant_best_ba": group.new_best_ba.median(), "median_delta_ba": group.delta_ba.median(),
                     "improved_count": int((group.delta_ba > 0).sum()), "worsened_count": int((group.delta_ba < 0).sum()), "tied_count": int((group.delta_ba == 0).sum()),
                     "old_ge_060_count": int(group.old_ge_060.sum()), "new_ge_060_count": int(group.new_ge_060.sum()),
                     "old_ge_065_count": int(group.old_ge_065.sum()), "new_ge_065_count": int(group.new_ge_065.sum()),
                     "old_maximum_participant_best_ba": group.old_best_ba.max(), "new_maximum_participant_best_ba": group.new_best_ba.max()})
    return pd.DataFrame(rows)


def save_modality_line_figure(participant: pd.DataFrame, modality: str, path: Path) -> None:
    """Render valence and arousal participant-best comparisons in one figure."""
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for axis, target in zip(axes, TARGETS):
        frame = participant.loc[(participant.target == target) & (participant.modality == modality)]
        axis.plot(frame.participant.astype(str), frame.old_best_ba, marker="o", linewidth=1.4, label="Old broad search")
        axis.plot(frame.participant.astype(str), frame.new_best_ba, marker="o", linewidth=1.4, label="New focused search")
        axis.axhline(.50, color="gray", linewidth=1, linestyle="--"); axis.axhline(.60, color="black", linewidth=1, linestyle=":")
        axis.set(title=target.title(), ylabel="Participant-best BA", ylim=(.35, .75)); axis.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Participant"); axes[-1].tick_params(axis="x", rotation=90)
    figure.suptitle(f"{modality.title()}: old vs focused participant-best balanced accuracy", y=.995); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)


def save_modality_delta_figure(participant: pd.DataFrame, modality: str, path: Path) -> None:
    """Render focused-minus-old BA by participant for both targets."""
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for axis, target in zip(axes, TARGETS):
        frame = participant.loc[(participant.target == target) & (participant.modality == modality)]
        colors = ["#2a7f62" if value > 0 else "#b24a4a" if value < 0 else "#808080" for value in frame.delta_ba]
        axis.bar(frame.participant.astype(str), frame.delta_ba, color=colors); axis.axhline(0, color="black", linewidth=1)
        axis.set(title=target.title(), ylabel="Focused − old BA", ylim=(-.16, .16))
    axes[-1].set_xlabel("Participant"); axes[-1].tick_params(axis="x", rotation=90)
    figure.suptitle(f"{modality.title()}: participant-best BA change", y=.995); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)


def write_all_modality_report(path: Path, summary: pd.DataFrame, overall: pd.DataFrame, winners: pd.DataFrame) -> None:
    """Write a concise modality-specific verdict based only on participant-best rows."""
    lines = ["# Old broad vs focused personalized classification: all modalities", "", "This read-only comparison selects one best mean outer-CV BA per participant × target × modality in each search. It does not use configuration-row counts as evidence because the searches differ in candidate spaces.", ""]
    for modality in MODALITIES:
        lines += [f"## {modality.title()}", "", "| Target | Old mean | Focused mean | Mean Δ | Old median | Focused median | Median Δ | Improved/worsened/tied | ≥0.60 old → focused | ≥0.65 old → focused |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for target in TARGETS:
            row = summary.loc[(summary.target == target) & (summary.modality == modality)].iloc[0]
            lines.append(f"| {target.title()} | {row.old_mean_participant_best_ba:.3f} | {row.new_mean_participant_best_ba:.3f} | {row.mean_delta_ba:+.3f} | {row.old_median_participant_best_ba:.3f} | {row.new_median_participant_best_ba:.3f} | {row.median_delta_ba:+.3f} | {int(row.improved_count)}/{int(row.worsened_count)}/{int(row.tied_count)} | {int(row.old_ge_060_count)} → {int(row.new_ge_060_count)} | {int(row.old_ge_065_count)} → {int(row.new_ge_065_count)} |")
        lines.append("")
    eeg = summary.loc[summary.modality.eq("eeg")].set_index("target")
    face = summary.loc[summary.modality.eq("face")].set_index("target")
    multi = summary.loc[summary.modality.eq("multimodal")].set_index("target")
    lines += [
        "## Verdict",
        "",
        f"**EEG:** yes. The focused search improved participant-best mean and median BA for both targets (valence mean Δ {eeg.loc['valence', 'mean_delta_ba']:+.3f}; arousal mean Δ {eeg.loc['arousal', 'mean_delta_ba']:+.3f}) and increased the BA ≥0.60 count from {int(eeg.loc['valence', 'old_ge_060_count'])} to {int(eeg.loc['valence', 'new_ge_060_count'])} for valence and from {int(eeg.loc['arousal', 'old_ge_060_count'])} to {int(eeg.loc['arousal', 'new_ge_060_count'])} for arousal.",
        "",
        f"**Face:** modest but uncertain improvement. Valence mean BA was effectively unchanged ({face.loc['valence', 'mean_delta_ba']:+.3f}) and its cohort median was lower; arousal mean BA increased by {face.loc['arousal', 'mean_delta_ba']:+.3f}. The BA ≥0.60 count increased from {int(face.loc['valence', 'old_ge_060_count'])} to {int(face.loc['valence', 'new_ge_060_count'])} for valence and from {int(face.loc['arousal', 'old_ge_060_count'])} to {int(face.loc['arousal', 'new_ge_060_count'])} for arousal, but the BA ≥0.65 count fell to {int(face.loc['valence', 'new_ge_065_count'])} and {int(face.loc['arousal', 'new_ge_065_count'])}, respectively. This is not yet a strong broad cohort-level gain.",
        "",
        f"**Multimodal:** no general cohort-level improvement. Valence declined in both mean and median (mean Δ {multi.loc['valence', 'mean_delta_ba']:+.3f}; median Δ {multi.loc['valence', 'median_delta_ba']:+.3f}); arousal mean and median also declined ({multi.loc['arousal', 'mean_delta_ba']:+.3f}; {multi.loc['arousal', 'median_delta_ba']:+.3f}). Some threshold counts and the focused arousal maximum were higher, but these do not outweigh the typical participant-level results.",
        "",
        f"Overall, the focused search improved personalized classification for the EEG branch—the intended optimization target—and produced modest Face gains, but not a general Multimodal gain: mean ΔBA was positive in {int((summary.mean_delta_ba > 0).sum())} of six target-modality combinations and negative in {int((summary.mean_delta_ba < 0).sum())}. This conclusion is based on participant-best comparisons, not incomparable configuration-row counts or a single maximum.",
    ]
    lines += ["", "## Overall best personalized model regardless of modality", "", "For each participant and target, this section selects the single highest mean outer-CV BA across EEG, Face, and Multimodal separately within the old and focused searches. Exact BA ties use the fixed order EEG, Face, Multimodal only to make metadata deterministic.", "", "| Target | Old mean | Focused mean | Mean Δ | Old median | Focused median | Median Δ | Improved/worsened/tied | ≥0.60 old → focused | ≥0.65 old → focused | Max old → focused |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for target in TARGETS:
        row = overall.loc[overall.target.eq(target)].iloc[0]
        lines.append(f"| {target.title()} | {row.old_mean_participant_best_ba:.3f} | {row.new_mean_participant_best_ba:.3f} | {row.mean_delta_ba:+.3f} | {row.old_median_participant_best_ba:.3f} | {row.new_median_participant_best_ba:.3f} | {row.median_delta_ba:+.3f} | {int(row.improved_count)}/{int(row.worsened_count)}/{int(row.tied_count)} | {int(row.old_ge_060_count)} → {int(row.new_ge_060_count)} | {int(row.old_ge_065_count)} → {int(row.new_ge_065_count)} | {row.old_maximum_participant_best_ba:.3f} → {row.new_maximum_participant_best_ba:.3f} |")
    lines += ["", "Winning-modality counts:", "", "| Target | Search | EEG | Face | Multimodal | Winner modality changed |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for target in TARGETS:
        group = winners.loc[winners.target.eq(target)]
        changed = int((group.old_best_modality != group.new_best_modality).sum())
        for label, column in (("Old", "old_best_modality"), ("Focused", "new_best_modality")):
            counts = group[column].value_counts()
            lines.append(f"| {target.title()} | {label} | {int(counts.get('eeg', 0))} | {int(counts.get('face', 0))} | {int(counts.get('multimodal', 0))} | {changed if label == 'Focused' else ''} |")
    overall_indexed = overall.set_index("target")
    valence, arousal = overall_indexed.loc["valence"], overall_indexed.loc["arousal"]
    lines += ["", f"**Overall verdict:** {'Yes' if valence.mean_delta_ba > 0 and arousal.mean_delta_ba > 0 else 'No'}—after allowing all three modalities to compete per participant, the focused search changed mean participant-best BA by {valence.mean_delta_ba:+.3f} for valence and {arousal.mean_delta_ba:+.3f} for arousal. The accompanying improved/worsened/tied and threshold counts above determine whether this is a cohort-level gain rather than a maximum-only result."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = load_all_modalities(OLD_PATH, "balanced_accuracy_mean")
    new = load_all_modalities(NEW_PATH, "mean_outer_cv_balanced_accuracy")
    participant = participant_best_all_modalities(old, new)
    summary = all_modality_summary(participant)
    overall_participant = overall_participant_best(old, new)
    overall = overall_summary(overall_participant)
    overall_participant.to_csv(OUTPUT / "old_vs_new_participant_best_any_modality.csv", index=False)
    overall.to_csv(OUTPUT / "old_vs_new_participant_best_any_modality_summary.csv", index=False)
    for target in TARGETS:
        frame = overall_participant.loc[overall_participant.target.eq(target)]
        save_overall_line_figure(frame, target, OUTPUT / f"overall_best_{target}.png")
        save_overall_delta_figure(frame, target, OUTPUT / f"overall_delta_{target}.png")
    write_all_modality_report(OUTPUT / "OLD_VS_NEW_EEG_COMPARISON.md", summary, overall, overall_participant)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
