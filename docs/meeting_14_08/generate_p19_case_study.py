#!/usr/bin/env python3
"""Create a read-only P19 case study from completed modeling outputs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IND = ROOT / "outputs/image_classification/stage_a_individual_v1"
GEN = ROOT / "outputs/image_classification/stage_a_general_v1"
PARTICIPANT = "P19"


def mask(frame: pd.DataFrame, **values: str) -> pd.Series:
    result = pd.Series(True, index=frame.index)
    for key, value in values.items():
        result &= frame[key].astype(str).eq(str(value))
    return result


def decode(feature: str) -> dict[str, str]:
    if feature.startswith("video_"):
        name, statistic = feature.removeprefix("video_").rsplit("_", 1)
        descriptions = {
            "irisdo_norm": "mean eyelid-opening distance, averaged across eyes and normalized by outer-eye distance",
            "eso_norm": "distance between left and right iris centers, normalized by outer-eye distance",
            "enso_norm": "distance from midpoint of both irises to subnasale approximation, normalized by outer-eye distance",
            "mnso_norm": "distance from upper-lip center to subnasale approximation, normalized by outer-eye distance",
            "mwo_norm": "mouth-corner width, normalized by outer-eye distance",
        }
        return {"feature_family": "face geometry", "channel_if_eeg": "", "frequency_band_if_eeg": "", "statistic_if_identifiable": statistic, "modality": "face", "plain_description": descriptions.get(name, name) + f"; trial-level {statistic}"}
    family, channel = feature.split("__", 1)
    labels = {"eeg_sd": "signal standard deviation", "eeg_se": "spectral entropy of the full power spectrum", "eeg_hm": "Hjorth mobility", "eeg_hc": "Hjorth complexity", "eeg_mf_hz": "median frequency", "eeg_bp": "band power", "eeg_se": "spectral entropy"}
    if family.startswith("eeg_bp_"):
        band = family.removeprefix("eeg_bp_")
        description = f"{band}-band power"
        statistic = "power"
    elif family.startswith("eeg_se_"):
        band = family.removeprefix("eeg_se_")
        description = f"spectral entropy within the {band} band"
        statistic = "spectral entropy"
    else:
        band = ""
        description = labels.get(family, family)
        statistic = description
    return {"feature_family": family, "channel_if_eeg": channel, "frequency_band_if_eeg": band, "statistic_if_identifiable": statistic, "modality": "EEG", "plain_description": f"{description} at {channel}"}


def best_for(summary: pd.DataFrame, *, target: str | None = None, modality: str | None = None) -> pd.Series:
    data = summary.copy()
    if target: data = data[data.target.eq(target)]
    if modality: data = data[data.modality.eq(modality)]
    return data.loc[data.balanced_accuracy_mean.idxmax()]


def fmt(row: pd.Series) -> str:
    return f"{row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_sd:.3f} (median {row.balanced_accuracy_median:.3f}; range {row.balanced_accuracy_min:.3f}–{row.balanced_accuracy_max:.3f})"


def main() -> None:
    summary = pd.read_csv(IND / "results_summary.csv")
    folds = pd.read_csv(IND / "fold_results.csv")
    params = pd.read_csv(IND / "best_hyperparameters.csv")
    selected = pd.read_csv(IND / "selected_features_by_fold.csv")
    general_folds = pd.read_csv(GEN / "fold_results.csv")
    p19 = summary[summary.participant.eq(PARTICIPANT)].copy()
    best_value = p19.balanced_accuracy_mean.max()
    tied = p19[p19.balanced_accuracy_mean.eq(best_value)].sort_values(["modality", "classifier"])
    winner = tied[(tied.modality == "eeg") & (tied.classifier == "knn")].iloc[0]
    best_valence, best_arousal = best_for(p19, target="valence"), best_for(p19, target="arousal")
    best_eeg, best_face, best_multi = best_for(p19, modality="eeg"), best_for(p19, modality="face"), best_for(p19, modality="multimodal")
    key = {"participant": PARTICIPANT, "target": winner.target, "modality": winner.modality, "classifier": winner.classifier, "feature_count_request": winner.feature_count_request}
    win_folds = folds[mask(folds, **key)].sort_values("fold")
    win_params = params[mask(params, **key)].sort_values("fold")
    win_selected = selected[mask(selected, **key)]
    stable = (win_selected[win_selected.selected]
              .groupby("feature", as_index=False)
              .agg(number_of_folds_selected=("fold", "nunique"), mean_feature_score_if_available=("feature_score", "mean"))
              .sort_values(["number_of_folds_selected", "mean_feature_score_if_available", "feature"], ascending=[False, False, True]))
    stable["selection_frequency_percent"] = 100 * stable.number_of_folds_selected / win_folds.fold.nunique()
    decoded = stable.feature.map(decode).apply(pd.Series).reset_index(drop=True)
    stable = pd.concat([stable.reset_index(drop=True), decoded], axis=1)
    stable = stable[["feature", "number_of_folds_selected", "selection_frequency_percent", "mean_feature_score_if_available", "feature_family", "channel_if_eeg", "frequency_band_if_eeg", "statistic_if_identifiable", "modality", "plain_description"]]
    stable.to_csv(OUT / "p19_best_model_feature_stability.csv", index=False)
    top = stable.head(10)

    matched_general = general_folds[(general_folds.held_out_participant.eq(PARTICIPANT)) & mask(general_folds, target=winner.target, modality=winner.modality, classifier=winner.classifier, feature_count_request=winner.feature_count_request)].iloc[0]
    best_general_same_target = general_folds[(general_folds.held_out_participant.eq(PARTICIPANT)) & general_folds.target.eq(winner.target)].nlargest(1, "balanced_accuracy").iloc[0]
    totals = win_folds[["tn", "fp", "fn", "tp"]].sum()
    common_parameters = win_params.best_parameters.value_counts().idxmax()
    channels = stable[stable.modality.eq("EEG")].groupby("channel_if_eeg").number_of_folds_selected.sum().sort_values(ascending=False)
    bands = stable[stable.frequency_band_if_eeg.ne("")].groupby("frequency_band_if_eeg").number_of_folds_selected.sum().sort_values(ascending=False)
    trial_table = pd.read_csv(ROOT / "derived/P19/Image_Experiment/runs/p19_final/merged/p19_image_trial_dataset.csv")
    valence_low = int((trial_table.valence_rating < 4).sum())
    valence_high = int((trial_table.valence_rating >= 4).sum())

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, target in zip(axes, ["valence", "arousal"]):
        data = p19[p19.target.eq(target)].groupby(["modality", "classifier"], as_index=False).balanced_accuracy_mean.max()
        pivot = data.pivot(index="classifier", columns="modality", values="balanced_accuracy_mean").reindex(index=["knn", "svm", "gnb"], columns=["eeg", "face", "multimodal"])
        pivot.plot(kind="bar", ax=axis)
        axis.axhline(.5, color="black", linestyle="--", linewidth=1)
        axis.set(title=target.capitalize(), xlabel="Classifier", ylabel="Best mean balanced accuracy", ylim=(0, .8))
    fig.suptitle("P19 individual-model comparison", y=1.02); fig.tight_layout(); fig.savefig(OUT / "p19_model_comparison.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2)); ax.plot(win_folds.fold, win_folds.balanced_accuracy, marker="o"); ax.axhline(winner.balanced_accuracy_mean, color="black", linestyle="--", label=f"Mean {winner.balanced_accuracy_mean:.3f}"); ax.axhline(.5, color="gray", linestyle=":", label="Balanced-accuracy chance"); ax.set(xticks=win_folds.fold, ylim=(0, 1), xlabel="Outer CV fold", ylabel="Balanced accuracy", title="P19 best individual model: outer-fold performance"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "p19_best_model_folds.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.6)); ax.barh(top.feature.iloc[::-1], top.selection_frequency_percent.iloc[::-1]); ax.set(xlim=(0, 100), xlabel="Outer folds selected (%)", title="P19 best model: selected EEG features"); fig.tight_layout(); fig.savefig(OUT / "p19_top_features.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2)); names=["Individual\nmatched config", "General LOSO\nmatched config", "General LOSO\nbest arousal config"]; values=[winner.balanced_accuracy_mean, matched_general.balanced_accuracy, best_general_same_target.balanced_accuracy]; ax.bar(names, values, color=["#4477AA", "#CC6677", "#DDCC77"]); ax.axhline(.5, color="black", linestyle="--"); ax.set(ylim=(0, 1), ylabel="Balanced accuracy", title="P19: personalized versus unseen-participant evaluation"); fig.tight_layout(); fig.savefig(OUT / "p19_individual_vs_general.png", dpi=180); plt.close(fig)

    tie_text = "; ".join(f"{row.modality} {row.classifier} {row.feature_count_request}" for row in tied.itertuples())
    table = "| Target | Modality | Classifier | Features | Balanced accuracy | SD | Range | Frequent hyperparameters |\n|---|---|---|---|---:|---:|---:|---|\n"
    for label, row in [("Best overall tie", winner), ("Best valence", best_valence), ("Best arousal", best_arousal), ("Best face", best_face)]:
        matching = params[mask(params, participant=PARTICIPANT, target=row.target, modality=row.modality, classifier=row.classifier, feature_count_request=row.feature_count_request)]
        table += f"| {label}: {row.target} | {row.modality} | {row.classifier} | {row.feature_count_request} | {row.balanced_accuracy_mean:.3f} | {row.balanced_accuracy_sd:.3f} | {row.balanced_accuracy_min:.3f}–{row.balanced_accuracy_max:.3f} | `{matching.best_parameters.value_counts().idxmax()}` |\n"
    feature_lines = "\n".join(f"- `{row.feature}` — {row.plain_description}; selected in {int(row.number_of_folds_selected)}/5 folds ({row.selection_frequency_percent:.0f}%), mean F score {row.mean_feature_score_if_available:.2f}." for row in top.itertuples())
    report = f"""# P19 Individual Emotion-Classification Case Study

## Why P19

P19 belongs to the **Very clean** QC group and had 120 usable trials for both targets and all three modalities. Its rating classes were: valence LOW/HIGH = {valence_low}/{valence_high}; arousal LOW/HIGH = {int(win_folds.low_count.iloc[0])}/{int(win_folds.high_count.iloc[0])}. These are moderately imbalanced, so raw accuracy alone would be misleading; balanced accuracy remains the primary metric.

## Best individual result

{table}

The highest P19 mean balanced accuracy was a four-way numerical tie at **{best_value:.3f}**: {tie_text}. This report uses **arousal, EEG, kNN, Top 10** as the representative winner because it is the direct EEG-only result. The tied multimodal kNN result selected the same EEG-only feature set, so it did not demonstrate an added face-feature benefit.

The kNN model classifies a trial according to the LOW/HIGH ratings of its most similar training trials in the selected EEG-feature space. Each outer fold used 96 P19 trials for training/tuning and 24 untouched P19 trials for testing.

Fold balanced accuracies were {", ".join(f"{value:.3f}" for value in win_folds.balanced_accuracy)}; mean {winner.balanced_accuracy_mean:.3f}, SD {winner.balanced_accuracy_sd:.3f}, median {winner.balanced_accuracy_median:.3f}, range {winner.balanced_accuracy_min:.3f}–{winner.balanced_accuracy_max:.3f}. Fold confusion-matrix totals were TN={int(totals.tn)}, FP={int(totals.fp)}, FN={int(totals.fn)}, TP={int(totals.tp)}. Tuning selected: {"; ".join(f"fold {r.fold}: {r.best_parameters}" for r in win_params.itertuples())}. The most frequent combination was `{common_parameters}`.

## Valence vs arousal

P19 was more predictable for **arousal**: best arousal {fmt(best_arousal)} versus best valence {fmt(best_valence)}. This is an exploratory within-person comparison, not a significance test.

## EEG vs face vs multimodal

The best EEG result was {fmt(best_eeg)}. The best face result was {fmt(best_face)}. The best multimodal result tied the EEG result, but its selected predictors were EEG-only; therefore multimodal input did **not** improve on EEG for P19’s top result.

## Most informative features

These features were consistently selected by the representative kNN classifier for P19; they are not established neural biomarkers.

{feature_lines}

Selected features were concentrated in frontal/central/posterior channels: {", ".join(f"{channel} ({int(count)} selections)" for channel, count in channels.items())}. Band-specific selections were dominated by {", ".join(f"{band} ({int(count)})" for band, count in bands.items())}; other repeatedly selected features were Hjorth mobility/complexity, median frequency, and full-spectrum spectral entropy. Feature names and definitions come directly from the frozen feature extractor: band-power features integrate Welch PSD in the named band, while spectral-entropy features apply Shannon entropy to the relevant PSD values.

## Individual vs unseen-participant generalization

**Individual nested CV:** trained and tested only within P19, representative arousal EEG kNN Top 10 = {winner.balanced_accuracy_mean:.3f}.

**General LOSO, matched configuration:** trained on the other nine Very clean participants and tested entirely on P19, arousal EEG kNN Top 10 = {matched_general.balanced_accuracy:.3f}; absolute difference = {winner.balanced_accuracy_mean - matched_general.balanced_accuracy:.3f}.

**General LOSO, best P19 arousal configuration:** {best_general_same_target.modality} {best_general_same_target.classifier} {best_general_same_target.feature_count_request} = {best_general_same_target.balanced_accuracy:.3f}. This best-vs-best comparison uses different configurations and is descriptive only.

The difference between personalized and unseen-participant performance is consistent with P19-specific affective patterns being more learnable after personal calibration. One participant cannot establish that conclusion statistically.

## Interpretation for the thesis

P19 is a useful case study because its individualized arousal prediction was above balanced-accuracy chance and reasonably stable across folds, whereas P19 was weaker when completely unseen by models trained on other people. This supports investigating personalized calibration alongside subject-independent approaches, without generalizing from P19 to the wider cohort.

## What I can say in the meeting

- "P19 was in the Very clean QC group and had 120 usable trials for both valence and arousal."
- "P19’s strongest individualized result was arousal classification from EEG features, with mean balanced accuracy {winner.balanced_accuracy_mean:.3f}."
- "The result was a tie with some multimodal fits because their winning feature sets contained EEG features only; face features did not add to that result."
- "The consistently selected features included beta-band power at PO8 and Fz, plus Hjorth mobility and complexity at Fz."
- "When P19 was held out entirely from a general model, the matched configuration fell to {matched_general.balanced_accuracy:.3f}, close to or below balanced-accuracy chance."
- "This is compatible with participant-specific mappings that may benefit from personal calibration, but it is not a statistical conclusion from one participant."
"""
    (OUT / "P19_CASE_STUDY.md").write_text(report, encoding="utf-8")
    print(f"Best P19 target: {winner.target}")
    print(f"Best P19 modality: {winner.modality}")
    print(f"Best classifier: {winner.classifier}")
    print(f"Best feature setting: {winner.feature_count_request}")
    print(f"Mean balanced accuracy ± SD: {winner.balanced_accuracy_mean:.3f} ± {winner.balanced_accuracy_sd:.3f}")
    print(f"Range: {winner.balanced_accuracy_min:.3f}–{winner.balanced_accuracy_max:.3f}")
    print(f"Class counts LOW/HIGH: {int(win_folds.low_count.iloc[0])}/{int(win_folds.high_count.iloc[0])}")
    print("Top 5 selected features: " + ", ".join(top.feature.head(5)))
    print(f"Matched individual-vs-general: {winner.balanced_accuracy_mean:.3f} vs {matched_general.balanced_accuracy:.3f}")


if __name__ == "__main__":
    main()
