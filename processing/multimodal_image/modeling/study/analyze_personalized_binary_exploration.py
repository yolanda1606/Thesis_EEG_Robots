#!/usr/bin/env python3
"""Summarize completed personalized binary-classification explorations.

This is a read-only analysis of existing ``outputs/image_classification`` CSVs.
It deliberately does not fit a model or read raw acquisition data.  The output
tables treat each outer-CV configuration as a descriptive observation; they do
not treat the correlated configurations from one participant as independent
statistical tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "image_classification"
EXPERIMENTS = {
    "no_ica": "balanced_accuracy_mean",
    "individual_optimized_top15": "mean_balanced_accuracy",
    "feature_family_top5": "mean_balanced_accuracy",
    "label_ambiguity_top5": "mean_balanced_accuracy",
}
THRESHOLDS = (0.55, 0.60, 0.65)


def summarize(values: pd.Series) -> dict[str, float | int]:
    """Return descriptive BA distribution measures for one configuration group."""
    return {
        "n_configurations": len(values),
        "median_ba": values.median(),
        "mean_ba": values.mean(),
        "upper_quartile_ba": values.quantile(0.75),
        "maximum_ba": values.max(),
        **{f"fraction_ba_ge_{int(threshold * 100)}": (values >= threshold).mean() for threshold in THRESHOLDS},
    }


def factor_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build comparable within-experiment factor and interaction summaries."""
    plans = {
        "no_ica": [("classifier",), ("modality",), ("feature_count_request",), ("classifier", "modality")],
        "individual_optimized_top15": [
            ("classifier",), ("modality",), ("feature_count_request",),
            ("classifier", "modality"), ("classifier", "feature_count_request"),
        ],
        "feature_family_top5": [
            ("feature_family",), ("channel_subset",), ("classifier",),
            ("feature_family", "classifier"), ("feature_family", "channel_subset"),
            ("classifier", "channel_subset"),
        ],
        "label_ambiguity_top5": [("label_condition",), ("classifier",), ("label_condition", "classifier")],
    }
    rows: list[dict[str, object]] = []
    for name, frame in frames.items():
        score = EXPERIMENTS[name]
        for target, target_frame in frame.groupby("target", sort=True):
            for columns in plans[name]:
                for values, group in target_frame.groupby(list(columns), dropna=False, sort=True):
                    if not isinstance(values, tuple):
                        values = (values,)
                    rows.append({
                        "experiment": name,
                        "target": target,
                        "factor": " × ".join(columns),
                        "level": " × ".join(str(value) for value in values),
                        **summarize(group[score]),
                    })
    return pd.DataFrame(rows).sort_values(["experiment", "factor", "target", "mean_ba"], ascending=[True, True, True, False])


def paired_classifier_rows(frame: pd.DataFrame, experiment: str) -> list[dict[str, object]]:
    """Compare LogReg to each alternative using identical non-classifier settings."""
    if experiment == "label_ambiguity_top5":
        frame = frame.loc[frame["label_condition"] == "standard"].copy()
    keys = [column for column in (
        "participant", "target", "modality", "feature_family", "channel_subset", "feature_count_request", "label_condition"
    ) if column in frame]
    logreg = frame.loc[frame["classifier"] == "logreg"].set_index(keys)[EXPERIMENTS[experiment]]
    rows: list[dict[str, object]] = []
    for comparator in sorted(set(frame["classifier"]) - {"logreg"}):
        other = frame.loc[frame["classifier"] == comparator].set_index(keys)[EXPERIMENTS[experiment]]
        joined = pd.concat([logreg.rename("logreg"), other.rename("comparator")], axis=1).dropna()
        for target, subset in joined.groupby(level="target", sort=True):
            delta = subset["logreg"] - subset["comparator"]
            rows.append({
                "comparison_type": "matched_logreg_vs_classifier",
                "experiment": experiment,
                "target": target,
                "comparison": f"logreg - {comparator}",
                "n_matched_configurations": len(delta),
                "mean_delta_ba": delta.mean(),
                "median_delta_ba": delta.median(),
                "fraction_logreg_better": (delta > 0).mean(),
            })
    return rows


def comparison_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create matched classifier and label-condition comparisons."""
    rows: list[dict[str, object]] = []
    for experiment in ("individual_optimized_top15", "feature_family_top5", "label_ambiguity_top5"):
        rows.extend(paired_classifier_rows(frames[experiment], experiment))

    family = frames["feature_family_top5"]
    rows.extend(paired_classifier_rows(family.loc[family["feature_family"] == "paper_compact"].copy(), "feature_family_top5"))
    for row in rows[-6:]:
        row["comparison_type"] = "matched_paper_compact_logreg_vs_classifier"

    ambiguity = frames["label_ambiguity_top5"]
    keys = ["participant", "target", "modality", "classifier", "feature_count_request"]
    paired = ambiguity.pivot(index=keys, columns="label_condition", values="mean_balanced_accuracy").dropna()
    for target, subset in paired.groupby(level="target", sort=True):
        delta = subset["exclude_midpoint"] - subset["standard"]
        rows.append({
            "comparison_type": "matched_exclude_midpoint_vs_standard",
            "experiment": "label_ambiguity_top5",
            "target": target,
            "comparison": "exclude_midpoint - standard",
            "n_matched_configurations": len(delta),
            "mean_delta_ba": delta.mean(),
            "median_delta_ba": delta.median(),
            "fraction_logreg_better": (delta > 0).mean(),
        })
    return pd.DataFrame(rows)


def participant_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join baseline distributions and QC/trial information at participant-target level."""
    baseline = frames["no_ica"]
    grouped = baseline.groupby(["participant", "target"])["balanced_accuracy_mean"].agg(
        n_configurations="size", median_ba="median", mean_ba="mean", upper_quartile_ba=lambda x: x.quantile(.75),
        maximum_ba="max", fraction_ba_ge_55=lambda x: (x >= .55).mean(), fraction_ba_ge_60=lambda x: (x >= .60).mean(),
    ).reset_index()
    fold = pd.read_csv(OUTPUT_ROOT / "no_ica" / "fold_results.csv")
    fold_summary = fold.groupby(["participant", "target"])["balanced_accuracy"].agg(
        pooled_fold_ba_sd="std", pooled_fold_ba_min="min", pooled_fold_ba_max="max"
    ).reset_index()
    qc = pd.read_csv(OUTPUT_ROOT / "no_ica" / "resolved_participants.csv")
    output = grouped.merge(fold_summary, on=["participant", "target"], how="left").merge(qc, on="participant", how="left")

    optimized = frames["individual_optimized_top15"]
    optimized_summary = optimized.groupby(["participant", "target"])["mean_balanced_accuracy"].agg(
        optimized_n_configurations="size", optimized_median_ba="median", optimized_maximum_ba="max", optimized_fraction_ba_ge_60=lambda x: (x >= .60).mean()
    ).reset_index()
    return output.merge(optimized_summary, on=["participant", "target"], how="left").sort_values(["target", "maximum_ba"], ascending=[True, False])


def selected_hyperparameter_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize values selected by inner CV, without mistaking them for a grid ablation."""
    rows: list[dict[str, object]] = []
    for experiment in ("individual_optimized_top15", "feature_family_top5", "label_ambiguity_top5"):
        frame = frames[experiment]
        for row_index, result in frame.iterrows():
            for fold in json.loads(result["best_hyperparameters"]):
                for parameter, value in fold["parameters"].items():
                    rows.append({
                        "experiment": experiment,
                        "configuration_row": row_index,
                        "target": result["target"],
                        "classifier": result["classifier"],
                        "parameter": parameter,
                        "selected_value": str(value),
                        "outer_configuration_ba": result[EXPERIMENTS[experiment]],
                    })
    selected = pd.DataFrame(rows)
    output_rows: list[dict[str, object]] = []
    for values, group in selected.groupby(["experiment", "target", "classifier", "parameter", "selected_value"], sort=True):
        experiment, target, classifier, parameter, selected_value = values
        output_rows.append({
            "experiment": experiment,
            "target": target,
            "classifier": classifier,
            "parameter": parameter,
            "selected_value": selected_value,
            "n_selected_outer_folds": len(group),
            "n_configurations_with_selection": group["configuration_row"].nunique(),
            **summarize(group["outer_configuration_ba"]),
        })
    return pd.DataFrame(output_rows).sort_values(["experiment", "target", "classifier", "parameter", "selected_value"])


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns without an index and with readable decimals."""
    display = frame.loc[:, columns].copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def write_report(destination: Path, factors: pd.DataFrame, comparisons: pd.DataFrame, participants: pd.DataFrame, hyperparameters: pd.DataFrame) -> None:
    """Write concise findings, limitations, and a deliberately reduced next search."""
    def factor(experiment: str, target: str, name: str) -> pd.DataFrame:
        return factors.query("experiment == @experiment and target == @target and factor == @name")

    logreg = comparisons.query("comparison_type == 'matched_logreg_vs_classifier'")
    paper = comparisons.query("comparison_type == 'matched_paper_compact_logreg_vs_classifier'")
    midpoint = comparisons.query("comparison_type == 'matched_exclude_midpoint_vs_standard'")
    top = participants.loc[participants["participant"].isin(["P06", "P17", "P22", "P36", "P40"])]
    text = f"""# Personalized binary Image classification: exploratory-search audit

Generated from completed outputs only. No models were fit for this report; raw `data/` was not read or changed.

## What the four folders are

| Folder | Cohort and design | Completed configuration-level rows |
|---|---|---:|
| `no_ica` | Baseline: P01–P46, EEG/face/multimodal, kNN/SVM/GNB, requests all/5/10/20 | 3,312 |
| `individual_optimized_top15` | Follow-up restricted to P06/P17/P22/P36/P40; adds LogReg and ExtraTrees, requests 3/5/8/10/15/20/30/all | 1,050 |
| `feature_family_top5` | Same five participants; EEG-only feature-family/channel-subset representations and kNN/SVM/LogReg/GNB | 3,400 |
| `label_ambiguity_top5` | Same five; standard labels vs excluding rating 4, with EEG/face/multimodal and kNN/SVM/LogReg/GNB | 1,680 |

Every row is an outer-CV mean balanced accuracy (BA), using shuffled stratified outer CV (up to 5 folds) and fold-local inner CV (up to 3 folds) for selection. The standard label is LOW `<4`, HIGH `>=4`; midpoint exclusion is LOW `<=3`, HIGH `>=5`, dropping rating 4. `no_ica` has fold-level predictions/metrics; the three follow-ups retain only configuration-level outer means and selected fold hyperparameters, not held-out predictions.

The searches are **not one common benchmark**: the top-five studies are selected from the 46-person baseline, expand the model/count grids, and in the feature-family study use EEG-only representations. Claims below are therefore within-experiment comparisons unless marked exploratory.

## Opportunity-adjusted evidence

Use `factor_summary.csv` for every requested count, median, mean, upper quartile, maximum, and BA success fraction. Counts differ for valid reasons: face has fewer feasible feature-count requests; family/subset representations have different available dimensions; and `paper_compact` consequently has 1,120 of 3,400 feature-family rows. Raw numbers of threshold exceedances must not be compared without their denominators.

### Baseline (46 participants; apples-to-apples within this grid)

{markdown_table(factor('no_ica', 'valence', 'classifier'), ['level','n_configurations','median_ba','mean_ba','upper_quartile_ba','maximum_ba','fraction_ba_ge_60'])}

Valence and arousal were both weak overall (baseline medians 0.515 and 0.508 respectively). GNB has the best typical baseline BA; kNN is consistently lower. Face is modestly strongest for both targets in this baseline, so it would be incorrect to claim an EEG-only family changed the multimodal baseline.

### Expanded five-participant search (apples-to-apples within the five participants)

{markdown_table(factor('individual_optimized_top15', 'valence', 'classifier'), ['level','n_configurations','median_ba','mean_ba','upper_quartile_ba','maximum_ba','fraction_ba_ge_60'])}

LogReg is the most useful broad addition: paired with identical participant/target/modality/count settings, it beats kNN in 69.0% of pairs (mean ΔBA +0.024), ExtraTrees in 65.7% (+0.016), SVM in 58.1% (+0.009), and ties GNB overall (50.0%, +0.004). Its advantage is larger for valence versus kNN (+0.031) but present for both targets. This is robust relative to nearby *configurations*, not proof of a general population effect: all five participants were preselected for promising baseline maxima.

Feature count 10 is the strongest broadly supported count in this follow-up (especially arousal); count 8 is also competitive. Requests 3 and 5 are notably weak for arousal, while 20/30 add little typical performance. Face has the best typical valence BA but only five valid requests per classifier versus eight for EEG/multimodal.

### Feature-family experiment (EEG representations only)

{markdown_table(factor('feature_family_top5', 'valence', 'feature_family'), ['level','n_configurations','median_ba','mean_ba','upper_quartile_ba','maximum_ba','fraction_ba_ge_60'])}

For arousal, `all_eeg` has the best family-level typical BA (mean 0.546); for valence, time-statistical (only 60 rows) and Hjorth are strongest. `paper_compact` is middling for valence (mean 0.534) and weak for arousal (0.513), despite having far more opportunities than several families. That makes its isolated maxima particularly unpersuasive.

`paper_compact` exactly comprises 64 all-channel features: standard deviation; Hjorth mobility and complexity; median frequency; beta and gamma bandpower; beta and gamma spectral entropy, over Fz/C3/Cz/C4/Pz/PO7/Oz/PO8. Its declared subsets contain 32 features each (midline, posterior, frontal-central). It deliberately excludes broad-band spectral entropy and every unlisted feature type.

The central hypothesis is not supported globally. Across matched `paper_compact` cells, LogReg beats SVM in 55.7% (mean Δ +0.005), GNB in 52.1% (+0.004), and kNN in 63.2% (+0.016). Its stronger pockets are participant-specific (notably P17 valence/posterior and P36 arousal/frontal-central), while midline/paper_compact is consistently weak. All-channel and frontal-central `paper_compact` are therefore hypotheses worth a small confirmatory block, not a favored primary family.

### Label ambiguity (matched but changes the dataset)

{markdown_table(midpoint, ['target','n_matched_configurations','mean_delta_ba','median_delta_ba','fraction_logreg_better'])}

Excluding midpoint trials worsens arousal in 56.4% of matched settings (mean Δ −0.010) and helps valence in 60.7% (mean Δ +0.016). This is exploratory because the evaluation set, number of trials, and class balance change simultaneously; it should not replace standard labels without a prespecified target-specific validation.

## Participants and apparent learnability

{markdown_table(top, ['participant','target','median_ba','maximum_ba','fraction_ba_ge_60','optimized_median_ba','optimized_maximum_ba','optimized_fraction_ba_ge_60','eeg_usable_trials','eeg_retention_percent','qc_group'])}

P36 (both targets) and P17 valence have broad above-chance-looking configuration distributions, not just one high maximum; P06 arousal and P22 arousal are moderately promising. P40 is the clearest “isolated maximum” case: maxima of 0.703/0.642 but optimized medians 0.501/0.515 and only 8.6%/6.7% of configurations at BA ≥0.60. It should not drive design decisions.

Across all 46 participants, many target-specific distributions stay near 0.50. This alone does **not** identify noisy EEG: several near-chance participants have 100% retention, complete ratings, and “Very clean” QC, while P36 performs well despite 85% retention. Class composition by target is retained in `fold_results.csv`; the report's participant table includes retained-trial/QC context, but no fold predictions exist for the later studies to establish a mechanistic cause.

## Hyperparameters: what can and cannot be concluded

The saved hyperparameters are the *inner-CV winner per outer fold*, not performance for every candidate grid point. They can support pruning candidates that never win, but cannot establish a causal outer-BA ranking for a hyperparameter. In the optimized study, GNB selected `var_smoothing=1e-11` in every fold; this is a reasonable reduction. For LogReg, selected C values 0.01–1 have slightly better descriptive outer distributions than 10–100, and `class_weight=balanced` is modestly better. For SVM, RBF gamma 1 and the low-C 0.1 region are unproductive in the saved selections; linear or RBF gamma 0.001/0.01 are more defensible. These are pruning hypotheses, not estimates from a separate hyperparameter experiment.

`selected_hyperparameter_summary.csv` supplies the selection count and the same descriptive BA measures for every selected value. The count is outer folds that selected the value, not an independent count of model fits.

## Proposed reduced next experiment

**KEEP**

- Standard labels as the primary analysis; nested CV, fold-local selection, seed 42, and BA as the selection metric.
- LogReg alongside GNB and SVM: LogReg is consistently better than kNN/ExtraTrees and comparable-to-slightly-better than SVM/GNB in matched expanded settings.
- Feature requests 8 and 10 (plus `all` only as a prespecified reference); GNB `var_smoothing=1e-11`; LogReg C `0.01, 0.1, 1` with and without balanced weights; SVM linear plus RBF gamma `0.001, 0.01`, C `1, 10`.
- EEG `all_eeg` and Hjorth as primary EEG family representations. Retain face/multimodal only if the goal remains operational multimodal prediction, because the strongest baseline typical result was face rather than EEG.

**DROP**

- kNN and ExtraTrees from the primary focused search; neither is competitive with LogReg in matched settings.
- Feature count 30, and 3/5 for arousal; they consume search budget without robust benefit.
- `entropy_beta_gamma` and its midline/posterior variants; they are consistently near chance despite many opportunities.
- `paper_compact` midline; posterior is also not a general candidate. Do not use the P17/P36 high cells as global defaults.
- SVM RBF gamma 1 and the broad redundant SVM/LogReg grids unless a prespecified sensitivity block requires them.

**UNCERTAIN**

- `paper_compact × LogReg`: include only a small, explicitly secondary all-channel/frontal-central confirmation block; it is not supported as the main route to higher typical BA.
- Excluding rating-4 trials for valence only: a target-specific secondary label-sensitivity analysis is warranted; arousal evidence argues against it.
- Participant-specific model spaces: promising P36/P17/P06/P22 targets merit confirmation, but were selected after exploration and need a guard against winner’s curse.

Smallest reasonable next test: use the five selected participants but prespecify a confirmatory split/repeated-CV protocol, evaluate standard labels, feature counts 8/10, classifiers LogReg/GNB/SVM, and three representations (`all_eeg`, Hjorth-all-channel, plus secondary `paper_compact` all-channel/frontal-central). Hold one untouched evaluation procedure/seed set or use repeated outer CV; report each participant-target rather than only pooled maxima. Add label-midpoint exclusion only as a valence sensitivity analysis. This is substantially smaller than the previous grids while directly testing the remaining LogReg and paper-compact hypotheses.

For genuine participant-level above-chance claims, run a later, separately budgeted permutation test that repeats the **entire nested pipeline** (including feature selection and inner tuning) with shuffled labels, then compare the observed outer-CV BA to that participant-target null distribution. Permuting only final predictions or tuning on unpermuted labels would be invalid.
"""
    (destination / "PERSONALIZED_BINARY_EXPLORATION_AUDIT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "personalized_binary_exploration_audit_v3")
    args = parser.parse_args()
    frames = {name: pd.read_csv(OUTPUT_ROOT / name / "results_summary.csv") for name in EXPERIMENTS}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    factors = factor_table(frames)
    comparisons = comparison_table(frames)
    participants = participant_table(frames)
    hyperparameters = selected_hyperparameter_table(frames)
    factors.to_csv(args.output_dir / "factor_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "matched_comparisons.csv", index=False)
    participants.to_csv(args.output_dir / "participant_summary.csv", index=False)
    hyperparameters.to_csv(args.output_dir / "selected_hyperparameter_summary.csv", index=False)
    write_report(args.output_dir, factors, comparisons, participants, hyperparameters)
    print(f"Wrote audit to {args.output_dir}")


if __name__ == "__main__":
    main()
