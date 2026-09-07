#!/usr/bin/env python3
"""Read-only diagnostic of focused personalized Image-classifier results.

The script intentionally performs no fitting, resampling, threshold selection,
or model inference.  It combines frozen outer-CV artifacts with the existing
no-ICA merged Image tables to describe learnability signals per participant and
target.  The resulting categories are triage labels, not exclusions or model
selection rules.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

import train_classification as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = PROJECT_ROOT / "outputs/image_classification/focused_personalized_binary_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/image_classification"
ALL_EEG_COLUMNS = list(baseline.EEG_COLUMNS)
WEAK_BA = 0.55
ROBUST_BA = 0.65
SEVERE_IMBALANCE_FRACTION = 0.30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def configuration_columns() -> list[str]:
    """Return columns uniquely identifying one frozen candidate configuration."""
    return [
        "participant", "target", "classifier", "feature_family", "channel_subset",
        "feature_count_request", "analysis_tier", "available_feature_count",
    ]


def normalized_configuration_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Avoid CSV type inference changing feature-count configuration identities."""
    frame = frame.copy()
    frame["feature_count_request"] = frame["feature_count_request"].astype(str)
    return frame


def choose_best_configurations(results: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen BA, fold-SD tie-break to choose each descriptive best."""
    ordered = results.sort_values(
        ["participant", "target", "mean_outer_cv_balanced_accuracy", "fold_ba_sd"],
        ascending=[True, True, False, True], kind="stable",
    )
    return ordered.groupby(["participant", "target"], as_index=False).first()


def candidate_and_representation_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Describe candidate distributions and family/subset consistency."""
    keys = ["participant", "target"]
    candidate = results.groupby(keys)["mean_outer_cv_balanced_accuracy"].agg(
        median_candidate_ba="median", mean_candidate_ba="mean",
    ).reset_index()
    representation = results.groupby(keys + ["feature_family", "channel_subset"], as_index=False)[
        "mean_outer_cv_balanced_accuracy"
    ].median().rename(columns={"mean_outer_cv_balanced_accuracy": "representation_median_ba"})
    rows: list[dict[str, Any]] = []
    for values, group in representation.groupby(keys, sort=True):
        ranked = group.sort_values("representation_median_ba", ascending=False, kind="stable").reset_index(drop=True)
        first, second = ranked.iloc[0], ranked.iloc[1]
        rows.append({
            "participant": values[0], "target": values[1],
            "best_representation": f"{first.feature_family}|{first.channel_subset}",
            "best_representation_median_ba": first.representation_median_ba,
            "second_representation": f"{second.feature_family}|{second.channel_subset}",
            "second_representation_median_ba": second.representation_median_ba,
            "representation_median_spread_ba": first.representation_median_ba - second.representation_median_ba,
        })
    classifier = results.groupby(keys + ["classifier"], as_index=False)[
        "mean_outer_cv_balanced_accuracy"
    ].median().rename(columns={"mean_outer_cv_balanced_accuracy": "classifier_median_ba"})
    classifier_rows: list[dict[str, Any]] = []
    for values, group in classifier.groupby(keys, sort=True):
        ranked = group.sort_values("classifier_median_ba", ascending=False, kind="stable").reset_index(drop=True)
        classifier_rows.append({
            "participant": values[0], "target": values[1],
            "best_classifier_by_candidate_median": ranked.iloc[0].classifier,
            "best_classifier_candidate_median_ba": ranked.iloc[0].classifier_median_ba,
            "classifier_median_spread_ba": ranked.iloc[0].classifier_median_ba - ranked.iloc[-1].classifier_median_ba,
        })
    return candidate.merge(pd.DataFrame(rows), on=keys).merge(pd.DataFrame(classifier_rows), on=keys)


def best_prediction_metrics(predictions: pd.DataFrame, best: pd.Series) -> dict[str, float]:
    """Compute held-out sensitivity/specificity from the saved outer predictions."""
    mask = np.ones(len(predictions), dtype=bool)
    for column in configuration_columns():
        mask &= predictions[column].astype(str).to_numpy() == str(best[column])
    frame = predictions.loc[mask]
    if len(frame) != int(best.sample_size):
        raise ValueError(f"Unexpected held-out prediction count for {best.participant} {best.target}: {len(frame)}")
    truth = frame["true_label"].to_numpy(dtype=int)
    predicted = frame["predicted_label"].to_numpy(dtype=int)
    sensitivity = float((predicted[truth == 1] == 1).mean())
    specificity = float((predicted[truth == 0] == 0).mean())
    return {
        "oof_sensitivity": sensitivity,
        "oof_specificity": specificity,
        "oof_sensitivity_specificity_gap": sensitivity - specificity,
        "oof_absolute_sensitivity_specificity_gap": abs(sensitivity - specificity),
    }


def feature_stability(features: pd.DataFrame, best: pd.Series) -> dict[str, Any]:
    """Summarize selected-feature agreement for the best frozen configuration."""
    mask = np.ones(len(features), dtype=bool)
    for column in configuration_columns():
        mask &= features[column].astype(str).to_numpy() == str(best[column])
    selected = features.loc[mask & features["selected"].astype(bool), ["fold", "feature"]]
    sets = [set(group.feature) for _, group in selected.groupby("fold", sort=True)]
    jaccard = [len(left & right) / len(left | right) for left, right in combinations(sets, 2) if left | right]
    frequency = selected.groupby("feature").size().sort_values(ascending=False)
    outer_folds = int(best.outer_folds)
    stable = frequency.loc[frequency >= np.ceil(0.8 * outer_folds)]
    return {
        "mean_pairwise_selected_feature_jaccard": float(np.mean(jaccard)) if jaccard else np.nan,
        "n_features_selected_in_at_least_80pct_folds": int(len(stable)),
        "stable_selected_feature_names": ";".join(stable.index.tolist()),
    }


def separation_and_overlap(table: pd.DataFrame, target: str) -> dict[str, Any]:
    """Calculate descriptive all-EEG class separation without fitting a model."""
    ratings = pd.to_numeric(table[baseline.TARGET_COLUMNS[target]], errors="coerce")
    numeric = table[ALL_EEG_COLUMNS].apply(pd.to_numeric, errors="coerce")
    usable = ratings.notna() & numeric.notna().all(axis=1)
    x = numeric.loc[usable].to_numpy(dtype=float)
    y = (ratings.loc[usable].to_numpy() >= 4).astype(int)
    low, high = x[y == 0], x[y == 1]
    low_count, high_count = len(low), len(high)
    pooled_sd = np.sqrt(((low_count - 1) * low.var(axis=0, ddof=1) + (high_count - 1) * high.var(axis=0, ddof=1)) / (len(x) - 2))
    cohen_d = (high.mean(axis=0) - low.mean(axis=0)) / np.where(pooled_sd > 1e-12, pooled_sd, np.nan)
    ranks = np.apply_along_axis(rankdata, 0, x)
    high_rank_sum = ranks[y == 1].sum(axis=0)
    auc = (high_rank_sum - high_count * (high_count + 1) / 2) / (low_count * high_count)
    rank_biserial = 2 * auc - 1
    top_index = int(np.nanargmax(np.abs(rank_biserial)))

    # Standardization is descriptive and is calculated separately per
    # participant-target.  The ratio is a geometry summary, not a classifier.
    overall_sd = x.std(axis=0, ddof=1)
    z = (x - x.mean(axis=0)) / np.where(overall_sd > 1e-12, overall_sd, 1.0)
    centroid_distance = float(np.linalg.norm(z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0)))
    within = float((pdist(z[y == 0]).mean() + pdist(z[y == 1]).mean()) / 2)
    return {
        "retained_trials": int(len(x)), "low_count": int(low_count), "high_count": int(high_count),
        "minority_class_count": int(min(low_count, high_count)),
        "minority_class_fraction": float(min(low_count, high_count) / len(x)),
        "median_absolute_cohen_d_all_eeg": float(np.nanmedian(np.abs(cohen_d))),
        "maximum_absolute_cohen_d_all_eeg": float(np.nanmax(np.abs(cohen_d))),
        "maximum_absolute_rank_biserial_all_eeg": float(np.nanmax(np.abs(rank_biserial))),
        "top_rank_biserial_feature": ALL_EEG_COLUMNS[top_index],
        "top_rank_biserial_effect": float(rank_biserial[top_index]),
        "standardized_centroid_distance_all_eeg": centroid_distance,
        "mean_within_class_distance_all_eeg": within,
        "centroid_to_within_distance_ratio_all_eeg": centroid_distance / within,
    }


def classify_diagnostic(row: pd.Series, separation_ratio_q1: float) -> tuple[str, str]:
    """Return transparent, deliberately conservative descriptive triage labels."""
    if row.best_ba >= ROBUST_BA:
        return "unclear", "robust reference; no weakness mechanism assigned"
    if row.best_ba >= WEAK_BA:
        return "unclear", "not in focused weak stratum (<0.55 BA)"
    weak_separation = (
        row.maximum_absolute_rank_biserial_all_eeg < 0.30
        and row.centroid_to_within_distance_ratio_all_eeg <= separation_ratio_q1
    )
    representation_pattern = (
        row.representation_median_spread_ba >= 0.03
        and row.best_representation_median_ba >= 0.55
    )
    threshold_pattern = (
        row.severe_class_imbalance
        and row.oof_absolute_sensitivity_specificity_gap >= 0.15
    )
    if weak_separation:
        return "likely weak/no separable signal", "low all-EEG rank separation and low centroid-to-within ratio"
    if representation_pattern:
        return "possible representation limitation", "one representation has a materially higher candidate-median BA"
    if threshold_pattern:
        return "possible classifier/threshold limitation", "severe class imbalance plus asymmetric held-out sensitivity/specificity"
    return "unclear", "weak BA without a sufficiently specific descriptive pattern"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table with three-decimal floats."""
    display = frame.loc[:, columns].copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    return "\n".join([
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)],
    ])


def write_report(destination: Path, summary: pd.DataFrame) -> None:
    """Write a concise interpretation without implying causal diagnosis."""
    weak = summary.loc[summary.best_ba < WEAK_BA].sort_values(["target", "best_ba"])
    robust = summary.loc[summary.best_ba >= ROBUST_BA].sort_values(["target", "best_ba"], ascending=[True, False])
    group_rows = []
    for target, target_frame in summary.groupby("target", sort=True):
        for name, mask in (("weak (<0.55)", target_frame.best_ba < WEAK_BA), ("robust (>=0.65)", target_frame.best_ba >= ROBUST_BA)):
            subset = target_frame.loc[mask]
            group_rows.append({"target": target, "stratum": name, "n": len(subset), "median_best_ba": subset.best_ba.median(),
                               "median_candidate_ba": subset.median_candidate_ba.median(), "median_fold_sd": subset.fold_ba_sd.median(),
                               "median_minor_class_fraction": subset.minority_class_fraction.median(),
                               "median_rank_biserial_max": subset.maximum_absolute_rank_biserial_all_eeg.median(),
                               "median_centroid_ratio": subset.centroid_to_within_distance_ratio_all_eeg.median()})
    severe = summary.loc[summary.severe_class_imbalance].sort_values(["target", "participant"])
    text = f"""# Focused personalized binary Image search: weak-participant diagnostic

This is a read-only descriptive analysis of the completed focused-search CSVs and the existing no-ICA Image merged tables. It fits no models, performs no threshold selection, and changes no data, rankings, model outputs, or serialized pipelines.

## Definitions and limits

- Weak and robust are descriptive strata defined from the frozen participant-best outer-CV BA: weak `<0.55`; robust `>=0.65`.
- Sensitivity and specificity are aggregated held-out predictions of each participant-target's frozen best configuration. The completed run does not retain probabilities, so calibration and an optimal threshold cannot be determined here.
- Selected-feature stability is the mean pairwise Jaccard similarity of the five outer-fold selected feature sets for the frozen best configuration.
- LOW/HIGH separation uses the existing 120 `all_eeg` features only: pooled standardized mean differences, rank-biserial effects, and a standardized class-centroid-to-within-class-distance ratio. These are descriptive geometry measures, not fitted classifiers or inferential tests.
- Triage labels are intentionally conservative and do not identify exclusions, modify training, or establish causes.

## Weak versus robust strata

{markdown_table(pd.DataFrame(group_rows), ['target','stratum','n','median_best_ba','median_candidate_ba','median_fold_sd','median_minor_class_fraction','median_rank_biserial_max','median_centroid_ratio'])}

Weak cases generally show lower univariate and multivariate separation than robust references, but overlap remains substantial. Accordingly, low BA alone is not treated as evidence of a recording problem or of a threshold remedy.

## Focused weak cases

{markdown_table(weak, ['participant','target','best_ba','median_candidate_ba','fold_ba_sd','fold_ba_range','minority_class_fraction','oof_sensitivity','oof_specificity','mean_pairwise_selected_feature_jaccard','maximum_absolute_rank_biserial_all_eeg','centroid_to_within_distance_ratio_all_eeg','diagnostic_classification'])}

## Robust references

{markdown_table(robust, ['participant','target','best_ba','fold_ba_sd','minority_class_fraction','maximum_absolute_rank_biserial_all_eeg','centroid_to_within_distance_ratio_all_eeg'])}

## Imbalance flags

Severe imbalance is defined before interpretation as a minority-class fraction below 0.30. Threshold optimization is only marked *plausible* when severe imbalance coincides with an absolute held-out sensitivity/specificity gap of at least 0.15; it remains a hypothesis requiring leakage-safe inner-CV evaluation.

{markdown_table(severe, ['participant','target','low_count','high_count','minority_class_fraction','oof_sensitivity','oof_specificity','oof_absolute_sensitivity_specificity_gap','threshold_optimization_plausible'])}

P03, P13, and P28 remain included. Their individual results appear in the CSV and should be compared in a later pre-specified QC sensitivity analysis rather than used for automatic exclusion.

The full 92-row, machine-readable diagnostic is `focused_personalized_weak_participant_diagnostic.csv` beside this report.
"""
    destination.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = ["complete_search_results.csv", "fold_results.csv", "predictions_by_fold.csv", "selected_features_by_fold.csv", "run_manifest.json"]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing focused-search artifacts: {missing}")
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    results = normalized_configuration_frame(pd.read_csv(results_dir / "complete_search_results.csv"))
    folds = normalized_configuration_frame(pd.read_csv(results_dir / "fold_results.csv"))
    predictions = normalized_configuration_frame(pd.read_csv(results_dir / "predictions_by_fold.csv"))
    features = normalized_configuration_frame(pd.read_csv(results_dir / "selected_features_by_fold.csv"))
    best = choose_best_configurations(results)
    # Reduce the large fold artifacts to the 92 frozen best configurations once.
    # Repeatedly scanning all 622k feature rows for every participant-target is
    # unnecessary and can make this otherwise read-only diagnostic needlessly slow.
    best_keys = best.loc[:, configuration_columns()].copy()
    folds = folds.merge(best_keys, on=configuration_columns(), how="inner", validate="many_to_one")
    predictions = predictions.merge(best_keys, on=configuration_columns(), how="inner", validate="many_to_one")
    features = features.merge(best_keys, on=configuration_columns(), how="inner", validate="many_to_one")
    candidate_summary = candidate_and_representation_summary(results)
    output_rows: list[dict[str, Any]] = []
    for best_row in best.itertuples(index=False):
        best_series = pd.Series(best_row._asdict())
        participant_table = pd.read_csv(manifest["source_merged_tables"][best_series.participant])
        fold_mask = np.ones(len(folds), dtype=bool)
        for column in configuration_columns():
            fold_mask &= folds[column].astype(str).to_numpy() == str(best_series[column])
        fold_values = folds.loc[fold_mask, "balanced_accuracy"]
        row = {
            "participant": best_series.participant, "target": best_series.target,
            "best_classifier": best_series.classifier, "best_feature_family": best_series.feature_family,
            "best_channel_subset": best_series.channel_subset, "best_feature_count_request": best_series.feature_count_request,
            "best_ba": float(best_series.mean_outer_cv_balanced_accuracy), "fold_ba_sd": float(best_series.fold_ba_sd),
            "fold_ba_min": float(fold_values.min()), "fold_ba_max": float(fold_values.max()),
            "fold_ba_range": float(fold_values.max() - fold_values.min()),
            **best_prediction_metrics(predictions, best_series), **feature_stability(features, best_series),
            **separation_and_overlap(participant_table, best_series.target),
        }
        output_rows.append(row)
    diagnostic = pd.DataFrame(output_rows).merge(candidate_summary, on=["participant", "target"], how="left")
    diagnostic["severe_class_imbalance"] = diagnostic.minority_class_fraction < SEVERE_IMBALANCE_FRACTION
    diagnostic["threshold_optimization_plausible"] = diagnostic.severe_class_imbalance & (diagnostic.oof_absolute_sensitivity_specificity_gap >= 0.15)
    ratio_q1 = diagnostic.centroid_to_within_distance_ratio_all_eeg.quantile(0.25)
    labels = diagnostic.apply(lambda row: classify_diagnostic(row, ratio_q1), axis=1)
    diagnostic[["diagnostic_classification", "diagnostic_rationale"]] = pd.DataFrame(labels.tolist(), index=diagnostic.index)
    diagnostic = diagnostic.sort_values(["target", "participant"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "focused_personalized_weak_participant_diagnostic.csv"
    report_path = output_dir / "FOCUSED_PERSONALIZED_WEAK_PARTICIPANT_DIAGNOSTIC.md"
    diagnostic.to_csv(csv_path, index=False)
    write_report(report_path, diagnostic)
    print(f"Wrote {csv_path} ({len(diagnostic)} participant-target rows)")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
