#!/usr/bin/env python3
"""Describe the frozen EEG-only Image top-three models used for Robot transfer.

This is a read-only cohort analysis of the authoritative frozen transfer table.
It does not fit, rank, calibrate, or alter models, Image results, or Robot data.
"""
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
DEFAULT_SOURCE = ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv"
DEFAULT_OUTPUT = ROOT / "outputs/image_classification/focused_personalized_binary_v1/frozen_top3_analysis"
TARGET_ORDER = ["valence", "arousal"]
RANK_ORDER = [1, 2, 3]
REQUIRED_COLUMNS = {
    "participant", "target", "classifier", "feature_family", "feature_count_request", "feature_count_resolved",
    "mean_outer_cv_balanced_accuracy", "transfer_rank", "final_refit_best_parameters",
    "actual_selected_feature_names", "final_model_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_source(path: Path) -> pd.DataFrame:
    """Load and validate the immutable frozen EEG-only top-three table."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required frozen top-three columns: {missing}")
    if frame.duplicated(["participant", "target", "transfer_rank"]).any():
        raise ValueError("Frozen top-three table has duplicate participant/target/rank rows")
    if set(frame["target"].unique()).difference(TARGET_ORDER):
        raise ValueError("Frozen top-three table contains an unsupported target")
    if set(frame["transfer_rank"].unique()).difference(RANK_ORDER):
        raise ValueError("Frozen top-three table contains an unsupported transfer rank")
    feature_lists = frame["actual_selected_feature_names"].map(parse_feature_list)
    if feature_lists.map(lambda values: any(not value.startswith("eeg_") for value in values)).any():
        raise ValueError("Source is not EEG-only: a selected feature lacks the eeg_ prefix")
    frame = frame.copy()
    frame["selected_features"] = feature_lists
    frame["hyperparameters"] = frame["final_refit_best_parameters"].map(parse_hyperparameters)
    return frame


def parse_feature_list(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid selected-feature JSON: {value!r}") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("Selected features must be a JSON string list")
    return parsed


def parse_hyperparameters(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid frozen hyperparameter JSON: {value!r}") from error
    if not isinstance(parsed, dict):
        raise ValueError("Frozen hyperparameters must be a JSON object")
    return parsed


def frequency_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count a selected configuration overall, by target, and by rank."""
    rows: list[pd.DataFrame] = []
    for scope, group_columns in [("overall", []), ("target", ["target"]), ("rank", ["transfer_rank"]),
                                 ("target_rank", ["target", "transfer_rank"])]:
        counts = frame.groupby([*group_columns, column], dropna=False).size().rename("count").reset_index()
        counts["scope"] = scope
        totals = counts.groupby(group_columns, dropna=False)["count"].transform("sum") if group_columns else counts["count"].sum()
        counts["fraction"] = counts["count"] / totals
        rows.append(counts)
    return pd.concat(rows, ignore_index=True).sort_values(["scope", *[name for name in ["target", "transfer_rank", column] if name in frame]], kind="stable")


def hyperparameter_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in frame.itertuples(index=False):
        for parameter, value in item.hyperparameters.items():
            rows.append({"classifier": item.classifier, "parameter": parameter.replace("classifier__", ""),
                         "value": json.dumps(value, sort_keys=True), "target": item.target,
                         "transfer_rank": item.transfer_rank})
    long = pd.DataFrame(rows)
    outputs: list[pd.DataFrame] = []
    for scope, group_columns in [("overall", ["classifier", "parameter", "value"]),
                                 ("target", ["target", "classifier", "parameter", "value"])]:
        counts = long.groupby(group_columns).size().rename("count").reset_index()
        denominator = counts.groupby([name for name in group_columns if name != "value"])["count"].transform("sum")
        counts["fraction_within_classifier_parameter"] = counts["count"] / denominator
        counts["scope"] = scope
        outputs.append(counts)
    return pd.concat(outputs, ignore_index=True).sort_values(["scope", "classifier", "parameter", "count"], ascending=[True, True, True, False], kind="stable")


def feature_frequency(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, str]] = []
    for item in frame.itertuples(index=False):
        for feature in item.selected_features:
            descriptor, channel = feature.rsplit("__", maxsplit=1)
            rows.append({"target": item.target, "feature": feature, "descriptor": descriptor, "channel": channel})
    long = pd.DataFrame(rows)

    def count(column: str) -> pd.DataFrame:
        outputs: list[pd.DataFrame] = []
        for scope, groups in [("overall", [column]), ("target", ["target", column])]:
            values = long.groupby(groups).size().rename("count").reset_index()
            denominator = values.groupby([name for name in groups if name != column])["count"].transform("sum") if len(groups) > 1 else values["count"].sum()
            values["fraction_of_selected_features"] = values["count"] / denominator
            values["scope"] = scope
            outputs.append(values)
        return pd.concat(outputs, ignore_index=True).sort_values(["scope", "count", column], ascending=[True, False, True], kind="stable")

    return count("feature"), count("descriptor"), count("channel")


def diversity(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["participant", "target"], as_index=False).agg(
        n_top3_rows=("transfer_rank", "size"),
        unique_classifier_families=("classifier", "nunique"),
        unique_feature_families=("feature_family", "nunique"),
        classifier_families=("classifier", lambda values: ";".join(sorted(set(values)))),
        feature_families=("feature_family", lambda values: ";".join(sorted(set(values)))),
    ).sort_values(["participant", "target"])


def save_bar(values: pd.Series, title: str, ylabel: str, path: Path, horizontal: bool = False) -> None:
    fig, axis = plt.subplots(figsize=(8, max(3.5, 0.42 * len(values))))
    if horizontal:
        axis.barh(values.index.astype(str), values.to_numpy(), color="#35689a")
        axis.invert_yaxis()
        axis.set_xlabel(ylabel)
    else:
        axis.bar(values.index.astype(str), values.to_numpy(), color="#35689a")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_target_bar(frame: pd.DataFrame, category: str, title: str, path: Path) -> None:
    pivot = frame.groupby(["target", category]).size().unstack("target", fill_value=0).reindex(columns=TARGET_ORDER, fill_value=0)
    axis = pivot.plot(kind="bar", figsize=(8, 4.5), color=["#4477aa", "#cc6677"])
    axis.set_title(title); axis.set_ylabel("Frozen top-3 slots"); axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=25); axis.legend(title="Target")
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()


def save_heatmap(frame: pd.DataFrame, target: str, path: Path) -> None:
    matrix = pd.crosstab(frame.loc[frame.target.eq(target), "classifier"], frame.loc[frame.target.eq(target), "feature_family"])
    fig, axis = plt.subplots(figsize=(max(5, 1.1 * len(matrix.columns)), max(3.5, 0.75 * len(matrix.index))))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix.iloc[row, column])), ha="center", va="center")
    axis.set_title(f"Frozen classifier × feature family: {target}")
    fig.colorbar(image, ax=axis, label="Selected slots")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def save_ba_by_classifier(frame: pd.DataFrame, path: Path) -> None:
    labels, values, colors = [], [], []
    palette = {"valence": "#4477aa", "arousal": "#cc6677"}
    for target in TARGET_ORDER:
        for classifier in sorted(frame.classifier.unique()):
            subset = frame.loc[(frame.target.eq(target)) & (frame.classifier.eq(classifier)), "mean_outer_cv_balanced_accuracy"]
            if len(subset):
                labels.append(f"{target}\n{classifier}"); values.append(subset.to_numpy()); colors.append(palette[target])
    fig, axis = plt.subplots(figsize=(max(7, 0.9 * len(labels)), 4.5))
    boxes = axis.boxplot(values, patch_artist=True, labels=labels, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors): patch.set_facecolor(color)
    axis.set_title("Frozen Image balanced accuracy by classifier family")
    axis.set_ylabel("Outer-CV balanced accuracy"); axis.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def make_figures(frame: pd.DataFrame, feature_counts: pd.DataFrame, feature_frequency_table: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir()
    save_bar(frame.classifier.value_counts(), "Frozen classifier-family frequency", "Frozen top-3 slots", figures / "classifier_frequency.png")
    save_target_bar(frame, "classifier", "Frozen classifier-family frequency by target", figures / "classifier_frequency_by_target.png")
    save_target_bar(frame, "feature_family", "Frozen feature-family frequency by target", figures / "feature_family_frequency_by_target.png")
    for target in TARGET_ORDER:
        save_heatmap(frame, target, figures / f"classifier_feature_family_heatmap_{target}.png")
    save_target_bar(frame, "feature_count_resolved", "Resolved selected-feature count by target", figures / "feature_count_distribution.png")
    rank = frame.groupby(["transfer_rank", "classifier"]).size().unstack("classifier", fill_value=0).reindex(RANK_ORDER)
    rank.plot(kind="bar", figsize=(7, 4.5)); plt.title("Frozen classifier-family composition by rank"); plt.xlabel("Transfer rank"); plt.ylabel("Frozen top-3 slots"); plt.tight_layout(); plt.savefig(figures / "classifier_by_rank.png", dpi=170); plt.close()
    save_ba_by_classifier(frame, figures / "image_ba_by_classifier.png")
    for target in TARGET_ORDER:
        top = feature_frequency_table[(feature_frequency_table.scope.eq("target")) & (feature_frequency_table.target.eq(target))].nlargest(20, "count")
        save_bar(top.set_index("feature")["count"], f"Top selected EEG features: {target}", "Frozen top-3 selections", figures / f"selected_feature_frequency_{target}.png", horizontal=True)


def markdown_summary(frame: pd.DataFrame, classifier_frequency: pd.DataFrame, feature_frequency_table: pd.DataFrame,
                     diversity_table: pd.DataFrame) -> str:
    classifier = classifier_frequency[classifier_frequency.scope.eq("overall")][["classifier", "count", "fraction"]]
    top_features = feature_frequency_table[feature_frequency_table.scope.eq("overall")].nlargest(20, "count")[["feature", "count"]]
    diversity_counts = diversity_table.groupby(["unique_classifier_families", "unique_feature_families"]).size().rename("participant_target_count").reset_index()
    def table(values: pd.DataFrame) -> str:
        return values.to_markdown(index=False, floatfmt=".3f")
    return "\n".join([
        "# Frozen EEG-only top-3 selection analysis", "",
        f"Authoritative source: `{DEFAULT_SOURCE.relative_to(ROOT)}`.", "",
        f"Rows: {len(frame)}; participants: {frame.participant.nunique()}; targets: {', '.join(sorted(frame.target.unique()))}; ranks: 1–3.", "",
        "The source table is the frozen EEG-only transfer selection. This report is descriptive and does not perform model selection, fitting, calibration, or Robot analysis.", "",
        "## Classifier-family frequency", "", table(classifier), "",
        "## Top 20 selected EEG features", "", table(top_features), "",
        "## Participant × target top-3 diversity", "", table(diversity_counts), "",
    ])


def main() -> int:
    args = parse_args()
    source, output = args.source_csv.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis directory: {output}")
    frame = load_source(source)
    classifier_counts = frequency_table(frame, "classifier")
    family_counts = frequency_table(frame, "feature_family")
    feature_count_counts = frequency_table(frame, "feature_count_resolved")
    combinations = frame.groupby(["target", "classifier", "feature_family", "transfer_rank"]).size().rename("count").reset_index()
    hyperparameters = hyperparameter_frequency(frame)
    selected_features, descriptors, channels = feature_frequency(frame)
    diversity_table = diversity(frame)
    ba_groups = frame.groupby(["target", "classifier", "feature_family"], as_index=False).agg(
        n_models=("participant", "size"), mean_image_ba=("mean_outer_cv_balanced_accuracy", "mean"),
        median_image_ba=("mean_outer_cv_balanced_accuracy", "median"), min_image_ba=("mean_outer_cv_balanced_accuracy", "min"),
        max_image_ba=("mean_outer_cv_balanced_accuracy", "max"),
    )
    output.mkdir(parents=True)
    frame.drop(columns=["selected_features", "hyperparameters"]).to_csv(output / "frozen_top3_summary.csv", index=False)
    classifier_counts.to_csv(output / "classifier_frequency.csv", index=False)
    family_counts.to_csv(output / "feature_family_frequency.csv", index=False)
    combinations.to_csv(output / "classifier_feature_family_frequency.csv", index=False)
    feature_count_counts.to_csv(output / "feature_count_frequency.csv", index=False)
    hyperparameters.to_csv(output / "hyperparameter_frequency.csv", index=False)
    selected_features.to_csv(output / "selected_feature_frequency.csv", index=False)
    descriptors.to_csv(output / "selected_feature_descriptor_frequency.csv", index=False)
    channels.to_csv(output / "selected_feature_channel_frequency.csv", index=False)
    diversity_table.to_csv(output / "participant_top3_diversity.csv", index=False)
    ba_groups.to_csv(output / "image_ba_group_summary.csv", index=False)
    make_figures(frame, feature_count_counts, selected_features, output)
    (output / "frozen_top3_analysis.md").write_text(
        markdown_summary(frame, classifier_counts, selected_features, diversity_table), encoding="utf-8"
    )
    print(f"Wrote frozen EEG-only top-3 analysis: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}")
