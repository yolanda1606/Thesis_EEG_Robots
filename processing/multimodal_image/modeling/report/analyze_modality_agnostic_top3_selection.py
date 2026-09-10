#!/usr/bin/env python3
"""Summarize final modality-agnostic Image-domain top-3 model selections.

This is a read-only analysis of the final selection table.  It does not load,
refit, alter, or create any model artifact.  It only writes CSV summaries and
PNG figures to a new output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "outputs/image_classification/focused_personalized_binary_v1"
    / "top3_models_per_participant_target_modality.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/image_classification/focused_personalized_binary_v1"
    / "modality_agnostic_top3_analysis"
)
EXPECTED_MODALITIES = ("eeg", "face", "multimodal")
KEY_COLUMNS = ("participant", "target", "transfer_rank")
REQUIRED_COLUMNS = {
    "participant",
    "target",
    "transfer_rank",
    "modality",
    "classifier",
    "feature_family",
    "feature_count_resolved",
    "mean_outer_cv_balanced_accuracy",
    "final_refit_best_parameters",
    "actual_selected_feature_names",
    "final_model_path",
    "selection_scope",
    "model_use",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Authoritative final modality-agnostic top-3 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New directory for CSV summaries and PNG figures; must not exist.",
    )
    return parser.parse_args()


def validate_source(selection: pd.DataFrame, source_csv: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(selection.columns))
    if missing:
        raise ValueError(f"{source_csv}: missing required columns: {missing}")

    if selection.empty:
        raise ValueError(f"{source_csv}: contains no selections")
    if selection[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{source_csv}: null values in selection key {KEY_COLUMNS}")
    if selection.duplicated(list(KEY_COLUMNS)).any():
        duplicates = selection.loc[
            selection.duplicated(list(KEY_COLUMNS), keep=False), list(KEY_COLUMNS)
        ].sort_values(list(KEY_COLUMNS))
        raise ValueError(
            f"{source_csv}: duplicate participant/target/rank selections:\n"
            f"{duplicates.to_string(index=False)}"
        )

    observed_modalities = set(selection["modality"].dropna().astype(str))
    unexpected_modalities = sorted(observed_modalities - set(EXPECTED_MODALITIES))
    missing_modalities = sorted(set(EXPECTED_MODALITIES) - observed_modalities)
    if unexpected_modalities or missing_modalities:
        raise ValueError(
            "Expected exactly EEG, Face, and Multimodal selections; "
            f"observed={sorted(observed_modalities)}, "
            f"unexpected={unexpected_modalities}, missing={missing_modalities}"
        )

    expected_rows = (
        selection["participant"].nunique()
        * selection["target"].nunique()
        * selection["transfer_rank"].nunique()
    )
    if len(selection) != expected_rows:
        raise ValueError(
            f"{source_csv}: expected complete participant × target × rank coverage "
            f"({expected_rows} rows), found {len(selection)} rows"
        )


def frequency_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    counts = (
        frame.groupby(group_columns, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    denominator_columns = [column for column in group_columns if column != "modality"]
    if denominator_columns:
        counts["percentage"] = (
            counts["count"]
            / counts.groupby(denominator_columns, dropna=False)["count"].transform("sum")
            * 100.0
        )
    else:
        counts["percentage"] = counts["count"] / len(frame) * 100.0
    return counts.sort_values(group_columns).reset_index(drop=True)


def save_bar_chart(
    table: pd.DataFrame,
    *,
    category: str,
    value: str,
    title: str,
    output_path: Path,
    hue: str | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    if hue is None:
        ordered = table.sort_values(category)
        axis.bar(ordered[category].astype(str), ordered[value], color="#4C78A8")
    else:
        categories = list(dict.fromkeys(table[category].astype(str)))
        hues = list(dict.fromkeys(table[hue].astype(str)))
        positions = np.arange(len(categories))
        width = 0.8 / max(len(hues), 1)
        for index, hue_value in enumerate(hues):
            subset = table.loc[table[hue].astype(str) == hue_value].copy()
            values = (
                subset.set_index(category)[value]
                .reindex(categories, fill_value=0)
                .to_numpy()
            )
            axis.bar(
                positions - 0.4 + width / 2 + index * width,
                values,
                width=width,
                label=hue_value,
            )
        axis.set_xticks(positions, categories)
        axis.legend(title=hue)
    axis.set_title(title)
    axis.set_ylabel(value.replace("_", " ").title())
    axis.set_xlabel(category.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_heatmap(
    matrix: pd.DataFrame, *, title: str, output_path: Path) -> None:
    fig, axis = plt.subplots(
        figsize=(max(6, 1.2 * len(matrix.columns)), max(4, 0.7 * len(matrix.index)))
    )
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            axis.text(
                column_index,
                row_index,
                str(int(matrix.iat[row_index, column_index])),
                ha="center",
                va="center",
            )
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Top-3 slots")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_ba_distribution(selection: pd.DataFrame, output_path: Path) -> None:
    modalities = [modality for modality in EXPECTED_MODALITIES if modality in set(selection["modality"])]
    values = [
        selection.loc[selection["modality"] == modality, "mean_outer_cv_balanced_accuracy"]
        .dropna()
        .to_numpy()
        for modality in modalities
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=modalities, showmeans=True)
    axis.set_title("Image balanced-accuracy distribution by modality")
    axis.set_xlabel("Modality")
    axis.set_ylabel("Mean outer-CV balanced accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_csv = args.source_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not source_csv.is_file():
        raise FileNotFoundError(f"Authoritative source CSV does not exist: {source_csv}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}. "
            "Choose a new --output-dir."
        )

    selection = pd.read_csv(source_csv)
    validate_source(selection, source_csv)
    selection = selection.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=False)

    selection.to_csv(output_dir / "modality_agnostic_top3_selections.csv", index=False)

    overall_modality = frequency_table(selection, ["modality"])
    by_target_modality = frequency_table(selection, ["target", "modality"])
    by_rank_modality = frequency_table(selection, ["transfer_rank", "modality"])
    by_target_rank_modality = frequency_table(
        selection, ["target", "transfer_rank", "modality"]
    )
    modality_frequency = pd.concat(
        [
            overall_modality.assign(scope="overall"),
            by_target_modality.assign(scope="by_target"),
            by_rank_modality.assign(scope="by_rank"),
            by_target_rank_modality.assign(scope="by_target_and_rank"),
        ],
        ignore_index=True,
        sort=False,
    )
    modality_frequency.to_csv(output_dir / "modality_frequency.csv", index=False)

    classifier_by_modality = (
        selection.groupby(["modality", "classifier"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    classifier_by_modality["percentage_within_modality"] = (
        classifier_by_modality["count"]
        / classifier_by_modality.groupby("modality")["count"].transform("sum")
        * 100.0
    )
    classifier_by_target = (
        selection.groupby(["target", "classifier"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    classifier_by_target["percentage_within_target"] = (
        classifier_by_target["count"]
        / classifier_by_target.groupby("target")["count"].transform("sum")
        * 100.0
    )
    classifier_by_modality.to_csv(output_dir / "classifier_frequency_by_modality.csv", index=False)
    classifier_by_target.to_csv(output_dir / "classifier_frequency_by_target.csv", index=False)

    modality_classifier = pd.crosstab(selection["modality"], selection["classifier"])
    modality_classifier = modality_classifier.reindex(EXPECTED_MODALITIES, fill_value=0)
    modality_classifier.to_csv(output_dir / "modality_by_classifier.csv")

    modality_target = pd.crosstab(selection["modality"], selection["target"])
    modality_target = modality_target.reindex(EXPECTED_MODALITIES, fill_value=0)
    modality_target.to_csv(output_dir / "modality_by_target.csv")

    ba_summary = (
        selection.groupby("modality")["mean_outer_cv_balanced_accuracy"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    ba_summary.to_csv(output_dir / "image_ba_by_modality_summary.csv", index=False)

    diversity = (
        selection.groupby(["participant", "target"])["modality"]
        .agg(lambda values: tuple(sorted(set(values))))
        .rename("modalities")
        .reset_index()
    )
    diversity["unique_modality_count"] = diversity["modalities"].str.len()
    diversity["modality_set"] = diversity["modalities"].map(" + ".join)
    diversity["modality_diversity"] = np.where(
        diversity["unique_modality_count"] > 1,
        "mixed_modalities",
        "only_" + diversity["modalities"].str[0],
    )
    diversity = diversity.drop(columns="modalities").sort_values(["participant", "target"])
    diversity.to_csv(output_dir / "participant_top3_modality_diversity.csv", index=False)
    diversity_counts = (
        diversity.groupby(["target", "modality_diversity"])
        .size()
        .rename("count")
        .reset_index()
    )
    diversity_counts["percentage_within_target"] = (
        diversity_counts["count"]
        / diversity_counts.groupby("target")["count"].transform("sum")
        * 100.0
    )
    diversity_counts.to_csv(output_dir / "participant_top3_modality_diversity_summary.csv", index=False)

    rank1 = selection.loc[selection["transfer_rank"] == 1].copy()
    rank1_modality = frequency_table(rank1, ["target", "modality"])
    rank1_modality.to_csv(output_dir / "rank1_modality_frequency_by_target.csv", index=False)

    save_bar_chart(
        overall_modality,
        category="modality",
        value="count",
        title="Final Image top-3 modality frequency",
        output_path=output_dir / "modality_frequency_overall.png",
    )
    save_bar_chart(
        by_target_modality,
        category="modality",
        value="count",
        hue="target",
        title="Final Image top-3 modality frequency by target",
        output_path=output_dir / "modality_frequency_by_target.png",
    )
    save_bar_chart(
        by_rank_modality,
        category="modality",
        value="count",
        hue="transfer_rank",
        title="Final Image top-3 modality frequency by rank",
        output_path=output_dir / "modality_frequency_by_rank.png",
    )
    save_bar_chart(
        classifier_by_modality,
        category="classifier",
        value="count",
        hue="modality",
        title="Classifier frequency by modality",
        output_path=output_dir / "classifier_frequency_by_modality.png",
    )
    save_bar_chart(
        classifier_by_target,
        category="classifier",
        value="count",
        hue="target",
        title="Classifier frequency by target",
        output_path=output_dir / "classifier_frequency_by_target.png",
    )
    save_heatmap(
        modality_classifier,
        title="Modality × classifier top-3 slots",
        output_path=output_dir / "modality_by_classifier_heatmap.png",
    )
    save_heatmap(
        modality_target,
        title="Modality × target top-3 slots",
        output_path=output_dir / "modality_by_target_heatmap.png",
    )
    save_ba_distribution(selection, output_dir / "image_ba_by_modality.png")
    save_bar_chart(
        diversity_counts,
        category="modality_diversity",
        value="count",
        hue="target",
        title="Participant-level modality diversity across final top 3",
        output_path=output_dir / "participant_top3_modality_diversity.png",
    )
    save_bar_chart(
        rank1_modality,
        category="modality",
        value="count",
        hue="target",
        title="Rank-1 modality frequency by target",
        output_path=output_dir / "rank1_modality_frequency_by_target.png",
    )

    print(f"Read authoritative modality-agnostic selections: {source_csv}")
    print(f"Rows: {len(selection)}")
    print(f"Wrote CSV summaries and PNG figures: {output_dir}")


if __name__ == "__main__":
    main()
