"""Compare P01 trial ratings with matched OASIS normative mean ratings.

The comparison is descriptive. OASIS values are normative references, not
ground-truth labels or model targets for the participant.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_FILENAMES = (
    "rating_comparison.csv",
    "rating_comparison_report.md",
    "plots/p01_vs_oasis_valence.png",
    "plots/p01_vs_oasis_arousal.png",
)


def parse_arguments() -> argparse.Namespace:
    """Parse explicit input and output paths for one rating-comparison run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="Merged P01 trial CSV.")
    parser.add_argument("--oasis", type=Path, required=True, help="OASIS selected-images CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for final comparison artifacts.")
    return parser.parse_args()


def normalize_stimulus_name(value: object) -> str:
    """Normalize a P01 image filename or OASIS theme for verified matching."""
    text = str(value).strip().lower()
    if text.endswith(".jpg"):
        text = text[:-4]
    return re.sub(r"[^a-z0-9]+", "", text)


def read_and_match_trials(trials_path: Path, oasis_path: Path) -> pd.DataFrame:
    """Return the P01 table left-joined to OASIS by normalized image/theme name."""
    trials = pd.read_csv(trials_path)
    oasis = pd.read_csv(oasis_path, skiprows=2)
    oasis = oasis.rename(columns={oasis.columns[0]: "oasis_id"})

    required_trials = {"trigger", "stim_id", "category", "valence_rating", "arousal_rating"}
    required_oasis = {"oasis_id", "Theme", "Category", "Valence_mean", "Arousal_mean"}
    missing_trials = sorted(required_trials.difference(trials.columns))
    missing_oasis = sorted(required_oasis.difference(oasis.columns))
    if missing_trials:
        raise ValueError(f"Merged trial table is missing columns: {missing_trials}")
    if missing_oasis:
        raise ValueError(f"OASIS table is missing columns: {missing_oasis}")

    trials = trials.loc[:, ["trigger", "stim_id", "category", "valence_rating", "arousal_rating"]].copy()
    oasis = oasis.loc[:, ["oasis_id", "Theme", "Category", "Valence_mean", "Arousal_mean"]].copy()
    trials["normalized_stimulus"] = trials["stim_id"].map(normalize_stimulus_name)
    oasis["normalized_stimulus"] = oasis["Theme"].map(normalize_stimulus_name)

    if trials["normalized_stimulus"].duplicated().any():
        raise ValueError("Normalized P01 stimulus names are not unique.")
    if oasis["normalized_stimulus"].duplicated().any():
        raise ValueError("Normalized OASIS Theme values are not unique.")

    matched = trials.merge(
        oasis,
        on="normalized_stimulus",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_p01", "_oasis"),
    )
    matched["join_status"] = matched.pop("_merge").map({"both": "matched", "left_only": "unmatched"})
    matched = matched.rename(
        columns={
            "category": "p01_category",
            "Theme": "oasis_theme",
            "Category": "oasis_category",
            "Valence_mean": "oasis_valence_mean",
            "Arousal_mean": "oasis_arousal_mean",
        }
    )
    matched["valence_difference_p01_minus_oasis"] = matched["valence_rating"] - matched["oasis_valence_mean"]
    matched["arousal_difference_p01_minus_oasis"] = matched["arousal_rating"] - matched["oasis_arousal_mean"]
    return matched


def summary_statistics(frame: pd.DataFrame, participant_column: str, oasis_column: str) -> dict[str, float | int]:
    """Calculate paired descriptive statistics for one rating dimension."""
    paired = frame.loc[
        (frame["join_status"] == "matched") & frame[participant_column].notna() & frame[oasis_column].notna(),
        [participant_column, oasis_column],
    ]
    participant = paired[participant_column].astype(float)
    normative = paired[oasis_column].astype(float)
    difference = participant - normative
    pearson_r, pearson_p = pearsonr(normative, participant)
    spearman_rho, spearman_p = spearmanr(normative, participant)
    return {
        "matched_nonmissing_trials": int(len(paired)),
        "p01_mean": float(participant.mean()),
        "p01_sd": float(participant.std(ddof=1)),
        "oasis_mean": float(normative.mean()),
        "oasis_sd": float(normative.std(ddof=1)),
        "mean_signed_difference_p01_minus_oasis": float(difference.mean()),
        "mean_absolute_difference": float(difference.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
    }


def write_scatter_plot(frame: pd.DataFrame, participant_column: str, oasis_column: str, title: str, output_path: Path) -> None:
    """Write one P01-versus-OASIS scatter plot with an equal-rating line."""
    paired = frame.loc[
        (frame["join_status"] == "matched") & frame[participant_column].notna() & frame[oasis_column].notna(),
        [participant_column, oasis_column],
    ]
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.scatter(paired[oasis_column], paired[participant_column], color="#1f77b4", alpha=0.8, edgecolors="white", linewidths=0.5)
    axis.plot([1, 7], [1, 7], color="#555555", linestyle="--", linewidth=1.25, label="Equal ratings")
    axis.set(xlim=(1, 7), ylim=(1, 7), xlabel="OASIS normative mean rating", ylabel="P01 rating", title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_statistics_table(statistics: dict[str, dict[str, float | int]]) -> str:
    """Format valence and arousal statistics as a Markdown table."""
    labels = [
        ("Matched, non-missing trials", "matched_nonmissing_trials", "d"),
        ("P01 mean", "p01_mean", ".3f"),
        ("P01 standard deviation", "p01_sd", ".3f"),
        ("OASIS mean", "oasis_mean", ".3f"),
        ("OASIS standard deviation", "oasis_sd", ".3f"),
        ("Mean signed difference (P01 − OASIS)", "mean_signed_difference_p01_minus_oasis", ".3f"),
        ("Mean absolute difference", "mean_absolute_difference", ".3f"),
        ("RMSE", "rmse", ".3f"),
        ("Pearson r", "pearson_r", ".3f"),
        ("Pearson p-value", "pearson_p_value", ".4f"),
        ("Spearman rho", "spearman_rho", ".3f"),
        ("Spearman p-value", "spearman_p_value", ".4f"),
    ]
    rows = ["| Metric | Valence | Arousal |", "|---|---:|---:|"]
    for label, key, specification in labels:
        rows.append(f"| {label} | {statistics['valence'][key]:{specification}} | {statistics['arousal'][key]:{specification}} |")
    return "\n".join(rows)


def write_report(output_path: Path, matched: pd.DataFrame, statistics: dict[str, dict[str, float | int]]) -> None:
    """Write a concise interpretation and reproducible summary of the comparison."""
    unmatched = matched.loc[matched["join_status"] == "unmatched", ["trigger", "stim_id"]]
    unmatched_text = "\n".join(f"- Trigger {row.trigger}: `{row.stim_id}`" for row in unmatched.itertuples(index=False)) or "- None"
    report = f"""# P01 ratings versus OASIS normative ratings

## Interpretation scope

OASIS values are normative reference means, not ground truth and not P01 model
targets. This comparison describes agreement and scale-use differences between
P01's ratings and the selected OASIS reference values.

## Join method

P01 `stim_id` and OASIS `Theme` were lowercased, the `.jpg` suffix was removed
from P01 names, and remaining non-alphanumeric characters were removed. The
normalized names were required to be unique on both sides and were joined
one-to-one. {int((matched['join_status'] == 'matched').sum())} of {len(matched)} P01 trials matched.

The following P01 trial remained deliberately unmatched and was excluded from
paired statistics and plots:

{unmatched_text}

## Paired comparison statistics

All differences are calculated as P01 minus OASIS. Standard deviations use the
sample standard deviation (`ddof=1`).

{format_statistics_table(statistics)}

## Outputs

- `rating_comparison.csv` contains all 120 P01 trials, their join status, OASIS
  values when matched, and outcome-specific signed differences.
- `plots/p01_vs_oasis_valence.png` and `plots/p01_vs_oasis_arousal.png` plot
  OASIS values on the x-axis and P01 ratings on the y-axis. Their dashed diagonal
  line represents equal ratings, not a ground-truth target.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the comparison after checking that no approved output would be overwritten."""
    args = parse_arguments()
    expected_outputs = [args.output_dir / filename for filename in OUTPUT_FILENAMES]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {paths}")

    matched = read_and_match_trials(args.trials, args.oasis)
    if matched.loc[matched["join_status"] == "unmatched", "stim_id"].tolist() != ["Keyboard 3.jpg"]:
        raise ValueError("Expected only Keyboard 3.jpg to remain unmatched; inspect the input tables before proceeding.")

    statistics = {
        "valence": summary_statistics(matched, "valence_rating", "oasis_valence_mean"),
        "arousal": summary_statistics(matched, "arousal_rating", "oasis_arousal_mean"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=False)
    matched.to_csv(args.output_dir / "rating_comparison.csv", index=False)
    write_report(args.output_dir / "rating_comparison_report.md", matched, statistics)
    write_scatter_plot(matched, "valence_rating", "oasis_valence_mean", "P01 versus OASIS valence ratings", plots_dir / "p01_vs_oasis_valence.png")
    write_scatter_plot(matched, "arousal_rating", "oasis_arousal_mean", "P01 versus OASIS arousal ratings", plots_dir / "p01_vs_oasis_arousal.png")

    print(f"Wrote comparison outputs to {args.output_dir}")
    print(f"Matched trials: {int((matched['join_status'] == 'matched').sum())}/{len(matched)}")
    for outcome, values in statistics.items():
        print(f"{outcome}: n={values['matched_nonmissing_trials']}, Pearson r={values['pearson_r']:.3f}, Spearman rho={values['spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
