#!/usr/bin/env python3
"""Create a Markdown report from a completed Stage C individual regression run.

The report summarizes already-computed nested-CV results only.  It does not
read raw acquisition data or fit models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = ("valence", "arousal")
TIE_ORDER = {"eeg": 0, "face": 1, "multimodal": 2, "knn": 0, "ridge": 1, "svr": 2, "5": 0, "10": 1, "20": 2, "all": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed regression output directory.")
    parser.add_argument("--output", type=Path, help="Report path (defaults to STAGE_C_INDIVIDUAL_REGRESSION_REPORT.md in run directory).")
    return parser.parse_args()


def choose_best(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one reproducible minimum-RMSE configuration per participant/target."""
    candidates = summary.loc[
        summary.groupby(["participant", "target"])["rmse_mean"].transform("min").eq(summary["rmse_mean"])
    ].copy()
    candidates["feature_count_request"] = candidates["feature_count_request"].astype(str)
    candidates["_modality_order"] = candidates["modality"].map(TIE_ORDER)
    candidates["_regressor_order"] = candidates["regressor"].map(TIE_ORDER)
    candidates["_feature_order"] = candidates["feature_count_request"].map(TIE_ORDER)
    selected = (
        candidates.sort_values(
            ["participant", "target", "_modality_order", "_regressor_order", "_feature_order"], kind="stable"
        )
        .groupby(["participant", "target"], as_index=False)
        .first()
        .drop(columns=["_modality_order", "_regressor_order", "_feature_order"])
    )
    ties = candidates.groupby(["participant", "target"]).size().rename("minimum_rmse_tie_count").reset_index()
    return selected.merge(ties, on=["participant", "target"], validate="one_to_one"), candidates


def metric_summary(values: pd.Series) -> str:
    return (
        f"mean {values.mean():.3f}; median {values.median():.3f}; SD {values.std():.3f}; "
        f"range {values.min():.3f}-{values.max():.3f}"
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    table = frame.loc[:, columns].copy()
    for column in ("rmse_mean", "rmse_std", "mae_mean", "r2_mean", "explained_variance_mean"):
        if column in table:
            table[column] = table[column].map(lambda value: f"{value:.3f}")
    table = table.rename(
        columns={
            "participant": "Participant", "target": "Target", "modality": "Modality", "regressor": "Regressor",
            "feature_count_request": "Features", "rmse_mean": "RMSE", "rmse_std": "RMSE SD", "mae_mean": "MAE",
            "r2_mean": "R2", "explained_variance_mean": "Explained variance", "minimum_rmse_tie_count": "RMSE ties",
        }
    )
    headers = list(table.columns)
    rows = table.astype(str).values.tolist()
    # Avoid pandas.DataFrame.to_markdown(), which requires the optional
    # ``tabulate`` dependency and is not part of this project's environment.
    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"

    return [render(headers), render(["---"] * len(headers)), *(render(row) for row in rows)]


def generate_report(run_dir: Path, output: Path) -> None:
    summary_path = run_dir / "results_summary.csv"
    manifest_path = run_dir / "run_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("The run directory must contain results_summary.csv and run_manifest.json.")
    summary = pd.read_csv(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"participant", "target", "modality", "regressor", "feature_count_request", "rmse_mean", "rmse_std", "mae_mean", "r2_mean", "explained_variance_mean"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"results_summary.csv is missing columns: {', '.join(sorted(missing))}")
    if set(summary.target.unique()) != set(TARGETS):
        raise ValueError("Expected valence and arousal results.")

    best, candidates = choose_best(summary)
    participants = sorted(best.participant.unique())
    lines = [
        "# Stage C individual regression: all 46 participants", "",
        "## Scope and evaluation", "",
        f"This report summarizes the completed Stage C individual regression run for {len(participants)} participants ({', '.join(participants)}). "
        "It uses the already saved outer-fold results; no models were refit for this report.", "",
        f"Each participant-target configuration used {int(summary.folds.iloc[0])}-fold nested shuffled K-fold cross-validation. "
        f"Hyperparameters were selected in each training split using `{manifest['inner_selection_metric']}`. "
        "The reported outcomes are unconstrained continuous ratings.", "",
        "For each participant and target, the table below selects the configuration with the lowest mean outer-fold RMSE across the 36 evaluated combinations "
        "(3 modalities x 3 regressors x 4 feature-count requests). Exact RMSE ties are resolved deterministically in the order EEG, face, multimodal; KNN, ridge, SVR; 5, 10, 20, all. "
        "The `RMSE ties` column records how many configurations shared that exact minimum.", "",
        "## Participant-best performance", "",
    ]
    for target in TARGETS:
        values = best.loc[best.target.eq(target)]
        lines.extend([
            f"### {target.capitalize()}", "",
            f"Participant-best RMSE: {metric_summary(values.rmse_mean)}. "
            f"MAE: {metric_summary(values.mae_mean)}. "
            f"R2: {metric_summary(values.r2_mean)}. "
            f"Positive mean R2: {int(values.r2_mean.gt(0).sum())}/{len(values)} participants.", "",
        ])
    lines.extend(markdown_table(
        best.sort_values(["participant", "target"]),
        ["participant", "target", "modality", "regressor", "feature_count_request", "rmse_mean", "rmse_std", "mae_mean", "r2_mean", "explained_variance_mean", "minimum_rmse_tie_count"],
    ))
    lines.extend(["", "## Model-selection patterns", ""])
    for target in TARGETS:
        subset = best.loc[best.target.eq(target)]
        modality = "; ".join(f"{name}: {count}" for name, count in subset.modality.value_counts().sort_index().items())
        regressor = "; ".join(f"{name}: {count}" for name, count in subset.regressor.value_counts().sort_index().items())
        features = "; ".join(f"{name}: {count}" for name, count in subset.feature_count_request.astype(str).value_counts().reindex(["5", "10", "20", "all"], fill_value=0).items())
        lines.append(f"- {target.capitalize()} — modality: {modality}; regressor: {regressor}; feature request: {features}.")
    lines.extend(["", "## Strongest participant-best results", ""])
    for target in TARGETS:
        lines.append(f"### {target.capitalize()}")
        lines.append("")
        for row in best.loc[best.target.eq(target)].nsmallest(5, "rmse_mean").itertuples():
            lines.append(
                f"- {row.participant}: RMSE {row.rmse_mean:.3f}, MAE {row.mae_mean:.3f}, R2 {row.r2_mean:.3f}; "
                f"{row.modality}/{row.regressor}/{row.feature_count_request} features."
            )
        lines.append("")
    total_tied = int(best.minimum_rmse_tie_count.gt(1).sum())
    lines.extend([
        "## Interpretation and limitations", "",
        "These results estimate within-participant predictive performance using held-out folds, so they are appropriate for exploratory participant-specific calibration. "
        "However, choosing the lowest-RMSE option from 36 configurations per participant-target is an optimistic selection procedure, not an independent performance estimate for a pre-specified deployed model.", "",
        "R2 is retained without clipping: negative values mean the configuration performed worse than the fold's mean-rating baseline. "
        f"Exact minimum-RMSE ties occurred for {total_tied}/92 participant-target selections. Modality and regressor counts therefore describe selected configurations, not proof that one modality or algorithm is intrinsically superior.", "",
        "No hypothesis tests, permutation tests, or corrections for the configuration search are included. Results should be presented as exploratory unless evaluated on an independent held-out session or a pre-registered final-model protocol.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "STAGE_C_INDIVIDUAL_REGRESSION_REPORT.md"
    generate_report(run_dir, output)
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()
