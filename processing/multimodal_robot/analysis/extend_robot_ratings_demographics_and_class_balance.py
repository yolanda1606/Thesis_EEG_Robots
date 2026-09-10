#!/usr/bin/env python3
"""Exploratory demographics and label-balance extension for Robot ratings only.

This script reads the canonical 1–7 Robot ratings and canonical participant YAML
metadata. Demographics are linked through the user-authorized reconstructed
spreadsheet sequence (# 1 → P01 through # 46 → P46). It does not read EEG,
predictions, models, or transfer outputs.
"""
from __future__ import annotations

import argparse
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
    DISPLAY_TASKS,
    OUTCOMES,
    PARTICIPANTS,
    TASK_ORDER,
    TASK_TYPE_ORDER,
    condition_outputs,
    load_canonical_ratings,
    participant_task_types,
)


DEMOGRAPHICS_PATH = ROOT / "data/Thesis Organization - Participant Info.csv"
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_RESAMPLES = 10_000
MEASURES = [
    "overall_valence", "overall_arousal",
    "observation_valence", "observation_arousal",
    "interaction_valence", "interaction_arousal",
    "alone_valence", "alone_arousal",
    "fault_valence_delta", "fault_arousal_delta",
    "speed_valence_delta", "speed_arousal_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robot_ratings")
    return parser.parse_args()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return BH-FDR adjusted p-values while preserving missing values."""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return adjusted
    ordered = valid.sort_values()
    count = len(ordered)
    values = ordered.to_numpy() * count / np.arange(1, count + 1)
    values = np.minimum.accumulate(values[::-1])[::-1]
    adjusted.loc[ordered.index] = np.minimum(values, 1.0)
    return adjusted


def load_reconstructed_demographics() -> pd.DataFrame:
    """Load demographics with the explicitly authorized # → P## reconstruction."""
    raw = pd.read_csv(DEMOGRAPHICS_PATH, header=None)
    if len(raw) != 48:
        raise ValueError(f"Expected title, header, and 46 demographic rows; found {len(raw)} rows")
    demographics = raw.iloc[2:].copy()
    demographics.columns = raw.iloc[1].tolist()
    demographics = demographics.reset_index(drop=True)
    expected_columns = ["#", "Age", "Participant ID", "Gender", "Prior ", "Experience", "Group", "Date ", "Time", "Completed", "Notes"]
    if demographics.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected demographic columns: {demographics.columns.tolist()}")
    sequence = pd.to_numeric(demographics["#"], errors="raise").astype(int)
    if sequence.tolist() != list(range(1, 47)):
        raise ValueError("Demographic # field is not the complete 1–46 sequence")
    demographics["participant"] = [f"P{number:02d}" for number in sequence]
    explicit = demographics["Participant ID"].dropna().astype(str)
    if explicit.to_dict() != {0: "P01", 1: "P02"}:
        raise ValueError("P01/P02 direct demographic ID confirmations are not present as expected")
    demographics["age"] = pd.to_numeric(demographics["Age"], errors="raise")
    if demographics.age.isna().any() or demographics.Gender.isna().any() or demographics["Prior "].isna().any():
        raise ValueError("Age, Gender, and Prior robot experience must be complete")
    if set(demographics.Gender) != {"Male", "Female"}:
        raise ValueError(f"Unexpected recorded gender categories: {sorted(demographics.Gender.unique())}")
    if set(demographics["Prior "]) != {"Yes", "No"}:
        raise ValueError(f"Unexpected recorded Prior categories: {sorted(demographics['Prior '].unique())}")
    return demographics.rename(columns={
        "Gender": "gender", "Prior ": "prior_robot_experience", "Experience": "experience_free_text",
        "Group": "demographic_sheet_group", "Date ": "demographic_sheet_date",
        "Time": "demographic_sheet_time", "Completed": "demographic_sheet_completed",
        "Notes": "demographic_sheet_notes", "Participant ID": "demographic_sheet_explicit_id",
    })


def participant_measures(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create the pre-specified participant-level rating and delta measures."""
    overall = ratings.groupby(["participant", "target"], as_index=False).robot_rating.mean()
    overall = overall.pivot(index="participant", columns="target", values="robot_rating").add_prefix("overall_")
    aggregates = participant_task_types(ratings)
    types = aggregates.pivot(index=["participant", "target"], columns="task_type", values="rating").reset_index()
    types = types.melt(id_vars=["participant", "target"], var_name="task_type", value_name="rating")
    types = types.pivot(index="participant", columns=["task_type", "target"], values="rating")
    types.columns = [f"{task_type.lower()}_{target}" for task_type, target in types.columns]
    _, deltas = condition_outputs(ratings)
    delta_columns = ["participant", "target", "fault_effect_delta", "speed_effect_delta"]
    delta_data = deltas.loc[:, delta_columns].pivot(index="participant", columns="target")
    delta_data.columns = [f"{effect.replace('_effect_delta', '')}_{target}_delta" for effect, target in delta_data.columns]
    summary = overall.join(types).join(delta_data).reindex(PARTICIPANTS)
    if summary.index.isna().any() or set(MEASURES) != set(summary.columns):
        raise ValueError("Participant rating measures do not match the requested definition")
    return summary.reset_index(names="participant")


def summary_rows(demographics: pd.DataFrame) -> pd.DataFrame:
    age = demographics.age
    rows = [{
        "variable": "age", "level": "continuous", "n": len(age), "missing_n": int(age.isna().sum()),
        "mean": float(age.mean()), "sd": float(age.std(ddof=1)), "median": float(age.median()),
        "q1": float(age.quantile(.25)), "q3": float(age.quantile(.75)), "iqr": float(age.quantile(.75) - age.quantile(.25)),
        "min": float(age.min()), "max": float(age.max()), "note": "Original recorded age; analyzed continuously.",
    }]
    for variable in ["gender", "prior_robot_experience"]:
        for level, data in demographics.groupby(variable, sort=True):
            rows.append({
                "variable": variable, "level": level, "n": len(data), "missing_n": int(data[variable].isna().sum()),
                "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                "min": np.nan, "max": np.nan,
                "note": "Original recorded category; no categories were merged or recoded.",
            })
    rows.append({
        "variable": "experience_free_text", "level": "free_text", "n": len(demographics),
        "missing_n": int(demographics.experience_free_text.isna().sum()), "mean": np.nan, "sd": np.nan,
        "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan,
        "note": f"Preserved without ordinal recoding; '-' appears {int(demographics.experience_free_text.eq('-').sum())} times.",
    })
    return pd.DataFrame(rows)


def bootstrap_spearman_ci(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    pairs = pd.DataFrame({"x": x, "y": y}).dropna().to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(pairs), size=(BOOTSTRAP_RESAMPLES, len(pairs)))
    values = []
    for indices in sampled:
        sample = pairs[indices]
        value = stats.spearmanr(sample[:, 0], sample[:, 1]).statistic
        if np.isfinite(value):
            values.append(value)
    return tuple(np.quantile(values, [.025, .975])) if values else (np.nan, np.nan)


def group_descriptive(values: pd.Series) -> dict[str, float | int]:
    values = values.dropna()
    q1, q3 = values.quantile([.25, .75])
    return {"n": len(values), "median": float(values.median()), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def cliffs_delta(left: pd.Series, right: pd.Series) -> float:
    differences = left.dropna().to_numpy()[:, None] - right.dropna().to_numpy()[None, :]
    return float((differences > 0).mean() - (differences < 0).mean())


def demographic_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Run pre-specified exploratory tests and correct within demographic families."""
    rows = []
    for measure in MEASURES:
        pairs = frame[["age", measure]].dropna()
        test = stats.spearmanr(pairs.age, pairs[measure])
        ci_low, ci_high = bootstrap_spearman_ci(pairs.age, pairs[measure])
        rows.append({
            "demographic": "age", "analysis_type": "spearman_correlation", "measure": measure,
            "n": len(pairs), "group_1": pd.NA, "group_2": pd.NA,
            "group_1_n": pd.NA, "group_2_n": pd.NA, "group_1_median": pd.NA, "group_1_iqr": pd.NA,
            "group_2_median": pd.NA, "group_2_iqr": pd.NA,
            "test_statistic": float(test.statistic), "effect_size_name": "spearman_rho", "effect_size": float(test.statistic),
            "ci_95_low": ci_low, "ci_95_high": ci_high, "p_value_raw": float(test.pvalue),
        })
    for demographic, levels in [("gender", ["Female", "Male"]), ("prior_robot_experience", ["No", "Yes"])]:
        for measure in MEASURES:
            left = frame.loc[frame[demographic].eq(levels[0]), measure].dropna()
            right = frame.loc[frame[demographic].eq(levels[1]), measure].dropna()
            test = stats.mannwhitneyu(left, right, alternative="two-sided", method="asymptotic")
            left_summary, right_summary = group_descriptive(left), group_descriptive(right)
            rows.append({
                "demographic": demographic, "analysis_type": "mann_whitney_u", "measure": measure,
                "n": len(left) + len(right), "group_1": levels[0], "group_2": levels[1],
                "group_1_n": left_summary["n"], "group_2_n": right_summary["n"],
                "group_1_median": left_summary["median"], "group_1_iqr": left_summary["iqr"],
                "group_2_median": right_summary["median"], "group_2_iqr": right_summary["iqr"],
                "test_statistic": float(test.statistic), "effect_size_name": "cliffs_delta_group_1_minus_group_2",
                "effect_size": cliffs_delta(left, right), "ci_95_low": np.nan, "ci_95_high": np.nan,
                "p_value_raw": float(test.pvalue),
            })
    results = pd.DataFrame(rows)
    results["fdr_family"] = results.demographic.map({
        "age": "age_10_measures", "gender": "gender_10_measures", "prior_robot_experience": "experience_10_measures",
    })
    results["p_value_fdr_bh"] = results.groupby("fdr_family", group_keys=False).p_value_raw.apply(benjamini_hochberg)
    results["fdr_significant_0_05"] = results.p_value_fdr_bh.lt(.05)
    return results


def class_balance(ratings: pd.DataFrame) -> pd.DataFrame:
    """Describe LOW/HIGH label composition using the frozen deployment threshold."""
    ratings = ratings.assign(label=np.where(ratings.robot_rating.ge(4), "HIGH", "LOW"))
    rows = []
    specifications = [
        ("overall", "Overall", ["target"]),
        ("task", None, ["task", "task_display", "target"]),
        ("task_type", None, ["task_type", "target"]),
    ]
    for scope, fixed_group, dimensions in specifications:
        for keys, data in ratings.groupby(dimensions, sort=False):
            values = keys if isinstance(keys, tuple) else (keys,)
            lookup = dict(zip(dimensions, values))
            low, high = int(data.label.eq("LOW").sum()), int(data.label.eq("HIGH").sum())
            rows.append({
                "scope": scope, "group": fixed_group or lookup.get("task_display", lookup.get("task_type")),
                "target": lookup["target"], "n_total": len(data), "n_low": low, "n_high": high,
                "proportion_low": low / len(data), "proportion_high": high / len(data),
                "high_to_low_ratio": high / low if low else np.nan,
            })
    order = {task: index for index, task in enumerate([DISPLAY_TASKS[task] for task in TASK_ORDER])}
    order.update({task_type: index for index, task_type in enumerate(TASK_TYPE_ORDER)})
    return pd.DataFrame(rows).assign(_order=lambda data: data.group.map(order).fillna(-1)).sort_values(["scope", "_order", "target"], kind="stable").drop(columns="_order")


def add_stack_labels(axis: plt.Axes, bottom: np.ndarray, values: np.ndarray) -> None:
    for position, (lower, value) in enumerate(zip(bottom, values)):
        if value:
            axis.text(position, lower + value / 2, f"{value:.0%}", ha="center", va="center", fontsize=8)


def save_figures(output: Path, balance: pd.DataFrame, participants: pd.DataFrame, associations: pd.DataFrame) -> None:
    destinations = [
        output / "class_balance_overall.png", output / "class_balance_by_task.png", output / "class_balance_by_task_type.png",
        output / "age_vs_fault_deltas.png", output / "experience_vs_fault_deltas.png",
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite extension figures: " + ", ".join(map(str, existing)))
    colors = {"LOW": "#d95f02", "HIGH": "#1b9e77"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    overall = balance[balance.scope.eq("overall")].set_index("target").reindex(OUTCOMES)
    fig, axis = plt.subplots(figsize=(6, 4.5))
    positions = np.arange(len(overall)); low, high = overall.proportion_low.to_numpy(), overall.proportion_high.to_numpy()
    axis.bar(positions, low, color=colors["LOW"], label="LOW (<4)"); axis.bar(positions, high, bottom=low, color=colors["HIGH"], label="HIGH (≥4)")
    add_stack_labels(axis, np.zeros_like(low), low); add_stack_labels(axis, low, high)
    axis.set(xticks=positions, xticklabels=[item.title() for item in overall.index], ylim=(0, 1), ylabel="Proportion of all participant-task ratings", title="Overall Robot-rating validation-label balance")
    axis.legend(loc="upper right"); axis.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(destinations[0], dpi=220, bbox_inches="tight"); plt.close(fig)

    for scope, destination, title, groups in [
        ("task", destinations[1], "Robot-rating validation-label balance by task", [DISPLAY_TASKS[task] for task in TASK_ORDER]),
        ("task_type", destinations[2], "Robot-rating validation-label balance by task type", TASK_TYPE_ORDER),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11 if scope == "task" else 8.5, 4.7), sharey=True)
        for axis, target in zip(axes, OUTCOMES):
            data = balance[(balance.scope == scope) & (balance.target == target)].set_index("group").reindex(groups)
            positions = np.arange(len(data)); low, high = data.proportion_low.to_numpy(), data.proportion_high.to_numpy()
            axis.bar(positions, low, color=colors["LOW"], label="LOW (<4)"); axis.bar(positions, high, bottom=low, color=colors["HIGH"], label="HIGH (≥4)")
            add_stack_labels(axis, np.zeros_like(low), low); add_stack_labels(axis, low, high)
            axis.set(xticks=positions, xticklabels=groups, ylim=(0, 1), title=target.title()); axis.grid(axis="y", alpha=.25)
        axes[0].set_ylabel("Proportion"); axes[-1].legend(loc="upper right")
        fig.suptitle(title, y=1.02); fig.tight_layout(); fig.savefig(destination, dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=False)
    for axis, target in zip(axes, OUTCOMES):
        measure = f"fault_{target}_delta"
        data = participants.dropna(subset=["age", measure])
        axis.scatter(data.age, data[measure], color="#7570b3", alpha=.7)
        slope, intercept, *_ = stats.linregress(data.age, data[measure])
        x = np.array([data.age.min(), data.age.max()]); axis.plot(x, intercept + slope * x, color="black", linewidth=1)
        result = associations[(associations.demographic == "age") & (associations.measure == measure)].iloc[0]
        axis.axhline(0, color="0.5", linestyle="--", linewidth=.8)
        axis.set(xlabel="Age (years)", ylabel="FAULTY_FAST − CONTROL_FAST", title=f"{target.title()}: ρ={result.effect_size:.2f}, FDR p={result.p_value_fdr_bh:.3f}")
        axis.grid(alpha=.25)
    fig.suptitle("Exploratory age association with fault-response deltas", y=1.02); fig.tight_layout(); fig.savefig(destinations[3], dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4), sharey=False)
    for axis, target in zip(axes, OUTCOMES):
        measure = f"fault_{target}_delta"
        values = [participants.loc[participants.prior_robot_experience.eq(level), measure].dropna() for level in ["No", "Yes"]]
        axis.boxplot(values, tick_labels=["No", "Yes"], showfliers=False)
        for position, series in enumerate(values, start=1): axis.scatter(np.full(len(series), position) + rng.normal(0, .04, len(series)), series, color="#1b9e77", alpha=.6, s=18)
        result = associations[(associations.demographic == "prior_robot_experience") & (associations.measure == measure)].iloc[0]
        axis.axhline(0, color="0.5", linestyle="--", linewidth=.8)
        axis.set(xlabel="Prior robot experience", ylabel="FAULTY_FAST − CONTROL_FAST", title=f"{target.title()}: FDR p={result.p_value_fdr_bh:.3f}")
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Exploratory prior-experience comparison of fault-response deltas", y=1.02); fig.tight_layout(); fig.savefig(destinations[4], dpi=220, bbox_inches="tight"); plt.close(fig)


def extended_summary(balance: pd.DataFrame, associations: pd.DataFrame) -> str:
    overall = balance[balance.scope.eq("overall")]
    largest_imbalance = balance[balance.scope.eq("task")].sort_values("proportion_low").head(4)
    fdr = associations[associations.fdr_significant_0_05].copy()
    association_columns = ["demographic", "measure", "n", "effect_size_name", "effect_size", "p_value_raw", "p_value_fdr_bh", "fdr_significant_0_05"]
    return "\n".join([
        "# Extended Robot participant-ratings analysis", "",
        "This ratings-only extension uses no EEG, model predictions, or transfer outputs.", "",
        "## Task-level ratings", "", "Unchanged from `task_level_summary.csv` and the original ratings-only summary.", "",
        "## Task-type ratings", "", "Unchanged from `task_type_summary.csv` and the original ratings-only summary.", "",
        "## Fault effect", "", "Unchanged from `condition_comparison.csv`; paired figures remain unchanged.", "",
        "## Speed effect", "", "Unchanged from `condition_comparison.csv`; paired figures remain unchanged.", "",
        "## Valence-arousal space", "", "Unchanged; see `valence_arousal_space.png`.", "",
        "## Class imbalance", "", "LOW is defined as rating <4 and HIGH as rating ≥4 only to describe later deployment-validation labels. This does not alter the original 1–7 ratings analysis.", "", overall.to_markdown(index=False), "", "Most LOW-sparse task × target distributions:", "", largest_imbalance.to_markdown(index=False), "", "Accuracy = (TP + TN) / N. Balanced Accuracy = 0.5 × (Sensitivity_HIGH + Specificity_LOW). When HIGH is much more common than LOW, accuracy can appear acceptable despite poor LOW performance; balanced accuracy weights the two classes equally.", "",
        "## Exploratory demographic associations", "", "Demographic tests are exploratory. Age uses Spearman correlation with deterministic bootstrap 95% CIs. Gender and prior experience use Mann–Whitney U with Cliff's delta. Benjamini–Hochberg FDR correction is applied separately across the 12 participant-level measures within each demographic family.", "", associations.loc[:, association_columns].to_markdown(index=False), "", "FDR-significant rows:", "", fdr.loc[:, association_columns].to_markdown(index=False) if len(fdr) else "None.", "",
        "## Data/linkage limitations", "", "The demographic linkage is reconstructed from the ordered `#` field (# 1 → P01 through # 46 → P46), not from a complete original ID field. P01/P02 are directly confirmed; P14 is independently corroborated by the FAULTY_SLOW Stack exception; acquisition chronology supports the sequence. Rows 26–32 have scheduling/acquisition-date differences consistent with rescheduling. The demographic-sheet Group values for rows 27/28 conflict with the canonical YAMLs and were never used; all experimental group/task/condition metadata comes only from canonical participant YAML files.", "",
    ])


def main() -> int:
    args = parse_args(); output = args.output_dir.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Missing existing ratings-only output directory: {output}")
    artifacts = [
        "demographic_summary.csv", "participant_demographic_rating_summary.csv", "demographic_rating_associations.csv",
        "class_balance_summary.csv", "robot_ratings_summary_extended.md", "demographic_class_balance_manifest.json",
        "class_balance_overall.png", "class_balance_by_task.png", "class_balance_by_task_type.png",
        "age_vs_fault_deltas.png", "experience_vs_fault_deltas.png",
    ]
    existing = [output / name for name in artifacts if (output / name).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite extension outputs: " + ", ".join(map(str, existing)))
    demographics = load_reconstructed_demographics()
    ratings, _ = load_canonical_ratings()
    measures = participant_measures(ratings)
    participants = demographics.merge(measures, on="participant", validate="one_to_one")
    if len(participants) != 46 or participants.participant.nunique() != 46:
        raise ValueError("Reconstructed demographic linkage did not yield 46 unique participants")
    associations = demographic_associations(participants)
    balance = class_balance(ratings)
    summary = summary_rows(demographics)
    participants.to_csv(output / "participant_demographic_rating_summary.csv", index=False)
    summary.to_csv(output / "demographic_summary.csv", index=False)
    associations.to_csv(output / "demographic_rating_associations.csv", index=False)
    balance.to_csv(output / "class_balance_summary.csv", index=False)
    save_figures(output, balance, participants, associations)
    (output / "robot_ratings_summary_extended.md").write_text(extended_summary(balance, associations) + "\n", encoding="utf-8")
    manifest = {
        "purpose": "Exploratory demographics and descriptive validation-label balance for Robot ratings only",
        "uses_eeg_or_predictions": False,
        "demographic_source": str(DEMOGRAPHICS_PATH),
        "demographic_linkage": "Reconstructed under explicit authorization: # 1 → P01 through # 46 → P46",
        "linkage_evidence": ["P01/P02 direct IDs", "P14 FAULTY_SLOW corroboration", "acquisition chronology"],
        "linkage_limitations": ["Rows 26–32 schedule/acquisition differences", "rows 27/28 demographic-sheet Group conflicts"],
        "experimental_metadata_source": "Canonical participant YAMLs only; demographic-sheet Group excluded",
        "age_test": "Spearman rho with deterministic percentile-bootstrap 95% CI",
        "categorical_tests": "Mann–Whitney U with Cliff's delta",
        "fdr": "Benjamini–Hochberg within each demographic family across 12 measures",
        "class_rule": "LOW < 4; HIGH >= 4",
    }
    (output / "demographic_class_balance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "fdr_significant_associations": int(associations.fdr_significant_0_05.sum())}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
