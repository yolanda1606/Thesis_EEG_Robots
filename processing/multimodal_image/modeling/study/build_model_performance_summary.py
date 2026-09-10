#!/usr/bin/env python3
"""Build a read-only inventory of saved Image Experiment modeling results.

The script never trains, evaluates, or changes an existing experiment output.
It reads saved result summaries and writes three new audit files below
``outputs/modeling_summary/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUMMARY_ROOT = OUTPUT_ROOT / "modeling_summary"
INVENTORY_COLUMNS = [
    "task_type", "model_scope", "target", "modality", "model", "input_representation", "ica_status",
    "label_condition", "experiment", "best_primary_metric", "primary_metric_name", "secondary_metric",
    "secondary_metric_name", "participant_if_individual", "feature_count_or_family", "channel_subset",
    "hyperparameters_if_available", "n_participants", "source_run", "source_file", "notes",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse output locations; inputs remain read-only existing output trees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    """Load optional metadata, returning an empty mapping for absent/invalid JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def model_name(value: Any) -> str:
    """Use thesis-readable model names without inferring families not saved."""
    names = {"gnb": "GaussianNB", "knn": "kNN", "svm": "SVM", "svr": "SVR", "logreg": "Logistic Regression", "ridge": "Ridge", "elasticnet": "ElasticNet", "rf": "Random Forest", "random_forest": "Random Forest", "extra_trees": "Extra Trees", "cnn": "CNN"}
    return names.get(str(value).lower(), str(value))


def modality_name(value: Any) -> str:
    """Normalize modality spelling while keeping input representation separate."""
    names = {"eeg": "eeg", "face": "face", "video": "face", "multimodal": "multimodal"}
    return names.get(str(value).lower(), str(value))


def run_metadata(run_dir: Path) -> tuple[dict[str, Any], int, str, str]:
    """Return manifest, cohort size, source-run description, and ICA status."""
    manifest = load_json(run_dir / "run_manifest.json")
    if not manifest:
        manifest = load_json(run_dir / "run_config.json")
    settings = manifest.get("settings", {}) if isinstance(manifest.get("settings"), dict) else manifest
    participants = manifest.get("resolved_participants") or manifest.get("requested_participants") or manifest.get("participants") or settings.get("participants") or []
    source_run = str(settings.get("source_run") or manifest.get("source_run") or "not recorded")
    lower = f"{run_dir.name} {source_run}".lower()
    ica = "no-ICA" if "no_ica" in lower or "no-ica" in lower else ("ICA" if "ica" in lower else "not identifiable")
    return manifest, len(participants) if isinstance(participants, list) else 0, source_run, ica


def experiment_name(run_dir: Path) -> str:
    """Classify a saved output directory without using filesystem timestamps."""
    names = {
        "no_ica": "baseline no-ICA", "general_no_ica": "baseline no-ICA", "individual_optimized_top15": "optimized classifier search",
        "feature_family_top5": "feature-family diagnostic", "label_ambiguity_top5": "label-ambiguity diagnostic",
        "p10_individual_valence_smoke": "smoke test",
    }
    if run_dir.name in names:
        return names[run_dir.name]
    if run_dir.name.startswith("stage_"):
        return f"historical {run_dir.name}"
    return f"historical {run_dir.name}"


def row_template(**values: Any) -> dict[str, Any]:
    """Produce a complete normalized inventory row."""
    row = {column: "" for column in INVENTORY_COLUMNS}
    row.update(values)
    return row


def classical_rows(path: Path, task_type: str) -> list[dict[str, Any]]:
    """Normalize saved classical classification or regression summaries."""
    frame = pd.read_csv(path)
    run_dir = path.parent
    manifest, n_participants, source_run, ica_status = run_metadata(run_dir)
    experiment = experiment_name(run_dir)
    rows: list[dict[str, Any]] = []
    if task_type == "classification":
        required = {"mode", "target", "modality", "classifier", "balanced_accuracy_mean", "accuracy_mean"}
        if not required.issubset(frame.columns):
            return rows
        for item in frame.to_dict("records"):
            scope = "individual" if item["mode"] == "individual" else "general LOSO"
            participant = str(item.get("participant", "")) if scope == "individual" else ""
            rows.append(row_template(task_type="classification", model_scope=scope, target=item["target"], modality=modality_name(item["modality"]), model=model_name(item["classifier"]), input_representation="handcrafted EEG/face trial features", ica_status=ica_status, label_condition="standard LOW<4 / HIGH>=4", experiment=experiment, best_primary_metric=item["balanced_accuracy_mean"], primary_metric_name="balanced_accuracy", secondary_metric=item["accuracy_mean"], secondary_metric_name="accuracy", participant_if_individual=participant, feature_count_or_family=item.get("feature_count_resolved", item.get("feature_count_request", "")), n_participants=n_participants, source_run=source_run, source_file=str(path.relative_to(PROJECT_ROOT)), notes="mean outer-fold metric"))
    else:
        required = {"mode", "target", "modality", "regressor", "rmse_mean", "mae_mean"}
        if not required.issubset(frame.columns):
            return rows
        for item in frame.to_dict("records"):
            scope = "individual" if item["mode"] == "individual" else "general LOSO"
            participant = str(item.get("participant", "")) if scope == "individual" else ""
            rows.append(row_template(task_type="regression", model_scope=scope, target=item["target"], modality=modality_name(item["modality"]), model=model_name(item["regressor"]), input_representation="handcrafted EEG/face trial features", ica_status=ica_status, label_condition="continuous original rating", experiment=experiment, best_primary_metric=item["rmse_mean"], primary_metric_name="RMSE", secondary_metric=item["mae_mean"], secondary_metric_name="MAE", participant_if_individual=participant, feature_count_or_family=item.get("feature_count_resolved", item.get("feature_count_request", "")), n_participants=n_participants, source_run=source_run, source_file=str(path.relative_to(PROJECT_ROOT)), notes="mean outer-fold RMSE"))
    return rows


def focused_classification_rows(outputs_root: Path) -> list[dict[str, Any]]:
    """Read focused saved winner tables, retaining their diagnostic status."""
    rows: list[dict[str, Any]] = []
    optimized = outputs_root / "image_classification/individual_optimized_top15/best_per_participant_target.csv"
    if optimized.is_file():
        for item in pd.read_csv(optimized).to_dict("records"):
            rows.append(row_template(task_type="classification", model_scope="individual", target=item["target"], modality=modality_name(item["modality"]), model=model_name(item["classifier"]), input_representation="handcrafted EEG/face trial features", ica_status="no-ICA", label_condition="standard LOW<4 / HIGH>=4", experiment="optimized classifier search", best_primary_metric=item["balanced_accuracy"], primary_metric_name="balanced_accuracy", secondary_metric=item["accuracy"], secondary_metric_name="accuracy", participant_if_individual=item["participant"], feature_count_or_family=item["feature_count"], hyperparameters_if_available=item.get("best_hyperparameters", ""), n_participants=5, source_run="{participant_lower}_no_ica", source_file=str(optimized.relative_to(PROJECT_ROOT)), notes="nested-CV winner; included in current standard-result comparison"))
    family = outputs_root / "image_classification/feature_family_top5/best_per_participant_target.csv"
    if family.is_file():
        for item in pd.read_csv(family).to_dict("records"):
            rows.append(row_template(task_type="classification", model_scope="individual", target=item["target"], modality="eeg", model=model_name(item["winning_classifier"]), input_representation="handcrafted EEG feature family", ica_status="no-ICA", label_condition="standard LOW<4 / HIGH>=4", experiment="feature-family diagnostic", best_primary_metric=item["feature_family_balanced_accuracy"], primary_metric_name="balanced_accuracy", secondary_metric=item["feature_family_accuracy"], secondary_metric_name="accuracy", participant_if_individual=item["participant"], feature_count_or_family=item["winning_feature_family"], channel_subset=item["winning_channel_subset"], hyperparameters_if_available=item.get("winning_hyperparameters", ""), n_participants=5, source_run="{participant_lower}_no_ica", source_file=str(family.relative_to(PROJECT_ROOT)), notes="nested-CV winner; representation diagnostic"))
    ambiguity = outputs_root / "image_classification/label_ambiguity_top5/best_per_participant_target_condition.csv"
    if ambiguity.is_file():
        for item in pd.read_csv(ambiguity).to_dict("records"):
            condition = str(item["label_condition"])
            label = "standard LOW<4 / HIGH>=4" if condition == "standard" else "exclude_midpoint LOW<=3 / HIGH>=5"
            rows.append(row_template(task_type="classification", model_scope="individual", target=item["target"], modality=modality_name(item["modality"]), model=model_name(item["classifier"]), input_representation="handcrafted EEG/face trial features", ica_status="no-ICA", label_condition=label, experiment="label-ambiguity diagnostic", best_primary_metric=item["mean_balanced_accuracy"], primary_metric_name="balanced_accuracy", secondary_metric=item["mean_accuracy"], secondary_metric_name="accuracy", participant_if_individual=item["participant"], feature_count_or_family=item["feature_count"], hyperparameters_if_available=item.get("best_hyperparameters", ""), n_participants=5, source_run="{participant_lower}_no_ica", source_file=str(ambiguity.relative_to(PROJECT_ROOT)), notes="diagnostic; midpoint-excluded rows are not comparable with standard labels"))
    return rows


def multiclass_rows(outputs_root: Path) -> list[dict[str, Any]]:
    """Inventory saved three-class results without mixing them into binary summaries."""
    path = outputs_root / "image_multiclass/p19_three_class_v1/results_summary.csv"
    if not path.is_file():
        return []
    return [row_template(task_type="classification", model_scope="individual", target=item["target"], modality=modality_name(item["modality"]), model=model_name(item["classifier"]), input_representation="handcrafted EEG/face trial features", ica_status="not identifiable", label_condition="three-class LOW/NEUTRAL/HIGH", experiment="historical three-class P19", best_primary_metric=item["balanced_accuracy_mean"], primary_metric_name="balanced_accuracy", secondary_metric=item["accuracy_mean"], secondary_metric_name="accuracy", participant_if_individual=item["participant"], feature_count_or_family=item["feature_count_resolved"], n_participants=1, source_run="not recorded", source_file=str(path.relative_to(PROJECT_ROOT)), notes="multiclass result; excluded from binary-label best summary") for item in pd.read_csv(path).to_dict("records")]


def cnn_rows(outputs_root: Path) -> list[dict[str, Any]]:
    """Read saved raw time-series CNN general-regression aggregates."""
    rows: list[dict[str, Any]] = []
    for target in ("valence", "arousal"):
        path = outputs_root / f"cnn_all_participants_regression_general/{target}/aggregate_summary.csv"
        config_path = path.parent / "run_config.json"
        if not path.is_file():
            continue
        summary = pd.read_csv(path).iloc[0]
        config = load_json(config_path)
        rows.append(row_template(task_type="regression", model_scope="general LOSO", target=target, modality="eeg", model="CNN", input_representation="raw EEG time-series", ica_status="not identifiable", label_condition="continuous original rating", experiment="neural-network experiment", best_primary_metric=summary["mean_test_rmse"], primary_metric_name="RMSE", secondary_metric=summary.get("mean_test_mae", ""), secondary_metric_name="MAE", n_participants=len(config.get("participants", [])), source_run="not recorded", source_file=str(path.relative_to(PROJECT_ROOT)), notes="mean outer-test RMSE; aggregate also records dummy-RMSE deltas"))
    return rows


def unrecoverable_mlp_rows() -> list[dict[str, Any]]:
    """Document neural variants defined in code but lacking persistent outputs."""
    variants = (("MLP", "handcrafted EEG features"), ("MLP", "face features"), ("MLP", "flat multimodal handcrafted features"), ("Branched MLP", "branched EEG + face handcrafted features"))
    return [row_template(task_type="regression", model_scope="not recoverable", target="not recoverable", modality="EEG" if "EEG" in representation else ("Face" if representation == "face features" else "Multimodal"), model=model, input_representation=representation, ica_status="not recoverable", label_condition="not recoverable", experiment="neural-network experiment", primary_metric_name="RMSE", secondary_metric_name="MAE", source_file="no persistent output located", notes="result not recoverable from saved outputs") for model, representation in variants]


def collect_inventory(outputs_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Collect normalized rows only from recognized saved result files."""
    rows: list[dict[str, Any]] = []
    inspected: list[str] = []
    for task_type, directory in (("classification", outputs_root / "image_classification"), ("regression", outputs_root / "image_regression")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*/results_summary.csv")):
            inspected.append(str(path.relative_to(PROJECT_ROOT)))
            rows.extend(classical_rows(path, task_type))
    rows.extend(focused_classification_rows(outputs_root))
    for path in (outputs_root / "image_classification/individual_optimized_top15/best_per_participant_target.csv", outputs_root / "image_classification/feature_family_top5/best_per_participant_target.csv", outputs_root / "image_classification/label_ambiguity_top5/best_per_participant_target_condition.csv"):
        if path.is_file():
            inspected.append(str(path.relative_to(PROJECT_ROOT)))
    rows.extend(multiclass_rows(outputs_root)); inspected.append("outputs/image_multiclass/p19_three_class_v1/results_summary.csv")
    rows.extend(cnn_rows(outputs_root)); inspected.extend([str(path.relative_to(PROJECT_ROOT)) for path in sorted((outputs_root / "cnn_all_participants_regression_general").glob("*/aggregate_summary.csv"))])
    rows.extend(unrecoverable_mlp_rows())
    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    for column in ("best_primary_metric", "secondary_metric"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, sorted(set(inspected))


def candidate_rows(inventory: pd.DataFrame) -> pd.DataFrame:
    """Exclude non-comparable labels and duplicate standard diagnostic reruns from winners."""
    standard = inventory.label_condition.eq("standard LOW<4 / HIGH>=4") | inventory.task_type.eq("regression")
    not_duplicate_diagnostic = ~((inventory.experiment == "label-ambiguity diagnostic") & inventory.label_condition.eq("standard LOW<4 / HIGH>=4"))
    current_ica = inventory.ica_status.eq("no-ICA") | inventory.ica_status.eq("not identifiable")
    return inventory[standard & not_duplicate_diagnostic & current_ica & inventory.best_primary_metric.notna()].copy()


def select_best(inventory: pd.DataFrame, *, prefer_no_ica: bool) -> pd.DataFrame:
    """Select saved comparable rows by BA (high) or RMSE (low).

    ``prefer_no_ica`` is the current-result view.  The global historical view
    intentionally leaves it off, so it can report the actual saved maximum.
    """
    candidates = candidate_rows(inventory)
    groups = []
    for _, group in candidates.groupby(["task_type", "model_scope", "target", "modality"], dropna=False):
        if prefer_no_ica and (group.ica_status == "no-ICA").any():
            group = group[group.ica_status == "no-ICA"]
        ascending = group.primary_metric_name.iloc[0] == "RMSE"
        groups.append(group.sort_values("best_primary_metric", ascending=ascending, kind="stable").iloc[0])
    return pd.DataFrame(groups, columns=INVENTORY_COLUMNS) if groups else pd.DataFrame(columns=INVENTORY_COLUMNS)


def best_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return the no-ICA-preferred current-result summary."""
    return select_best(inventory, prefer_no_ica=True)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without external formatting dependencies."""
    if frame.empty:
        return "No recoverable saved results.\n"
    output = frame[columns].copy()
    for column in output.select_dtypes("number").columns:
        output[column] = output[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ") if pd.notna(value) else ""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in output.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows]) + "\n"


def build_markdown(inventory: pd.DataFrame, best: pd.DataFrame, inspected: list[str]) -> str:
    """Write thesis-readable scoped tables while retaining metric meanings."""
    lines = ["# Image Experiment modeling performance summary", "", "This read-only inventory uses saved result files as the source of truth. Classification is ranked by balanced accuracy (higher is better); regression is ranked by RMSE (lower is better). Midpoint-excluded and three-class diagnostics are retained in the inventory but excluded from standard binary best-model selection.", "", f"Recognized saved result files inspected: {len(inspected)}.", ""]
    sections = [("A. Individual classification", "classification", "individual"), ("B. General classification", "classification", "general LOSO"), ("C. Individual regression", "regression", "individual"), ("D. General regression", "regression", "general LOSO")]
    for heading, task, scope in sections:
        lines += [f"## {heading}", ""]
        if task == "classification" and scope == "individual":
            # This table and its following per-participant maximum table use
            # the same global standard-label eligibility, including historical
            # saved rows with unidentifiable ICA status.
            global_best = select_best(inventory, prefer_no_ica=False)
            subset = global_best[(global_best.task_type == task) & (global_best.model_scope == scope)]
            lines.append("Global maximum saved **standard binary** balanced accuracy by target and modality (historical standard-label runs included):")
            lines.append("")
        else:
            subset = best[(best.task_type == task) & (best.model_scope == scope)]
        lines.append(markdown_table(subset, ["target", "modality", "model", "best_primary_metric", "primary_metric_name", "secondary_metric", "secondary_metric_name", "participant_if_individual", "feature_count_or_family", "experiment", "source_file"]))
        if task == "classification" and scope == "individual":
            candidates = candidate_rows(inventory)
            grouped = candidates[(candidates.task_type == "classification") & (candidates.model_scope == "individual")].groupby(["target", "modality"])
            cohort_rows = []
            for (target, modality), group in grouped:
                per_participant = group.groupby("participant_if_individual").best_primary_metric.max()
                cohort_rows.append({"target": target, "modality": modality, "highest_participant_BA": per_participant.max(), "mean_best_per_participant_BA": per_participant.mean(), "participants": len(per_participant)})
            lines += ["Highest participant BA and mean of each participant's best saved standard BA:", "", markdown_table(pd.DataFrame(cohort_rows), ["target", "modality", "highest_participant_BA", "mean_best_per_participant_BA", "participants"])]
    lines += ["## E. Neural-network regression/classification experiments", "", markdown_table(inventory[inventory.experiment == "neural-network experiment"], ["task_type", "model_scope", "target", "modality", "model", "input_representation", "best_primary_metric", "primary_metric_name", "secondary_metric", "secondary_metric_name", "source_file", "notes"]), "## F. Overall best results", "", markdown_table(best, ["task_type", "model_scope", "target", "modality", "model", "best_primary_metric", "primary_metric_name", "secondary_metric", "secondary_metric_name", "participant_if_individual", "experiment", "source_file"])]
    return "\n".join(lines)


def print_terminal_summary(best: pd.DataFrame) -> None:
    """Print the separate no-ICA-preferred current-result view."""
    print("CURRENT NO-ICA-PREFERRED SUMMARY")
    for scope, title in (("individual", "INDIVIDUAL"), ("general LOSO", "GENERAL LOSO")):
        for task in ("classification", "regression"):
            print(f"{title} {task.upper()}")
            subset = best[(best.model_scope == scope) & (best.task_type == task)]
            for _, row in subset.iterrows():
                participant = f" | {row.participant_if_individual}" if row.participant_if_individual else ""
                print(f"{row.target} | {row.modality}: {row.model} {row.primary_metric_name}={row.best_primary_metric:.3f}{participant}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build all three summary artifacts from existing output files only."""
    outputs_root = args.outputs_root.resolve()
    summary_root = args.summary_root.resolve()
    inventory, inspected = collect_inventory(outputs_root)
    best = best_summary(inventory)
    summary_root.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(summary_root / "model_performance_inventory.csv", index=False)
    best.to_csv(summary_root / "best_model_summary.csv", index=False)
    (summary_root / "modeling_summary.md").write_text(build_markdown(inventory, best, inspected), encoding="utf-8")
    print_terminal_summary(best)
    print(f"Saved inventory: {summary_root}")
    return {"inventory": inventory, "best": best, "inspected": inspected, "summary_root": summary_root}


if __name__ == "__main__":
    run(parse_args())
