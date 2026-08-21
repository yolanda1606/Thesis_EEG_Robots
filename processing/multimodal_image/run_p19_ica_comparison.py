#!/usr/bin/env python3
"""Fair P19 EEG-only nested-CV comparison: canonical ICA versus no-ICA Image runs.

The validated train_classification evaluator is used directly. This wrapper only
selects two explicit, already processed P19 Image run tables and writes a
dedicated comparison output; it never changes either processed run or uses Robot
data. Face/multimodal modes are deliberately unavailable because p19_no_ica did
not regenerate video features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_p19_no_ica import compare_runs  # noqa: E402
from modeling.train_classification import EEG_COLUMNS, evaluate_individual, prepare_modality_data  # noqa: E402


PARTICIPANT = "P19"
RUNS = {"with_ICA": "p19_final", "no_ICA": "p19_no_ica"}
TARGETS = ("valence", "arousal")
MODELS = ("knn", "svm", "gnb")
FEATURE_COUNTS = ("all", "5", "10", "20")
OUTPUT_DIR = Path("outputs/image_classification/P19_ica_comparison")
ALLOWED_CONFIG_DIFFERENCES = {"eeg.ica.enabled", "resources.resource_root"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, default=Path("derived"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate comparability and planned grid without fitting or writing.")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a non-zero integer")
    return args


def run_path(derived_root: Path, run_name: str) -> Path:
    return derived_root / PARTICIPANT / "Image_Experiment" / "runs" / run_name


def config_differences(left: object, right: object, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[str] = []
        for key in sorted(set(left).union(right)):
            child = f"{prefix}.{key}" if prefix else key
            result.extend(config_differences(left.get(key, "<missing>"), right.get(key, "<missing>"), child))
        return result
    return [] if left == right else [prefix]


def audit(derived_root: Path) -> list[str]:
    canonical, no_ica = run_path(derived_root, RUNS["with_ICA"]), run_path(derived_root, RUNS["no_ICA"])
    passed, messages = compare_runs(canonical, no_ica)
    if not passed:
        raise ValueError("P19 canonical/no-ICA identifiers, labels, schema, or retained trials differ; classification is blocked")
    configs = []
    for run in (canonical, no_ica):
        path = run / "config/resolved_configuration.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Resolved configuration missing: {path}")
        configs.append(yaml.safe_load(path.read_text(encoding="utf-8")))
    differences = set(config_differences(*configs))
    unexpected = differences.difference(ALLOWED_CONFIG_DIFFERENCES)
    if unexpected:
        raise ValueError(f"Canonical/no-ICA configuration has unexpected differences: {sorted(unexpected)}")
    if configs[0]["eeg"]["ica"]["enabled"] is not True or configs[1]["eeg"]["ica"]["enabled"] is not False:
        raise ValueError("Expected ICA enabled only in canonical run and disabled only in no-ICA run")
    messages.append("Resolved preprocessing configuration differs only at ICA enabled state and installation-specific resource_root path.")
    messages.append("Video/face artifacts are absent from p19_no_ica by design; this wrapper evaluates EEG only.")
    return messages


def load_table(derived_root: Path, run_name: str) -> pd.DataFrame:
    path = run_path(derived_root, run_name) / "merged/p19_image_trial_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Merged trial table missing: {path}")
    table = pd.read_csv(path)
    if missing := sorted(set(EEG_COLUMNS).difference(table.columns)):
        raise ValueError(f"{run_name}: merged table lacks Image EEG columns: {missing}")
    return table


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["representation", "source_run", "target", "modality", "classifier", "feature_count_request", "feature_count_resolved"]
    return results.groupby(keys, dropna=False).agg(
        outer_folds=("fold", "size"), mean_nested_cv_balanced_accuracy=("balanced_accuracy", "mean"),
        sd_nested_cv_balanced_accuracy=("balanced_accuracy", "std"), mean_low_recall=("low_recall", "mean"),
        mean_high_recall=("recall", "mean"),
    ).reset_index()


def matched_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "modality", "classifier", "feature_count_request", "feature_count_resolved"]
    values = summary.pivot(index=keys, columns="representation", values="mean_nested_cv_balanced_accuracy").reset_index()
    if set(RUNS).difference(values.columns):
        raise RuntimeError("Both ICA representations are required for every matched configuration")
    values["delta_BA_no_ICA_minus_with_ICA"] = values["no_ICA"] - values["with_ICA"]
    return values.sort_values(keys).reset_index(drop=True)


def best_per_target(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (representation, target), frame in summary.groupby(["representation", "target"], sort=False):
        maximum = float(frame.mean_nested_cv_balanced_accuracy.max())
        tied = frame[np.isclose(frame.mean_nested_cv_balanced_accuracy, maximum, rtol=0.0, atol=1e-12)].copy()
        tied["exact_mean_BA_top_tie"] = len(tied) > 1
        rows.append(tied)
    return pd.concat(rows, ignore_index=True)


def write_summary(audit_messages: list[str], summary: pd.DataFrame, deltas: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# P19 Image Classification: ICA versus No-ICA", "", "This is an EEG-only, P19-only comparison. Both representations use the unchanged validated individual nested-CV evaluator; no Robot data was used.", "",
             "## Comparability audit", "", *[f"- {message}" for message in audit_messages], "",
             "## Best configurations", "", "| Representation | Target | Classifier | Requested features | Mean nested-CV BA | SD | LOW recall | HIGH recall | Exact top tie |", "|---|---|---|---|---:|---:|---:|---:|---|"]
    for row in best.itertuples(index=False):
        lines.append(f"| {row.representation} | {row.target} | {row.classifier} | {row.feature_count_request} | {row.mean_nested_cv_balanced_accuracy:.3f} | {row.sd_nested_cv_balanced_accuracy:.3f} | {row.mean_low_recall:.3f} | {row.mean_high_recall:.3f} | {bool(row.exact_mean_BA_top_tie)} |")
    lines.extend(["", "## Matched-configuration BA changes", "", "`delta_BA_no_ICA_minus_with_ICA` is descriptive. No formal significance test is performed; differences must be interpreted alongside outer-fold variability.", "", "| Target | Classifier | Features | With ICA BA | No-ICA BA | Delta |", "|---|---|---|---:|---:|---:|"])
    for row in deltas.itertuples(index=False):
        lines.append(f"| {row.target} | {row.classifier} | {row.feature_count_request} | {row.with_ICA:.3f} | {row.no_ICA:.3f} | {row.delta_BA_no_ICA_minus_with_ICA:+.3f} |")
    lines.extend(["", "## Interpretation boundary", "", "A higher Robot-domain compatibility result from a separate experiment does not determine Image model selection here. No-ICA is not called better based on a small mean BA difference alone. Face and multimodal configurations are outside this comparison because no face/video artifacts were regenerated for `p19_no_ica`."])
    (output_dir / "P19_ica_vs_no_ica_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_messages = audit(args.derived_root)
    tables = {representation: load_table(args.derived_root, run) for representation, run in RUNS.items()}
    prepared = {(representation, target): prepare_modality_data(table, PARTICIPANT, target, "eeg")[:2]
                for representation, table in tables.items() for target in TARGETS}
    if args.dry_run:
        print("Dry-run comparability audit: PASS")
        print("\n".join(audit_messages))
        print(f"Planned Image-only evaluations: {len(RUNS) * len(TARGETS) * len(MODELS) * len(FEATURE_COUNTS)} configurations; 5 outer folds each where class counts allow.")
        for (representation, target), (_, labels) in prepared.items():
            print(f"{representation}/{target}: usable={len(labels)}, LOW={(labels == 0).sum()}, HIGH={(labels == 1).sum()}")
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison output: {args.output_dir}")
    results_rows, selected_rows, parameter_rows = [], [], []
    for representation, source_run in RUNS.items():
        for target in TARGETS:
            features, labels = prepared[(representation, target)]
            for classifier in MODELS:
                for request in FEATURE_COUNTS:
                    results, selected, parameters = evaluate_individual(PARTICIPANT, features, labels, target, "eeg", classifier, request, args.seed, n_jobs=args.n_jobs)
                    for row in results:
                        row.update({"representation": representation, "source_run": source_run,
                                    "low_recall": float(row["tn"] / (row["tn"] + row["fp"])) if row["tn"] + row["fp"] else 0.0})
                    for row in selected: row.update({"representation": representation, "source_run": source_run})
                    for row in parameters: row.update({"representation": representation, "source_run": source_run})
                    results_rows.extend(results); selected_rows.extend(selected); parameter_rows.extend(parameters)
    results, selected, parameters = pd.DataFrame(results_rows), pd.DataFrame(selected_rows), pd.DataFrame(parameter_rows)
    summary, deltas = summarize(results), None
    deltas, best = matched_deltas(summary), best_per_target(summary)
    args.output_dir.mkdir(parents=True)
    results.to_csv(args.output_dir / "P19_ica_vs_no_ica_fold_results.csv", index=False)
    selected.to_csv(args.output_dir / "P19_ica_vs_no_ica_selected_features_by_fold.csv", index=False)
    parameters.to_csv(args.output_dir / "P19_ica_vs_no_ica_hyperparameters_by_fold.csv", index=False)
    summary.to_csv(args.output_dir / "P19_ica_vs_no_ica_results.csv", index=False)
    deltas.to_csv(args.output_dir / "P19_ica_vs_no_ica_matched_deltas.csv", index=False)
    best.to_csv(args.output_dir / "P19_ica_vs_no_ica_best_per_target.csv", index=False)
    write_summary(audit_messages, summary, deltas, best, args.output_dir)
    manifest = {"participant": PARTICIPANT, "comparison": "canonical_with_ICA_vs_experimental_no_ICA", "runs": {key: str(run_path(args.derived_root, value)) for key, value in RUNS.items()}, "modality": "eeg", "models": list(MODELS), "feature_counts": list(FEATURE_COUNTS), "nested_cv": "validated evaluate_individual: StratifiedKFold outer up to 5; inner up to 3", "scoring": "balanced_accuracy", "seed": args.seed, "robot_data_used": False, "comparability_audit": audit_messages}
    (args.output_dir / "P19_ica_vs_no_ica_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"P19 ICA/no-ICA EEG comparison complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
