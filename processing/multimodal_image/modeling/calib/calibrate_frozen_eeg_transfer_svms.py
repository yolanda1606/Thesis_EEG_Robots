#!/usr/bin/env python3
"""Create Image-only Platt-calibrated deployment wrappers for frozen EEG SVMs.

The original frozen SVM artifacts remain immutable.  This script does not read
Robot data, alter the top-3 ranking, evaluate Robot predictions, or optimize
classifier hyperparameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[4]
TARGET_COLUMNS = {"valence": "valence_rating", "arousal": "arousal_rating"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top3-csv", type=Path, default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/top3_models_per_participant_target.csv")
    parser.add_argument("--model-root", type=Path, default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/image_classification/focused_personalized_binary_v1/final_transfer_probability_calibration_v1")
    parser.add_argument("--dry-run", action="store_true", help="Run reproduction checks but write no calibration artifacts.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_table(row: pd.Series) -> Path:
    match = re.search(r"merged trial table\s+(.+)$", str(row.preprocessing_condition))
    if not match:
        raise ValueError(f"{row.participant} {row.target} R{row.transfer_rank}: no Image source table in preprocessing_condition")
    path = Path(match.group(1)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing declared Image source table: {path}")
    return path


def eligible_image_data(path: Path, target: str, candidates: list[str], selected: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    table = pd.read_csv(path)
    rating_column = TARGET_COLUMNS[target]
    missing = sorted(set(candidates + [rating_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{path}: missing frozen candidate features or rating: {missing}")
    candidate_data = table.loc[:, candidates].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[rating_column], errors="coerce")
    usable = candidate_data.notna().all(axis=1) & ratings.notna()
    candidate_data = candidate_data.loc[usable].reset_index(drop=True)
    selected_data = candidate_data.loc[:, selected].copy()
    labels = (ratings.loc[usable].reset_index(drop=True) >= 4).astype(int)
    if not np.isfinite(candidate_data.to_numpy()).all() or not np.isfinite(selected_data.to_numpy()).all():
        raise ValueError(f"{path}: non-finite data remain after eligibility filtering")
    return candidate_data, selected_data, labels


def frozen_record(row: pd.Series, model_root: Path) -> dict[str, Any]:
    path = Path(row.final_model_path).resolve()
    if path.parent != model_root or not path.is_file():
        raise ValueError(f"Frozen model absent or outside --model-root: {path}")
    pipeline = joblib.load(path)
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(classifier, SVC) or classifier.probability:
        raise ValueError(f"{path.name}: expected frozen SVC(probability=False)")
    candidates = list(pipeline.feature_names_in_)
    selector = pipeline.named_steps.get("selector")
    selected = list(selector.get_feature_names_out(pipeline.feature_names_in_)) if selector is not None else candidates.copy()
    selected_csv = json.loads(row.actual_selected_feature_names)
    if selected != selected_csv:
        raise ValueError(f"{path.name}: serialized selected features differ from authoritative top-3 row")
    if any(not name.startswith("eeg_") for name in selected):
        raise ValueError(f"{path.name}: non-EEG selected feature found")
    return {
        "participant": row.participant, "target": row.target, "transfer_rank": int(row.transfer_rank),
        "model_path": path, "model_sha256": sha256(path), "pipeline": pipeline,
        "classifier_parameters": classifier.get_params(deep=False), "candidate_features": candidates,
        "selected_features": selected, "image_source": source_table(row),
    }


def reconstruct_svm(parameters: dict[str, Any]) -> Pipeline:
    """Construct the fixed selected-feature SVM; no selection/search is performed."""
    return Pipeline([("scale", StandardScaler()), ("classifier", SVC(**parameters))])


def validate_reproduction(record: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, Pipeline]:
    candidates, selected, labels = eligible_image_data(record["image_source"], record["target"], record["candidate_features"], record["selected_features"])
    class_counts = labels.value_counts().to_dict()
    if len(class_counts) != 2:
        raise ValueError(f"Only one Image class is eligible: {class_counts}")
    reconstructed = reconstruct_svm(record["classifier_parameters"])
    reconstructed.fit(selected, labels)
    original_prediction = record["pipeline"].predict(candidates)
    reconstructed_prediction = reconstructed.predict(selected)
    if not np.array_equal(original_prediction, reconstructed_prediction):
        mismatch = int(np.count_nonzero(original_prediction != reconstructed_prediction))
        raise RuntimeError(f"Hard-prediction reproduction failed: {mismatch}/{len(labels)} mismatches")
    return candidates, selected, labels, reconstructed


def calibrate(record: dict[str, Any], selected: pd.DataFrame, labels: pd.Series) -> tuple[CalibratedClassifierCV, int]:
    minority_count = int(labels.value_counts().min())
    folds = min(5, minority_count)
    if folds < 2:
        raise ValueError(f"At least two calibration folds are required; minority Image class has {minority_count} rows")
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    estimator = reconstruct_svm(record["classifier_parameters"])
    calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=split, ensemble=False, n_jobs=1)
    calibrated.fit(selected, labels)
    return calibrated, folds


def serializable_record(record: dict[str, Any], eligible_rows: int, class_counts: dict[int, int], folds: int) -> dict[str, Any]:
    return {
        "participant": record["participant"], "target": record["target"], "transfer_rank": record["transfer_rank"],
        "original_frozen_model_path": str(record["model_path"]), "original_frozen_model_sha256": record["model_sha256"],
        "image_source": str(record["image_source"]), "image_source_sha256": sha256(record["image_source"]),
        "candidate_features": record["candidate_features"], "selected_features": record["selected_features"],
        "classifier_parameters": record["classifier_parameters"], "eligible_image_rows": eligible_rows,
        "image_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "calibration_method": "CalibratedClassifierCV(method='sigmoid', ensemble=False)",
        "calibration_cv": f"StratifiedKFold(n_splits={folds}, shuffle=True, random_state=42)",
        "hard_prediction_reproduction": "exact match with original frozen SVM on eligible Image rows",
        "robot_data_used": False,
    }


def main() -> int:
    args = parse_args()
    top3, model_root, output = args.top3_csv.resolve(), args.model_root.resolve(), args.output_dir.resolve()
    rows = pd.read_csv(top3)
    rows = rows[rows.classifier.eq("svm")].sort_values(["participant", "target", "transfer_rank"]).reset_index(drop=True)
    if len(rows) != 85:
        raise ValueError(f"Expected 85 frozen SVM rows, found {len(rows)}")
    if not args.dry_run and output.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration directory: {output}")
    results: list[dict[str, Any]] = []
    artifacts: list[tuple[dict[str, Any], CalibratedClassifierCV]] = []
    for row in rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        base = {"participant": row.participant, "target": row.target, "transfer_rank": int(row.transfer_rank), "status": "failed"}
        try:
            record = frozen_record(row_series, model_root)
            _, selected, labels, _ = validate_reproduction(record)
            minority = int(labels.value_counts().min())
            folds = min(5, minority)
            result = base | {
                "status": "preflight_ok", "eligible_image_rows": int(len(labels)),
                "low_count": int((labels == 0).sum()), "high_count": int((labels == 1).sum()),
                "calibration_cv_folds": folds, "hard_prediction_reproduction": True,
                "original_model_path": str(record["model_path"]), "original_model_sha256": record["model_sha256"],
                "image_source": str(record["image_source"]), "message": "",
            }
            if not args.dry_run:
                calibrated, folds = calibrate(record, selected, labels)
                payload = serializable_record(record, len(labels), labels.value_counts().to_dict(), folds)
                artifacts.append((payload, calibrated))
                result["status"] = "calibrated"
            results.append(result)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            results.append(base | {"hard_prediction_reproduction": False, "message": str(error)})
    result_frame = pd.DataFrame(results)
    success = int(result_frame.status.eq("preflight_ok" if args.dry_run else "calibrated").sum())
    if args.dry_run:
        print(result_frame.to_csv(index=False))
        print(json.dumps({"mode": "dry_run", "svm_rows": len(rows), "success": success, "failures": int(len(rows) - success), "robot_data_used": False}, indent=2))
        return 0 if success == len(rows) else 2
    output.mkdir(parents=True)
    artifact_dir = output / "artifacts"
    artifact_dir.mkdir()
    for payload, calibrated in artifacts:
        stem = f"{payload['participant']}_{payload['target']}_rank{payload['transfer_rank']}_calibrated.joblib"
        artifact_path = artifact_dir / stem
        joblib.dump({"calibrated_probability_model": calibrated, "deployment_metadata": payload}, artifact_path)
        mask = ((result_frame.participant == payload["participant"]) & (result_frame.target == payload["target"]) & (result_frame.transfer_rank == payload["transfer_rank"]))
        result_frame.loc[mask, "calibrated_artifact_path"] = str(artifact_path.resolve())
        result_frame.loc[mask, "calibrated_artifact_sha256"] = sha256(artifact_path)
    result_frame.to_csv(output / "calibration_results.csv", index=False)
    manifest = {
        "purpose": "Image-only calibrated probability deployment layer for frozen EEG SVM transfer models",
        "command": " ".join([sys.executable, *sys.argv]), "top3_csv": str(top3), "top3_csv_sha256": sha256(top3),
        "model_root": str(model_root), "svm_rows_expected": 85, "success_count": success,
        "failure_count": int(len(rows) - success), "robot_data_used": False,
        "original_frozen_artifacts_modified": False,
        "selection_or_hyperparameter_optimization_performed": False,
        "calibration_method": "Image-only cross-validated Platt sigmoid calibration with fixed selected features and frozen SVM parameters",
        "hard_prediction_requirement": "A reconstructed full-Image fixed SVM must exactly reproduce original frozen hard predictions before calibration.",
        "later_robot_output_requirement": "Report original frozen SVM hard class separately from calibrated P(HIGH).",
        "results_file": str((output / "calibration_results.csv").resolve()),
    }
    (output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"svm_rows": len(rows), "success": success, "failures": int(len(rows) - success), "output": str(output), "robot_data_used": False}, indent=2))
    return 0 if success == len(rows) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
