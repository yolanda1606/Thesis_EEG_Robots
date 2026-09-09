#!/usr/bin/env python3
"""Create Image-only Platt-calibration artifacts for final modality-agnostic SVMs.

The source is the authoritative final Image-domain modality-agnostic top-3
table. Frozen models and their rankings are immutable. No Robot input path,
Robot feature, Robot window, Robot rating, or Robot output is read by this
script.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[3]
TARGET_COLUMNS = {"valence": "valence_rating", "arousal": "arousal_rating"}
EXPECTED_MODALITIES = {"eeg", "face", "multimodal"}
KEY_COLUMNS = ["participant", "target", "transfer_rank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top3-csv",
        type=Path,
        default=ROOT
        / "outputs/image_classification/focused_personalized_binary_v1"
        / "top3_models_per_participant_target_modality.csv",
        help="Authoritative final Image-domain modality-agnostic top-3 table.",
    )
    parser.add_argument(
        "--image-output-root",
        type=Path,
        default=ROOT / "outputs/image_classification/focused_personalized_binary_v1",
        help="Root that must contain each authoritative frozen Image model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/image_classification/focused_personalized_binary_v1"
        / "modality_agnostic_probability_calibration_v1",
        help="New directory for calibration artifacts; must not already exist.",
    )
    parser.add_argument(
        "--existing-calibration-dir",
        type=Path,
        default=ROOT
        / "outputs/image_classification/focused_personalized_binary_v1"
        / "final_transfer_probability_calibration_v1",
        help=(
            "Optional existing Image-only calibration directory to audit for "
            "hash identity only. Its artifacts are never reused or modified."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Image-only preflight and reproduction checks; write nothing.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of an immutable file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image_source(row: pd.Series, project_root: Path = ROOT) -> Path:
    """Resolve the declared no-ICA Image trial table for one frozen model row."""
    condition = str(row.preprocessing_condition)
    absolute = re.search(r"merged trial table\s+(.+)$", condition)
    if absolute:
        source = Path(absolute.group(1)).resolve()
    else:
        expected_run = f"{str(row.participant).lower()}_no_ica"
        declared_template = "no_ica source run {participant_lower}_no_ica"
        if expected_run not in condition and declared_template not in condition:
            raise ValueError(
                f"{row.participant} {row.target} R{row.transfer_rank}: "
                "preprocessing_condition does not declare the expected no-ICA Image source run"
            )
        source = (
            project_root
            / "derived"
            / str(row.participant)
            / "Image_Experiment"
            / "runs"
            / expected_run
            / "merged"
            / f"{str(row.participant).lower()}_image_trial_dataset.csv"
        ).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing declared Image source table: {source}")
    return source


def eligible_image_data(
    source: Path, target: str, candidates: list[str], selected: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read only the frozen candidate schema and Image labels needed for calibration."""
    table = pd.read_csv(source)
    rating_column = TARGET_COLUMNS[target]
    missing = sorted(set(candidates + [rating_column]).difference(table.columns))
    if missing:
        raise ValueError(f"{source}: missing frozen candidate features or rating: {missing}")
    candidate_data = table.loc[:, candidates].apply(pd.to_numeric, errors="coerce")
    ratings = pd.to_numeric(table[rating_column], errors="coerce")
    usable = candidate_data.notna().all(axis=1) & ratings.notna()
    candidate_data = candidate_data.loc[usable].reset_index(drop=True)
    selected_data = candidate_data.loc[:, selected].copy()
    labels = (ratings.loc[usable].reset_index(drop=True) >= 4).astype(int)
    if not np.isfinite(candidate_data.to_numpy()).all():
        raise ValueError(f"{source}: non-finite values remain after Image eligibility filtering")
    return candidate_data, selected_data, labels


def validate_source_rows(rows: pd.DataFrame) -> None:
    required = {
        "participant",
        "target",
        "transfer_rank",
        "modality",
        "classifier",
        "preprocessing_condition",
        "actual_selected_feature_names",
        "final_model_path",
        "selection_scope",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Top-3 CSV missing required columns: {missing}")
    if rows.duplicated(KEY_COLUMNS).any():
        raise ValueError("Top-3 CSV contains duplicate participant/target/transfer-rank rows")
    if set(rows.modality.dropna()) != EXPECTED_MODALITIES:
        raise ValueError(
            f"Top-3 CSV must contain exactly {sorted(EXPECTED_MODALITIES)}; "
            f"found {sorted(set(rows.modality.dropna()))}"
        )
    if set(rows.selection_scope.dropna()) != {"modality_agnostic_image_domain_top3"}:
        raise ValueError("Top-3 CSV is not the authoritative modality-agnostic Image-domain selection")


def artifact_path(directory: Path, record: dict[str, Any]) -> Path:
    return directory / "artifacts" / (
        f"{record['participant']}_{record['target']}_rank{record['transfer_rank']}_calibrated.joblib"
    )


def existing_artifact_state(directory: Path, record: dict[str, Any]) -> str:
    """Classify an old artifact by exact frozen-model hash; never reuse it."""
    path = artifact_path(directory, record)
    if not path.is_file():
        return "missing"
    try:
        metadata = joblib.load(path).get("deployment_metadata", {})
    except Exception as error:  # A corrupt external artifact is incompatible.
        return f"unreadable: {error}"
    if metadata.get("original_frozen_model_sha256") != record["model_sha256"]:
        return "hash_mismatch"
    return "matching_but_not_reused"


def frozen_record(
    row: pd.Series, image_output_root: Path, project_root: Path = ROOT
) -> dict[str, Any]:
    """Load and validate one authoritative frozen SVC without altering it."""
    model_path = Path(row.final_model_path).resolve()
    if not model_path.is_file() or not model_path.is_relative_to(image_output_root):
        raise ValueError(
            f"Frozen model absent or outside --image-output-root: {model_path}"
        )
    pipeline = joblib.load(model_path)
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(classifier, SVC) or classifier.probability:
        raise ValueError(f"{model_path.name}: expected frozen SVC(probability=False)")
    candidates = list(pipeline.feature_names_in_)
    selector = pipeline.named_steps.get("selector")
    selected = (
        list(selector.get_feature_names_out(pipeline.feature_names_in_))
        if selector is not None
        else candidates.copy()
    )
    serialized_selected = json.loads(row.actual_selected_feature_names)
    if selected != serialized_selected:
        raise ValueError(
            f"{model_path.name}: selected features differ from authoritative top-3 row"
        )
    modality = str(row.modality)
    if modality == "eeg" and any(not feature.startswith("eeg_") for feature in candidates):
        raise ValueError(f"{model_path.name}: EEG model has non-EEG candidate features")
    if modality == "face" and any(not feature.startswith("video_") for feature in candidates):
        raise ValueError(f"{model_path.name}: Face model has non-Face candidate features")
    if modality == "multimodal" and not (
        any(feature.startswith("eeg_") for feature in candidates)
        and any(feature.startswith("video_") for feature in candidates)
    ):
        raise ValueError(f"{model_path.name}: Multimodal model lacks EEG or Face candidate features")
    return {
        "participant": str(row.participant),
        "target": str(row.target),
        "transfer_rank": int(row.transfer_rank),
        "modality": modality,
        "model_path": model_path,
        "model_sha256": sha256(model_path),
        "pipeline": pipeline,
        "classifier_parameters": classifier.get_params(deep=False),
        "candidate_features": candidates,
        "selected_features": selected,
        "image_source": resolve_image_source(row, project_root),
    }


def reconstruct_svm(parameters: dict[str, Any]) -> Pipeline:
    """Construct the fixed selected-feature SVM; neither search nor selection runs."""
    return Pipeline([("scale", StandardScaler()), ("classifier", SVC(**parameters))])


def validate_reproduction(
    record: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Require exact frozen hard-prediction reproduction on eligible Image rows."""
    candidates, selected, labels = eligible_image_data(
        record["image_source"],
        record["target"],
        record["candidate_features"],
        record["selected_features"],
    )
    if labels.nunique() != 2:
        raise ValueError(f"Only one eligible Image class remains: {labels.value_counts().to_dict()}")
    reconstructed = reconstruct_svm(record["classifier_parameters"])
    reconstructed.fit(selected, labels)
    original_prediction = record["pipeline"].predict(candidates)
    reconstructed_prediction = reconstructed.predict(selected)
    if not np.array_equal(original_prediction, reconstructed_prediction):
        mismatches = int(np.count_nonzero(original_prediction != reconstructed_prediction))
        raise RuntimeError(
            f"Hard-prediction reproduction failed: {mismatches}/{len(labels)} mismatches"
        )
    return selected, labels


def calibrate(
    record: dict[str, Any], selected: pd.DataFrame, labels: pd.Series
) -> tuple[CalibratedClassifierCV, int]:
    minority_count = int(labels.value_counts().min())
    folds = min(5, minority_count)
    if folds < 2:
        raise ValueError(
            f"At least two Image-only calibration folds are required; "
            f"minority class has {minority_count} rows"
        )
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    calibrated = CalibratedClassifierCV(
        estimator=reconstruct_svm(record["classifier_parameters"]),
        method="sigmoid",
        cv=split,
        ensemble=False,
        n_jobs=1,
    )
    calibrated.fit(selected, labels)
    return calibrated, folds


def deployment_metadata(
    record: dict[str, Any], eligible_rows: int, labels: pd.Series, folds: int
) -> dict[str, Any]:
    """Return auditable identity metadata stored beside the calibrated wrapper."""
    return {
        "participant": record["participant"],
        "target": record["target"],
        "transfer_rank": record["transfer_rank"],
        "modality": record["modality"],
        "original_frozen_model_path": str(record["model_path"]),
        "original_frozen_model_sha256": record["model_sha256"],
        "classifier_parameters": record["classifier_parameters"],
        "candidate_features": record["candidate_features"],
        "selected_features": record["selected_features"],
        "image_source": str(record["image_source"]),
        "image_source_sha256": sha256(record["image_source"]),
        "eligible_image_rows": eligible_rows,
        "image_class_counts": {
            str(label): int(count) for label, count in labels.value_counts().to_dict().items()
        },
        "label_definition": "LOW < 4; HIGH >= 4",
        "calibration_method": "CalibratedClassifierCV(method='sigmoid', ensemble=False)",
        "calibration_cv": f"StratifiedKFold(n_splits={folds}, shuffle=True, random_state=42)",
        "hard_prediction_reproduction": "exact match with original frozen SVM on eligible Image rows",
        "robot_data_used": False,
    }


def main() -> int:
    args = parse_args()
    top3 = args.top3_csv.resolve()
    image_output_root = args.image_output_root.resolve()
    output = args.output_dir.resolve()
    existing = args.existing_calibration_dir.resolve()
    if not top3.is_file():
        raise FileNotFoundError(f"Missing authoritative modality-agnostic top-3 CSV: {top3}")
    if not image_output_root.is_dir():
        raise FileNotFoundError(f"Missing Image output root: {image_output_root}")
    if not args.dry_run and output.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration directory: {output}")

    all_rows = pd.read_csv(top3)
    validate_source_rows(all_rows)
    rows = (
        all_rows.loc[all_rows.classifier.str.lower().eq("svm")]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    results: list[dict[str, Any]] = []
    artifacts: list[tuple[dict[str, Any], CalibratedClassifierCV]] = []
    for row in rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        base = {
            "participant": row.participant,
            "target": row.target,
            "transfer_rank": int(row.transfer_rank),
            "modality": row.modality,
            "status": "failed",
        }
        try:
            record = frozen_record(row_series, image_output_root)
            selected, labels = validate_reproduction(record)
            folds = min(5, int(labels.value_counts().min()))
            result = base | {
                "status": "preflight_ok",
                "eligible_image_rows": len(labels),
                "low_count": int((labels == 0).sum()),
                "high_count": int((labels == 1).sum()),
                "calibration_cv_folds": folds,
                "hard_prediction_reproduction": True,
                "original_model_path": str(record["model_path"]),
                "original_model_sha256": record["model_sha256"],
                "image_source": str(record["image_source"]),
                "existing_artifact_state": existing_artifact_state(existing, record),
                "message": "",
            }
            if not args.dry_run:
                calibrated, folds = calibrate(record, selected, labels)
                metadata = deployment_metadata(record, len(labels), labels, folds)
                artifacts.append((metadata, calibrated))
                result["status"] = "calibrated"
            results.append(result)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            results.append(base | {"hard_prediction_reproduction": False, "message": str(error)})

    result_frame = pd.DataFrame(results)
    success_status = "preflight_ok" if args.dry_run else "calibrated"
    success = int(result_frame.status.eq(success_status).sum())
    if args.dry_run:
        print(result_frame.to_csv(index=False))
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "svm_rows": len(rows),
                    "success": success,
                    "failures": len(rows) - success,
                    "robot_data_used": False,
                },
                indent=2,
            )
        )
        return 0 if success == len(rows) else 2

    output.mkdir(parents=True, exist_ok=False)
    artifacts_dir = output / "artifacts"
    artifacts_dir.mkdir()
    for metadata, calibrated in artifacts:
        path = artifact_path(output, metadata)
        joblib.dump(
            {"calibrated_probability_model": calibrated, "deployment_metadata": metadata}, path
        )
        mask = pd.Series(True, index=result_frame.index)
        for key in KEY_COLUMNS:
            mask &= result_frame[key].astype(str).eq(str(metadata[key]))
        result_frame.loc[mask, "calibrated_artifact_path"] = str(path.resolve())
        result_frame.loc[mask, "calibrated_artifact_sha256"] = sha256(path)
    result_frame.to_csv(output / "calibration_results.csv", index=False)
    manifest = {
        "purpose": "Image-only calibrated probability deployment layer for final modality-agnostic frozen SVMs",
        "command": " ".join([sys.executable, *sys.argv]),
        "top3_csv": str(top3),
        "top3_csv_sha256": sha256(top3),
        "image_output_root": str(image_output_root),
        "existing_calibration_dir_audited_only": str(existing),
        "svm_rows_expected": len(rows),
        "success_count": success,
        "failure_count": len(rows) - success,
        "robot_data_used": False,
        "original_frozen_artifacts_modified": False,
        "selection_or_hyperparameter_optimization_performed": False,
        "calibration_method": "Image-only cross-validated Platt sigmoid calibration with fixed selected features and frozen SVM parameters",
        "hard_prediction_requirement": "A reconstructed fixed SVM must exactly reproduce original frozen hard predictions before calibration.",
        "results_file": str((output / "calibration_results.csv").resolve()),
    }
    (output / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "svm_rows": len(rows),
                "success": success,
                "failures": len(rows) - success,
                "output": str(output),
                "robot_data_used": False,
            },
            indent=2,
        )
    )
    return 0 if success == len(rows) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
