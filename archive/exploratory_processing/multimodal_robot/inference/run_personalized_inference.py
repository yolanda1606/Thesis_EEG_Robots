#!/usr/bin/env python3
"""Freeze Image-trained personalized models and apply them to Robot windows.

``prepare-models`` uses Image data only.  ``infer`` receives ICA/no-ICA as a
condition label plus feature paths; it never retrains or changes a frozen model.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from processing.multimodal_image.modeling import train_classification as tc

KEYS = ["participant", "task", "segment_id", "window_id", "window_start_s", "window_end_s"]


@dataclass(frozen=True)
class ModelSpec:
    target: str
    modality: str
    classifier: str
    feature_count_request: str

    @property
    def identifier(self) -> str:
        return f"{self.target}__{self.modality}__{self.classifier}__{self.feature_count_request}"


def parse_spec(value: str) -> ModelSpec:
    pieces = value.split(":")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError("--model-spec must be target:modality:classifier:feature_count")
    target, modality, classifier, count = pieces
    if target not in tc.TARGET_COLUMNS or modality not in {"eeg", "face", "multimodal"} or classifier not in {"knn", "svm", "gnb"} or count not in {"all", "5", "10", "20"}:
        raise argparse.ArgumentTypeError(f"Invalid model spec: {value}")
    return ModelSpec(target, modality, classifier, count)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select-models", "prepare-models", "preflight", "infer", "report"))
    parser.add_argument("--participant", required=True)
    parser.add_argument("--model-spec", action="append", type=parse_spec,
                        help="Repeat: target:modality:classifier:feature_count")
    parser.add_argument("--model-output-dir", type=Path, required=True)
    parser.add_argument("--image-derived-root", type=Path, default=ROOT / "derived")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", choices=("ICA", "no-ICA"))
    parser.add_argument("--robot-eeg-csv", type=Path)
    parser.add_argument("--robot-video-csv", type=Path)
    parser.add_argument("--inference-output-dir", type=Path)
    parser.add_argument("--ica-inference-dir", type=Path)
    parser.add_argument("--no-ica-inference-dir", type=Path)
    parser.add_argument("--report-output-dir", type=Path)
    parser.add_argument("--stage-b-results", type=Path, default=ROOT / "outputs/image_classification/stage_b_individual_v1/results_summary.csv")
    parser.add_argument("--selection-file", type=Path)
    result = parser.parse_args()
    result.model_spec = result.model_spec or []
    if len({spec.identifier for spec in result.model_spec}) != len(result.model_spec):
        parser.error("Duplicate --model-spec entries")
    if result.stage == "select-models" and not result.selection_file:
        parser.error("select-models requires --selection-file")
    if result.stage != "select-models" and not result.model_spec and not result.selection_file:
        parser.error("supply --model-spec or --selection-file")
    if result.stage in {"preflight", "infer"}:
        if not result.condition:
            parser.error("preflight/infer requires --condition")
        if result.stage == "infer" and not result.inference_output_dir:
            parser.error("infer requires --inference-output-dir")
        needs_eeg = any(s.modality in {"eeg", "multimodal"} for s in result.model_spec)
        needs_face = any(s.modality in {"face", "multimodal"} for s in result.model_spec)
        if needs_eeg and not result.robot_eeg_csv: parser.error("selected EEG/multimodal model requires --robot-eeg-csv")
        if needs_face and not result.robot_video_csv: parser.error("selected face/multimodal model requires --robot-video-csv")
    if result.stage == "report" and (not result.ica_inference_dir or not result.no_ica_inference_dir or not result.report_output_dir):
        parser.error("report requires --ica-inference-dir, --no-ica-inference-dir, and --report-output-dir")
    return result


def resolved_specs(a: argparse.Namespace) -> list[ModelSpec]:
    if a.model_spec: return a.model_spec
    payload = json.loads(a.selection_file.read_text(encoding="utf-8"))
    return [ModelSpec(**row) for row in payload["model_specs"]]


def select_models(a: argparse.Namespace) -> None:
    """Recover three participant-specific representatives per target from Stage B."""
    results = pd.read_csv(a.stage_b_results)
    rows = []
    for target in ("valence", "arousal"):
        data = results[(results["mode"] == "individual") & (results.participant == a.participant) & (results.target == target)].copy()
        if data.empty: raise ValueError(f"No Stage B individual results for {a.participant} {target}")
        # One representative per distinct BA rank.  Exact ties with the same
        # resolved dimensionality retain the simplest explicit request.
        data["_request_order"] = data.feature_count_request.map({"5": 0, "10": 1, "20": 2, "all": 3})
        candidates = []
        for _, tied in data.groupby("balanced_accuracy_mean", sort=False):
            tied = tied.sort_values(["feature_count_resolved", "_request_order", "balanced_accuracy_sd", "modality", "classifier"], kind="stable")
            candidates.append(tied.iloc[0])
        for rank, row in enumerate(sorted(candidates, key=lambda r: -r.balanced_accuracy_mean)[:3], 1):
            rows.append({"target": target, "modality": row.modality, "classifier": row.classifier, "feature_count_request": str(row.feature_count_request), "rank": rank, "mean_outer_balanced_accuracy": float(row.balanced_accuracy_mean), "tie_resolution": "simplest explicit request among exact mean-BA ties with the same effective feature dimensionality"})
    output = {"participant": a.participant, "source": str(a.stage_b_results.resolve()), "model_specs": [{k: r[k] for k in ("target", "modality", "classifier", "feature_count_request")} for r in rows], "selection": rows}
    if a.selection_file.exists(): raise FileExistsError(f"Refusing to overwrite selection file: {a.selection_file}")
    a.selection_file.parent.mkdir(parents=True, exist_ok=True); a.selection_file.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Selected six Stage-B representatives: {a.selection_file}")


def internal_splits(labels: pd.Series, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = tc.valid_stratified_splits(labels, 5, "final personalized Image-only CV")
    return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))


def fit_final_model(features: pd.DataFrame, labels: pd.Series, spec: ModelSpec, seed: int) -> tuple[Any, dict[str, Any]]:
    count = tc.resolved_feature_count(spec.feature_count_request, features.shape[1])
    pipeline, grid = tc.make_pipeline(spec.classifier, count, seed)
    if spec.classifier == "svm":
        pipeline.set_params(classifier__probability=True)
    splits = internal_splits(labels, seed)
    if spec.classifier == "knn":
        limit = min(len(train) for train, _ in splits)
        grid = [{**grid[0], "classifier__n_neighbors": [k for k in grid[0]["classifier__n_neighbors"] if k <= limit]}]
    search = GridSearchCV(pipeline, grid, scoring="balanced_accuracy", cv=splits, n_jobs=1, refit=True, error_score="raise")
    search.fit(features, labels)
    fitted = search.best_estimator_
    selector = fitted.named_steps.get("selector")
    selected = list(features.columns[selector.get_support()]) if selector is not None else list(features.columns)
    classifier = fitted.named_steps["classifier"]
    if not hasattr(classifier, "predict_proba"):
        raise RuntimeError(f"{spec.identifier}: final classifier lacks predict_proba")
    classes = [int(x) for x in classifier.classes_]
    if 1 not in classes:
        raise RuntimeError(f"{spec.identifier}: HIGH class 1 is absent from classes_={classes}")
    metadata = {
        "spec": asdict(spec), "identifier": spec.identifier, "best_parameters": search.best_params_,
        "internal_cv_best_balanced_accuracy": float(search.best_score_), "usable_image_rows": int(len(labels)),
        "low_count": int((labels == 0).sum()), "high_count": int((labels == 1).sum()),
        "candidate_features": list(features.columns), "selected_features": selected,
        "classifier_classes": classes, "high_probability_column": int(classes.index(1)),
        "supports_predict_proba": True,
        "image_feature_min": {name: float(features[name].min()) for name in selected},
        "image_feature_max": {name: float(features[name].max()) for name in selected},
    }
    return fitted, metadata


def prepare_models(a: argparse.Namespace) -> None:
    out = a.model_output_dir.resolve()
    if out.exists(): raise FileExistsError(f"Refusing to overwrite model output directory: {out}")
    table, source = tc.load_participant_table(a.image_derived_root.resolve(), a.participant)
    out.mkdir(parents=True)
    models = []
    for spec in resolved_specs(a):
        x, y, _ = tc.prepare_modality_data(table, a.participant, spec.target, spec.modality)
        fitted, metadata = fit_final_model(x, y, spec, a.seed)
        model_path = out / f"{spec.identifier}.joblib"
        joblib.dump(fitted, model_path)
        metadata["model_file"] = model_path.name
        training_path = out / f"{spec.identifier}__image_selected_features.csv"
        x.loc[:, metadata["selected_features"]].to_csv(training_path, index=False)
        metadata["image_selected_features_file"] = training_path.name
        models.append(metadata)
    manifest = {"participant": a.participant, "stage": "Image-only final deployment model preparation",
                "label_definition": "LOW=0 when rating < 4; HIGH=1 when rating >= 4",
                "image_source": str(source), "seed": a.seed,
                "tie_resolution": "Face has 10 candidate features; exact BA tie among requests 10/20/all retained request 10 as the simplest explicit effective dimensionality.",
                "models": models}
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{**m["spec"], "identifier": m["identifier"], "selected_feature_count": len(m["selected_features"]), "best_parameters": json.dumps(m["best_parameters"], sort_keys=True), "classifier_classes": json.dumps(m["classifier_classes"]), "high_probability_column": m["high_probability_column"]} for m in models]).to_csv(out / "deployment_model_summary.csv", index=False)
    print(f"Prepared {len(models)} Image-only deployment models: {out}")


def robot_eeg_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEYS + ["channel"])
    if missing := required.difference(frame): raise ValueError(f"EEG CSV missing: {sorted(missing)}")
    values = [c for c in frame if c.startswith("eeg_") and c != "eeg_coverage"]
    wide = frame.pivot(index=KEYS, columns="channel", values=values)
    wide.columns = [f"{feature}__{channel}" for feature, channel in wide.columns]
    return wide.reset_index()


def robot_input(spec: ModelSpec, eeg: pd.DataFrame | None, video: pd.DataFrame | None) -> pd.DataFrame:
    if spec.modality == "eeg": return eeg.copy() if eeg is not None else pd.DataFrame()
    if spec.modality == "face": return video.copy() if video is not None else pd.DataFrame()
    if eeg is None or video is None: raise ValueError("Multimodal inference needs both EEG and video data")
    if eeg.duplicated(KEYS).any() or video.duplicated(KEYS).any():
        raise ValueError("EEG/video window identity keys must each be unique before multimodal joining")
    # A video feature row defines a valid face-covered interval.  Retain only
    # the one-to-one intersection, while rejecting a video row lacking EEG.
    audit = eeg[KEYS].merge(video[KEYS], on=KEYS, how="outer", indicator=True)
    if audit["_merge"].eq("right_only").any():
        raise ValueError("Video contains window identities absent from EEG")
    return eeg.merge(video, on=KEYS, how="inner", validate="one_to_one")


def shift_diagnostics(model: Any, metadata: dict[str, Any], x: pd.DataFrame, keys: pd.DataFrame, condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Image-reference OOD diagnostics only in final selected dimensions."""
    selected = metadata["selected_features"]
    candidates = metadata["candidate_features"]
    selected_indices = [candidates.index(name) for name in selected]
    scaler = model.named_steps["scale"]
    z = (x[selected].to_numpy(dtype=float) - scaler.mean_[selected_indices]) / scaler.scale_[selected_indices]
    train = pd.read_csv(Path(metadata["_model_dir"]) / metadata["image_selected_features_file"])[selected].to_numpy(dtype=float)
    train_z = (train - scaler.mean_[selected_indices]) / scaler.scale_[selected_indices]
    distances = np.sqrt(((z[:, None, :] - train_z[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    lower = np.array([metadata["image_feature_min"][name] for name in selected])
    upper = np.array([metadata["image_feature_max"][name] for name in selected])
    outside = (x[selected].to_numpy(dtype=float) < lower) | (x[selected].to_numpy(dtype=float) > upper)
    window = keys.copy(); window["condition"] = condition; window["model_id"] = metadata["identifier"]
    window["nearest_image_distance"] = distances
    window["outside_selected_count"] = outside.sum(axis=1)
    window["outside_selected_fraction"] = outside.mean(axis=1)
    window["max_abs_image_z"] = np.abs(z).max(axis=1)
    window["median_abs_image_z"] = np.median(np.abs(z), axis=1)
    summaries = []
    groups = {"all_selected": selected}
    if metadata["spec"]["modality"] == "multimodal":
        groups.update({"eeg_selected": [f for f in selected if f.startswith("eeg_")], "face_selected": [f for f in selected if f.startswith("video_")]})
    for group, names in groups.items():
        if not names: continue
        idx = [selected.index(name) for name in names]
        subset_outside, subset_z = outside[:, idx], np.abs(z[:, idx])
        temporary = keys[["task"]].copy(); temporary["nearest_image_distance"] = distances
        temporary["has_outside"] = subset_outside.any(axis=1)
        temporary["outside_fraction"] = subset_outside.mean(axis=1)
        temporary["median_abs_z"] = np.median(subset_z, axis=1); temporary["max_abs_z"] = subset_z.max(axis=1)
        for task, part in temporary.groupby("task", dropna=False):
            q1, q3 = part.nearest_image_distance.quantile([.25, .75])
            summaries.append({"condition": condition, "model_id": metadata["identifier"], "task": task, "feature_group": group,
                              "window_count": len(part), "median_nearest_image_distance": float(part.nearest_image_distance.median()),
                              "iqr_nearest_image_distance": float(q3-q1), "fraction_windows_with_outside": float(part.has_outside.mean()),
                              "fraction_feature_values_outside": float(part.outside_fraction.mean()), "median_abs_image_z": float(part.median_abs_z.median()),
                              "maximum_abs_image_z": float(part.max_abs_z.max())})
    return window, pd.DataFrame(summaries)


def infer(a: argparse.Namespace) -> None:
    manifest = json.loads((a.model_output_dir / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest["participant"] != a.participant: raise ValueError("participant does not match frozen model manifest")
    lookup = {m["identifier"]: m for m in manifest["models"]}
    eeg = robot_eeg_table(a.robot_eeg_csv) if a.robot_eeg_csv else None
    video = pd.read_csv(a.robot_video_csv) if a.robot_video_csv else None
    if video is not None:
        if missing := set(KEYS).difference(video): raise ValueError(f"Video CSV missing: {sorted(missing)}")
        video = video.loc[:, KEYS + [c for c in video if c.startswith("video_")]].copy()
    out = a.inference_output_dir.resolve()
    if out.exists(): raise FileExistsError(f"Refusing to overwrite inference output directory: {out}")
    predictions = []; windows = []; summaries = []
    ranks = {target: 0 for target in tc.TARGET_COLUMNS}
    for spec in a.model_spec:
        ranks[spec.target] += 1
        metadata = lookup.get(spec.identifier)
        if metadata is None: raise ValueError(f"Frozen model absent: {spec.identifier}")
        frame = robot_input(spec, eeg, video)
        selected, candidates = metadata["selected_features"], metadata["candidate_features"]
        if missing := set(candidates).difference(frame): raise ValueError(f"{spec.identifier}: Robot feature columns missing: {sorted(missing)}")
        x = frame[candidates].apply(pd.to_numeric, errors="raise")
        valid = np.isfinite(x.to_numpy()).all(axis=1)
        frame, x = frame.loc[valid].reset_index(drop=True), x.loc[valid].reset_index(drop=True)
        if x.empty: raise ValueError(f"{spec.identifier}: no Robot windows have finite selected features")
        model = joblib.load(a.model_output_dir / metadata["model_file"])
        probability = model.predict_proba(x)[:, metadata["high_probability_column"]]
        result = frame[KEYS].copy(); result["condition"] = a.condition; result["model_id"] = spec.identifier
        result["target"] = spec.target; result["model_rank"] = ranks[spec.target]; result["modality"] = spec.modality; result["classifier"] = spec.classifier; result["feature_request"] = spec.feature_count_request
        result["predicted_class"] = model.predict(x); result["high_probability"] = probability
        predictions.append(result)
        metadata = {**metadata, "_model_dir": str(a.model_output_dir.resolve())}
        window, summary = shift_diagnostics(model, metadata, x, frame[KEYS], a.condition)
        windows.append(window); summaries.append(summary)
    out.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_csv(out / "window_predictions.csv", index=False)
    pd.concat(windows, ignore_index=True).to_csv(out / "selected_feature_shift_window_diagnostics.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(out / "selected_feature_shift_task_summary.csv", index=False)
    (out / "inference_manifest.json").write_text(json.dumps({"participant": a.participant, "condition": a.condition, "model_manifest": str((a.model_output_dir / 'model_manifest.json').resolve()), "robot_eeg_csv": str(a.robot_eeg_csv) if a.robot_eeg_csv else None, "robot_video_csv": str(a.robot_video_csv) if a.robot_video_csv else None}, indent=2) + "\n", encoding="utf-8")
    print(f"Inference complete: {out}")


def preflight(a: argparse.Namespace) -> None:
    """Validate frozen-model/Robot compatibility without producing predictions."""
    manifest = json.loads((a.model_output_dir / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest["participant"] != a.participant: raise ValueError("participant does not match frozen model manifest")
    lookup = {m["identifier"]: m for m in manifest["models"]}
    eeg = robot_eeg_table(a.robot_eeg_csv) if a.robot_eeg_csv else None
    video = pd.read_csv(a.robot_video_csv) if a.robot_video_csv else None
    if video is not None: video = video.loc[:, KEYS + [c for c in video if c.startswith("video_")]].copy()
    rows = []
    for spec in a.model_spec:
        metadata = lookup.get(spec.identifier)
        if metadata is None: raise ValueError(f"Frozen model absent: {spec.identifier}")
        frame = robot_input(spec, eeg, video)
        selected, candidates = metadata["selected_features"], metadata["candidate_features"]
        missing = sorted(set(candidates).difference(frame))
        valid = np.zeros(len(frame), dtype=bool) if missing else np.isfinite(frame[candidates].apply(pd.to_numeric, errors="raise").to_numpy()).all(axis=1)
        eeg_only = 0
        if spec.modality == "multimodal":
            eeg_only = int(len(eeg[KEYS].drop_duplicates().merge(video[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True).query("_merge == 'left_only'")))
        rows.append({"model_id": spec.identifier, "condition": a.condition, "modality": spec.modality,
                     "robot_windows": len(frame), "selected_feature_count": len(selected), "pipeline_input_feature_count": len(candidates), "missing_pipeline_features": ";".join(missing),
                     "eeg_windows_excluded_without_video": eeg_only, "valid_windows_with_finite_selected_features": int(valid.sum()),
                     "windows_excluded_nonfinite_selected_features": int((~valid).sum()), "compatible": bool(not missing and valid.any())})
    print(pd.DataFrame(rows).to_json(orient="records", indent=2))


def probability_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "condition", "target", "model_id", "model_rank", "modality", "classifier", "feature_request", "task"]
    return predictions.groupby(keys, dropna=False).agg(n_windows=("high_probability", "size"), mean_high_probability=("high_probability", "mean"), median_high_probability=("high_probability", "median"), sd_high_probability=("high_probability", "std"), fraction_predicted_high=("predicted_class", "mean")).reset_index()


def comparison(ica: pd.DataFrame, no_ica: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    keys = KEYS + ["target", "model_id", "model_rank", "modality", "classifier", "feature_request"]
    merged = ica.merge(no_ica, on=keys, suffixes=("_ica", "_no_ica"), validate="one_to_one")
    rows = []
    for group, frame in merged.groupby(["target", "model_id", "model_rank", "modality", "classifier", "feature_request", "task"], dropna=False):
        diff = np.abs(frame.high_probability_ica - frame.high_probability_no_ica)
        corr = frame.high_probability_ica.corr(frame.high_probability_no_ica) if len(frame) > 1 else np.nan
        rows.append(dict(zip(["target", "model_id", "model_rank", "modality", "classifier", "feature_request", "task"], group)) | {"n_matched_windows": len(frame), "mean_abs_probability_difference": diff.mean(), "median_abs_probability_difference": diff.median(), "maximum_abs_probability_difference": diff.max(), "pearson_probability_correlation": corr})
    result = pd.DataFrame(rows)
    selected = shift[shift.feature_group.eq("all_selected")]
    pivot = selected.pivot(index=["model_id", "task"], columns="condition", values=["median_nearest_image_distance", "fraction_feature_values_outside", "median_abs_image_z"])
    if not pivot.empty:
        for metric, label in [("median_nearest_image_distance", "lower_distance_condition"), ("fraction_feature_values_outside", "lower_outside_fraction_condition"), ("median_abs_image_z", "lower_median_abs_z_condition")]:
            values = pivot[metric]
            result = result.merge(pd.DataFrame({"model_id": values.index.get_level_values(0), "task": values.index.get_level_values(1), label: np.where(values["ICA"] < values["no-ICA"], "ICA", np.where(values["ICA"] > values["no-ICA"], "no-ICA", "tie"))}), on=["model_id", "task"], how="left")
    return result


def agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, target, task), group in predictions.groupby(["condition", "target", "task"]):
        wide = group.pivot(index=KEYS, columns="model_rank", values=["predicted_class", "high_probability"])
        ranks = sorted(group.model_rank.unique())
        labels = wide["predicted_class"][ranks]
        probabilities = wide["high_probability"][ranks]
        common = {"condition": condition, "target": target, "task": task, "n_all_three_windows": len(wide), "unanimous_agreement_rate": float((labels.nunique(axis=1) == 1).mean()), "unanimous_high_rate": float((labels == 1).all(axis=1).mean()), "unanimous_low_rate": float((labels == 0).all(axis=1).mean()), "disagreement_rate": float((labels.nunique(axis=1) > 1).mean()), "majority_vote_high_fraction": float((labels.sum(axis=1) >= 2).mean())}
        for left, right in [(1, 2), (1, 3), (2, 3)]:
            common[f"class_agreement_{left}_{right}"] = float((labels[left] == labels[right]).mean())
            common[f"probability_correlation_{left}_{right}"] = float(probabilities[left].corr(probabilities[right]))
        rows.append(common)
    return pd.DataFrame(rows)


def _model_label(row: pd.Series) -> str:
    request = "All" if str(row.feature_request) == "all" else f"Top-{row.feature_request}"
    return f"{row.modality.title()} {row.classifier.upper()} {request}"


def _segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Split sliding-window trajectories at missing/non-consecutive windows."""
    frame = frame.sort_values("window_center_s")
    starts = frame.window_start_s.to_numpy(dtype=float)
    breaks = np.r_[True, np.diff(starts) > 1.000001]
    return [part for _, part in frame.assign(_segment=breaks.cumsum()).groupby("_segment")]


def trajectory_figures(predictions: pd.DataFrame, out: Path, participant: str, shift: pd.DataFrame | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for target, data in predictions.groupby("target"):
        tasks = list(data.task.drop_duplicates()); fig, axes = plt.subplots(len(tasks), 1, figsize=(11, 2.15 * len(tasks)), sharex=False)
        axes = np.atleast_1d(axes)
        for axis, task in zip(axes, tasks):
            subset = data[data.task.eq(task)].copy(); subset["window_center_s"] = (subset.window_start_s + subset.window_end_s) / 2
            colors = {rank: color for rank, color in zip(sorted(subset.model_rank.unique()), ["#1b9e77", "#377eb8", "#984ea3"])}
            for (condition, rank, modality), frame in subset.groupby(["condition", "model_rank", "modality"]):
                if modality == "face" and condition != "ICA":
                    continue
                style = "-" if condition == "ICA" else "--"
                label = _model_label(frame.iloc[0]) + ("" if modality == "face" else f" — {condition}")
                first = True
                for segment in _segments(frame):
                    axis.plot(segment.window_center_s, segment.high_probability, color=colors[rank], ls=style, lw=.7, alpha=.55, marker="o", ms=2.2, label=label if first else None)
                    first = False
            # Descriptive consensus only: common windows across the three ranks.
            for condition, style in [("ICA", "-"), ("no-ICA", "--")]:
                view = subset[((subset.condition.eq(condition)) & (subset.modality.ne("face"))) | ((subset.modality.eq("face")) & (subset.condition.eq("ICA")))]
                wide = view.pivot(index="window_center_s", columns="model_rank", values="high_probability").dropna()
                if len(wide):
                    median, low, high = wide.median(axis=1), wide.min(axis=1), wide.max(axis=1)
                    axis.fill_between(wide.index, low, high, color="0.35", alpha=.10)
                    axis.plot(wide.index, median, color="0.15", ls=style, lw=2.0, alpha=.9, label=f"Top-3 median — {condition}")
            if shift is not None:
                flagged = shift[(shift.task.eq(task)) & shift.shift_flag]
                if len(flagged):
                    labels = [f"{condition}: " + ",".join("R" + str(rank) for rank in group.model_rank) for condition, group in flagged.groupby("condition")]
                    axis.text(.99, .05, "Shift warning: " + "; ".join(labels), ha="right", va="bottom", transform=axis.transAxes, fontsize=7, color="#8b0000")
            axis.axhline(.5, color="black", ls=":", lw=.9); axis.set_ylim(0, 1); axis.set_ylabel("P(HIGH)"); axis.set_title(task.replace("pick_place", "Pick & Place").replace("shape_sorter_observation", "Shape Sorter Observation").replace("shape_sorter_interaction", "Shape Sorter Interaction").replace("shape_sorter_alone", "Shape Sorter Alone").replace("sisyphus", "Sisyphus").replace("stack", "Stack"), loc="left", fontsize=9)
        axes[0].legend(ncol=2, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1)); axes[-1].set_xlabel("Time within task (s)")
        fig.suptitle(f"{participant} Robot: model probability of HIGH {target} (raw sliding windows)"); fig.tight_layout(rect=(0, 0, .79, .98))
        stem = out / f"{participant}_{target}_probability_timeline"; fig.savefig(stem.with_suffix(".png"), dpi=170); plt.close(fig)
        # Compact consensus companion, retaining no fitted ensemble semantics.
        fig, axes = plt.subplots(len(tasks), 1, figsize=(8.5, 1.65 * len(tasks))); axes = np.atleast_1d(axes)
        for axis, task in zip(axes, tasks):
            subset = data[data.task.eq(task)].copy(); subset["window_center_s"] = (subset.window_start_s + subset.window_end_s) / 2
            for condition, style in [("ICA", "-"), ("no-ICA", "--")]:
                view = subset[((subset.condition.eq(condition)) & (subset.modality.ne("face"))) | ((subset.modality.eq("face")) & (subset.condition.eq("ICA")))]
                wide = view.pivot(index="window_center_s", columns="model_rank", values="high_probability").dropna()
                if len(wide): axis.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color="0.35", alpha=.15); axis.plot(wide.index, wide.median(axis=1), color="0.15", ls=style, lw=1.6, label=condition)
            axis.axhline(.5, color="black", ls=":", lw=.8); axis.set_ylim(0, 1); axis.set_ylabel(task)
        axes[0].legend(fontsize=7); axes[-1].set_xlabel("Time within task (s)"); fig.suptitle(f"{participant} {target}: descriptive top-3 median P(HIGH)"); fig.tight_layout()
        stem = out / f"{participant}_{target}_probability_consensus"; fig.savefig(stem.with_suffix(".png"), dpi=170); plt.close(fig)


def robot_ratings(participant: str) -> pd.DataFrame:
    """Load canonical post-task Robot ratings for descriptive comparison only."""
    ratings = pd.read_csv(ROOT / "data" / "Robot Ratings - Robot Ratings.csv")
    rows = []
    task_names = {"pick_place": "Pick and Place", "shape_sorter_observation": "Shape Sorter Observation", "stack": "Stack", "sisyphus": "Sisyphus", "shape_sorter_interaction": "Shape Sorter Interaction", "shape_sorter_alone": "Shape Sorter Alone"}
    subset = ratings[ratings["Participant ID"].astype(str).eq(participant)]
    for task, label in task_names.items():
        row = subset[subset["Robot Experiment"].astype(str).str.casefold().eq(label.casefold())]
        if len(row) != 1: raise ValueError(f"Expected one canonical Robot rating for {participant} {label}; found {len(row)}")
        for target, column in (("valence", "Valence"), ("arousal", "Arousal")):
            value = float(row.iloc[0][column]); rows.append({"participant": participant, "task": task, "target": target, "robot_rating": value, "robot_rating_class": "HIGH" if value >= 4 else "LOW"})
    return pd.DataFrame(rows)


def shift_warning_table(shift: pd.DataFrame, metadata: pd.DataFrame, participant: str) -> pd.DataFrame:
    data = shift[shift.feature_group.eq("all_selected")].merge(metadata, on="model_id", validate="many_to_one")
    data["participant"] = participant
    data["model_label"] = data.apply(_model_label, axis=1)
    data["median_abs_z"] = data["median_abs_image_z"]
    data["out_of_range_fraction"] = data["fraction_feature_values_outside"]
    data["shift_flag"] = data["median_nearest_image_distance"] > 10.0
    data["shift_reason"] = np.where(data.shift_flag, "median nearest Image-training distance > 10 in selected standardized feature space", "")
    return data[["participant", "target", "task", "model_rank", "model_label", "condition", "median_nearest_image_distance", "median_abs_z", "out_of_range_fraction", "shift_flag", "shift_reason"]]


def verdict_summary(summary: pd.DataFrame, agreement_data: pd.DataFrame, ratings: pd.DataFrame, metadata: pd.DataFrame, warnings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, task, condition), group in summary.groupby(["target", "task", "condition"]):
        group = group.copy().sort_values("model_rank")
        group["model_label"] = group.apply(_model_label, axis=1)
        if len(group) != 3: continue
        medians = group.median_high_probability.to_numpy(); verdicts = np.where(medians >= .5, "HIGH", "LOW")
        row = {"participant": group.participant.iloc[0], "task": task, "target": target, "condition": condition, "n_valid_windows": int(group.n_windows.min())}
        for item in group.itertuples():
            prefix = f"rank{item.model_rank}"; verdict = "HIGH" if item.median_high_probability >= .5 else "LOW"
            row.update({f"{prefix}_model_label": item.model_label, f"{prefix}_mean_high_probability": item.mean_high_probability, f"{prefix}_median_high_probability": item.median_high_probability, f"{prefix}_fraction_high_windows": item.fraction_predicted_high, f"{prefix}_task_verdict": verdict})
        consensus = float(np.median(medians)); verdict = "HIGH" if consensus >= .5 else "LOW"
        row.update({"consensus_median_probability": consensus, "consensus_verdict": verdict, "models_voting_HIGH": int((verdicts == "HIGH").sum()), "models_voting_LOW": int((verdicts == "LOW").sum()), "unanimous_verdict": bool(len(set(verdicts)) == 1)})
        agree = agreement_data[(agreement_data.target.eq(target)) & (agreement_data.task.eq(task)) & (agreement_data.condition.eq(condition))]
        row["agreement_fraction"] = float(agree.iloc[0].unanimous_agreement_rate) if len(agree) else np.nan
        rate = ratings[(ratings.task.eq(task)) & (ratings.target.eq(target))].iloc[0]
        row.update({"robot_rating": rate.robot_rating, "robot_rating_class": rate.robot_rating_class, "consensus_matches_rating": bool(verdict == rate.robot_rating_class)})
        for rank, vote in enumerate(verdicts, 1): row[f"rank{rank}_matches_rating"] = bool(vote == rate.robot_rating_class)
        flag = warnings[(warnings.target.eq(target)) & (warnings.task.eq(task)) & (warnings.condition.eq(condition)) & warnings.shift_flag]
        row["shift_warning"] = "None" if flag.empty else f"{condition}: " + ", ".join("R" + str(x) for x in flag.model_rank)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        pairs = result.pivot(index=["participant", "task", "target"], columns="condition", values="consensus_verdict")
        result = result.merge(pd.DataFrame({"participant": pairs.index.get_level_values(0), "task": pairs.index.get_level_values(1), "target": pairs.index.get_level_values(2), "ica_no_ica_same_consensus_verdict": pairs["ICA"].eq(pairs["no-ICA"]).to_numpy()}), on=["participant", "task", "target"], how="left")
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(x) for x in frame.columns]
    rows = [[str(value) for value in row] for row in frame.fillna("").itertuples(index=False, name=None)]
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def report(a: argparse.Namespace) -> None:
    out = a.report_output_dir.resolve()
    if out.exists(): raise FileExistsError(f"Refusing to overwrite report output directory: {out}")
    ica_dir, no_dir = a.ica_inference_dir.resolve(), a.no_ica_inference_dir.resolve()
    ica, no = pd.read_csv(ica_dir / "window_predictions.csv"), pd.read_csv(no_dir / "window_predictions.csv")
    shift = pd.concat([pd.read_csv(ica_dir / "selected_feature_shift_task_summary.csv"), pd.read_csv(no_dir / "selected_feature_shift_task_summary.csv")], ignore_index=True)
    face = ica[ica.modality.eq("face")].merge(no[no.modality.eq("face")], on=KEYS + ["model_id"], suffixes=("_ica", "_no"), validate="one_to_one")
    face_max = float(np.abs(face.high_probability_ica-face.high_probability_no).max()) if len(face) else np.nan
    if face_max > 1e-12: raise RuntimeError(f"Face-only ICA/no-ICA control failed: max probability difference={face_max}")
    predictions = pd.concat([ica, no], ignore_index=True); summary = probability_summary(predictions); compare = comparison(ica, no, shift); agree = agreement(predictions)
    metadata = predictions[["model_id", "target", "model_rank", "modality", "classifier", "feature_request"]].drop_duplicates()
    warnings = shift_warning_table(shift, metadata, a.participant)
    verdicts = verdict_summary(summary, agree, robot_ratings(a.participant), metadata.assign(model_label=metadata.apply(_model_label, axis=1)), warnings)
    out.mkdir(parents=True); predictions.to_csv(out / f"{a.participant}_robot_window_predictions.csv", index=False)
    shift.to_csv(out / f"{a.participant}_robot_feature_shift.csv", index=False); warnings.to_csv(out / f"{a.participant}_robot_shift_warnings.csv", index=False); summary.to_csv(out / f"{a.participant}_robot_task_summary.csv", index=False); compare.to_csv(out / f"{a.participant}_ica_no_ica_comparison.csv", index=False); agree.to_csv(out / f"{a.participant}_top3_model_agreement.csv", index=False); verdicts.to_csv(out / f"{a.participant}_robot_task_verdict_summary.csv", index=False)
    trajectory_figures(predictions, out / "figures", a.participant, warnings)
    compact = verdicts[["task", "target", "condition", "consensus_median_probability", "consensus_verdict", "agreement_fraction", "robot_rating", "robot_rating_class", "consensus_matches_rating", "shift_warning"]].copy()
    compact.columns = ["Task", "Target", "Condition", "Median consensus P(HIGH)", "Consensus verdict", "Model agreement", "Robot rating", "Rating class", "Match?", "Shift warning"]
    lines = [f"# {a.participant} personalized Image-to-Robot inference", "", f"Face-only control maximum absolute ICA/no-ICA P(HIGH) difference: {face_max:.16g}.", "", "Predictions are frozen-model P(HIGH), not Robot emotion ground truth. Robot ratings are descriptive post-task comparisons only.", "", "## Task-level descriptive consensus", "", markdown_table(compact), "", "See numerical CSVs for task summaries, condition-specific shift warnings, ICA/no-ICA comparisons, and top-three agreement."]
    (out / f"{a.participant}_inference_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report complete: {out}")


def main() -> int:
    a = args()
    if a.stage == "select-models": select_models(a)
    else:
        a.model_spec = resolved_specs(a)
    if a.stage == "select-models": pass
    elif a.stage == "prepare-models": prepare_models(a)
    elif a.stage == "preflight": preflight(a)
    elif a.stage == "infer": infer(a)
    else: report(a)
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
