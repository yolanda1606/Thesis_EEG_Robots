#!/usr/bin/env python3
"""Minimal read-only robot pipeline: validation, health checks, and shared-crop review only."""
from __future__ import annotations

import argparse
import copy
import json
import platform as platform_module
import shutil
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import mne
import numpy as np
import yaml

from continuous import ProgressReporter, alignment_for_segmented_task, alignment_for_task, extract_segmented_task, extract_task, has_eeg_segments

HERE = Path(__file__).resolve().parent
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participant-config", type=Path, required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--raw-root", type=Path,
                   help="Override configured raw-data root. A participant directory remains supported for compatibility.")
    p.add_argument("--platform", choices=("windows", "linux"),
                   help="Raw-root platform; defaults to the current operating system.")
    p.add_argument("--tasks", nargs="+", choices=TASKS,
                   help="Optional subset of robot tasks; defaults to all six tasks.")
    p.add_argument("--resource-root", type=Path, default=HERE.parent / "multimodal_image" / "models")
    p.add_argument("--output-root", type=Path, default=Path("derived"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--overwrite-run", action="store_true")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--health-check-only", action="store_true")
    mode.add_argument("--crop-preview-only", action="store_true")
    mode.add_argument("--validate-alignment-only", action="store_true")
    mode.add_argument("--continuous-features", action="store_true")
    p.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Temporary shared crop candidate; does not change YAML.")
    p.add_argument("--save-crop", action="store_true", help="Save --crop to participant-level YAML after crop preview; remains unreviewed unless --approve-crop is supplied.")
    p.add_argument("--approve-crop", action="store_true", help="Explicitly approve a --save-crop candidate after preview.")
    p.add_argument("--show-progress", action="store_true", help="Show participant/task/stage progress for alignment and feature runs.")
    p.add_argument("--video-progress", action="store_true", help="Show throttled MediaPipe/video frame progress.")
    p.add_argument("--eeg-progress", action="store_true", help="Show major continuous EEG preparation and feature stages.")
    p.add_argument("--timing", action="store_true", help="Show elapsed time for major stages and tasks.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("Participant YAML must contain a mapping")
    return value


def raw_roots() -> dict[str, str]:
    """Load the shared platform-specific Robot raw-data roots."""
    value = load(HERE / "configs" / "raw_roots.yaml").get("raw_roots")
    if not isinstance(value, dict) or not all(isinstance(value.get(name), str) for name in ("windows", "linux")):
        raise ValueError("configs/raw_roots.yaml must define string windows and linux raw_roots")
    return {name: value[name] for name in ("windows", "linux")}


def resolved_platform(platform_name: str | None) -> str:
    if platform_name is not None:
        return platform_name
    return "windows" if platform_module.system().casefold().startswith("win") else "linux"


def relative_task_parts(value: str, participant_dir: str) -> tuple[str, ...]:
    """Return a configured task path beneath its participant directory."""
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    parts = windows_path.parts if not posix_path.is_absolute() else posix_path.parts
    participant_indices = [index for index, part in enumerate(parts) if part.casefold() == participant_dir.casefold()]
    if windows_path.is_absolute() or posix_path.is_absolute():
        if not participant_indices:
            raise ValueError(f"Configured task path is not beneath raw_participant_dir: {value}")
        parts = parts[participant_indices[-1]:]
    if not parts or parts[0].casefold() != participant_dir.casefold() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Configured task path must be relative to the raw-data root beneath {participant_dir}: {value}")
    return tuple(parts)


def localize_task_paths(
    config: dict[str, Any],
    raw_root: Path | None,
    platform_name: str | None = None,
    configured_roots: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve root-relative task files without altering the source YAML.

    Priority is ``--raw-root``, then the shared configured root for
    ``--platform`` (or the detected platform), then legacy configured paths.
    ``--raw-root`` may be either the raw-data root or, for compatibility, the
    participant directory accepted by earlier pipeline versions.
    """
    participant_dir = PureWindowsPath(str(config.get("raw_participant_dir", ""))).name
    if not participant_dir:
        raise ValueError("Participant YAML requires raw_participant_dir")
    selected_platform = resolved_platform(platform_name)
    root_is_participant_dir = False
    if raw_root is not None:
        selected_root: str | Path = raw_root.resolve(strict=True)
        root_is_participant_dir = raw_root.name.casefold() == participant_dir.casefold()
    else:
        roots = configured_roots if configured_roots is not None else raw_roots()
        root = roots.get(selected_platform)
        if root is None:
            return config
        selected_root = root

    def local_path(value: str) -> str:
        parts = relative_task_parts(value, participant_dir)
        if root_is_participant_dir:
            parts = parts[1:]
        if selected_platform == "windows" and raw_root is None:
            return str(PureWindowsPath(str(selected_root)).joinpath(*parts))
        return str(Path(selected_root).joinpath(*parts))

    def local_item(item: Any) -> Any:
        if isinstance(item, str):
            return local_path(item)
        if not isinstance(item, dict):
            return item
        result = copy.deepcopy(item)
        if isinstance(result.get("file"), str):
            result["file"] = local_path(result["file"])
        if isinstance(result.get("segments"), list):
            for segment in result["segments"]:
                if isinstance(segment, dict) and isinstance(segment.get("file"), str):
                    segment["file"] = local_path(segment["file"])
        return result

    result = copy.deepcopy(config)
    for task in result.get("tasks", {}).values():
        if isinstance(task, dict):
            for modality in ("eeg", "video", "vision_log", "robot_metrics"):
                if modality in task:
                    task[modality] = local_item(task[modality])
    return result


def selected_files(modality: Any) -> list[Path]:
    """Return canonical selected paths: a scalar path or an ordered segment list."""
    if isinstance(modality, str):
        return [Path(modality)]
    if not isinstance(modality, dict):
        return []
    if "segments" in modality:
        segments = modality["segments"]
        if not isinstance(segments, list):
            return []
        files = []
        for item in segments:
            value = item.get("file") if isinstance(item, dict) else item
            if isinstance(value, str):
                files.append(Path(value))
        return files
    return [Path(modality["file"])] if isinstance(modality.get("file"), str) else []


def unavailable(modality: Any) -> bool:
    return isinstance(modality, dict) and modality.get("unavailable") is True


def modality_reason(modality: Any) -> str:
    return modality.get("reason", "not selected") if isinstance(modality, dict) else "not selected"


def multimodal_available(task: dict[str, Any]) -> bool:
    return all(
        selected_files(task.get(name)) and not (isinstance(task.get(name), dict) and task[name].get("unusable"))
        for name in ("eeg", "video", "vision_log")
    )


def processing_skip_reason(task: dict[str, Any]) -> str | None:
    """Return an explicit configured reason not to process a multimodal task."""
    for name in ("eeg", "video", "vision_log"):
        item = task.get(name)
        if unavailable(item):
            return f"{name} unavailable: {modality_reason(item)}"
        if isinstance(item, dict) and item.get("unusable") is True:
            return f"{name} unusable: {modality_reason(item)}"
        if not selected_files(item):
            return f"{name} has no selected file"
    return None


def validate(config: dict[str, Any], participant: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []; warnings: list[str] = []
    if config.get("participant") != participant:
        errors.append("--participant does not match YAML participant")
    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        errors.append("YAML must contain exactly the six expected robot task entries")
        return errors, warnings
    video_settings = config.get("video_settings")
    crop = video_settings.get("crop") if isinstance(video_settings, dict) else None
    face = video_settings.get("face_detection") if isinstance(video_settings, dict) else None
    if not isinstance(crop, dict) or not isinstance(face, dict) or not isinstance(crop.get("reviewed"), bool) or not isinstance(face.get("crop_validated"), bool):
        errors.append("participant-level video_settings.crop and face_detection schema is invalid")
    else:
        values = [crop.get(k) for k in ("x", "y", "width", "height")]
        populated = [value is not None for value in values]
        if any(populated) and (not all(isinstance(v, int) for v in values) or crop["x"] < 0 or crop["y"] < 0 or crop["width"] <= 0 or crop["height"] <= 0):
            errors.append("configured participant crop must have non-negative x/y and positive integer width/height")
        if crop["reviewed"] and not all(populated):
            errors.append("reviewed participant crop must have coordinates")
    for task_id, task in tasks.items():
        if not isinstance(task, dict): errors.append(f"{task_id}: task must be a mapping"); continue
        if not isinstance(task.get("category"), str): errors.append(f"{task_id}: missing category")
        for name in ("eeg", "video", "vision_log"):
            item = task.get(name)
            if item is None: errors.append(f"{task_id}.{name}: missing selected modality"); continue
            files = selected_files(item)
            if unavailable(item):
                if files or not isinstance(item.get("reason"), str): errors.append(f"{task_id}.{name}: unavailable modality requires a reason and no selected path")
                continue
            if isinstance(item, dict) and "segments" in item:
                segments = item["segments"]
                if not isinstance(segments, list) or not segments or not all(isinstance(segment, dict) and isinstance(segment.get("file"), str) and isinstance(segment.get("order"), int) for segment in segments):
                    errors.append(f"{task_id}.{name}: segments must be non-empty ordered file mappings")
                elif [segment["order"] for segment in segments] != list(range(1, len(segments) + 1)):
                    errors.append(f"{task_id}.{name}: segment order must be consecutive starting at 1")
                if len(files) != len(set(files)): errors.append(f"{task_id}.{name}: duplicate segment path")
            if not files: errors.append(f"{task_id}.{name}: selected path is required unless explicitly unavailable")
            for path in files:
                if not path.is_file(): errors.append(f"{task_id}.{name}: selected path does not exist: {path}")
        metrics = task.get("robot_metrics")
        if metrics is not None:
            if unavailable(metrics):
                if selected_files(metrics) or not isinstance(metrics.get("reason"), str): errors.append(f"{task_id}.robot_metrics: unavailable modality requires a reason and no selected path")
            else:
                files = selected_files(metrics)
                if not files: errors.append(f"{task_id}.robot_metrics: selected path is required unless explicitly unavailable")
                for path in files:
                    if not path.is_file(): errors.append(f"{task_id}.robot_metrics: selected path does not exist: {path}")
    return errors, warnings


def eeg_check(item: Any, *, status_events_expected: bool = True) -> dict[str, Any]:
    files = selected_files(item)
    if not files: return {"status": "UNAVAILABLE", "reason": modality_reason(item)}
    records = []
    try:
        for path in files:
            raw = mne.io.read_raw_bdf(str(path), preload=False, verbose="ERROR")
            status = next((c for c in raw.ch_names if c.lower() == "status"), None)
            events = mne.find_events(raw, stim_channel=status, shortest_event=1, verbose="ERROR") if status else np.empty((0, 3), int)
            records.append({"file": str(path), "sampling_hz": float(raw.info["sfreq"]), "sample_count": int(raw.n_times), "duration_s": float(raw.n_times/raw.info["sfreq"]), "channel_count": len(raw.ch_names), "channels": raw.ch_names, "status_channel": status, "status_trigger_count": int(len(events)), "unique_status_codes": sorted({int(e[2]) for e in events})})
    except Exception as exc:
        return {"status": "ERROR", "reason": str(exc), "segments": records}
    status_events_found = any(row["status_trigger_count"] for row in records)
    warning = status_events_expected and not status_events_found
    return {
        "status": "PASS_WITH_WARNINGS" if warning else "PASS",
        "reason": "No BDF Status events" if warning else None,
        "status_events_expected": status_events_expected,
        "status_events_found": status_events_found,
        "segmented": len(records) > 1,
        "segments": records,
    }


def video_check(item: Any) -> dict[str, Any]:
    files = selected_files(item)
    if not files: return {"status": "UNAVAILABLE", "reason": modality_reason(item)}
    path = files[0]
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    metadata_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_indices: list[int] = []
    if opened:
        ok, _ = cap.read()
        if ok:
            decoded_indices.append(0)
        if metadata_frames > 1:
            for index in sorted({metadata_frames // 2, metadata_frames - 1} - {0}):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, _ = cap.read()
                if ok:
                    decoded_indices.append(index)
        else:
            # Some AVI codecs expose no reliable frame-count metadata.  Probe two
            # further sequential frames rather than declaring such files broken.
            for index in (1, 2):
                ok, _ = cap.read()
                if ok:
                    decoded_indices.append(index)
    cap.release()
    result = {
        "file": str(path),
        "fps": fps,
        "metadata_frame_count": metadata_frames,
        "decoded_frame_count": len(decoded_indices),
        "decoded_frame_indices": decoded_indices,
        "resolution": [width, height],
        "metadata_duration_s": metadata_frames / fps if metadata_frames > 0 and fps > 0 else None,
    }
    if not decoded_indices:
        return {"status": "ERROR", "reason": "no decodable frames", **result}
    return {"status": "PASS", **result}


def health(config: dict[str, Any]) -> dict[str, Any]:
    result = {"participant": config["participant"], "tasks": {}}
    for task_id, task in config["tasks"].items():
        log_files = selected_files(task["vision_log"]); metric_files = selected_files(task.get("robot_metrics"))
        eeg = eeg_check(task["eeg"], status_events_expected=task_id != "shape_sorter_alone")
        video = video_check(task["video"])
        vision_log = {"status": "PASS" if log_files and log_files[0].is_file() else "UNAVAILABLE", "file": str(log_files[0]) if log_files else None}
        result["tasks"][task_id] = {
            "eeg": eeg,
            "video": video,
            "vision_log": vision_log,
            "robot_metrics": {"status": "PASS" if metric_files and metric_files[0].is_file() else "NOT_APPLICABLE", "file": str(metric_files[0]) if metric_files else None},
            "multimodal_usable": eeg["status"] == "PASS" and video["status"] == "PASS" and vision_log["status"] == "PASS",
        }
    return result


def temporary_crop(config: dict[str, Any], candidate: list[int] | None) -> dict[str, Any] | None:
    if candidate: return dict(zip(("x", "y", "width", "height"), candidate))
    crop = config["video_settings"]["crop"]
    fields = ("x", "y", "width", "height")
    return {k: crop[k] for k in fields} if all(crop.get(k) is not None for k in fields) else None


def crop_preview(config: dict[str, Any], resource_root: Path, destination: Path, candidate: list[int] | None) -> dict[str, Any]:
    crop = temporary_crop(config, candidate); rows = []; skipped = []
    frames: list[tuple[str, np.ndarray]] = []
    for task_id, task in config["tasks"].items():
        files = selected_files(task["video"])
        if not files:
            skipped.append({"task": task_id, "reason": modality_reason(task["video"])})
            continue
        cap = cv2.VideoCapture(str(files[0]))
        if not cap.isOpened():
            cap.release()
            skipped.append({"task": task_id, "reason": "selected video could not be opened"})
            continue
        metadata_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if metadata_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, metadata_frames // 2)
        ok, frame = cap.read()
        if not ok and metadata_frames > 0:
            # A metadata-derived middle frame can be unreliable; try the first
            # decodable frame before excluding an otherwise readable video.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if ok:
            frames.append((task_id, frame))
        else:
            skipped.append({"task": task_id, "reason": f"no decodable representative frame (metadata_frame_count={metadata_frames})"})
    if not frames: raise ValueError("No readable usable task videos for crop preview")
    detector = None
    if crop:
        x, y, w, h = (crop[k] for k in ("x", "y", "width", "height"))
        if min(x, y) < 0 or w <= 0 or h <= 0: raise ValueError("Crop must have non-negative x/y and positive width/height")
        model = resource_root / "face_landmarker.task"
        if not model.is_file(): raise FileNotFoundError(f"Face-landmark model not found: {model}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.IMAGE, num_faces=1))
    panels = []; face_panels = []
    for task_id, frame in frames:
        panel = frame.copy(); detected = None
        if crop:
            x, y, w, h = (crop[k] for k in ("x", "y", "width", "height"))
            if x+w > frame.shape[1] or y+h > frame.shape[0]:
                if detector: detector.close()
                raise ValueError(f"Shared crop is outside {task_id} frame dimensions {frame.shape[1]}x{frame.shape[0]}")
            cv2.rectangle(panel, (x,y), (x+w,y+h), (0,180,255), 2)
            region = frame[y:y+h, x:x+w].copy(); image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(region, cv2.COLOR_BGR2RGB)); result = detector.detect(image); detected = bool(result.face_landmarks)
            if detected:
                for point in result.face_landmarks[0]: cv2.circle(region, (int(point.x * region.shape[1]), int(point.y * region.shape[0])), 1, (0, 220, 0), -1)
            original = cv2.resize(panel, (480,360)); cropped = cv2.resize(region, (480,360)); cv2.putText(cropped, f"face={'yes' if detected else 'no'}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2); face_panels.append(np.hstack([original, cropped]))
        label = f"{task_id}: face={'yes' if detected else 'no'}" if crop else f"{task_id}: no candidate crop configured"
        cv2.putText(panel, label, (10,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2); panels.append(cv2.resize(panel, (480,360)))
        rows.append({"task": task_id, "video": str(selected_files(config["tasks"][task_id]["video"])[0]), "sampled_frames": 1, "face_detections": int(bool(detected)), "face_detected": detected})
    if detector: detector.close()
    sheet = np.vstack(panels) if len(panels) == 1 else np.vstack([np.hstack(panels[i:i+2] if len(panels[i:i+2]) == 2 else [panels[i], np.zeros_like(panels[i])]) for i in range(0,len(panels),2)])
    destination.mkdir(parents=True, exist_ok=False); image_path = destination / "shared_crop_representative_frames.jpg"; cv2.imwrite(str(image_path), sheet)
    face_path = None
    if crop:
        face_sheet = np.vstack(face_panels) if len(face_panels) == 1 else np.vstack([np.hstack(face_panels[i:i+2] if len(face_panels[i:i+2]) == 2 else [face_panels[i], np.zeros_like(face_panels[i])]) for i in range(0,len(face_panels),2)])
        face_path = destination / "crop_face_landmark_validation.jpg"; cv2.imwrite(str(face_path), face_sheet)
    return {"crop": crop, "representative_tasks": rows, "skipped_tasks": skipped, "sampled_frame_count": len(rows), "face_detection_count": sum(row["face_detections"] for row in rows), "image": str(image_path), "face_validation_image": str(face_path) if face_path else None, "automatic_approval": False}


def continuous_run(config: dict[str, Any], participant: str, resource_root: Path, run: Path, extract: bool, task_ids: tuple[str, ...], progress: ProgressReporter | None = None) -> dict[str, Any]:
    """Validate all robot timelines, then optionally extract supported tasks."""
    alignment_rows: list[dict[str, Any]] = []; qcs: list[dict[str, Any]] = []
    eeg_tables = []; video_tables = []; merged_tables = []
    for task_index, task_id in enumerate(task_ids, start=1):
        task = config["tasks"][task_id]
        if progress:
            progress.task_start(task_index, len(task_ids), task_id)
        skip_reason = processing_skip_reason(task)
        if skip_reason:
            qc = {
                "task": task_id,
                "method": "unsupported_configured_modality",
                "skip_reason": skip_reason,
                "eeg_duration_s": None,
                "video_duration_s": None,
                "status_event_count": 0,
                "original_anchor_count": 0,
                "retained_anchor_count": 0,
                "rejected_anchor_count": 0,
                "matched_anchor_count": 0,
                "aligned_overlap_s": 0.0,
                "offset_s": None,
                "drift_s_per_s": None,
                "rmse_s": None,
                "max_residual_s": None,
                "median_absolute_residual_s": None,
            }
            qcs.append(qc)
            if progress:
                progress.task_skipped(task_id, skip_reason)
            else:
                print(f"{task_id}: skipped: {skip_reason}")
            continue
        segmented = has_eeg_segments(task)
        if segmented:
            qc, pairs, segment_results = alignment_for_segmented_task(task, task_id, str(config.get("raw_participant_dir", "")), progress)
            raws = []
        else:
            qc, pairs, raws = alignment_for_task(task, task_id, str(config.get("raw_participant_dir", "")), progress)
        qcs.append(qc); alignment_rows.extend(pairs)
        if progress and progress.show_progress:
            progress.alignment(qc)
        else:
            print(f"{task_id}: EEG={qc['eeg_duration_s']:.3f}s, video={qc['video_duration_s']:.3f}s, Status={qc['status_event_count']}, anchors={qc['matched_anchor_count']}, model={qc['method']}, offset={qc['offset_s']}, drift={qc['drift_s_per_s']}, RMSE={qc['rmse_s']}, max={qc['max_residual_s']}, overlap={qc['aligned_overlap_s']:.3f}s")
        if extract and (qc.get("accepted", True) if segmented else not qc["method"].startswith("unsupported")):
            prepared = dict(task); prepared["_crop"] = config["video_settings"]["crop"]
            if segmented:
                eeg, video, merged = extract_segmented_task(participant, task_id, prepared, segment_results, resource_root, progress)
            else:
                eeg, video, merged = extract_task(participant, task_id, prepared, qc, raws, resource_root, progress)
            if len(eeg): eeg_tables.append(eeg)
            if len(video): video_tables.append(video)
            if len(merged): merged_tables.append(merged)
        if progress:
            progress.task_done(task_id)
    align_dir = run / "alignment"; align_dir.mkdir(parents=True, exist_ok=False)
    (align_dir / "alignment_metadata.json").write_text(json.dumps({"tasks": qcs}, indent=2), encoding="utf-8")
    pd = __import__("pandas")
    pd.DataFrame(alignment_rows).to_csv(align_dir / "synchronization_anchors.csv", index=False)
    summary: dict[str, Any] = {
        "participant": participant,
        "mode": "continuous_features" if extract else "alignment_validation",
        "alignment": qcs,
        "eeg_preprocessing": {
            "reference": "common_average",
            "filter": {"l_freq_hz": 1.0, "h_freq_hz": 40.0, "method": "butterworth_iir", "order": 4},
            "ica": "disabled",
            "continuous_window": {"length_s": 2.0, "step_s": 1.0},
        },
    }
    if extract:
        feature_dir = run / "features"; feature_dir.mkdir(parents=True, exist_ok=False)
        eeg_df = pd.concat(eeg_tables, ignore_index=True) if eeg_tables else pd.DataFrame()
        video_df = pd.concat(video_tables, ignore_index=True) if video_tables else pd.DataFrame()
        merged_df = pd.concat(merged_tables, ignore_index=True) if merged_tables else pd.DataFrame()
        if {"epoch_index", "trigger"}.intersection(eeg_df.columns) or {"epoch_index", "trigger"}.intersection(video_df.columns):
            raise RuntimeError("Image-specific epoch/trigger fields leaked into robot features")
        if len(eeg_df) and eeg_df.duplicated(["participant", "task", "segment_id", "window_id", "channel"]).any():
            raise RuntimeError("Duplicate EEG participant/task/window/channel IDs")
        eeg_feature_columns = [name for name in eeg_df if name.startswith("eeg_")]
        if len(eeg_df) and not np.isfinite(eeg_df[eeg_feature_columns].to_numpy(dtype=float)).all():
            raise RuntimeError("Non-finite EEG feature value")
        for name, frame in (("eeg_window_features.csv", eeg_df), ("video_window_features.csv", video_df), ("multimodal_window_features.csv", merged_df)):
            frame.to_csv(feature_dir / name, index=False)
        if len(eeg_df) and (eeg_df["sample_count"] != 500).any(): raise RuntimeError("EEG window sample count validation failed")
        if len(merged_df) and merged_df["window_id"].duplicated().any(): raise RuntimeError("Duplicate multimodal window IDs")
        summary["window_counts"] = {"eeg_rows": int(len(eeg_df)), "video_windows": int(len(video_df)), "multimodal_windows": int(len(merged_df))}
    return summary



def main() -> int:
    a = args()
    if a.save_crop and not a.crop: raise ValueError("--save-crop requires --crop")
    if a.approve_crop and not a.save_crop: raise ValueError("--approve-crop requires --save-crop")
    if a.save_crop and not a.crop_preview_only: raise ValueError("--save-crop is available only with --crop-preview-only")
    if not a.run_name or a.run_name in {".", ".."} or "/" in a.run_name or "\\" in a.run_name:
        raise ValueError("--run-name must be one directory name")
    config_path = a.participant_config.resolve(strict=True)
    config = localize_task_paths(load(config_path), a.raw_root, a.platform)
    errors, warnings = validate(config, a.participant)
    if errors: raise ValueError("Validation failed: " + " | ".join(errors))
    for warning in warnings: print(f"WARNING: {warning}")
    root = a.output_root.resolve(); run = root / a.participant / "Robot_Experiment" / ("health_checks" if a.health_check_only else "runs") / a.run_name
    if run.exists():
        if not a.overwrite_run: raise FileExistsError(f"Run exists: {run}")
        shutil.rmtree(run)
    run.mkdir(parents=True); (run / "resolved_participant.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {"participant": a.participant, "mode": "validate"}
    if a.health_check_only: summary = health(config); summary["mode"] = "health_check"
    if a.crop_preview_only:
        summary = crop_preview(config, a.resource_root.resolve(), run / "video" / "crop_preview", a.crop); summary["mode"] = "crop_preview"
        if a.save_crop:
            config["video_settings"]["crop"] = {"reviewed": bool(a.approve_crop), **dict(zip(("x","y","width","height"), a.crop))}
            config["video_settings"]["face_detection"] = {"crop_validated": bool(a.approve_crop), "notes": "Manually supplied shared crop; explicitly approved." if a.approve_crop else "Manually supplied shared crop; requires manual review."}
            with config_path.open("w", encoding="utf-8") as f: yaml.safe_dump(config, f, sort_keys=False)
            (run / "resolved_participant.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if a.validate_alignment_only or a.continuous_features:
        task_ids = tuple(a.tasks) if a.tasks else TASKS
        progress = ProgressReporter(a.show_progress, a.video_progress, a.eeg_progress, a.timing)
        progress.participant_start(a.participant, len(task_ids))
        summary = continuous_run(config, a.participant, a.resource_root.resolve(), run, a.continuous_features, task_ids, progress)
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if a.validate_alignment_only or a.continuous_features:
        progress.participant_done(a.participant, run)
    if a.health_check_only:
        for task_id, task in summary["tasks"].items():
            print(f"{task_id}: EEG={task['eeg']['status']}, video={task['video']['status']}, log={task['vision_log']['status']}")
    if a.crop_preview_only:
        for row in summary["representative_tasks"]:
            print(f"{row['task']}: face detections={row['face_detections']}/{row['sampled_frames']}")
        for row in summary["skipped_tasks"]:
            print(f"{row['task']}: skipped crop preview: {row['reason']}")
        print(f"Crop-preview detections: {summary['face_detection_count']}/{summary['sampled_frame_count']} representative frames")
    print(f"{a.participant} {summary['mode']} complete: {run}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc: print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
