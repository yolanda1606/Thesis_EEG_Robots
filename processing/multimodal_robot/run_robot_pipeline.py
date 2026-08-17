#!/usr/bin/env python3
"""Minimal read-only robot pipeline: validation, health checks, and shared-crop review only."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import mne
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
TASKS = ("pick_place", "shape_sorter_observation", "stack", "sisyphus", "shape_sorter_interaction", "shape_sorter_alone")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participant-config", type=Path, required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--resource-root", type=Path, default=HERE.parent / "multimodal_image" / "models")
    p.add_argument("--output-root", type=Path, default=Path("derived"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--overwrite-run", action="store_true")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--health-check-only", action="store_true")
    mode.add_argument("--crop-preview-only", action="store_true")
    p.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Temporary shared crop candidate; does not change YAML.")
    p.add_argument("--save-crop", action="store_true", help="Save --crop to participant-level YAML after crop preview; remains unreviewed unless --approve-crop is supplied.")
    p.add_argument("--approve-crop", action="store_true", help="Explicitly approve a --save-crop candidate after preview.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("Participant YAML must contain a mapping")
    return value


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


def eeg_check(item: Any) -> dict[str, Any]:
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
    warning = not any(row["status_trigger_count"] for row in records)
    return {"status": "PASS_WITH_WARNINGS" if warning else "PASS", "reason": "No BDF Status events" if warning else None, "segmented": len(records) > 1, "segments": records}


def video_check(item: Any) -> dict[str, Any]:
    files = selected_files(item)
    if not files: return {"status": "UNAVAILABLE", "reason": modality_reason(item)}
    path = files[0]; cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened(); fps = float(cap.get(cv2.CAP_PROP_FPS)); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); ok, _ = cap.read(); cap.release()
    if not opened or fps <= 0 or frames <= 0 or not ok: return {"status": "ERROR", "file": str(path), "reason": "zero-frame/broken video", "fps": fps, "frame_count": frames, "resolution": [w, h]}
    return {"status": "PASS", "file": str(path), "fps": fps, "frame_count": frames, "duration_s": frames/fps, "resolution": [w, h]}


def health(config: dict[str, Any]) -> dict[str, Any]:
    result = {"participant": config["participant"], "tasks": {}}
    for task_id, task in config["tasks"].items():
        log_files = selected_files(task["vision_log"]); metric_files = selected_files(task.get("robot_metrics"))
        result["tasks"][task_id] = {"eeg": eeg_check(task["eeg"]), "video": video_check(task["video"]), "vision_log": {"status": "PASS" if log_files and log_files[0].is_file() else "UNAVAILABLE", "file": str(log_files[0]) if log_files else None}, "robot_metrics": {"status": "PASS" if metric_files and metric_files[0].is_file() else "NOT_APPLICABLE", "file": str(metric_files[0]) if metric_files else None}, "multimodal_usable": multimodal_available(task)}
    return result


def temporary_crop(config: dict[str, Any], candidate: list[int] | None) -> dict[str, Any] | None:
    if candidate: return dict(zip(("x", "y", "width", "height"), candidate))
    crop = config["video_settings"]["crop"]
    fields = ("x", "y", "width", "height")
    return {k: crop[k] for k in fields} if all(crop.get(k) is not None for k in fields) else None


def crop_preview(config: dict[str, Any], resource_root: Path, destination: Path, candidate: list[int] | None) -> dict[str, Any]:
    crop = temporary_crop(config, candidate); rows = []
    frames: list[tuple[str, np.ndarray]] = []
    for task_id, task in config["tasks"].items():
        if unavailable(task["video"]): continue
        files = selected_files(task["video"])
        if not files: continue
        cap = cv2.VideoCapture(str(files[0])); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2); ok, frame = cap.read(); cap.release()
        if ok: frames.append((task_id, frame))
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
    return {"crop": crop, "representative_tasks": rows, "sampled_frame_count": len(rows), "face_detection_count": sum(row["face_detections"] for row in rows), "image": str(image_path), "face_validation_image": str(face_path) if face_path else None, "automatic_approval": False}



def main() -> int:
    a = args()
    if a.save_crop and not a.crop: raise ValueError("--save-crop requires --crop")
    if a.approve_crop and not a.save_crop: raise ValueError("--approve-crop requires --save-crop")
    if a.save_crop and not a.crop_preview_only: raise ValueError("--save-crop is available only with --crop-preview-only")
    if not a.run_name or a.run_name in {".", ".."} or "/" in a.run_name or "\\" in a.run_name:
        raise ValueError("--run-name must be one directory name")
    config_path = a.participant_config.resolve(strict=True); config = load(config_path)
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
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if a.health_check_only:
        for task_id, task in summary["tasks"].items():
            print(f"{task_id}: EEG={task['eeg']['status']}, video={task['video']['status']}, log={task['vision_log']['status']}")
    if a.crop_preview_only:
        for row in summary["representative_tasks"]:
            print(f"{row['task']}: face detections={row['face_detections']}/{row['sampled_frames']}")
        print(f"Crop-preview detections: {summary['face_detection_count']}/{summary['sampled_frame_count']} representative frames")
    print(f"{a.participant} {summary['mode']} complete: {run}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc: print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
