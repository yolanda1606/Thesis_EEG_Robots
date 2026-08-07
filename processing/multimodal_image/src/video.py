from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision

from .validation import available_image_codes


LANDMARK_COUNT = 478


def _crop_config(config: dict[str, Any]) -> dict[str, Any]:
    crop = config["video"]["crop"]
    if not (0 <= crop["left"] < crop["right"] and 0 <= crop["top"] < crop["bottom"]):
        raise ValueError("Crop coordinates are invalid")
    return crop


def _landmarker(config: dict[str, Any], model: Path, mode):
    settings = config["video"]["face_landmarker"]
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=mode, num_faces=1,
        min_face_detection_confidence=settings["min_detection_confidence"],
        min_face_presence_confidence=settings["min_presence_confidence"],
        min_tracking_confidence=settings["min_tracking_confidence"],
    ))


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def create_crop_preview(paths: dict[str, Path], config: dict[str, Any], destination: Path, logger) -> list[str]:
    """Write five representative original/crop/landmark preview images only."""
    crop = _crop_config(config)
    capture = cv2.VideoCapture(str(paths["video"]))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = sorted({0, total // 4, total // 2, 3 * total // 4, max(total - 1, 0)})
    # The run setup creates the empty planned folder. Refuse to place previews
    # beside any earlier preview rather than overwriting it.
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Crop-preview directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    with _landmarker(config, paths["model"], vision.RunningMode.IMAGE) as landmarker:
        for index, position in enumerate(positions, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read preview frame {position}")
            original = frame.copy()
            cv2.rectangle(original, (crop["left"], crop["top"]), (crop["right"], crop["bottom"]), (0, 180, 255), 2)
            cv2.putText(original, f"Original frame {position + 1}: proposed crop", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            region = frame[crop["top"]:crop["bottom"], crop["left"]:crop["right"]]
            enlarged = cv2.resize(region, None, fx=float(crop["scale"]), fy=float(crop["scale"]), interpolation=cv2.INTER_CUBIC)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)))
            for landmark in result.face_landmarks[:1]:
                for point in landmark:
                    x, y = int(point.x * enlarged.shape[1]), int(point.y * enlarged.shape[0])
                    cv2.circle(enlarged, (x, y), 1, (0, 220, 0), -1)
            label = "Green points: detected landmarks" if result.face_landmarks else "No landmarks detected"
            cv2.putText(enlarged, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            combined = np.hstack([_fit(original, 640, 480), _fit(enlarged, 640, 480)])
            out = destination / f"crop_preview_{index:02d}_frame_{position + 1:05d}.jpg"
            if not cv2.imwrite(str(out), combined):
                raise RuntimeError(f"Could not write crop preview: {out}")
            outputs.append(str(out))
    capture.release()
    logger.info("Wrote %d crop-preview images to %s", len(outputs), destination)
    return outputs


def _vision_log(path: Path, config: dict[str, Any]) -> tuple[dict[int, dict[str, str]], dict[int, float]]:
    codes = available_image_codes(config)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_frame = {int(float(row["Frame_Count"])): row for row in rows}
    onsets = {int(float(row["Trigger"])): float(row["Experiment_Time"]) for row in rows
              if row.get("Trigger", "").strip() not in ("", "0", "0.0") and int(float(row["Trigger"])) in codes}
    return by_frame, onsets


def _facial_measures(landmarks: np.ndarray, indices: dict[str, int], width: int, height: int) -> dict[str, float]:
    xy = landmarks[:, :2] * np.asarray([width, height], dtype=float)
    point = lambda name: xy[indices[name]]
    distance = lambda a, b: float(np.linalg.norm(a - b))
    scale = distance(point("left_eye_outer"), point("right_eye_outer"))
    if scale <= 0:
        return {name: float("nan") for name in ("irisdo_norm", "eso_norm", "enso_norm", "mnso_norm", "mwo_norm")}
    irisdo = (distance(point("left_upper_eyelid"), point("left_lower_eyelid")) + distance(point("right_upper_eyelid"), point("right_lower_eyelid"))) / 2.0
    iris_mid = (point("left_iris_center") + point("right_iris_center")) / 2.0
    return {"irisdo_norm": irisdo / scale, "eso_norm": distance(point("left_iris_center"), point("right_iris_center")) / scale,
            "enso_norm": distance(iris_mid, point("subnasale_approximation")) / scale,
            "mnso_norm": distance(point("upper_lip_center"), point("subnasale_approximation")) / scale,
            "mwo_norm": distance(point("mouth_left"), point("mouth_right")) / scale}


def process_video(paths: dict[str, Path], config: dict[str, Any], landmarks_path: Path, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    crop = _crop_config(config)
    if not crop["approved"]:
        raise ValueError("Crop is not approved. Run --crop-preview-only and approve the coordinates first.")
    log_by_frame, onsets = _vision_log(paths["vision_log"], config)
    capture = cv2.VideoCapture(str(paths["video"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_input_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    window = float(config["video"]["analysis_window_s"])
    expected_selected_frames = int(round(len(onsets) * window * fps))
    logger.info("Video landmarks: starting %d input frames; approximately %d image-window frames will be analysed", total_input_frames, expected_selected_frames)
    frames: list[dict[str, Any]] = []
    landmarks: list[np.ndarray] = []
    frame_number = 0
    with _landmarker(config, paths["model"], vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            log_row = log_by_frame.get(frame_number, {})
            experiment_time = float(log_row.get("Experiment_Time", (frame_number - 1) / fps))
            active = [trigger for trigger, onset in onsets.items() if onset <= experiment_time < onset + window]
            if not active:
                continue
            trigger = active[0]
            region = frame[crop["top"]:crop["bottom"], crop["left"]:crop["right"]]
            enlarged = cv2.resize(region, None, fx=float(crop["scale"]), fy=float(crop["scale"]), interpolation=cv2.INTER_CUBIC)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)), int(round(experiment_time * 1000)))
            row: dict[str, Any] = {"trigger": trigger, "frame_number": frame_number, "timestamp": log_row.get("Timestamp", ""),
                                   "experiment_time_s": experiment_time, "face_detected": False}
            array = np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
            if result.face_landmarks:
                array = np.asarray([[point.x, point.y, point.z] for point in result.face_landmarks[0]], dtype=np.float32)
                if array.shape == (LANDMARK_COUNT, 3):
                    row["face_detected"] = True
                    row.update(_facial_measures(array, config["video"]["landmark_indices"], enlarged.shape[1], enlarged.shape[0]))
            frames.append(row)
            landmarks.append(array)
            if len(frames) % int(config["output"]["video_progress_interval_frames"]) == 0:
                logger.info("Video landmarks: %d/%d image-window frames analysed", len(frames), expected_selected_frames)
    capture.release()
    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    if landmarks_path.exists():
        raise FileExistsError(f"Refusing to overwrite landmarks: {landmarks_path}")
    np.savez_compressed(landmarks_path, landmarks=np.stack(landmarks), frame_number=np.asarray([row["frame_number"] for row in frames]),
                        trigger=np.asarray([row["trigger"] for row in frames]), coordinate_order=np.asarray(["x_normalized", "y_normalized", "z_normalized"]))
    frame_table = pd.DataFrame(frames)
    measures = ["irisdo_norm", "eso_norm", "enso_norm", "mnso_norm", "mwo_norm"]
    summaries = []
    for trigger, group in frame_table.groupby("trigger"):
        detected = group[group["face_detected"]]
        item = {"trigger": int(trigger), "video_frame_count": int(len(group)), "face_detected_frames": int(len(detected)), "face_detection_rate": float(len(detected) / len(group))}
        for measure in measures:
            item[f"video_{measure}_mean"] = float(detected[measure].mean()) if measure in detected else float("nan")
            item[f"video_{measure}_std"] = float(detected[measure].std()) if measure in detected else float("nan")
        summaries.append(item)
    logger.info("Video processing retained %d image-window frames across %d trials", len(frame_table), len(summaries))
    return frame_table, pd.DataFrame(summaries)
