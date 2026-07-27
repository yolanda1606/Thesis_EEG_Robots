#!/usr/bin/env python3
"""Non-destructive offline face-landmark extraction for an image baseline video."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

# Avoid MediaPipe/Matplotlib trying to cache under a read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


OUTPUT_VIDEO = "overlay_face_mesh_IMAGE_BASELINE.avi"
OUTPUT_FEATURES = "face_features_IMAGE_BASELINE.csv"
OUTPUT_LANDMARKS = "face_landmarks_IMAGE_BASELINE.npz"
LANDMARK_COUNT = 478

# MediaPipe canonical landmark indices.
KEY_POINTS = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "chin": 152,
    "mouth_upper": 13,
    "mouth_lower": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_brow": 105,
    "left_upper_eye": 159,
    "right_brow": 334,
    "right_upper_eye": 386,
    "left_lower_eyelid": 145,
    "right_lower_eyelid": 374,
    "left_iris_center": 468,
    "right_iris_center": 473,
    "subnasale": 2,
    "upper_lip_center": 0,
}

CONTOURS = [
    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
     379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
     127, 162, 21, 54, 103, 67, 109, 10],
    [33, 160, 158, 133, 153, 144, 33],
    [263, 387, 385, 362, 380, 373, 263],
    [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
     17, 84, 181, 91, 146, 61],
    [70, 63, 105, 66, 107],
    [336, 296, 334, 293, 300],
]

CSV_FIELDS = [
    "participant", "profile", "frame_count", "timestamp", "experiment_time",
    "trigger", "face_detected", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "eye_distance_px", "mouth_open_norm", "mouth_width_norm", "nose_chin_norm",
    "left_brow_eye_norm", "right_brow_eye_norm",
    "irisdo_px", "eso_px", "enso_px", "mnso_px", "mwo_px",
]


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).resolve().parents[1] / "multimodal_image" / "models" / "face_landmarker.task"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_presence_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    video = args.video.expanduser().resolve(strict=True)
    log = args.log.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()

    if not video.is_file() or not log.is_file() or not model.is_file():
        raise ValueError("Video, log, and model must each be regular files")
    if out_dir in (video.parent, log.parent):
        raise ValueError("Output directory must be a dedicated subdirectory")
    if out_dir in video.parents or out_dir in log.parents:
        raise ValueError("An input file cannot be located inside the output directory")

    final_paths = [out_dir / name for name in (OUTPUT_VIDEO, OUTPUT_FEATURES, OUTPUT_LANDMARKS)]
    existing = [str(path) for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(existing))
    return video, log, model, out_dir


def normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip().lower(): (value.strip() if value is not None else "")
            for key, value in row.items()}


def load_log(path: Path) -> tuple[dict[int, dict[str, str]], list[dict[str, str]]]:
    by_frame: dict[int, dict[str, str]] = {}
    ordered: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = normalized_row(raw_row)
            ordered.append(row)
            value = row.get("frame_count", "")
            try:
                by_frame[int(float(value))] = row
            except ValueError:
                pass
    return by_frame, ordered


def point_xy(landmarks: np.ndarray, index: int, width: int, height: int) -> np.ndarray:
    return landmarks[index, :2] * np.array([width, height], dtype=np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def extract_features(landmarks: np.ndarray, width: int, height: int) -> dict[str, float | int]:
    pixel_points = {name: point_xy(landmarks, idx, width, height)
                    for name, idx in KEY_POINTS.items()}
    eye_distance = distance(pixel_points["left_eye_outer"], pixel_points["right_eye_outer"])
    scale = eye_distance if eye_distance > 1e-6 else math.nan
    all_xy = landmarks[:, :2] * np.array([width, height], dtype=np.float32)
    min_xy = np.floor(np.nanmin(all_xy, axis=0)).astype(int)
    max_xy = np.ceil(np.nanmax(all_xy, axis=0)).astype(int)
    min_xy = np.maximum(min_xy, [0, 0])
    max_xy = np.minimum(max_xy, [width - 1, height - 1])

    def norm(a: str, b: str) -> float:
        return distance(pixel_points[a], pixel_points[b]) / scale

    iris_midpoint = (
        pixel_points["left_iris_center"] + pixel_points["right_iris_center"]
    ) / 2.0
    left_eyelid_opening = distance(
        pixel_points["left_upper_eye"], pixel_points["left_lower_eyelid"]
    )
    right_eyelid_opening = distance(
        pixel_points["right_upper_eye"], pixel_points["right_lower_eyelid"]
    )

    return {
        "bbox_x": int(min_xy[0]), "bbox_y": int(min_xy[1]),
        "bbox_w": int(max_xy[0] - min_xy[0] + 1),
        "bbox_h": int(max_xy[1] - min_xy[1] + 1),
        "eye_distance_px": eye_distance,
        "mouth_open_norm": norm("mouth_upper", "mouth_lower"),
        "mouth_width_norm": norm("mouth_left", "mouth_right"),
        "nose_chin_norm": norm("nose_tip", "chin"),
        "left_brow_eye_norm": norm("left_brow", "left_upper_eye"),
        "right_brow_eye_norm": norm("right_brow", "right_upper_eye"),
        "irisdo_px": (left_eyelid_opening + right_eyelid_opening) / 2.0,
        "eso_px": distance(pixel_points["left_iris_center"],
                           pixel_points["right_iris_center"]),
        "enso_px": distance(iris_midpoint, pixel_points["subnasale"]),
        "mnso_px": distance(pixel_points["upper_lip_center"],
                            pixel_points["subnasale"]),
        "mwo_px": distance(pixel_points["mouth_left"], pixel_points["mouth_right"]),
    }


def draw_overlay(frame: np.ndarray, landmarks: np.ndarray | None,
                 features: dict[str, float | int] | None, frame_count: int,
                 trigger: str) -> None:
    height, width = frame.shape[:2]
    if landmarks is None or features is None:
        cv2.putText(frame, "NO FACE DETECTED", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        xy = np.rint(landmarks[:, :2] * [width, height]).astype(int)
        for x, y in xy:
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 1, (80, 180, 80), -1, cv2.LINE_AA)
        for contour in CONTOURS:
            pts = xy[contour].reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 220, 255), 1, cv2.LINE_AA)
        for idx in KEY_POINTS.values():
            x, y = xy[idx]
            cv2.circle(frame, (int(x), int(y)), 3, (255, 80, 255), -1, cv2.LINE_AA)
        x, y = int(features["bbox_x"]), int(features["bbox_y"])
        w, h = int(features["bbox_w"]), int(features["bbox_h"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)

    cv2.rectangle(frame, (0, 0), (min(width, 520), 58), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {frame_count}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Trigger: {trigger if trigger != '' else 'N/A'}", (12, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def blank_feature_row(args: argparse.Namespace, frame_count: int,
                      log_row: dict[str, str]) -> dict[str, object]:
    return {
        "participant": args.participant, "profile": args.profile,
        "frame_count": frame_count, "timestamp": log_row.get("timestamp", ""),
        "experiment_time": log_row.get("experiment_time", ""),
        "trigger": log_row.get("trigger", ""), "face_detected": 0,
        "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
        "eye_distance_px": "", "mouth_open_norm": "", "mouth_width_norm": "",
        "nose_chin_norm": "", "left_brow_eye_norm": "", "right_brow_eye_norm": "",
        "irisdo_px": "", "eso_px": "", "enso_px": "", "mnso_px": "", "mwo_px": "",
    }


def main() -> int:
    args = parse_args()
    video_path, log_path, model_path, out_dir = validate_paths(args)
    log_by_frame, log_ordered = load_log(log_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid FPS or dimensions")

    out_dir.mkdir(parents=True, exist_ok=True)
    final_video = out_dir / OUTPUT_VIDEO
    final_csv = out_dir / OUTPUT_FEATURES
    final_npz = out_dir / OUTPUT_LANDMARKS
    temp_video = out_dir / (".partial_" + OUTPUT_VIDEO)
    temp_csv = out_dir / (".partial_" + OUTPUT_FEATURES)
    temp_npz = out_dir / (".partial_" + OUTPUT_LANDMARKS)
    temp_paths = [temp_video, temp_csv, temp_npz]
    if any(path.exists() for path in temp_paths):
        capture.release()
        raise FileExistsError("Partial output exists; remove it manually before retrying")

    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not open the overlay video writer")

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=args.min_detection_confidence,
        min_face_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    landmarks_by_frame: list[np.ndarray] = []
    frame_count = 0
    detected_count = 0

    try:
        with temp_csv.open("x", encoding="utf-8", newline="") as csv_handle, \
                vision.FaceLandmarker.create_from_options(options) as landmarker:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_count += 1
                log_row = log_by_frame.get(
                    frame_count,
                    log_ordered[frame_count - 1] if frame_count <= len(log_ordered) else {},
                )
                row = blank_feature_row(args, frame_count, log_row)
                timestamp_ms = int(round((frame_count - 1) * 1000.0 / fps))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)

                frame_landmarks: np.ndarray | None = None
                features = None
                if result.face_landmarks:
                    candidate = np.asarray(
                        [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]],
                        dtype=np.float32,
                    )
                    if candidate.shape == (LANDMARK_COUNT, 3):
                        frame_landmarks = candidate
                        features = extract_features(candidate, width, height)
                        row.update(features)
                        row["face_detected"] = 1
                        detected_count += 1

                if frame_landmarks is None:
                    frame_landmarks = np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
                    overlay_landmarks = None
                else:
                    overlay_landmarks = frame_landmarks
                landmarks_by_frame.append(frame_landmarks)
                draw_overlay(frame, overlay_landmarks, features, frame_count,
                             str(row["trigger"]))
                writer.write(frame)
                csv_writer.writerow(row)

        np.savez_compressed(
            temp_npz,
            landmarks=np.stack(landmarks_by_frame, axis=0),
            frame_count=np.arange(1, frame_count + 1, dtype=np.int64),
            participant=np.asarray(args.participant),
            profile=np.asarray(args.profile),
            coordinate_order=np.asarray(["x_normalized", "y_normalized", "z_normalized"]),
        )
        # NumPy appends .npz when the supplied path does not end in exactly '.npz'.
        actual_temp_npz = Path(str(temp_npz) + ".npz")
        if actual_temp_npz.exists():
            temp_npz = actual_temp_npz
        writer.release()
        capture.release()
        temp_video.replace(final_video)
        temp_csv.replace(final_csv)
        temp_npz.replace(final_npz)
    except Exception:
        writer.release()
        capture.release()
        for path in temp_paths + [Path(str(temp_npz) + ".npz")]:
            if path.exists():
                path.unlink()
        raise

    detection_pct = 100.0 * detected_count / frame_count if frame_count else 0.0
    print(f"Overlay video: {final_video}")
    print(f"Feature CSV: {final_csv}")
    print(f"Landmarks NPZ: {final_npz}")
    print(f"Total frames processed: {frame_count}")
    print(f"Face detected: {detected_count}/{frame_count} ({detection_pct:.2f}%)")
    if detected_count < frame_count:
        print(f"WARNING: No face detected in {frame_count - detected_count} frame(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
