#!/usr/bin/env python3
"""Read P01 videos and assess whether facial landmarks are usable.

This script never writes to the raw-data tree.  It samples frames from every
P01 video and writes derived summaries only to an output directory outside
``data/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from process_image_baseline_face_features import LANDMARK_COUNT, extract_features


RESULT_COLUMNS = [
    "participant", "task", "video_file", "log_file", "profile",
    "sampled_frame", "video_time_s", "face_detected", "bbox_w", "bbox_h",
    "face_width_fraction", "face_height_fraction",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participant-dir", type=Path,
        default=root / "data" / "P01_2026-05-26",
        help="Raw participant directory to inspect (read-only).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=root / "analysis" / "outputs" / "P01_face_usability",
        help="New derived-output directory; it must not be inside data/.",
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parents[1] / "multimodal_image" / "models" / "face_landmarker.task",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=10,
        help="Analyse every Nth frame; default 10 samples roughly three frames per second.",
    )
    parser.add_argument(
        "--roi", default=None,
        help="Optional crop as left,top,right,bottom pixels before detection.",
    )
    parser.add_argument(
        "--roi-scale", type=float, default=1.0,
        help="Optional enlargement applied to the crop before detection.",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    participant_dir = args.participant_dir.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    model = args.model.expanduser().resolve(strict=True)
    if not participant_dir.is_dir() or not model.is_file():
        raise ValueError("Participant directory and MediaPipe model must exist.")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")
    if args.roi_scale <= 0:
        raise ValueError("--roi-scale must be positive.")
    data_root = Path(__file__).resolve().parents[1] / "data"
    if out_dir == data_root or data_root in out_dir.parents:
        raise ValueError("Output directory must be outside the raw data directory.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")
    return participant_dir, out_dir, model


def parse_roi(value: str | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        left, top, right, bottom = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--roi must be four comma-separated integers") from exc
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"ROI must lie inside the {width}x{height} video frame")
    return left, top, right, bottom


def paired_log(video: Path) -> Path | None:
    expected = video.with_name(video.name.replace("vision_video_", "vision_log_", 1)).with_suffix(".csv")
    return expected if expected.is_file() else None


def profile_from_name(video: Path) -> str:
    stem = video.stem
    return stem.split("_", 3)[-1] if stem.startswith("vision_video_") else stem


def judgement(detection_pct: float, median_width_fraction: float | None) -> str:
    if detection_pct >= 90 and median_width_fraction is not None and median_width_fraction >= 0.12:
        return "usable"
    if detection_pct >= 60 and median_width_fraction is not None and median_width_fraction >= 0.08:
        return "borderline"
    return "not_usable_without_review"


def inspect_video(
    video: Path, participant: str, task: str, log: Path | None, args: argparse.Namespace,
    model: Path, results: list[dict[str, object]],
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {video}")
    roi = parse_roi(args.roi, width, height)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE if roi is not None else vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=args.min_detection_confidence,
        min_face_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    sampled = 0
    detected = 0
    widths: list[float] = []
    heights: list[float] = []
    frame_index = 0
    profile = profile_from_name(video)

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if (frame_index - 1) % args.frame_stride:
                    continue
                sampled += 1
                timestamp_ms = int(round((frame_index - 1) * 1000.0 / fps))
                detection_frame = frame
                if roi is not None:
                    left, top, right, bottom = roi
                    detection_frame = frame[top:bottom, left:right]
                if args.roi_scale != 1.0:
                    detection_frame = cv2.resize(
                        detection_frame, None, fx=args.roi_scale, fy=args.roi_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                detection_height, detection_width = detection_frame.shape[:2]
                rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image) if roi is not None else landmarker.detect_for_video(image, timestamp_ms)
                row: dict[str, object] = {
                    "participant": participant,
                    "task": task,
                    "video_file": str(video),
                    "log_file": str(log) if log else "",
                    "profile": profile,
                    "sampled_frame": frame_index,
                    "video_time_s": round((frame_index - 1) / fps, 3),
                    "face_detected": 0,
                    "bbox_w": "",
                    "bbox_h": "",
                    "face_width_fraction": "",
                    "face_height_fraction": "",
                }
                if result.face_landmarks:
                    landmarks = np.asarray(
                        [[point.x, point.y, point.z] for point in result.face_landmarks[0]],
                        dtype=np.float32,
                    )
                    if landmarks.shape == (LANDMARK_COUNT, 3):
                        features = extract_features(landmarks, detection_width, detection_height)
                        face_width = float(features["bbox_w"]) / detection_width
                        face_height = float(features["bbox_h"]) / detection_height
                        row.update({
                            "face_detected": 1,
                            "bbox_w": int(features["bbox_w"]),
                            "bbox_h": int(features["bbox_h"]),
                            "face_width_fraction": round(face_width, 6),
                            "face_height_fraction": round(face_height, 6),
                        })
                        detected += 1
                        widths.append(face_width)
                        heights.append(face_height)
                results.append(row)
    finally:
        capture.release()

    detection_pct = 100.0 * detected / sampled if sampled else 0.0
    median_width = float(np.median(widths)) if widths else None
    median_height = float(np.median(heights)) if heights else None
    return {
        "participant": participant,
        "task": task,
        "profile": profile,
        "video_file": str(video),
        "vision_log_file": str(log) if log else None,
        "video_fps": fps,
        "video_width_px": width,
        "video_height_px": height,
        "video_total_frames_reported": total_frames,
        "frame_stride": args.frame_stride,
        "roi": args.roi,
        "roi_scale": args.roi_scale,
        "sampled_frames": sampled,
        "detected_frames": detected,
        "detection_pct": round(detection_pct, 3),
        "median_face_width_fraction": round(median_width, 6) if median_width is not None else None,
        "median_face_height_fraction": round(median_height, 6) if median_height is not None else None,
        "usability_judgement": judgement(detection_pct, median_width),
    }


def main() -> int:
    args = parse_args()
    participant_dir, out_dir, model = validate(args)
    videos = sorted(participant_dir.glob("*/vision_video_*.avi"))
    if not videos:
        raise FileNotFoundError(f"No videos found under {participant_dir}")
    participant = participant_dir.name.split("_")[0]
    results: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for video in videos:
        task = video.parent.name
        try:
            summaries.append(inspect_video(video, participant, task, paired_log(video), args, model, results))
        except (RuntimeError, ValueError) as exc:
            errors.append({"video_file": str(video), "error": f"{type(exc).__name__}: {exc}"})

    out_dir.mkdir(parents=True)
    with (out_dir / "sampled_frame_detection.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    (out_dir / "video_usability_summary.json").write_text(
        json.dumps({"videos": summaries, "errors": errors}, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "video_usability_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = list(summaries[0]) if summaries else ["video_file", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Read {len(videos)} raw videos; no raw files were changed.")
    print(f"Derived results: {out_dir}")
    for item in summaries:
        print(f"{item['task']} | {item['profile']} | {item['detection_pct']}% | {item['usability_judgement']}")
    if errors:
        print(f"Videos with errors: {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
