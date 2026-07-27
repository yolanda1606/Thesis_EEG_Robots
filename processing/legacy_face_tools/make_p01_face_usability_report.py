#!/usr/bin/env python3
"""Create a plain-language P01 face-usability report outside raw data."""

from __future__ import annotations

import argparse
import csv
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

from process_image_baseline_face_features import LANDMARK_COUNT


TASK_ORDER = [
    "Image Experiment",
    "Pick and Place",
    "Shape Sorter Observation",
    "Stack",
    "Sisyphus",
    "Shape Sorter Interaction",
    "Shape Sorter Alone",
]
ROI = (80, 100, 330, 360)
ROI_SCALE = 3.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detection-csv", type=Path,
        default=root / "analysis" / "outputs" / "P01_face_usability_cropped" / "sampled_frame_detection.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=root / "analysis" / "outputs" / "P01_face_usability_report",
    )
    parser.add_argument(
        "--model", type=Path,
        default=root / "processing" / "multimodal_image" / "models" / "face_landmarker.task",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def detected_rows(rows: list[dict[str, str]], task: str, video: str) -> list[dict[str, str]]:
    return [row for row in rows if row["task"] == task and row["video_file"] == video and row["face_detected"] == "1"]


def coefficient_of_variation(values: list[float]) -> float | None:
    if not values:
        return None
    mean = float(np.mean(values))
    return 100.0 * float(np.std(values)) / mean if mean else None


def annotate_frame(video: Path, frame_number: int, model: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video}")
    left, top, right, bottom = ROI
    crop = frame[top:bottom, left:right]
    enlarged = cv2.resize(crop, None, fx=ROI_SCALE, fy=ROI_SCALE, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        raise RuntimeError(f"No face landmarks found in selected frame from {video}")
    landmarks = result.face_landmarks[0]
    if len(landmarks) != LANDMARK_COUNT:
        raise RuntimeError("Unexpected landmark count")
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 180, 0), 2)
    for point in landmarks:
        x = int(round(left + point.x * (right - left)))
        y = int(round(top + point.y * (bottom - top)))
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x, y), 1, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, "P01 face-landmark example", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Orange: crop used before detection | Green: landmarks", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Could not write annotation: {destination}")


def main() -> int:
    args = parse_args()
    csv_path = args.detection_csv.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    model = args.model.expanduser().resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")
    rows = read_rows(csv_path)
    examples = out_dir / "examples"
    examples.mkdir(parents=True)
    summaries: list[dict[str, object]] = []

    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        video = task_rows[0]["video_file"]
        run_rows = [row for row in task_rows if row["video_file"] == video]
        found = detected_rows(rows, task, video)
        width_values = [float(row["face_width_fraction"]) for row in found]
        height_values = [float(row["face_height_fraction"]) for row in found]
        detected_pct = 100.0 * len(found) / len(run_rows) if run_rows else 0.0
        example = found[len(found) // 2] if found else None
        image_name = task.lower().replace(" ", "_") + ".jpg"
        image_path = examples / image_name
        if example is not None:
            annotate_frame(Path(video), int(example["sampled_frame"]), model, image_path)
        summaries.append({
            "task": task,
            "profile": run_rows[0]["profile"],
            "sampled_frames": len(run_rows),
            "usable_frames": len(found),
            "usable_pct": detected_pct,
            "width_cv_pct": coefficient_of_variation(width_values),
            "height_cv_pct": coefficient_of_variation(height_values),
            "example_image": f"examples/{image_name}" if example is not None else None,
        })

    report = [
        "# P01 facial-landmark usability report",
        "",
        "## Plain summary",
        "",
        "P01's face can be detected in all seven task types when the participant side of the image is cropped and enlarged before landmark detection. The full camera image was not suitable for this detector, even though the face is visible to a person viewing the video.",
        "",
        "The percentages below describe sampled video frames, not every original frame. The source usability check sampled every tenth frame (about three frames per second). No raw file was changed.",
        "",
        "## How to read stability",
        "",
        "Stability is based on how much the detected face width and height change across usable sampled frames. Smaller percentages mean a steadier face size in the image. This is a simple camera-placement check, not a measure of facial expression.",
        "",
        "| Experiment | Usable sampled frames | Face-size stability | Result |",
        "|---|---:|---|---|",
    ]
    for item in summaries:
        stability = "not available"
        if item["width_cv_pct"] is not None:
            stability = f"width {item['width_cv_pct']:.1f}%; height {item['height_cv_pct']:.1f}%"
        report.append(
            f"| {item['task']} ({item['profile']}) | {item['usable_frames']}/{item['sampled_frames']} ({item['usable_pct']:.1f}%) | {stability} | usable with the P01 crop |"
        )
    report.extend([
        "",
        "## Example detected frames",
        "",
        "The orange rectangle is the fixed P01 crop. Green dots are detected face landmarks. These pictures are new derived files stored outside `data/`.",
        "",
    ])
    for item in summaries:
        report.extend([f"### {item['task']}", "", f"![Detected landmarks]({item['example_image']})", ""])
    report.extend([
        "## What this means",
        "",
        "- The P01 robot videos are suitable for a careful landmark-based test after cropping.",
        "- The crop is specific to P01 and must not be assumed to work for other participants without their own check.",
        "- Stack had two sampled frames without a detected face in the three-frames-per-second check. This should be retained as missing landmark data, not filled in.",
        "- The short 13-frame Sisyphus repeat is not used as the task example. The earliest longer Sisyphus recording was used because it is the closest video to the main Sisyphus EEG recording start.",
        "",
        "## Proposed processing path",
        "",
        "```text",
        "Raw video and vision log (read only)",
        "        ↓",
        "Choose the approved recording run",
        "        ↓",
        "Crop the participant area and enlarge it",
        "        ↓",
        "Convert video colour from BGR to RGB",
        "        ↓",
        "Detect face landmarks",
        "        ↓",
        "Keep detection status and mark failed frames as missing",
        "        ↓",
        "Check face-size stability and remove only clearly unusable windows",
        "        ↓",
        "Calculate approved landmark or facial-geometry features",
        "        ↓",
        "Aggregate features into planned time windows",
        "        ↓",
        "Link to triggers, robot condition, ratings, and EEG only after synchronization checks",
        "```",
        "",
        "## Important limits",
        "",
        "- Landmark detection does not identify a person's emotional state by itself.",
        "- The robot ratings are condition-level ratings, so they cannot label individual robot events or frames as happy, sad, stressed, or similar states.",
        "- Full facial processing should be run only after the crop rule, run-selection rule, and feature set are approved.",
    ])
    (out_dir / "P01_face_usability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (out_dir / "P01_face_usability_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = list(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote report and {len(summaries)} annotated examples to {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
