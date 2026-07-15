# P01 Image Baseline Face-Feature Pipeline

## Purpose

This offline pipeline extracts facial landmarks and simple geometric features from the first participant's Image Experiment emotional-baseline video. It is non-destructive: the raw AVI and original vision log CSV are opened only for reading, while all generated data is stored in a separate `analysis_outputs` directory.

The implementation is in `analysis/process_image_baseline_face_features.py`. It uses OpenCV for video input, annotation, and output, and MediaPipe Face Landmarker for single-face tracking and landmark estimation.

## Processing flow

For every video frame, the pipeline:

1. Reads the corresponding vision-log record, matched primarily by `Frame_Count`.
2. Runs MediaPipe Face Landmarker in video mode with tracking enabled.
3. Stores 478 normalized 3D face landmarks when a face is detected.
4. Calculates a face bounding box and selected facial-geometry features.
5. Draws landmarks, facial contours, selected key points, the bounding box, frame number, and trigger on the overlay frame.
6. Marks failed detections with `NO FACE DETECTED` and stores `NaN` landmarks for that frame.

The script refuses to overwrite completed outputs. Files are first written with temporary names and are moved to their final names only after successful processing.

## Selected features

The feature CSV contains participant and experiment identifiers, synchronized log values, a detection flag, a landmark-derived bounding box, and the following geometry features:

| Feature | Description |
|---|---|
| `eye_distance_px` | Pixel distance between the outer corners of the two eyes. |
| `mouth_open_norm` | Distance between the upper and lower inner-lip points, divided by eye distance. |
| `mouth_width_norm` | Distance between the left and right mouth corners, divided by eye distance. |
| `nose_chin_norm` | Distance from the nose tip to the chin, divided by eye distance. |
| `left_brow_eye_norm` | Distance between the selected left eyebrow and upper-eye points, divided by eye distance. |
| `right_brow_eye_norm` | Distance between the selected right eyebrow and upper-eye points, divided by eye distance. |
| `irisdo_px` | Mean of the left and right vertical eyelid-opening distances, used as an approximate iris-diameter/open-eye measure. |
| `eso_px` | Distance between the left and right iris centers (eye separation). |
| `enso_px` | Distance from the midpoint between the iris centers to the subnasale approximation (eye–nose separation). |
| `mnso_px` | Distance from the upper-lip center to the subnasale approximation (mouth–nose separation). |
| `mwo_px` | Distance between the left and right mouth corners (mouth width). |

Normalization by eye distance reduces the influence of face size and camera distance. These measurements are simple geometry descriptors intended as inputs for later emotional-baseline modelling; they are not direct emotion classifications.

The five acronym-based measurements are stored in pixels to follow their distance-based definitions. Their exact MediaPipe landmark implementation is:

| Measurement | MediaPipe landmarks and calculation |
|---|---|
| `IRISDO` | Mean of eyelid distances 159–145 and 386–374. |
| `ESO` | Iris-center distance 468–473. |
| `ENSO` | Distance from the midpoint of 468 and 473 to landmark 2. |
| `MNSO` | Distance from upper-lip center 0 to landmark 2. |
| `MWO` | Mouth-corner distance 61–291. |

MediaPipe does not provide a landmark explicitly named *subnasale*. Landmark 2 is therefore used as the closest practical approximation to the point below the nostrils. `IRISDO` is defined as the bilateral mean because the source definition describes a single approximate iris-diameter measurement. These implementation choices should be stated when comparing results with work based on manually annotated anthropometric landmarks.

The CSV also includes `participant`, `profile`, `frame_count`, `timestamp`, `experiment_time`, `trigger`, `face_detected`, and the bounding-box fields `bbox_x`, `bbox_y`, `bbox_w`, and `bbox_h`.

## Landmark data

The compressed NPZ stores:

- `landmarks`: array shaped `(frames, 478, 3)`, ordered as normalized `x`, normalized `y`, and normalized `z` coordinates.
- `frame_count`: one-based video frame numbers.
- `participant` and `profile`: run identifiers.
- `coordinate_order`: labels describing the three landmark dimensions.

Frames without a detected face contain `NaN` values in the landmark array. The CSV geometry fields for those frames are left empty and `face_detected` is set to `0`.

## P01 run summary

The pipeline processed the P01 Image Experiment baseline video successfully:

- Total frames: 22,754
- Face detected: 22,722 frames (99.86%)
- Face not detected: 32 frames (0.14%)
- Overlay video: 640 × 480 pixels at 30 FPS

Generated outputs are located in:

`data/P01_2026-05-26/Image Experiment/analysis_outputs/`

- `overlay_face_mesh_IMAGE_BASELINE.avi`
- `face_features_IMAGE_BASELINE.csv`
- `face_landmarks_IMAGE_BASELINE.npz`

The 32 failed detections are retained explicitly rather than removed, preserving alignment between the video, feature table, landmark array, and original vision log.
