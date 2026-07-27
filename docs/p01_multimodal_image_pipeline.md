# P01 Image Experiment multimodal pipeline

## Purpose

This pipeline creates a derived, trial-level P01 Image Experiment dataset from read-only EEG, video, frame-log, and ratings files. It never writes inside `data/`.

The first completed run is:

```text
derived/P01/Image_Experiment/runs/p01_image_initial_v2/
```

It contains 120 rating/video trials and 119 retained EEG trials. EEG trigger 326 (`Present 1.jpg`, `LAHV`) was removed by AutoReject and remains as a row with missing EEG features in the merged table.

## Pipeline files and responsibilities

| File | Role |
|---|---|
| `processing/multimodal_image/configs/p01_image_initial.yaml` | P01 input paths, electrode mapping, event codes, EEG settings, feature bands, and approved crop coordinates. |
| `processing/multimodal_image/run_image_pipeline.py` | Command-line entry point. Creates one new derived run directory, records the command and software versions, and coordinates each stage. |
| `src/validation.py` | Checks required inputs, ratings columns, unique image triggers, EEG header/events, video metadata, and the 120 shared trial codes. Creates input checksums. |
| `src/alignment.py` | Matches image triggers between EEG and video logs. Saves per-trigger onset pairs and the video-to-EEG offset/drift model. |
| `src/eeg.py` | Applies the approved EEG sequence: channel mapping, common-average reference, fourth-order 0.1–40 Hz Butterworth IIR filter, image epochs, ICA/motion review, optional approved channel interpolation, and AutoReject. |
| `src/features.py` | Extracts per-trial/per-channel EEG standard deviation, Shannon entropy, band-specific entropy, Hjorth mobility/complexity, median frequency, and delta/theta/alpha/beta/gamma band power. Gamma is 30–40 Hz. |
| `src/video.py` | Creates crop previews; processes only the 2-second image window; preserves frame timestamps and landmarks; calculates normalised `IRISDO`, `ESO`, `ENSO`, `MNSO`, and `MWO`; creates per-trial summaries. |
| `src/export.py` | Writes CSV and JSON tables without overwriting existing files. |
| `src/configuration.py` | Strictly merges shared, Image Experiment, and participant YAML files. Unknown keys are errors. |
| `src/health.py` | Performs read-only EEG, ratings, video-log, and synchronization health checks. It does not preprocess EEG or extract features. |

## Configuration files

New runs use three layered files:

```text
configs/pipeline/eeg_video_defaults.yaml
    -> configs/experiments/image_experiment.yaml
    -> configs/participants/P01.yaml
```

The pipeline prints these sources at startup and writes the fully resolved YAML
to `config/resolved_configuration.yaml` in every new derived run. The earlier
single P01 YAML remains temporarily supported with a visible deprecation warning.

## P01 processing sequence

```text
Read-only raw inputs
  ├── Ratings and triggers ──> validation and trial identifiers
  ├── EEG BDF ───────────────> CAR -> 0.1–40 Hz IIR -> epochs -> ICA review
  │                              -> AutoReject -> cleaned epochs -> EEG features
  └── Video and frame log ───> approved crop -> landmarks -> facial features
                                      ↓
                         trigger-based EEG/video alignment
                                      ↓
                           120-row merged trial dataset
```

## Main outputs

All files below are inside `derived/P01/Image_Experiment/runs/p01_image_initial_v2/`.

| Output | Contents |
|---|---|
| `manifest/run_manifest.json` | Command, configuration, package versions, and run details. |
| `manifest/input_manifest.json` | Input paths, sizes, and SHA-256 checksums. |
| `manifest/validation_summary.json` | Counts of EEG, ratings, video, and matching image triggers. |
| `eeg/cleaned_epochs/p01_image_cleaned-epo.fif` | 119 retained clean EEG epochs. |
| `eeg/features/eeg_epoch_features.csv` | 952 rows: 119 epochs × 8 channels. |
| `eeg/quality_control/eeg_qc.json` | Retained/rejected epochs, ICA decision, and interpolation status. |
| `eeg/quality_control/ica_motion_correlation.csv` | Motion correlations for eight ICA components. |
| `video/landmarks/p01_image_landmarks.npz` | Derived normalised 3D landmarks, selected frame numbers, and trigger codes. |
| `video/features/video_frame_features.csv` | 7,205 image-window frames and frame-level facial values. |
| `video/features/video_trial_features.csv` | 120 rows with face-detection rate and facial mean/standard deviation values. |
| `alignment/trigger_alignment.csv` | EEG/video onset pair for every image trigger. |
| `alignment/alignment_model.json` | Offset and drift summary. |
| `merged/p01_image_trial_dataset.csv` | Ratings, stimulus details, EEG features, and facial features in one 120-row table. |
| `logs/pipeline.log` | Stage-level processing log. |

## How to repeat this for another participant

Do not reuse the P01 configuration unchanged.

1. Copy `p01_image_initial.yaml` to a new participant-specific configuration file.
2. Change `participant`, `participant_dir`, and the EEG, ratings, video, and frame-log filenames to that participant's actual files.
3. Set `video.crop.approved: false` and enter a provisional participant crop. Each participant needs their own crop review.
4. Run a dry run. It creates no files.
5. Run validation only. Confirm the expected trial count and unique shared triggers.
6. Run `--crop-preview-only`; review the beginning, middle, and end crop images.
7. Record the approved crop in that participant's configuration by setting `approved: true`.
8. Run the complete pipeline with a new descriptive `--run-name`.
9. Review the log, EEG QC, trial-level video table, alignment table, and merged table before using the data.

Example dry run for a future participant:

```bash
cd /home/sysgen/Projects/Yolanda/Thesis_EEG_Robots
source eeg_venv/bin/activate

python -u processing/multimodal_image/run_image_pipeline.py \
  --participant P02 \
  --experiment image \
  --config processing/multimodal_image/configs/p02_image_initial.yaml \
  --output-root derived \
  --run-name p02_image_dry_run \
  --dry-run \
  --verbose
```

The subsequent crop-preview, validation, and full commands use the same arguments, with the relevant stage flags and a new run name each time. Existing run directories are never overwritten.

## Current limitations and decisions needed before wider use

- P01 video detection was 100% in the selected image windows. This does not prove that the same crop or detection rate will work for another participant.
- The P01 run produced an MNE warning that ICA was fitted on baseline-corrected epochs and that eight components may be numerically unstable. The P01 result is a useful technical run, but the ICA fitting rule should be reviewed before treating the cleaned EEG as final.
- The pipeline does not include asymmetry, false-nearest-neighbour, diffuse-slowing, spike, burst, or suppression features. They remain deferred.
- No channel was approved for interpolation in P01. The pipeline only interpolates channels explicitly listed in the participant configuration after quality review.
- The merged table intentionally retains all 120 rating/video rows. Use an explicit EEG-valid field or the missing EEG feature values to exclude the one EEG-rejected trial in EEG-based analysis.

## Workspace cleanup decision

The active landmark model is stored with the pipeline at:

```text
processing/multimodal_image/models/face_landmarker.task
```

The old face tools are retained under `processing/legacy_face_tools/`. Their
old P01 outputs are retained under `derived/P01/legacy_face_usability/`.

The active pipeline does not import the retained legacy tools. Legacy source
scripts are under `processing/legacy_image_tools/`; the legacy feature dataset
is under `derived/P01/legacy_image_experiment/`. Duplicate P01 acquisition
copies were removed only after byte-for-byte comparison with `data/`.
