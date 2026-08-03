# Image pipeline command guide

This guide explains the commands available in
`processing/multimodal_image/run_image_pipeline.py`.

The pipeline reads files in `data/` but does not change them. All new results
are written under `derived/`. Use a new `--run-name` every time: a run will
stop rather than overwrite an existing folder.

## Start in the project folder

```bash
cd /home/sysgen/Projects/Yolanda/Thesis_EEG_Robots
source eeg_venv/bin/activate
```

The examples below use P01 and the Image Experiment. Replace the participant
file and run name when working with another participant.

## Normal configuration used in the examples

```bash
--config processing/multimodal_image/configs/experiments/image_experiment.yaml \
--participant-config processing/multimodal_image/configs/participants/P01.yaml \
--resource-root processing/multimodal_image/models \
--output-root derived
```

The pipeline combines settings in this order:

1. Shared settings in `configs/pipeline/eeg_video_defaults.yaml`.
2. Image Experiment settings given by `--config`.
3. Participant settings given by `--participant-config`.
4. Relevant command-line settings.

## Safe inspection commands

### Show all available options

```bash
python -u processing/multimodal_image/run_image_pipeline.py --help
```

Shows the built-in list of options. It does not read data or create files.

### Check settings and input paths without creating files

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_dry_run_example \
  --dry-run \
  --verbose
```

`--dry-run` displays the resolved settings, selected input files, and intended
output folder. It creates no output folder or files.

### Check whether the main input files match each other

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_validation_example \
  --validate-only \
  --save-qc \
  --verbose
```

`--validate-only` checks the selected files and image-trial matching. It saves
the run settings, input records, validation summary, trigger alignment, and a
log under `derived/.../runs/<run name>/`. It does not preprocess EEG or video.
`--save-qc` has no additional effect in this mode, but is safe to include.

### Run the health checks only

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_health_example \
  --health-check-only \
  --verbose
```

`--health-check-only` checks EEG, ratings, video, frame log, and timing
alignment without preprocessing. It writes its results under
`derived/P01/Image_Experiment/health_checks/<run name>/`.

Do not combine it with crop preview, validation, or processing options.

### Make face-crop preview images

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_crop_preview_example \
  --crop-preview-only \
  --verbose
```

`--crop-preview-only` saves a small set of cropped video images for checking
whether the selected area contains the participant's face. Review these before
running face-landmark processing. The images are saved under
`derived/.../runs/<run name>/video/crop_preview/`.

## Processing commands

These commands create derived data only. They do not change `data/`.

### EEG preprocessing

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_eeg_example \
  --preprocess-eeg \
  --save-clean-epochs \
  --save-qc \
  --verbose
```

`--preprocess-eeg` runs the configured EEG preparation steps. `--save-clean-epochs`
saves the resulting trial segments. `--save-qc` saves the EEG quality-control
summary and any ICA movement-correlation table.

### EEG preprocessing and EEG features

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_eeg_features_example \
  --preprocess-eeg \
  --extract-eeg-features \
  --save-clean-epochs \
  --save-qc \
  --verbose
```

`--extract-eeg-features` creates one set of configured EEG measurements per
trial. It requires `--preprocess-eeg` in the same command.

### Video landmark processing

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_video_example \
  --preprocess-video \
  --save-landmarks \
  --verbose
```

`--preprocess-video` detects face landmarks in the configured video area and
writes frame-level face measurements. It requires `--save-landmarks`, which
saves the detailed landmark file.

### Video processing and trial-level face features

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_video_features_example \
  --preprocess-video \
  --extract-video-features \
  --save-landmarks \
  --verbose
```

`--extract-video-features` makes one face-measurement summary per trial. It
requires `--preprocess-video` in the same command.

### Full multimodal run

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root derived \
  --run-name p01_full_example \
  --preprocess-eeg \
  --preprocess-video \
  --extract-eeg-features \
  --extract-video-features \
  --merge-modalities \
  --save-clean-epochs \
  --save-landmarks \
  --save-qc \
  --verbose
```

This is the complete run: it prepares EEG and video, saves the optional
detailed files, calculates trial-level EEG and face measurements, and combines
them with the ratings in one trial-level table.

## Meaning of every option

| Option | Meaning |
|---|---|
| `--config PATH` | Required. The experiment settings file. Older one-file settings files are also accepted. |
| `--participant-config PATH` | Participant-specific settings. Use this with the three-file setup. |
| `--pipeline-config PATH` | Shared settings file. Normally leave it at its default location. |
| `--participant ID` | Optional check that the participant ID agrees with the participant settings file. |
| `--experiment NAME` | Optional check that the experiment name agrees with the experiment settings file. |
| `--resource-root PATH` | Folder containing the face-landmark model. This is separate from scientific settings so it can differ between computers. |
| `--output-root PATH` | Top-level folder for new derived results. Default: `derived`. It must not be inside `data/`. |
| `--run-name NAME` | Required name for this run's output folder. It must be new. |
| `--dry-run` | Displays settings, inputs, and intended outputs without creating files. |
| `--validate-only` | Checks selected inputs and trial matching; does not preprocess EEG or video. |
| `--health-check-only` | Runs read-only health checks and writes a health report. |
| `--crop-preview-only` | Creates face-crop preview images only. |
| `--preprocess-eeg` | Runs EEG preparation. |
| `--preprocess-video` | Runs face-landmark processing on the video. Requires `--save-landmarks`. |
| `--extract-eeg-features` | Creates trial-level EEG features. Requires `--preprocess-eeg`. |
| `--extract-video-features` | Creates trial-level face features. Requires `--preprocess-video`. |
| `--merge-modalities` | Combines ratings with any EEG and face features created in that same run. |
| `--save-clean-epochs` | Saves prepared EEG trial segments. Used with `--preprocess-eeg`. |
| `--save-landmarks` | Saves detailed face-landmark data. Required with `--preprocess-video`. |
| `--save-qc` | Saves EEG quality-control files when EEG preprocessing is run. |
| `--log-level DEBUG/INFO/WARNING` | Sets how much information appears in the terminal. Default: `INFO`. |
| `--verbose` | Same practical effect as choosing the most detailed `DEBUG` log level. |
| `-h` or `--help` | Shows the built-in option list and exits. |

## Important option rules

- Use only one of `--health-check-only`, `--crop-preview-only`, and
  `--validate-only` at a time.
- `--health-check-only` cannot be combined with any processing option.
- `--extract-eeg-features` needs `--preprocess-eeg`.
- `--extract-video-features` needs `--preprocess-video`.
- `--preprocess-video` needs `--save-landmarks`.
- `--merge-modalities` can be used with EEG features, video features, or both.
  It combines only results made in the current run.
- A run name cannot be reused. Choose a new name instead of deleting or
  overwriting an existing run folder.
- The pipeline protects raw data by refusing an output folder inside `data/`.

## Where results go

Most runs write to:

```text
derived/<participant>/Image_Experiment/runs/<run name>/
```

Health-check runs write to:

```text
derived/<participant>/Image_Experiment/health_checks/<run name>/
```

Each created run includes the resolved settings, input manifest, run manifest,
validation summary, and log. Processing runs can additionally contain EEG,
video, timing-alignment, and merged-data folders according to the options used.
