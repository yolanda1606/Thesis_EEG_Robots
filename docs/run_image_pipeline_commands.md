# Image pipeline command guide

This guide explains the commands available in
`processing/multimodal_image/run_image_pipeline.py`.

The pipeline reads files in `data/` but does not change them. All new results
are written under `derived/`. By default a run stops rather than overwriting an
existing folder. Use `--overwrite-run` only to replace the exact requested run
directory after reviewing it.

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
summary, any ICA movement-correlation table, the AutoReject epoch-by-channel
decision graph and its event-aligned CSV, plus a participant-level
event-related spectral-power QC plot and CSV, plus a single Fz
event-related time-frequency figure.

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

`--extract-video-features` makes one face-landmark-derived geometric-feature summary per trial. It
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
| `--overwrite-run` | Explicitly replaces only the exact participant/experiment/mode/run-name directory. It is ignored by `--dry-run`. |
| `--dry-run` | Displays settings, inputs, and intended outputs without creating files. |
| `--validate-only` | Checks selected inputs and trial matching; does not preprocess EEG or video. |
| `--health-check-only` | Runs read-only health checks and writes a health report. |
| `--crop-preview-only` | Creates face-crop preview images only. |
| `--preprocess-eeg` | Runs EEG preparation. |
| `--preprocess-video` | Runs face-landmark processing on the video. Requires `--save-landmarks`. |
| `--extract-eeg-features` | Creates trial-level EEG features. Requires `--preprocess-eeg`. |
| `--extract-video-features` | Creates trial-level face-landmark-derived geometric features. Requires `--preprocess-video`. |
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
  overwriting an existing run folder, unless you explicitly use `--overwrite-run`
  for that exact directory.
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
Directories are created lazily immediately before their first file is written,
so successful runs do not contain empty output branches.

## EEG QC outputs

With `--preprocess-eeg --save-qc`, EEG QC files are written beneath
`eeg/quality_control/`. `autoreject_epoch_channel_log.png` is AutoReject's
standard epoch-by-channel display: good observations, interpolated
observations, bad observations, and rejected epochs. Its accompanying CSV
keeps each pre-rejection epoch index aligned to its trigger ID and records the
AutoReject decision.

`event_related_band_power_by_category.png` is descriptive participant-level QC,
not statistical evidence that emotional category caused an EEG difference. It
uses the complete cleaned event-locked epoch (-0.5 to +2.0 s), computes
time-resolved Morlet power in the shared configured bands, normalizes each
trial/channel/frequency to the -0.5 to 0.0 s baseline as dB
(`10 * log10(power / baseline power)`), and summarizes retained trials in the
HAHV, HALV, LAHV, and LALV trigger categories. The companion CSV contains the
plotted curves. Model EEG features remain deliberately restricted to 0.0 to
2.0 s.

AutoReject's epoch-wise channel interpolation is disabled by default for this
eight-channel montage (`allow_epoch_channel_interpolation: false`). AutoReject
still learns participant-specific thresholds and records bad channel
observations, but affected epochs are dropped instead of being repaired. This
can increase rejected epochs. Separately approved whole-channel interpolation
in the participant configuration remains unchanged; retained epochs otherwise
contain only recorded channels.

`event_related_time_frequency_Fz_by_category.png` is the single requested
time-frequency QC figure. It has four Fz panels (HAHV, LALV, HALV, LAHV), uses
Morlet power from 4 to 30 Hz at 20 log-spaced frequencies with
`n_cycles = frequency / 2`, and uses MNE's `logratio` normalization relative
to the -0.5 to 0.0 s baseline. It is descriptive QC only and does not change
epoch or feature data.
